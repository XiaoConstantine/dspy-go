package optimizers

import (
	"context"
	"crypto/sha256"
	jsonv1 "encoding/json"
	jsonv2 "encoding/json/v2"
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/sourcegraph/conc/pool"
)

// gepaEvaluationCase captures one example-level evaluation. Future reflective
// mutation can reuse these records to construct failure-focused minibatches.
type gepaEvaluationCase struct {
	Example          core.Example
	Outputs          map[string]any
	Score            float64
	Err              error
	Feedback         string
	FeedbackTarget   string
	FeedbackMetadata map[string]any
	TraceEvidence    []string
}

type gepaCachedEvaluationCase struct {
	Outputs          map[string]any
	Score            float64
	Err              error
	Feedback         string
	FeedbackTarget   string
	FeedbackMetadata map[string]any
	TraceEvidence    []string
}

// gepaCandidateEvaluation captures a candidate evaluation against a stable
// example batch. This is the minimal adapter seam needed for future DSPy-style
// minibatch selection, reflective dataset construction, and acceptance logic.
type gepaCandidateEvaluation struct {
	Candidate    *GEPACandidate
	Cases        []gepaEvaluationCase
	TotalScore   float64
	AverageScore float64
}

type gepaEvaluationAdapter struct {
	baseProgram core.Program
	batch       []core.Example
	scorer      core.Scorer
	// concurrentScoring lets scorer and FeedbackEvaluator calls overlap. It is
	// false for a legacy core.Metric, which was never required to be safe for
	// concurrent use.
	concurrentScoring bool
}

func (a *gepaEvaluationAdapter) subset(caseIndexes []int) *gepaEvaluationAdapter {
	if a == nil {
		return nil
	}
	if len(caseIndexes) == 0 || len(caseIndexes) >= len(a.batch) {
		return a
	}

	batch := make([]core.Example, 0, len(caseIndexes))
	for _, caseIndex := range caseIndexes {
		if caseIndex < 0 || caseIndex >= len(a.batch) {
			continue
		}
		batch = append(batch, cloneEvaluationExample(a.batch[caseIndex]))
	}
	if len(batch) == 0 {
		return a
	}

	return &gepaEvaluationAdapter{
		baseProgram:       a.baseProgram,
		batch:             batch,
		scorer:            a.scorer,
		concurrentScoring: a.concurrentScoring,
	}
}

func (g *GEPA) newEvaluationAdapter(program core.Program, dataset core.Dataset, metric core.Metric) *gepaEvaluationAdapter {
	return &gepaEvaluationAdapter{
		baseProgram: program,
		batch:       g.materializeEvaluationBatch(dataset),
		scorer:      core.MetricScorer(metric),
	}
}

func (g *GEPA) newEvaluationAdapterForExamples(program core.Program, examples []core.Example, metric core.Metric) *gepaEvaluationAdapter {
	return newGEPAScorerAdapter(program, examples, core.MetricScorer(metric), false)
}

func newGEPAScorerAdapter(program core.Program, examples []core.Example, scorer core.Scorer, concurrentScoring bool) *gepaEvaluationAdapter {
	return &gepaEvaluationAdapter{
		baseProgram:       program,
		batch:             cloneEvaluationExamples(examples),
		scorer:            scorer,
		concurrentScoring: concurrentScoring,
	}
}

func newGEPAExampleDataset(examples []core.Example) core.Dataset {
	return datasets.NewSimpleDataset(examples)
}

// evaluationBatchForGeneration visits every training example once per epoch.
// The final batch may be smaller: padding it would overweight repeated cases.
// Deriving the shuffle from the epoch avoids coupling it to proposal RNG calls
// and lets a checkpoint reconstruct its batch without persisting a cursor.
func (g *GEPA) evaluationBatchForGeneration(examples []core.Example, generation int) []core.Example {
	if len(examples) == 0 {
		return nil
	}
	size := max(1, g.config.EvaluationBatchSize)
	batchesPerEpoch := 1 + (len(examples)-1)/size
	generation = max(0, generation)
	epoch := generation / batchesPerEpoch
	seed := g.config.RandomSeed
	if seed == 0 {
		seed = g.state.StartTime.UnixNano()
	}
	order := rand.New(rand.NewSource(seed + int64(epoch))).Perm(len(examples))
	start := (generation % batchesPerEpoch) * size
	end := start + min(size, len(examples)-start)
	batch := make([]core.Example, 0, end-start)
	for _, index := range order[start:end] {
		batch = append(batch, cloneEvaluationExample(examples[index]))
	}
	return batch
}

func (g *GEPA) materializeEvaluationBatch(dataset core.Dataset) []core.Example {
	if dataset == nil {
		return nil
	}

	limit := g.config.EvaluationBatchSize
	if limit <= 0 {
		// Preserve legacy GEPA behavior from the pre-adapter evaluation loop:
		// non-positive batch sizes still consume exactly one example because the
		// old post-increment break condition fired after the first scored case.
		limit = 1
	}

	batch := make([]core.Example, 0, limit)
	dataset.Reset()
	for {
		example, hasNext := dataset.Next()
		if !hasNext {
			break
		}

		batch = append(batch, example)
		if len(batch) >= limit {
			break
		}
	}

	return batch
}

func (g *GEPA) evaluateCandidateWithAdapter(ctx context.Context, candidate *GEPACandidate, adapter *gepaEvaluationAdapter) *gepaCandidateEvaluation {
	if adapter == nil {
		return &gepaCandidateEvaluation{Candidate: candidate}
	}

	if g != nil && g.state != nil && candidate != nil {
		g.state.BeginCandidateCaseRecording(candidate.ID)
	}

	modifiedProgram := g.applyCandidate(adapter.baseProgram, candidate)
	result := &gepaCandidateEvaluation{
		Candidate: candidate,
		Cases:     make([]gepaEvaluationCase, 0, len(adapter.batch)),
	}

	if len(adapter.batch) == 0 || adapter.scorer == nil {
		return result
	}

	candidateCtx := context.WithValue(ctx, candidateIDKey, candidate.ID)
	cases := make([]gepaEvaluationCase, len(adapter.batch))
	ran := make([]bool, len(adapter.batch))
	var spent atomic.Int64

	// Program executions of one candidate run concurrently; results keep batch
	// order, so scores and reflection evidence do not depend on completion order.
	// Scoring and feedback calls overlap only when the adapter allows it.
	var scoring sync.Mutex
	workers := pool.New().WithMaxGoroutines(max(g.config.ConcurrencyLevel, 1))
	for i, example := range adapter.batch {
		workers.Go(func() {
			if ctx.Err() != nil {
				return
			}
			evalCase := gepaEvaluationCase{Example: example}
			cacheKey := evaluationCaseCacheKey(candidate, example)
			if cached := g.state.GetEvaluationCaseCache(cacheKey); cached != nil {
				evalCase.Outputs = cloneStringAnyMap(cached.Outputs)
				evalCase.Err = cached.Err
				evalCase.Score = cached.Score
				evalCase.Feedback = cached.Feedback
				evalCase.FeedbackTarget = cached.FeedbackTarget
				evalCase.FeedbackMetadata = cloneStringAnyMap(cached.FeedbackMetadata)
				evalCase.TraceEvidence = append([]string(nil), cached.TraceEvidence...)
			} else {
				// Capture only traces emitted by this example, including internal
				// module calls whose inputs differ from the program's outer input.
				traces := &gepaCaseTraces{}
				outputs, err := modifiedProgram.Execute(context.WithValue(candidateCtx, caseTracesKey, traces), example.Inputs)
				evalCase.Outputs = outputs
				evalCase.Err = err
				evalCase.TraceEvidence = traces.evidence()
				if !adapter.concurrentScoring {
					scoring.Lock()
				}
				cacheable := true
				if err == nil {
					var scoreErr error
					evalCase.Score, scoreErr = adapter.scorer.Score(ctx, example, outputs)
					switch {
					case scoreErr == nil:
					case ctx.Err() != nil:
						evalCase.Err = scoreErr
					default:
						// An unscorable example counts as zero, so it cannot
						// raise the average the way a dropped case would. It is
						// not cached: the scorer may succeed on a later attempt.
						logging.GetLogger().Warn(ctx, "GEPA: scorer failed: %v", scoreErr)
						evalCase.Score = 0
						cacheable = false
					}
				}
				g.populateEvaluationCaseFeedback(ctx, candidate, &evalCase)
				if !adapter.concurrentScoring {
					scoring.Unlock()
				}
				spent.Add(1)
				if cacheable {
					g.state.UpsertEvaluationCaseCache(cacheKey, &gepaCachedEvaluationCase{
						Outputs:          outputs,
						Score:            evalCase.Score,
						Err:              evalCase.Err,
						Feedback:         evalCase.Feedback,
						FeedbackTarget:   evalCase.FeedbackTarget,
						FeedbackMetadata: evalCase.FeedbackMetadata,
						TraceEvidence:    evalCase.TraceEvidence,
					})
				}
			}
			cases[i], ran[i] = evalCase, true
		})
	}
	workers.Wait()

	for i, evalCase := range cases {
		if !ran[i] {
			continue
		}
		if evalCase.Err == nil {
			result.TotalScore += evalCase.Score
		}
		result.Cases = append(result.Cases, evalCase)
	}
	spentMetricCalls := int(spent.Load())

	// A failed case counts as zero in the average; dropping it would let a
	// candidate that fails most examples outscore one that answers them all.
	if len(result.Cases) > 0 {
		result.AverageScore = result.TotalScore / float64(len(result.Cases))
	}
	if spentMetricCalls > 0 && g != nil && g.state != nil {
		g.state.AddMetricCalls(spentMetricCalls)
	}

	if g != nil && g.state != nil && candidate != nil {
		candidate.Fitness = result.AverageScore
		g.state.FinalizeCandidateCaseFitness(candidate, result.AverageScore)
	}

	return result
}

func (g *GEPA) populateEvaluationCaseFeedback(ctx context.Context, candidate *GEPACandidate, evalCase *gepaEvaluationCase) {
	if evalCase == nil {
		return
	}
	if g == nil || g.config == nil {
		return
	}
	if g.config.FeedbackEvaluator == nil && !g.config.AddFormatFailureAsFeedback {
		return
	}

	var feedback *GEPAFeedback
	if g.config.FeedbackEvaluator != nil {
		feedback = normalizeGEPAFeedback(g.config.FeedbackEvaluator.EvaluateFeedback(
			ctx,
			evalCase.Example.Outputs,
			evalCase.Outputs,
			&GEPAFeedbackContext{
				Candidate: candidate,
				Example:   cloneEvaluationExample(evalCase.Example),
				Outputs:   cloneStringAnyMap(evalCase.Outputs),
				Err:       evalCase.Err,
			},
		))
	}

	if evalCase.Err != nil && g.config.AddFormatFailureAsFeedback {
		target := ""
		if feedback != nil {
			target = feedback.TargetComponent
		}
		if target == "" && candidate != nil {
			target = strings.TrimSpace(candidate.ModuleName)
		}

		failureFeedback := "Execution failure: " + strings.TrimSpace(evalCase.Err.Error())
		switch {
		case feedback == nil:
			feedback = &GEPAFeedback{
				Feedback:        failureFeedback,
				TargetComponent: target,
			}
		case feedback.Feedback == "":
			feedback.Feedback = failureFeedback
			if feedback.TargetComponent == "" {
				feedback.TargetComponent = target
			}
		default:
			feedback.Feedback = strings.TrimSpace(feedback.Feedback + "\n" + failureFeedback)
			if feedback.TargetComponent == "" {
				feedback.TargetComponent = target
			}
		}
	}

	feedback = normalizeGEPAFeedback(feedback)
	if feedback == nil {
		return
	}

	evalCase.Feedback = feedback.Feedback
	evalCase.FeedbackTarget = feedback.TargetComponent
	evalCase.FeedbackMetadata = cloneStringAnyMap(feedback.Metadata)
}

func cloneCachedEvaluationCase(cached *gepaCachedEvaluationCase) *gepaCachedEvaluationCase {
	if cached == nil {
		return nil
	}

	return &gepaCachedEvaluationCase{
		Outputs:          cloneStringAnyMap(cached.Outputs),
		Score:            cached.Score,
		Err:              cached.Err,
		Feedback:         cached.Feedback,
		FeedbackTarget:   cached.FeedbackTarget,
		FeedbackMetadata: cloneStringAnyMap(cached.FeedbackMetadata),
		TraceEvidence:    append([]string(nil), cached.TraceEvidence...),
	}
}

func evaluationCaseCacheKey(candidate *GEPACandidate, example core.Example) string {
	return candidateComponentCacheKey(candidate) + "|" + exampleCacheKey(example)
}

func candidateComponentCacheKey(candidate *GEPACandidate) string {
	return stableHashStringAnyMap(map[string]any{
		"component_texts": cloneCandidateComponentTexts(candidate),
	})
}

func exampleCacheKey(example core.Example) string {
	return stableHashStringAnyMap(map[string]any{
		"inputs":  example.Inputs,
		"outputs": example.Outputs,
	})
}

func stableHashStringAnyMap(value map[string]any) string {
	rendered, err := jsonv2.Marshal(value, jsonv1.DefaultOptionsV1(), jsonv2.Deterministic(true))
	if err != nil {
		return fmt.Sprintf("fallback:%v", value)
	}

	sum := sha256.Sum256(rendered)
	return fmt.Sprintf("%x", sum)
}

func cloneGEPACandidateEvaluation(evaluation *gepaCandidateEvaluation) *gepaCandidateEvaluation {
	if evaluation == nil {
		return nil
	}

	cloned := &gepaCandidateEvaluation{
		TotalScore:   evaluation.TotalScore,
		AverageScore: evaluation.AverageScore,
	}
	if evaluation.Candidate != nil {
		cloned.Candidate = CloneCandidate(evaluation.Candidate)
	}
	if len(evaluation.Cases) > 0 {
		cloned.Cases = make([]gepaEvaluationCase, len(evaluation.Cases))
		for i, evalCase := range evaluation.Cases {
			cloned.Cases[i] = gepaEvaluationCase{
				Example:          cloneEvaluationExample(evalCase.Example),
				Outputs:          cloneStringAnyMap(evalCase.Outputs),
				Score:            evalCase.Score,
				Err:              evalCase.Err,
				Feedback:         evalCase.Feedback,
				FeedbackTarget:   evalCase.FeedbackTarget,
				FeedbackMetadata: cloneStringAnyMap(evalCase.FeedbackMetadata),
				TraceEvidence:    append([]string(nil), evalCase.TraceEvidence...),
			}
		}
	}

	return cloned
}

func cloneEvaluationExamples(examples []core.Example) []core.Example {
	if len(examples) == 0 {
		return nil
	}

	cloned := make([]core.Example, len(examples))
	for i, example := range examples {
		cloned[i] = cloneEvaluationExample(example)
	}
	return cloned
}

func cloneEvaluationExample(example core.Example) core.Example {
	return core.Example{
		Inputs:  cloneStringAnyMap(example.Inputs),
		Outputs: cloneStringAnyMap(example.Outputs),
	}
}

func cloneStringAnyMap(values map[string]any) map[string]any {
	if len(values) == 0 {
		return nil
	}

	cloned := make(map[string]any, len(values))
	for key, value := range values {
		// This is intentionally a shallow value copy. Evaluation examples and
		// outputs currently hold scalar/string-like values, and reflection only
		// needs map isolation rather than recursive deep-copying.
		cloned[key] = value
	}

	return cloned
}
