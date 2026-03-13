package optimizers

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
)

// gepaEvaluationCase captures one example-level evaluation. Future reflective
// mutation can reuse these records to construct failure-focused minibatches.
type gepaEvaluationCase struct {
	Example core.Example
	Outputs map[string]interface{}
	Score   float64
	Err     error
}

type gepaCachedEvaluationCase struct {
	Outputs map[string]interface{}
	Score   float64
	Err     error
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
	metric      core.Metric
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
		baseProgram: a.baseProgram,
		batch:       batch,
		metric:      a.metric,
	}
}

func (g *GEPA) newEvaluationAdapter(program core.Program, dataset core.Dataset, metric core.Metric) *gepaEvaluationAdapter {
	return &gepaEvaluationAdapter{
		baseProgram: program,
		batch:       g.materializeEvaluationBatch(dataset),
		metric:      metric,
	}
}

func (g *GEPA) newEvaluationAdapterForExamples(program core.Program, examples []core.Example, metric core.Metric) *gepaEvaluationAdapter {
	return &gepaEvaluationAdapter{
		baseProgram: program,
		batch:       cloneEvaluationExamples(examples),
		metric:      metric,
	}
}

func newGEPAExampleDataset(examples []core.Example) core.Dataset {
	return datasets.NewSimpleDataset(examples)
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

	modifiedProgram := g.applyCandidate(adapter.baseProgram, candidate)
	result := &gepaCandidateEvaluation{
		Candidate: candidate,
		Cases:     make([]gepaEvaluationCase, 0, len(adapter.batch)),
	}

	if len(adapter.batch) == 0 || adapter.metric == nil {
		return result
	}

	candidateCtx := context.WithValue(ctx, candidateIDKey, candidate.ID)
	totalScore := 0.0
	scoredCases := 0

	for _, example := range adapter.batch {
		evalCase := gepaEvaluationCase{
			Example: example,
		}
		cacheKey := evaluationCaseCacheKey(candidate, example)
		if g != nil && g.state != nil {
			if cached := g.state.GetEvaluationCaseCache(cacheKey); cached != nil {
				evalCase.Outputs = cloneStringAnyMap(cached.Outputs)
				evalCase.Err = cached.Err
				evalCase.Score = cached.Score
			} else {
				outputs, err := modifiedProgram.Execute(candidateCtx, example.Inputs)
				evalCase.Outputs = outputs
				evalCase.Err = err
				if err == nil {
					evalCase.Score = adapter.metric(example.Outputs, outputs)
				}
				g.state.UpsertEvaluationCaseCache(cacheKey, &gepaCachedEvaluationCase{
					Outputs: outputs,
					Score:   evalCase.Score,
					Err:     err,
				})
			}
		} else {
			outputs, err := modifiedProgram.Execute(candidateCtx, example.Inputs)
			evalCase.Outputs = outputs
			evalCase.Err = err
			if err == nil {
				evalCase.Score = adapter.metric(example.Outputs, outputs)
			}
		}
		if evalCase.Err == nil {
			totalScore += evalCase.Score
			scoredCases++
		}

		result.Cases = append(result.Cases, evalCase)
	}

	if scoredCases > 0 {
		result.TotalScore = totalScore
		result.AverageScore = totalScore / float64(scoredCases)
	}

	return result
}

func cloneCachedEvaluationCase(cached *gepaCachedEvaluationCase) *gepaCachedEvaluationCase {
	if cached == nil {
		return nil
	}

	return &gepaCachedEvaluationCase{
		Outputs: cloneStringAnyMap(cached.Outputs),
		Score:   cached.Score,
		Err:     cached.Err,
	}
}

func evaluationCaseCacheKey(candidate *GEPACandidate, example core.Example) string {
	return candidateComponentCacheKey(candidate) + "|" + exampleCacheKey(example)
}

func candidateComponentCacheKey(candidate *GEPACandidate) string {
	return stableHashStringAnyMap(map[string]interface{}{
		"component_texts": cloneCandidateComponentTexts(candidate),
	})
}

func exampleCacheKey(example core.Example) string {
	return stableHashStringAnyMap(map[string]interface{}{
		"inputs":  example.Inputs,
		"outputs": example.Outputs,
	})
}

func stableHashStringAnyMap(value map[string]interface{}) string {
	rendered, err := json.Marshal(value)
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
				Example: cloneEvaluationExample(evalCase.Example),
				Outputs: cloneStringAnyMap(evalCase.Outputs),
				Score:   evalCase.Score,
				Err:     evalCase.Err,
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

func cloneStringAnyMap(values map[string]interface{}) map[string]interface{} {
	if len(values) == 0 {
		return nil
	}

	cloned := make(map[string]interface{}, len(values))
	for key, value := range values {
		// This is intentionally a shallow value copy. Evaluation examples and
		// outputs currently hold scalar/string-like values, and reflection only
		// needs map isolation rather than recursive deep-copying.
		cloned[key] = value
	}

	return cloned
}
