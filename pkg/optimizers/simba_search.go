package optimizers

import (
	"context"
	"fmt"
	"maps"
	"math"
	"reflect"
	"slices"
	"sort"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// SIMBAPredictorTrace records one successful predictor invocation, not the
// enclosing program's inputs or outputs. Name is stable across program clones.
type SIMBAPredictorTrace struct {
	Name    string
	Inputs  map[string]any
	Outputs map[string]any
}

type simbaPoolEntry struct {
	program   core.Program
	scores    []float64
	fullScore *float64
}

func (p simbaPoolEntry) score() float64 {
	// A known full-set result takes precedence over noisy minibatches. In
	// particular, an overfitting minibatch winner cannot displace the incumbent
	// as the greedy parent after it failed full-set validation.
	if p.fullScore != nil {
		return *p.fullScore
	}
	var total float64
	for _, score := range p.scores {
		total += score
	}
	return total / float64(len(p.scores))
}

type simbaBucket struct {
	best, worst              Trajectory
	gap, maxScore, maxAvgGap float64
}

type simbaSearch struct {
	pool    []simbaPoolEntry // Entry zero is always the original baseline.
	buckets []simbaBucket
}

// topPrograms is DSPy's top_k_plus_baseline: reserve a slot for the baseline
// even when all k highest-scoring programs beat it. Invalid entries stay in
// history but cannot be sampled while a valid program exists.
func (s *SIMBA) topPrograms() []int {
	indices := make([]int, len(s.search.pool))
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(i, j int) bool {
		return simbaPoolScore(s.search.pool[indices[i]]) > simbaPoolScore(s.search.pool[indices[j]])
	})
	indices = indices[:min(len(indices), s.config.NumCandidates)]
	if len(indices) > 0 && !slices.Contains(indices, 0) {
		indices[len(indices)-1] = 0
	}
	return indices
}

func simbaPoolScore(entry simbaPoolEntry) float64 {
	score := entry.score()
	if math.IsNaN(score) {
		return math.Inf(-1)
	}
	return score
}

func (s *SIMBA) sampleSource(fallback core.Program) core.Program {
	if s.search == nil {
		return fallback
	}
	var indices []int
	var scores []float64
	for _, i := range s.topPrograms() {
		score := simbaPoolScore(s.search.pool[i])
		if math.IsInf(score, 1) {
			return s.search.pool[i].program
		}
		if !math.IsInf(score, -1) {
			indices = append(indices, i)
			scores = append(scores, score)
		}
	}
	if len(indices) == 0 {
		return fallback
	}
	chosen := s.temperatureSample(scores, s.getCurrentTemperature(s.state.CurrentStep))
	return s.search.pool[indices[chosen]].program
}

func (s *SIMBA) rememberCandidates(candidates []core.Program, scores []float64, selected core.Program, fullScore float64) {
	for i := 1; i < len(candidates); i++ {
		entry := simbaPoolEntry{program: candidates[i], scores: []float64{scores[i]}}
		// selectBestCandidate returns the chosen candidate, not a clone. Map
		// identity makes this work even for modules containing function fields.
		if reflect.DeepEqual(candidates[i].Modules, selected.Modules) {
			entry.fullScore = &fullScore
		}
		s.search.pool = append(s.search.pool, entry)
	}
}

// prepareStep gathers evidence on THIS minibatch before any candidate is built.
// Unlike DSPy's repeated stochastic rollouts, evaluate each top program once per
// example: this bounds cost and keeps the comparisons reproducible offline.
func (s *SIMBA) prepareStep(ctx context.Context, examples []core.Example, scorer core.Scorer) ([]core.Example, error) {
	batch, err := s.sampleMiniBatchFromExamples(ctx, examples)
	if err != nil {
		return nil, err
	}
	s.search.buckets = nil
	if s.config.DisableTrajectoryTracking || s.config.StrategyMode == "instruction_only" {
		return batch, nil
	}
	indices := s.topPrograms()
	traces := make([][]Trajectory, len(indices))
	scores := make([]float64, len(indices))
	errs := make([]error, len(indices))
	p := s.createLLMPool()
	for i, index := range indices {
		p.Go(func() {
			scores[i], traces[i], errs[i] = s.evaluateBatch(ctx, s.search.pool[index].program, batch, scorer)
		})
	}
	p.Wait()
	for i, err := range errs {
		if err != nil {
			return nil, err
		}
		index := indices[i]
		s.search.pool[index].scores = append(s.search.pool[index].scores, scores[i])
		for j := range traces[i] {
			traces[i][j].ProgramID = fmt.Sprintf("pool_%d", index)
		}
	}
	for j := range batch {
		var bucket []Trajectory
		for _, programTraces := range traces {
			trajectory := programTraces[j]
			// Execution/scoring failures have no evidence and cannot become demos
			// or an artificial zero that outranks a valid negative score.
			if trajectory.Prediction != nil && !math.IsNaN(trajectory.Score) && !math.IsInf(trajectory.Score, 0) {
				bucket = append(bucket, trajectory)
			}
		}
		if len(bucket) == 0 {
			continue
		}
		sort.SliceStable(bucket, func(a, b int) bool { return bucket[a].Score > bucket[b].Score })
		var total float64
		for _, trajectory := range bucket {
			total += trajectory.Score
		}
		best, worst := bucket[0], bucket[len(bucket)-1]
		s.search.buckets = append(s.search.buckets, simbaBucket{
			best: best, worst: worst, gap: best.Score - worst.Score,
			maxScore: best.Score, maxAvgGap: best.Score - total/float64(len(bucket)),
		})
	}
	sort.SliceStable(s.search.buckets, func(i, j int) bool {
		a, b := s.search.buckets[i], s.search.buckets[j]
		if a.gap != b.gap {
			return a.gap > b.gap
		}
		if a.maxScore != b.maxScore {
			return a.maxScore > b.maxScore
		}
		return a.maxAvgGap > b.maxAvgGap
	})
	return batch, ctx.Err()
}

// simbaPredictors assigns stable names even to nested predictors with identical
// signatures. Alias references to the same predictor receive only one name.
func simbaPredictors(program core.Program) map[string]*modules.Predict {
	result := make(map[string]*modules.Predict)
	seen := make(map[*modules.Predict]bool)
	for _, name := range slices.Sorted(maps.Keys(program.Modules)) {
		for i, predictor := range demoPredictors(program.Modules[name]) {
			if !seen[predictor] {
				result[fmt.Sprintf("%s/%d", name, i)] = predictor
				seen[predictor] = true
			}
		}
	}
	return result
}

// traceForward isolates observations per example; parallel examples never share
// a trace buffer. Record only declared fields so provider metadata is not a demo.
func simbaTraceForward(ctx context.Context, program core.Program, inputs map[string]any) (map[string]any, []SIMBAPredictorTrace, error) {
	names := make(map[*modules.Predict]string)
	for name, predictor := range simbaPredictors(program) {
		names[predictor] = name
	}
	var mu sync.Mutex
	var traces []SIMBAPredictorTrace
	ctx = modules.WithPredictionObserver(ctx, func(p *modules.Predict, inputs, outputs map[string]any) {
		name, ok := names[p]
		if !ok {
			return
		}
		trace := SIMBAPredictorTrace{Name: name, Inputs: make(map[string]any), Outputs: make(map[string]any)}
		for _, field := range p.GetSignature().Inputs {
			value, exists := inputs[field.Name]
			if !exists {
				return
			}
			trace.Inputs[field.Name] = value
		}
		for _, field := range p.GetSignature().Outputs {
			value, exists := outputs[field.Name]
			if !exists {
				return
			}
			trace.Outputs[field.Name] = value
		}
		mu.Lock()
		traces = append(traces, trace)
		mu.Unlock()
	})
	prediction, err := program.Forward(ctx, inputs)
	return prediction, traces, err
}

func (s *SIMBA) limitProposalDemos(program core.Program) {
	if s.config.MaxDemos <= 0 {
		return
	}
	for _, predictor := range simbaPredictors(program) {
		demos := predictor.GetDemos()
		if len(demos) > s.config.MaxDemos {
			predictor.SetDemos(slices.Clone(demos[len(demos)-s.config.MaxDemos:]))
		}
	}
}

func (s *SIMBA) appendDemo(program core.Program, bucket simbaBucket) (core.Program, bool) {
	good := bucket.best
	if s.config.MaxDemos <= 0 || good.Prediction == nil || !(good.Score >= s.config.DemoScoreThreshold) || math.IsInf(good.Score, 0) {
		return program, false
	}
	last := make(map[string]SIMBAPredictorTrace)
	for _, trace := range good.PredictorTraces {
		last[trace.Name] = trace
	}
	variant := program.Clone()
	changed := false
	for name, predictor := range simbaPredictors(variant) {
		trace, ok := last[name]
		if !ok {
			continue
		}
		// Keep the newest demos and make room for exactly one successful trace.
		// Unlike DSPy's Poisson dropping this is a strict per-predictor cap.
		demos := predictor.GetDemos()
		keep := min(len(demos), s.config.MaxDemos-1)
		demos = append(slices.Clone(demos[len(demos)-keep:]), core.Example{
			Inputs: maps.Clone(trace.Inputs), Outputs: maps.Clone(trace.Outputs),
		})
		predictor.SetDemos(demos)
		changed = true
	}
	return variant, changed
}

func simbaFormatPredictorTraces(traces []SIMBAPredictorTrace) string {
	var text string
	for _, trace := range traces {
		text += fmt.Sprintf("Predictor %s:\n  Inputs: %v\n  Outputs: %v\n", trace.Name, trace.Inputs, trace.Outputs)
	}
	return text
}

func (s *SIMBA) generateBucketRuleCandidate(ctx context.Context, parent core.Program, bucket simbaBucket) (core.Program, error) {
	variant := parent.Clone()
	targets := simbaInstructionModules(variant)
	if len(targets) == 0 {
		return variant, nil
	}
	target := targets[s.proposalIntn(ctx, len(targets))]
	name := target.GetModuleType()
	for path, predictor := range simbaPredictors(variant) {
		if target == predictor {
			name = path
			break
		}
	}
	var prompt string
	if bucket.gap > 0 {
		prompt = s.buildTrajectoryAnalysisPrompt([]Trajectory{bucket.best}, []Trajectory{bucket.worst})
	} else if bucket.best.Success {
		prompt = s.buildSuccessPatternAnalysisPrompt([]Trajectory{bucket.best})
	}
	if prompt == "" || s.analyzerModel == nil {
		return s.perturbProgram(ctx, parent)
	}
	prompt += fmt.Sprintf("\n\nTarget predictor: %s\nSignature: %s\nCurrent instruction: %s\nWrite RULE lines only for this predictor's sub-task. Use other predictors' traces as context, not as this predictor's inputs or outputs.",
		name, target.GetSignature().String(), target.GetSignature().Instruction)
	core.RecordModelCall(ctx, s.analyzerModel)
	response, err := s.analyzerModel.Generate(ctx, s.withIntrospection(prompt))
	if err != nil {
		return s.perturbProgram(ctx, parent)
	}
	rules := s.parseRulesFromAnalysis(response.Content)
	if len(rules) == 0 {
		return s.perturbProgram(ctx, parent)
	}
	signature := target.GetSignature()
	signature.Instruction, err = s.appendRuleToInstruction(ctx, signature.Instruction, rules)
	if err != nil {
		return variant, err
	}
	target.SetSignature(signature)
	return variant, nil
}
