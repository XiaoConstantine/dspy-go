package optimizers

import (
	"context"

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

		outputs, err := modifiedProgram.Execute(candidateCtx, example.Inputs)
		evalCase.Outputs = outputs
		evalCase.Err = err
		if err == nil {
			evalCase.Score = adapter.metric(example.Outputs, outputs)
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
