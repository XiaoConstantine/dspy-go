package optimizers

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
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

func (g *GEPA) materializeEvaluationBatch(dataset core.Dataset) []core.Example {
	if dataset == nil {
		return nil
	}

	// Preserve existing GEPA semantics: a non-positive batch size means
	// "materialize the full dataset" instead of truncating the batch.
	limit := g.config.EvaluationBatchSize
	batchCapacity := limit
	if batchCapacity < 0 {
		batchCapacity = 0
	}

	batch := make([]core.Example, 0, batchCapacity)
	dataset.Reset()
	for {
		example, hasNext := dataset.Next()
		if !hasNext {
			break
		}

		batch = append(batch, example)
		if limit > 0 && len(batch) >= limit {
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
		result.AverageScore = totalScore / float64(scoredCases)
	}

	return result
}
