package optimize

import (
	"context"
	"fmt"
)

// HarnessExampleResult records one evaluator outcome for one example.
type HarnessExampleResult struct {
	ExampleID string
	Result    *EvalResult
}

// HarnessRunResult aggregates a deterministic evaluation run.
type HarnessRunResult struct {
	Results           []HarnessExampleResult
	AverageScore      float64
	PassedExamples    int
	FailedExamples    int
	CompletedExamples int
	EvaluationErrors  int
}

// Harness runs an evaluator across a fixed example set while isolating each run
// behind a fresh agent clone.
type Harness struct {
	Evaluator     AgentEvaluator
	PassThreshold float64
}

// Run evaluates each example sequentially using a fresh clone of the base agent.
func (h *Harness) Run(ctx context.Context, baseAgent OptimizableAgent, examples []AgentExample) (*HarnessRunResult, error) {
	if h == nil || h.Evaluator == nil {
		return nil, fmt.Errorf("optimize: nil harness evaluator")
	}
	if baseAgent == nil {
		return nil, fmt.Errorf("optimize: nil base agent")
	}

	result := &HarnessRunResult{
		Results: make([]HarnessExampleResult, 0, len(examples)),
	}

	for _, example := range examples {
		candidateAgent, err := baseAgent.Clone()
		if err != nil {
			return nil, fmt.Errorf("optimize: clone agent for example %q: %w", example.ID, err)
		}

		evalResult, err := h.Evaluator.Evaluate(ctx, candidateAgent, example)
		if err != nil {
			result.EvaluationErrors++
			evalResult = evaluationFailureResult(err)
		}
		if evalResult == nil {
			result.EvaluationErrors++
			evalResult = evaluationFailureResult(fmt.Errorf("nil evaluation result"))
		}

		result.Results = append(result.Results, HarnessExampleResult{
			ExampleID: example.ID,
			Result:    evalResult,
		})
		result.CompletedExamples++
		result.AverageScore += evalResult.Score
		if evalResult.Score >= h.passThreshold() {
			result.PassedExamples++
		} else {
			result.FailedExamples++
		}
	}

	if result.CompletedExamples > 0 {
		result.AverageScore /= float64(result.CompletedExamples)
	}

	return result, nil
}

func (h *Harness) passThreshold() float64 {
	if h == nil || h.PassThreshold <= 0 {
		return 1.0
	}

	return h.PassThreshold
}

func evaluationFailureResult(err error) *EvalResult {
	return &EvalResult{
		Score: 0,
		SideInfo: &SideInfo{
			Diagnostics: map[string]interface{}{
				"evaluation_error": err.Error(),
			},
			Scores: map[string]float64{
				"evaluation_success": 0,
			},
		},
	}
}
