package optimize

import (
	"context"
	"errors"
	"maps"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockOptimizableAgent struct {
	artifacts    AgentArtifacts
	outputs      map[string]map[string]interface{}
	errs         map[string]error
	lastTrace    *agents.ExecutionTrace
	executeCount int
	memory       agents.Memory
}

func newMockOptimizableAgent() *mockOptimizableAgent {
	return &mockOptimizableAgent{
		artifacts: AgentArtifacts{
			Text: make(map[ArtifactKey]string),
			Int:  make(map[string]int),
			Bool: make(map[string]bool),
		},
		outputs: make(map[string]map[string]interface{}),
		errs:    make(map[string]error),
		memory:  agents.NewInMemoryStore(),
	}
}

func (m *mockOptimizableAgent) Execute(_ context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	m.executeCount++
	key, _ := input["id"].(string)

	output := maps.Clone(m.outputs[key])
	if output == nil {
		output = map[string]interface{}{}
	}
	if _, ok := output["count"]; !ok {
		output["count"] = m.executeCount
	}

	trace := &agents.ExecutionTrace{
		AgentID:        "mock-agent",
		AgentType:      "mock",
		Task:           key,
		Input:          maps.Clone(input),
		Output:         maps.Clone(output),
		Status:         agents.TraceStatusSuccess,
		StartedAt:      time.Now().Add(-25 * time.Millisecond),
		CompletedAt:    time.Now(),
		ProcessingTime: 25 * time.Millisecond,
		TokenUsage: map[string]int64{
			"total": int64(10 + m.executeCount),
		},
		ToolUsageCount: map[string]int{
			"lookup": 1,
		},
		TerminationCause: "finish",
	}

	if err := m.errs[key]; err != nil {
		trace.Status = agents.TraceStatusFailure
		trace.Error = err.Error()
		m.lastTrace = trace
		return output, err
	}

	m.lastTrace = trace
	return output, nil
}

func (m *mockOptimizableAgent) GetCapabilities() []core.Tool { return nil }

func (m *mockOptimizableAgent) GetMemory() agents.Memory { return m.memory }

func (m *mockOptimizableAgent) GetArtifacts() AgentArtifacts { return m.artifacts.Clone() }

func (m *mockOptimizableAgent) SetArtifacts(artifacts AgentArtifacts) error {
	m.artifacts = artifacts.Clone()
	return nil
}

func (m *mockOptimizableAgent) Clone() (OptimizableAgent, error) {
	cloned := newMockOptimizableAgent()
	cloned.artifacts = m.artifacts.Clone()
	cloned.outputs = cloneOutputFixtures(m.outputs)
	cloned.errs = cloneErrorFixtures(m.errs)
	return cloned, nil
}

func (m *mockOptimizableAgent) LastExecutionTrace() *agents.ExecutionTrace {
	return m.lastTrace.Clone()
}

func TestDeterministicEvaluator_Evaluate_ExactMatch(t *testing.T) {
	agent := newMockOptimizableAgent()
	agent.outputs["case-1"] = map[string]interface{}{
		"answer": "42",
	}

	evaluator := NewDeterministicEvaluator(nil)
	result, err := evaluator.Evaluate(context.Background(), agent, AgentExample{
		ID: "case-1",
		Inputs: map[string]interface{}{
			"id": "case-1",
		},
		Outputs: map[string]interface{}{
			"answer": "42",
		},
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.SideInfo)
	require.NotNil(t, result.SideInfo.Trace)

	assert.Equal(t, 1.0, result.Score)
	assert.Equal(t, 1.0, result.SideInfo.Scores["output:answer"])
	assert.Equal(t, 1.0, result.SideInfo.Scores["output_match"])
	assert.Equal(t, 1.0, result.SideInfo.Scores["execution_success"])
	assert.Contains(t, result.SideInfo.PassedTests, "output:answer")
	assert.Empty(t, result.SideInfo.FailedTests)
	assert.Equal(t, "success", result.SideInfo.Diagnostics["trace_status"])
	assert.Equal(t, "finish", result.SideInfo.Diagnostics["termination_cause"])
	assert.Equal(t, map[string]int{"lookup": 1}, result.SideInfo.Diagnostics["tool_usage_count"])
	assert.Equal(t, int64(11), result.SideInfo.Tokens["total"])
	assert.Equal(t, "42", result.SideInfo.Trace.Output["answer"])

	result.SideInfo.Trace.Output["answer"] = "mutated"
	assert.Equal(t, "42", agent.lastTrace.Output["answer"])
}

func TestDeterministicEvaluator_Evaluate_ExecutionErrorReturnsZeroScore(t *testing.T) {
	agent := newMockOptimizableAgent()
	agent.errs["broken"] = errors.New("tool crashed")

	evaluator := NewDeterministicEvaluator(nil)
	result, err := evaluator.Evaluate(context.Background(), agent, AgentExample{
		ID: "broken",
		Inputs: map[string]interface{}{
			"id": "broken",
		},
		Outputs: map[string]interface{}{
			"answer": "never",
		},
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.SideInfo)

	assert.Equal(t, 0.0, result.Score)
	assert.Equal(t, 0.0, result.SideInfo.Scores["execution_success"])
	assert.Equal(t, "tool crashed", result.SideInfo.Diagnostics["execution_error"])
	assert.Equal(t, "failure", result.SideInfo.Diagnostics["trace_status"])
}

func TestDeterministicEvaluator_Evaluate_PartialMatchTracksMismatches(t *testing.T) {
	agent := newMockOptimizableAgent()
	agent.outputs["partial"] = map[string]interface{}{
		"answer": "42",
		"count":  float64(1),
	}

	evaluator := NewDeterministicEvaluator(nil)
	result, err := evaluator.Evaluate(context.Background(), agent, AgentExample{
		ID: "partial",
		Inputs: map[string]interface{}{
			"id": "partial",
		},
		Outputs: map[string]interface{}{
			"answer": "42",
			"count":  1,
			"extra":  "expected",
		},
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.SideInfo)

	assert.InDelta(t, 1.0/3.0, result.Score, 0.000001)
	assert.Equal(t, 1.0, result.SideInfo.Scores["output:answer"])
	assert.Equal(t, 0.0, result.SideInfo.Scores["output:count"])
	assert.Equal(t, 0.0, result.SideInfo.Scores["output:extra"])
	assert.Contains(t, result.SideInfo.PassedTests, "output:answer")
	assert.ElementsMatch(t, []string{"output:count", "output:extra"}, result.SideInfo.FailedTests)

	mismatches, ok := result.SideInfo.Diagnostics["mismatches"].(map[string]interface{})
	require.True(t, ok)
	countMismatch, ok := mismatches["count"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, 1, countMismatch["expected"])
	assert.Equal(t, float64(1), countMismatch["actual"])
	extraMismatch, ok := mismatches["extra"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, true, extraMismatch["missing"])
}

func TestDeterministicEvaluator_Evaluate_ComparatorErrorReturnsZeroScore(t *testing.T) {
	agent := newMockOptimizableAgent()
	evaluator := NewDeterministicEvaluator(OutputComparatorFunc(func(ex AgentExample, actual map[string]interface{}) (*ComparisonResult, error) {
		return nil, errors.New("compare failed")
	}))

	result, err := evaluator.Evaluate(context.Background(), agent, AgentExample{
		ID: "compare-error",
		Inputs: map[string]interface{}{
			"id": "compare-error",
		},
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.SideInfo)

	assert.Equal(t, 0.0, result.Score)
	assert.Equal(t, 1.0, result.SideInfo.Scores["execution_success"])
	assert.Equal(t, 0.0, result.SideInfo.Scores["comparison_success"])
	assert.Equal(t, "compare failed", result.SideInfo.Diagnostics["comparison_error"])
}

func TestHarness_Run_ClonesAgentPerExample(t *testing.T) {
	baseAgent := newMockOptimizableAgent()
	harness := &Harness{
		Evaluator: NewDeterministicEvaluator(nil),
	}

	runResult, err := harness.Run(context.Background(), baseAgent, []AgentExample{
		{
			ID: "first",
			Inputs: map[string]interface{}{
				"id": "first",
			},
			Outputs: map[string]interface{}{
				"count": 1,
			},
		},
		{
			ID: "second",
			Inputs: map[string]interface{}{
				"id": "second",
			},
			Outputs: map[string]interface{}{
				"count": 1,
			},
		},
	})

	require.NoError(t, err)
	require.NotNil(t, runResult)
	require.Len(t, runResult.Results, 2)

	assert.Equal(t, 2, runResult.CompletedExamples)
	assert.Equal(t, 2, runResult.PassedExamples)
	assert.Equal(t, 0, runResult.FailedExamples)
	assert.Equal(t, 1.0, runResult.AverageScore)
	assert.Equal(t, 0, baseAgent.executeCount)
	assert.Equal(t, "first", runResult.Results[0].ExampleID)
	assert.Equal(t, "second", runResult.Results[1].ExampleID)
}

func TestHarness_Run_UsesConfigurablePassThreshold(t *testing.T) {
	baseAgent := newMockOptimizableAgent()
	baseAgent.outputs["threshold"] = map[string]interface{}{
		"a": "match",
		"b": "match",
		"c": "wrong",
	}

	harness := &Harness{
		Evaluator:     NewDeterministicEvaluator(nil),
		PassThreshold: 0.6,
	}

	runResult, err := harness.Run(context.Background(), baseAgent, []AgentExample{
		{
			ID: "threshold",
			Inputs: map[string]interface{}{
				"id": "threshold",
			},
			Outputs: map[string]interface{}{
				"a": "match",
				"b": "match",
				"c": "expected",
			},
		},
	})

	require.NoError(t, err)
	require.NotNil(t, runResult)
	require.Len(t, runResult.Results, 1)

	assert.InDelta(t, 2.0/3.0, runResult.AverageScore, 0.000001)
	assert.Equal(t, 1, runResult.PassedExamples)
	assert.Equal(t, 0, runResult.FailedExamples)
}

func TestHarness_Run_ContinuesAfterEvaluatorError(t *testing.T) {
	baseAgent := newMockOptimizableAgent()
	comparator := OutputComparatorFunc(func(ex AgentExample, actual map[string]interface{}) (*ComparisonResult, error) {
		if ex.ID == "bad" {
			return nil, errors.New("bad comparison")
		}
		return &ComparisonResult{
			Score: 1,
			Scores: map[string]float64{
				"output_match": 1,
			},
		}, nil
	})

	harness := &Harness{
		Evaluator: NewDeterministicEvaluator(comparator),
	}

	runResult, err := harness.Run(context.Background(), baseAgent, []AgentExample{
		{
			ID: "bad",
			Inputs: map[string]interface{}{
				"id": "bad",
			},
		},
		{
			ID: "good",
			Inputs: map[string]interface{}{
				"id": "good",
			},
		},
	})

	require.NoError(t, err)
	require.NotNil(t, runResult)
	require.Len(t, runResult.Results, 2)

	assert.Equal(t, 0, runResult.EvaluationErrors)
	assert.Equal(t, 1, runResult.PassedExamples)
	assert.Equal(t, 1, runResult.FailedExamples)
	assert.Equal(t, "bad comparison", runResult.Results[0].Result.SideInfo.Diagnostics["comparison_error"])
	assert.Equal(t, 1.0, runResult.Results[1].Result.Score)
}

func TestHarness_Run_RecordsCustomEvaluatorErrorsAndContinues(t *testing.T) {
	baseAgent := newMockOptimizableAgent()
	harness := &Harness{
		Evaluator: agentEvaluatorFunc(func(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error) {
			if ex.ID == "bad" {
				return nil, errors.New("evaluator exploded")
			}
			return &EvalResult{
				Score: 1,
				SideInfo: &SideInfo{
					Scores: map[string]float64{"evaluation_success": 1},
				},
			}, nil
		}),
	}

	runResult, err := harness.Run(context.Background(), baseAgent, []AgentExample{
		{ID: "bad"},
		{ID: "good"},
	})

	require.NoError(t, err)
	require.NotNil(t, runResult)

	assert.Equal(t, 1, runResult.EvaluationErrors)
	assert.Equal(t, 1, runResult.PassedExamples)
	assert.Equal(t, 1, runResult.FailedExamples)
	assert.Equal(t, "evaluator exploded", runResult.Results[0].Result.SideInfo.Diagnostics["evaluation_error"])
}

func TestHarness_Run_EmptyExamples(t *testing.T) {
	baseAgent := newMockOptimizableAgent()
	harness := &Harness{
		Evaluator: NewDeterministicEvaluator(nil),
	}

	runResult, err := harness.Run(context.Background(), baseAgent, nil)
	require.NoError(t, err)
	require.NotNil(t, runResult)

	assert.Empty(t, runResult.Results)
	assert.Equal(t, 0, runResult.CompletedExamples)
	assert.Equal(t, 0.0, runResult.AverageScore)
	assert.Equal(t, 0, runResult.PassedExamples)
	assert.Equal(t, 0, runResult.FailedExamples)
}

type agentEvaluatorFunc func(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error)

func (f agentEvaluatorFunc) Evaluate(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error) {
	return f(ctx, agent, ex)
}

func cloneOutputFixtures(input map[string]map[string]interface{}) map[string]map[string]interface{} {
	if input == nil {
		return nil
	}

	cloned := make(map[string]map[string]interface{}, len(input))
	for key, value := range input {
		cloned[key] = maps.Clone(value)
	}

	return cloned
}

func cloneErrorFixtures(input map[string]error) map[string]error {
	if input == nil {
		return nil
	}

	cloned := make(map[string]error, len(input))
	for key, value := range input {
		cloned[key] = value
	}

	return cloned
}
