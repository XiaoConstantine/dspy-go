package main

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	modrlm "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToAgentExamples_NormalizesTasks(t *testing.T) {
	examples := toAgentExamples([]datasets.OolongTask{{
		TaskID:   "sample-task",
		Context:  "context body",
		Question: "What is the answer?",
		Answer:   "42",
	}})

	require.Len(t, examples, 1)
	assert.Equal(t, "sample-task", examples[0].ID)
	assert.Equal(t, "context body", examples[0].Inputs["context"])
	assert.Equal(t, "What is the answer?", examples[0].Inputs["query"])
	assert.Equal(t, "42", examples[0].Outputs["answer"])
}

func TestSplitExamples_UsesValidationFraction(t *testing.T) {
	examples := make([]optimize.AgentExample, 6)
	train, validation := splitExamples(examples, 0.25)
	assert.Len(t, train, 5)
	assert.Len(t, validation, 1)
}

func TestSliceTasks_UsesDeterministicOffset(t *testing.T) {
	tasks := datasets.SampleOolongTasks()
	sliced := datasets.SliceOolongTasks(tasks, 1, 2)

	require.Len(t, sliced, 2)
	assert.Equal(t, tasks[1].Normalize().ID, sliced[0].Normalize().ID)
	assert.Equal(t, tasks[2].Normalize().ID, sliced[1].Normalize().ID)
}

func TestOolongEvaluator_ScoresAnswerAndInteraction(t *testing.T) {
	agent := &stubOptimizableAgent{
		output: map[string]interface{}{"answer": "Answer: 750"},
		trace: &agents.ExecutionTrace{
			Steps: []agents.TraceStep{
				{Index: 1, Tool: "explore", Success: true},
			},
			ToolUsageCount:   map[string]int{"explore": 1},
			TerminationCause: "final_answer",
		},
	}

	result, err := (oolongEvaluator{}).Evaluate(context.Background(), agent, optimize.AgentExample{
		Outputs: map[string]interface{}{"answer": "750"},
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, 1.0, result.Score)
	assert.Equal(t, 1.0, result.SideInfo.Scores["answer_match"])
	assert.Equal(t, 1.0, result.SideInfo.Scores["context_interaction"])
	assert.Equal(t, 1.0, result.SideInfo.Scores["termination"])
}

func TestSummarizeRun_AggregatesTraceMetrics(t *testing.T) {
	run := &optimize.HarnessRunResult{
		AverageScore:      0.75,
		PassedExamples:    1,
		FailedExamples:    1,
		CompletedExamples: 2,
		Results: []optimize.HarnessExampleResult{
			{
				ExampleID: "task-1",
				Result: &optimize.EvalResult{
					Score: 1,
					SideInfo: &optimize.SideInfo{
						Trace: &agents.ExecutionTrace{
							Steps:            []agents.TraceStep{{Index: 1}, {Index: 2}},
							TerminationCause: "state_final",
							ContextMetadata: map[string]interface{}{
								modrlm.TraceMetadataSubLLMCallCount: 2,
								modrlm.TraceMetadataSubRLMCallCount: 1,
							},
						},
					},
				},
			},
			{
				ExampleID: "task-2",
				Result: &optimize.EvalResult{
					Score: 0.5,
					SideInfo: &optimize.SideInfo{
						Trace: &agents.ExecutionTrace{
							Steps:            []agents.TraceStep{{Index: 1}},
							TerminationCause: "max_iterations",
							ContextMetadata: map[string]interface{}{
								modrlm.TraceMetadataSubLLMCallCount: 0,
								modrlm.TraceMetadataSubRLMCallCount: 0,
							},
						},
					},
				},
			},
		},
	}

	summary := summarizeRun(run)
	assert.InDelta(t, 0.75, summary.AverageScore, 0.000001)
	assert.Equal(t, 1, summary.PassedExamples)
	assert.Equal(t, 2, summary.CompletedExamples)
	assert.InDelta(t, 1.5, summary.AverageSteps, 0.000001)
	assert.InDelta(t, 1.0, summary.AverageSubLLM, 0.000001)
	assert.InDelta(t, 0.5, summary.AverageSubRLM, 0.000001)
	assert.Equal(t, []string{"task-1", "task-2"}, summary.ExampleIDs)
	assert.Equal(t, 1, summary.TerminationCounts["state_final"])
	assert.Equal(t, 1, summary.TerminationCounts["max_iterations"])
}

func TestBuildEvalReport_IncludesValidationObjectives(t *testing.T) {
	report := buildEvalReport(evalReportInput{
		Config: evalConfig{
			Provider:   "gemini",
			Model:      "gemini-test",
			TaskSource: "embedded",
		},
		Tasks:               []datasets.OolongTask{{ID: "task-1"}},
		TrainingCount:       3,
		ValidationCount:     1,
		SeedIterationPrompt: "seed",
		OptimizedPrompt:     "optimized",
		Baseline:            &optimize.HarnessRunResult{},
		Optimized:           &optimize.HarnessRunResult{},
		BestValidation: &optimize.GEPACandidateEvaluation{
			AverageScore: 0.8,
			Fitness: optimizeFitnessStub{
				SuccessRate:    0.8,
				OutputQuality:  0.9,
				Efficiency:     0.7,
				Robustness:     0.6,
				Generalization: 0.75,
				Diversity:      0.5,
				Innovation:     0.4,
				WeightedScore:  0.77,
			}.toMultiObjective(),
		},
	})

	assert.Equal(t, "task-1", report.TaskIDs[0])
	assert.Equal(t, "optimized", report.OptimizedIterationPrompt)
	assert.InDelta(t, 0.77, report.BestValidationWeightedFitness, 0.000001)
	assert.InDelta(t, 0.8, report.BestValidationAverageScore, 0.000001)
	assert.InDelta(t, 0.9, report.BestValidationObjectives["output_quality"], 0.000001)
}

type optimizeFitnessStub struct {
	SuccessRate    float64
	OutputQuality  float64
	Efficiency     float64
	Robustness     float64
	Generalization float64
	Diversity      float64
	Innovation     float64
	WeightedScore  float64
}

func (s optimizeFitnessStub) toMultiObjective() *optimizers.MultiObjectiveFitness {
	return &optimizers.MultiObjectiveFitness{
		SuccessRate:    s.SuccessRate,
		OutputQuality:  s.OutputQuality,
		Efficiency:     s.Efficiency,
		Robustness:     s.Robustness,
		Generalization: s.Generalization,
		Diversity:      s.Diversity,
		Innovation:     s.Innovation,
		WeightedScore:  s.WeightedScore,
	}
}

type stubOptimizableAgent struct {
	output    map[string]interface{}
	trace     *agents.ExecutionTrace
	artifacts optimize.AgentArtifacts
}

func (s *stubOptimizableAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	return core.ShallowCopyMap(s.output), nil
}

func (s *stubOptimizableAgent) GetCapabilities() []core.Tool { return nil }

func (s *stubOptimizableAgent) GetMemory() agents.Memory { return nil }

func (s *stubOptimizableAgent) GetArtifacts() optimize.AgentArtifacts {
	return s.artifacts.Clone()
}

func (s *stubOptimizableAgent) SetArtifacts(artifacts optimize.AgentArtifacts) error {
	s.artifacts = artifacts.Clone()
	return nil
}

func (s *stubOptimizableAgent) Clone() (optimize.OptimizableAgent, error) {
	cloned := *s
	cloned.artifacts = s.artifacts.Clone()
	if s.trace != nil {
		cloned.trace = s.trace.Clone()
	}
	return &cloned, nil
}

func (s *stubOptimizableAgent) LastExecutionTrace() *agents.ExecutionTrace {
	if s.trace == nil {
		return nil
	}
	return s.trace.Clone()
}
