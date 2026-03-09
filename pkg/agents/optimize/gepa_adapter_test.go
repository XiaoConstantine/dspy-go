package optimize

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGEPAAgentOptimizer_SeedCandidateRoundTrip(t *testing.T) {
	optimizer := NewGEPAAgentOptimizer(nil, nil, GEPAAdapterConfig{
		ArtifactKeys:    []ArtifactKey{ArtifactSkillPack, ArtifactToolPolicy},
		PrimaryArtifact: ArtifactSkillPack,
	})

	seed := AgentArtifacts{
		Text: map[ArtifactKey]string{
			ArtifactSkillPack:  "Use repo-specific debugging steps.",
			ArtifactToolPolicy: "Prefer deterministic tools first.",
		},
		Int: map[string]int{
			"max_iterations": 6,
		},
		Bool: map[string]bool{
			"parallel_tools": true,
		},
	}

	candidate, err := optimizer.SeedCandidate(seed)
	require.NoError(t, err)
	require.NotNil(t, candidate)

	assert.Equal(t, string(ArtifactSkillPack), candidate.ModuleName)
	assert.Equal(t, seed.Text[ArtifactSkillPack], candidate.Instruction)
	assert.Equal(t, []string{"skill_pack", "tool_policy"}, candidate.Metadata[gepaMetadataArtifactKeysKey])

	decoded, err := optimizer.CandidateArtifacts(candidate)
	require.NoError(t, err)
	assert.Equal(t, seed, decoded)

	candidate.Instruction = "Use faster repo triage."
	decoded, err = optimizer.CandidateArtifacts(candidate)
	require.NoError(t, err)
	assert.Equal(t, "Use faster repo triage.", decoded.Text[ArtifactSkillPack])
	assert.Equal(t, seed.Text[ArtifactToolPolicy], decoded.Text[ArtifactToolPolicy])
}

func TestGEPAAgentOptimizer_MaterializeAgentFallsBackToFactory(t *testing.T) {
	optimizer := NewGEPAAgentOptimizer(cloneFailAgent{newMockOptimizableAgent()}, nil, DefaultGEPAAdapterConfig()).
		WithFactory(func(artifacts AgentArtifacts) (OptimizableAgent, error) {
			agent := newMockOptimizableAgent()
			require.NoError(t, agent.SetArtifacts(artifacts))
			return agent, nil
		})

	artifacts := AgentArtifacts{
		Text: map[ArtifactKey]string{
			ArtifactSkillPack: "factory seeded",
		},
	}

	agent, err := optimizer.MaterializeAgent(artifacts)
	require.NoError(t, err)
	require.NotNil(t, agent)
	assert.Equal(t, artifacts.Clone(), agent.GetArtifacts())
}

func TestGEPAAgentOptimizer_EvaluateCandidateBuildsFitnessAndTraces(t *testing.T) {
	baseAgent := newMockOptimizableAgent()
	baseAgent.outputs["first"] = map[string]interface{}{
		"answer": "42",
	}
	baseAgent.outputs["second"] = map[string]interface{}{
		"answer": "wrong",
	}

	optimizer := NewGEPAAgentOptimizer(baseAgent, NewDeterministicEvaluator(nil), GEPAAdapterConfig{
		PassThreshold:   1.0,
		PrimaryArtifact: ArtifactSkillPack,
	})

	candidate, err := optimizer.SeedCandidate(AgentArtifacts{
		Text: map[ArtifactKey]string{
			ArtifactSkillPack: "Use the repo debugging guide.",
		},
	})
	require.NoError(t, err)

	evaluation, err := optimizer.EvaluateCandidate(context.Background(), candidate, []AgentExample{
		{
			ID: "first",
			Inputs: map[string]interface{}{
				"id": "first",
			},
			Outputs: map[string]interface{}{
				"answer": "42",
			},
		},
		{
			ID: "second",
			Inputs: map[string]interface{}{
				"id": "second",
			},
			Outputs: map[string]interface{}{
				"answer": "expected",
			},
		},
	})

	require.NoError(t, err)
	require.NotNil(t, evaluation)
	require.NotNil(t, evaluation.Run)
	require.NotNil(t, evaluation.Fitness)
	require.Len(t, evaluation.Traces, 2)

	assert.Equal(t, 2, evaluation.Run.CompletedExamples)
	assert.Equal(t, 1, evaluation.Run.PassedExamples)
	assert.Equal(t, 1, evaluation.Run.FailedExamples)
	assert.Equal(t, 0, evaluation.Run.EvaluationErrors)
	assert.InDelta(t, 0.5, evaluation.AverageScore, 0.000001)

	assert.InDelta(t, 0.5, evaluation.Fitness.SuccessRate, 0.000001)
	assert.InDelta(t, 0.5, evaluation.Fitness.OutputQuality, 0.000001)
	assert.InDelta(t, 0.75, evaluation.Fitness.Generalization, 0.000001)
	assert.Greater(t, evaluation.Fitness.Efficiency, 0.0)
	assert.Equal(t, evaluation.Fitness.WeightedScore, candidate.Fitness)

	assert.True(t, evaluation.Traces[0].Success)
	assert.False(t, evaluation.Traces[1].Success)
	assert.Equal(t, "first", evaluation.Traces[0].ContextData["example_id"])
	assert.Contains(t, evaluation.Traces[0].ContextData, gepaMetadataTraceSummaryKey)
	assert.Contains(t, evaluation.Traces[0].ContextData, gepaMetadataTraceEvidenceKey)
	assert.Contains(t, evaluation.Traces[1].ContextData[gepaMetadataTraceEvidenceKey], "failed_test=output:answer")
	assert.Contains(t, evaluation.Traces[1].ContextData, "diagnostics")
}

func TestBuildMultiObjectiveFitness_CountsEvaluationFailuresOnce(t *testing.T) {
	run := &HarnessRunResult{
		Results: []HarnessExampleResult{
			{
				ExampleID: "ok",
				Result: &EvalResult{
					Score: 1,
					SideInfo: &SideInfo{
						LatencyMS: 100,
					},
				},
			},
			{
				ExampleID: "err",
				Result:    evaluationFailureResult(errors.New("boom")),
			},
		},
		AverageScore:      0.5,
		PassedExamples:    1,
		FailedExamples:    1,
		CompletedExamples: 2,
		EvaluationErrors:  1,
	}

	fitness := buildMultiObjectiveFitness(run)
	require.NotNil(t, fitness)
	assert.InDelta(t, 0.5, fitness.Robustness, 0.000001)
}

func TestScoreConsistency_SingleNilResult(t *testing.T) {
	assert.Equal(t, 0.0, scoreConsistency([]HarnessExampleResult{
		{ExampleID: "nil-result"},
	}))
}

func TestSingleResultEfficiency_SkipsMissingToolUsageData(t *testing.T) {
	result := &EvalResult{
		Score: 0.5,
		SideInfo: &SideInfo{
			LatencyMS: 1000,
		},
	}

	assert.InDelta(t, 0.5, singleResultEfficiency(result), 0.000001)
	assert.Equal(t, -1, toolCallCount(result.SideInfo))
}

func TestBuildGEPATrace_PreservesExpectedOutputsSeparately(t *testing.T) {
	trace := buildGEPATrace(
		&optimizers.GEPACandidate{
			ID:         "candidate-1",
			ModuleName: "skill_pack",
		},
		AgentExample{
			ID: "example-1",
			Inputs: map[string]interface{}{
				"question": "What is 6 * 7?",
			},
			Outputs: map[string]interface{}{
				"answer": "42",
			},
		},
		&EvalResult{
			Score: 0,
			SideInfo: &SideInfo{
				LatencyMS: 25,
			},
		},
		1.0,
	)

	assert.Nil(t, trace.Outputs)
	assert.Equal(t, map[string]interface{}{"answer": "42"}, trace.ContextData["expected_outputs"])
	assert.Equal(t, 25*time.Millisecond, trace.Duration)
}

type cloneFailAgent struct {
	*mockOptimizableAgent
}

func (a cloneFailAgent) Clone() (OptimizableAgent, error) {
	return nil, errors.New("clone failed")
}
