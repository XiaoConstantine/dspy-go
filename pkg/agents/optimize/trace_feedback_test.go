package optimize

import (
	"context"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGEPAAgentOptimizer_BuildEngineConfig_PropagatesAdvancedControls(t *testing.T) {
	agent := newTargetAwareMockAgent()
	customFeedback := optimizers.GEPAFeedbackEvaluatorFunc(func(_ context.Context, _, _ map[string]interface{}, _ *optimizers.GEPAFeedbackContext) *optimizers.GEPAFeedback {
		return &optimizers.GEPAFeedback{
			Feedback:        "custom feedback",
			TargetComponent: "root.system",
		}
	})

	optimizer := NewGEPAAgentOptimizer(
		agent,
		agentEvaluatorFunc(func(context.Context, OptimizableAgent, AgentExample) (*EvalResult, error) {
			return &EvalResult{Score: 1.0}, nil
		}),
		GEPAAdapterConfig{
			PopulationSize:             4,
			MaxGenerations:             3,
			ReflectionFreq:             1,
			SearchBatchSize:            4,
			StagnationLimit:            60,
			ValidationSplit:            0,
			ValidationFrequency:        2,
			EvalConcurrency:            2,
			PassThreshold:              0.5,
			PrimaryArtifact:            ArtifactSkillPack,
			MaxMetricCalls:             7,
			ScoreThreshold:             0.9,
			MaxRuntime:                 3 * time.Second,
			FeedbackEvaluator:          customFeedback,
			AddFormatFailureAsFeedback: true,
		},
	)

	engineConfig := optimizer.buildEngineConfig(agent.GetArtifacts(), 11)
	assert.Equal(t, 2, engineConfig.ValidationFrequency)
	assert.Equal(t, 7, engineConfig.MaxMetricCalls)
	assert.Equal(t, 0.9, engineConfig.ScoreThreshold)
	assert.Equal(t, 3*time.Second, engineConfig.MaxRuntime)
	assert.True(t, engineConfig.AddFormatFailureAsFeedback)

	feedback := engineConfig.FeedbackEvaluator.EvaluateFeedback(context.Background(), nil, nil, &optimizers.GEPAFeedbackContext{})
	require.NotNil(t, feedback)
	assert.Equal(t, "custom feedback", feedback.Feedback)
	assert.Equal(t, "root.system", feedback.TargetComponent)
}

func TestGEPAAgentOptimizer_DefaultTraceFeedbackUsesStableTargetIDs(t *testing.T) {
	agent := newTargetAwareMockAgent()
	optimizer := NewGEPAAgentOptimizer(
		agent,
		agentEvaluatorFunc(func(context.Context, OptimizableAgent, AgentExample) (*EvalResult, error) {
			return &EvalResult{Score: 1.0}, nil
		}),
		GEPAAdapterConfig{
			PrimaryArtifact: ArtifactSkillPack,
		},
	)

	engineConfig := optimizer.buildEngineConfig(agent.GetArtifacts(), 1)
	require.NotNil(t, engineConfig.FeedbackEvaluator)

	feedback := engineConfig.FeedbackEvaluator.EvaluateFeedback(context.Background(), nil, map[string]interface{}{
		"score":          0.25,
		"error":          "loop exhausted the budget",
		"trace_status":   "failed",
		"termination":    "budget_exhausted",
		"failed_tests":   []string{"search"},
		"trace_evidence": []string{"failed_test=search", "sub_rlm_calls=2"},
	}, &optimizers.GEPAFeedbackContext{
		Candidate: &optimizers.GEPACandidate{
			ModuleName: intArtifactModuleName("max_turns"),
		},
	})

	require.NotNil(t, feedback)
	assert.Equal(t, "root.max_turns", feedback.TargetComponent)
	assert.Contains(t, feedback.Feedback, "score=0.25")
	assert.Contains(t, feedback.Feedback, "issue=loop exhausted the budget")
	assert.Contains(t, feedback.Feedback, "trace_status=failed")
	assert.Contains(t, feedback.Feedback, "failed_tests=search")
	assert.Equal(t, "deterministic_trace_feedback", feedback.Metadata["source"])
	assert.Equal(t, "root.max_turns", feedback.Metadata["optimization_target_id"])
}
