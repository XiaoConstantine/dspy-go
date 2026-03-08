package optimize

import (
	"context"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestGEPAAgentOptimizer_Optimize_RunsSharedEvolutionLoop(t *testing.T) {
	setupAgentGEPAMockLLM(t)

	optimizer := NewGEPAAgentOptimizer(
		newMockOptimizableAgent(),
		agentEvaluatorFunc(func(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error) {
			return &EvalResult{
				Score: 1.0,
				SideInfo: &SideInfo{
					Scores: map[string]float64{
						"output_match": 1.0,
					},
					Diagnostics: map[string]interface{}{},
				},
			}, nil
		}),
		GEPAAdapterConfig{
			PopulationSize:  3,
			MaxGenerations:  2,
			ReflectionFreq:  0,
			ValidationSplit: 0,
			EvalConcurrency: 2,
			PassThreshold:   0.5,
			PrimaryArtifact: ArtifactSkillPack,
		},
	)

	result, err := optimizer.Optimize(context.Background(), GEPAOptimizeRequest{
		SeedArtifacts: AgentArtifacts{
			Text: map[ArtifactKey]string{
				ArtifactSkillPack: "Use the repository debugging guide.",
			},
		},
		TrainingExamples: []AgentExample{
			{ID: "train-1"},
		},
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.BestCandidate)
	require.NotNil(t, result.OptimizationState)
	require.Len(t, result.OptimizationState.PopulationHistory, 2)
	assert.Equal(t, 1, result.TrainingExampleCount)
	assert.Equal(t, 0, result.ValidationExampleCount)
	assert.NotEmpty(t, result.BestArtifacts.Text[ArtifactSkillPack])

	secondGeneration := result.OptimizationState.PopulationHistory[1]
	require.NotNil(t, secondGeneration)
	require.NotEmpty(t, secondGeneration.Candidates)
	for _, candidate := range secondGeneration.Candidates {
		artifacts, decodeErr := optimizer.CandidateArtifacts(candidate)
		require.NoError(t, decodeErr)
		assert.NotEmpty(t, artifacts.Text[ArtifactSkillPack])
	}
}

func TestGEPAAgentOptimizer_Optimize_UsesValidationExamplesForFinalSelection(t *testing.T) {
	setupAgentGEPAMockLLM(t)

	optimizer := NewGEPAAgentOptimizer(
		newMockOptimizableAgent(),
		agentEvaluatorFunc(func(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error) {
			skill := strings.ToLower(agent.GetArtifacts().Text[ArtifactSkillPack])
			score := 0.6
			if ex.ID == "validation" {
				score = 0.2
				if strings.Contains(skill, "carefully") {
					score = 1.0
				}
			}
			return &EvalResult{
				Score: score,
				SideInfo: &SideInfo{
					Scores: map[string]float64{
						"artifact_score": score,
					},
					Diagnostics: map[string]interface{}{},
				},
			}, nil
		}),
		GEPAAdapterConfig{
			PopulationSize:  3,
			MaxGenerations:  1,
			ReflectionFreq:  0,
			ValidationSplit: 0,
			EvalConcurrency: 1,
			PassThreshold:   0.5,
			PrimaryArtifact: ArtifactSkillPack,
		},
	)

	result, err := optimizer.Optimize(context.Background(), GEPAOptimizeRequest{
		SeedArtifacts: AgentArtifacts{
			Text: map[ArtifactKey]string{
				ArtifactSkillPack: "Use the repository debugging guide.",
			},
		},
		TrainingExamples: []AgentExample{
			{ID: "training"},
		},
		ValidationExamples: []AgentExample{
			{ID: "validation"},
		},
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.BestCandidate)
	require.NotNil(t, result.BestValidationEvaluation)

	assert.Contains(t, strings.ToLower(result.BestCandidate.Instruction), "carefully")
	assert.InDelta(t, 1.0, result.BestValidationEvaluation.Fitness.OutputQuality, 0.000001)
	assert.InDelta(t, 0.95, result.BestValidationEvaluation.Fitness.WeightedScore, 0.000001)
	assert.Equal(t, 1, result.ValidationExampleCount)
}

func setupAgentGEPAMockLLM(t *testing.T) {
	t.Helper()

	mockLLM := &testutil.MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: `1. Carefully analyze the repository state before taking action.
2. Thoroughly inspect the available evidence and prefer deterministic fixes.
3. Systematically verify the result before concluding.`,
	}, nil)

	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	core.SetDefaultLLM(mockLLM)
	core.GlobalConfig.TeacherLLM = mockLLM
	t.Cleanup(func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	})
}
