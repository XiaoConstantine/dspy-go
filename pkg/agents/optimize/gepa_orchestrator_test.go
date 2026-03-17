package optimize

import (
	"context"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestGEPAAgentOptimizer_Optimize_UsesMainlineGEPACompile(t *testing.T) {
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
	require.GreaterOrEqual(t, result.OptimizationState.CurrentGeneration, 1)
	assert.Equal(t, 1, result.TrainingExampleCount)
	assert.Equal(t, 0, result.ValidationExampleCount)
	assert.NotEmpty(t, result.BestArtifacts.Text[ArtifactSkillPack])

	secondGeneration := result.OptimizationState.PopulationHistory[1]
	require.NotNil(t, secondGeneration)
	require.NotEmpty(t, secondGeneration.Candidates)
	for _, candidate := range secondGeneration.Candidates {
		artifacts, decodeErr := optimizer.candidateArtifactsWithBase(candidate, AgentArtifacts{
			Text: map[ArtifactKey]string{
				ArtifactSkillPack: "Use the repository debugging guide.",
			},
		})
		require.NoError(t, decodeErr)
		assert.NotEmpty(t, artifacts.Text[ArtifactSkillPack])
	}
}

func TestGEPAAgentOptimizer_BuildEngineConfig_UsesSearchBatchAndStagnationLimit(t *testing.T) {
	optimizer := NewGEPAAgentOptimizer(
		newMockOptimizableAgent(),
		agentEvaluatorFunc(func(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error) {
			return &EvalResult{Score: 1.0}, nil
		}),
		GEPAAdapterConfig{
			PopulationSize:  4,
			MaxGenerations:  3,
			ReflectionFreq:  1,
			SearchBatchSize: 4,
			StagnationLimit: 60,
			EvalConcurrency: 2,
			PassThreshold:   0.5,
			PrimaryArtifact: ArtifactSkillPack,
		},
	)

	engineConfig := optimizer.buildEngineConfig(13, 3)
	assert.Equal(t, 4, engineConfig.EvaluationBatchSize)
	assert.Equal(t, 60, engineConfig.StagnationLimit)
	assert.InDelta(t, 3.0/16.0, engineConfig.ValidationSplit, 0.000001)

	engineConfig = optimizer.buildEngineConfig(3, 0)
	assert.Equal(t, 3, engineConfig.EvaluationBatchSize)
	assert.Zero(t, engineConfig.ValidationSplit)
}

func TestGEPAAgentOptimizer_Optimize_PrefersEngineValidationWinner(t *testing.T) {
	setupAgentGEPAMockLLM(t)

	optimizer := NewGEPAAgentOptimizer(
		newMockOptimizableAgent(),
		agentEvaluatorFunc(func(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error) {
			score := 0.2
			if strings.Contains(strings.ToLower(agent.GetArtifacts().Text[ArtifactSkillPack]), "carefully") {
				if ex.ID == "validation" {
					score = 1.0
				}
			} else if ex.ID == "training" {
				score = 1.0
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
	require.NotNil(t, result.OptimizationState)
	require.NotNil(t, result.OptimizationState.BestValidationCandidate)

	assert.NotNil(t, result.OptimizationState.BestCandidate)
	assert.Equal(t, result.OptimizationState.BestValidationCandidate.ID, result.BestCandidate.ID)
	assert.NotEqual(t, result.OptimizationState.BestCandidate.ID, result.BestCandidate.ID)
	assert.Equal(t, result.BestCandidate.ID, result.BestValidationEvaluation.Candidate.ID)
	assert.InDelta(t, 1.0, result.BestValidationEvaluation.Fitness.OutputQuality, 0.000001)
	assert.InDelta(t, 1.0, result.BestValidationEvaluation.AverageScore, 0.000001)
	assert.Equal(t, 1, result.ValidationExampleCount)
}

func TestGEPAAgentOptimizer_EvaluateBestCandidateOnValidation_UsesStateBestValidationCandidate(t *testing.T) {
	setupAgentGEPAMockLLM(t)

	optimizer := NewGEPAAgentOptimizer(
		newMockOptimizableAgent(),
		agentEvaluatorFunc(func(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error) {
			score := 0.2
			if strings.Contains(strings.ToLower(agent.GetArtifacts().Text[ArtifactSkillPack]), "carefully") {
				score = 1.0
			}
			return &EvalResult{
				Score: score,
				SideInfo: &SideInfo{
					Scores:      map[string]float64{"artifact_score": score},
					Diagnostics: map[string]interface{}{},
				},
			}, nil
		}),
		GEPAAdapterConfig{
			EvalConcurrency: 1,
			PassThreshold:   0.5,
			PrimaryArtifact: ArtifactSkillPack,
		},
	)

	seed := AgentArtifacts{
		Text: map[ArtifactKey]string{
			ArtifactSkillPack: "Use the repository debugging guide.",
		},
	}
	archiveCandidate, err := optimizer.SeedCandidate(seed)
	require.NoError(t, err)
	archiveCandidate.ID = "archive"

	historyCandidate := optimizers.CloneCandidate(archiveCandidate)
	historyCandidate.ID = "history-best"
	historyCandidate.Instruction = "Carefully analyze before acting."
	historyCandidate.ComponentTexts = map[string]string{
		string(ArtifactSkillPack): historyCandidate.Instruction,
	}

	state := optimizers.NewGEPAState()
	state.BestCandidate = archiveCandidate
	state.BestValidationCandidate = historyCandidate
	state.ParetoArchive = []*optimizers.GEPACandidate{archiveCandidate}
	state.PopulationHistory = []*optimizers.Population{
		{
			Generation: 1,
			Candidates: []*optimizers.GEPACandidate{
				archiveCandidate,
				historyCandidate,
			},
			BestCandidate: archiveCandidate,
		},
	}

	bestCandidate, bestEvaluation, err := optimizer.evaluateBestCandidateOnValidation(context.Background(), state, seed, []AgentExample{{ID: "validation"}})
	require.NoError(t, err)
	require.NotNil(t, bestCandidate)
	require.NotNil(t, bestEvaluation)
	assert.Equal(t, historyCandidate.ID, bestCandidate.ID)
	assert.Equal(t, historyCandidate.ID, bestEvaluation.Candidate.ID)
	assert.Contains(t, strings.ToLower(bestEvaluation.Artifacts.Text[ArtifactSkillPack]), "carefully")
}

func TestBestCandidateFromState_PrefersBestValidationCandidate(t *testing.T) {
	state := optimizers.NewGEPAState()
	state.BestCandidate = &optimizers.GEPACandidate{ID: "training-best", Fitness: 0.9}
	state.BestValidationCandidate = &optimizers.GEPACandidate{ID: "validation-best", Fitness: 0.4}

	best := bestCandidateFromState(state)
	require.NotNil(t, best)
	assert.Equal(t, "validation-best", best.ID)
}

func TestBestCandidateFromState_FallsBackToBestArchiveCandidate(t *testing.T) {
	state := optimizers.NewGEPAState()
	state.BestCandidate = nil
	state.ParetoArchive = []*optimizers.GEPACandidate{
		{ID: "candidate-b", Fitness: 0.7},
		{ID: "candidate-a", Fitness: 0.7},
		{ID: "candidate-c", Fitness: 0.9},
	}
	state.ArchiveFitnessMap = map[string]*optimizers.MultiObjectiveFitness{
		"candidate-b": {WeightedScore: 0.7},
		"candidate-a": {WeightedScore: 0.7},
		"candidate-c": {WeightedScore: 0.9},
	}

	best := bestCandidateFromState(state)
	require.NotNil(t, best)
	assert.Equal(t, "candidate-c", best.ID)
}

func TestGEPAAgentOptimizer_Optimize_StoresAggregateCandidateFitnessForSyntheticProgram(t *testing.T) {
	setupAgentGEPAMockLLM(t)

	optimizer := NewGEPAAgentOptimizer(
		newMockOptimizableAgent(),
		agentEvaluatorFunc(func(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error) {
			score := 0.0
			if ex.ID == "first" {
				score = 1.0
			}
			return &EvalResult{
				Score: score,
				SideInfo: &SideInfo{
					Scores:      map[string]float64{"artifact_score": score},
					Diagnostics: map[string]interface{}{},
				},
			}, nil
		}),
		GEPAAdapterConfig{
			PopulationSize:  1,
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
			{ID: "first"},
			{ID: "second"},
		},
	})

	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.BestCandidate)
	require.NotNil(t, result.OptimizationState)

	metrics := result.OptimizationState.CandidateMetrics[result.BestCandidate.ID]
	require.NotNil(t, metrics)
	fitness, ok := metrics.Metadata["multi_objective_fitness"].(*optimizers.MultiObjectiveFitness)
	require.True(t, ok)
	require.NotNil(t, fitness)

	assert.InDelta(t, 0.5, result.BestCandidate.Fitness, 0.000001)
	assert.InDelta(t, 0.5, metrics.AverageFitness, 0.000001)
	assert.InDelta(t, 0.5, fitness.OutputQuality, 0.000001)
	assert.InDelta(t, 0.5, fitness.SuccessRate, 0.000001)
}

func TestGEPA_BootstrapPopulationFromSeed_RejectsReinitialization(t *testing.T) {
	setupAgentGEPAMockLLM(t)

	engine, err := optimizers.NewGEPA(optimizers.DefaultGEPAConfig())
	require.NoError(t, err)

	seed := &optimizers.GEPACandidate{
		ID:          "seed",
		ModuleName:  "skill_pack",
		Instruction: "Use the repository debugging guide.",
	}

	require.NoError(t, engine.BootstrapPopulationFromSeed(context.Background(), seed))
	err = engine.BootstrapPopulationFromSeed(context.Background(), seed)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already been initialized")
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
