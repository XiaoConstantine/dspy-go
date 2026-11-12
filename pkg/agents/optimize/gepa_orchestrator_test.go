package optimize

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

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
					Diagnostics: map[string]any{},
				},
			}, nil
		}),
		GEPAAdapterConfig{
			MaxGenerations:  2,
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
	require.Len(t, result.OptimizationState.PopulationHistory, 1)
	assert.Equal(t, "perfect_validation", result.OptimizationState.StopReason)
	assert.Equal(t, 1, result.TrainingExampleCount)
	assert.Equal(t, 0, result.ValidationExampleCount)
	assert.NotEmpty(t, result.BestArtifacts.Text[ArtifactSkillPack])

	secondGeneration := result.OptimizationState.PopulationHistory[0]
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
			MaxGenerations:  3,
			SearchBatchSize: 4,
			StagnationLimit: 60,
			EvalConcurrency: 2,
			PassThreshold:   0.5,
			PrimaryArtifact: ArtifactSkillPack,
		},
	)

	engineConfig := optimizer.buildEngineConfig(AgentArtifacts{
		Text: map[ArtifactKey]string{
			ArtifactSkillPack: "Use the repository debugging guide.",
		},
	}, 13)
	assert.Equal(t, 4, engineConfig.EvaluationBatchSize)
	assert.Equal(t, 60, engineConfig.StagnationLimit)
	assert.Zero(t, engineConfig.ValidationSplit)
	assert.Nil(t, engineConfig.ValidationExamples)

	engineConfig = optimizer.buildEngineConfig(AgentArtifacts{
		Text: map[ArtifactKey]string{
			ArtifactSkillPack: "Use the repository debugging guide.",
		},
	}, 3)
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
				if ex.ID == "training" {
					score = 1.0
				}
			} else if ex.ID == "validation" {
				score = 0.8
			}
			return &EvalResult{
				Score: score,
				SideInfo: &SideInfo{
					Scores: map[string]float64{
						"artifact_score": score,
					},
					Diagnostics: map[string]any{},
				},
			}, nil
		}),
		GEPAAdapterConfig{
			MaxGenerations:  2,
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
	assert.Equal(t, result.OptimizationState.BestCandidate.ID, result.BestCandidate.ID)
	require.Len(t, result.OptimizationState.PopulationHistory[1].Candidates, 2, "a training improvement is retained even if its validation score is lower")
	assert.Equal(t, result.BestCandidate.ID, result.BestValidationEvaluation.Candidate.ID)
	assert.InDelta(t, 0.8, result.BestValidationEvaluation.Fitness.OutputQuality, 0.000001)
	assert.InDelta(t, 0.8, result.BestValidationEvaluation.AverageScore, 0.000001)
	assert.Equal(t, 1, result.ValidationExampleCount)
}

func TestGEPAAgentOptimizer_Optimize_MultiArtifactValidationFixture(t *testing.T) {
	setupAgentMultiArtifactGEPAMockLLM(t)

	optimizer := NewGEPAAgentOptimizer(
		newMockOptimizableAgent(),
		agentEvaluatorFunc(func(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error) {
			artifacts := agent.GetArtifacts()
			skill := strings.ToLower(artifacts.Text[ArtifactSkillPack])
			toolPolicy := strings.ToLower(artifacts.Text[ArtifactToolPolicy])
			maxTurns := artifacts.Int["max_turns"]

			score := 0.1
			switch ex.ID {
			case "train":
				switch {
				case strings.Contains(toolPolicy, "evidence-seeking"):
					score = 0.8
				case strings.Contains(skill, "carefully"):
					score = 0.6
				case maxTurns == 12:
					score = 0.4
				}
			case "validation":
				switch {
				case strings.Contains(toolPolicy, "evidence-seeking"):
					score = 1.0
				case strings.Contains(skill, "carefully"):
					score = 0.3
				case maxTurns == 12:
					score = 0.6
				}
			}

			return &EvalResult{
				Score: score,
				SideInfo: &SideInfo{
					Scores: map[string]float64{
						"artifact_score": score,
					},
					Diagnostics: map[string]any{
						"skill_pack":  artifacts.Text[ArtifactSkillPack],
						"tool_policy": artifacts.Text[ArtifactToolPolicy],
						"max_turns":   maxTurns,
					},
				},
			}, nil
		}),
		GEPAAdapterConfig{
			MaxGenerations:  4,
			ValidationSplit: 0,
			EvalConcurrency: 1,
			PassThreshold:   0.5,
			PrimaryArtifact: ArtifactToolPolicy,
			ArtifactKeys:    []ArtifactKey{ArtifactSkillPack, ArtifactToolPolicy},
			IntMutationPlans: map[string]IntMutationConfig{
				"max_turns": {Min: 8, Max: 24, Step: 4},
			},
		},
	)

	seed := AgentArtifacts{
		Text: map[ArtifactKey]string{
			ArtifactSkillPack:  "Use the repository debugging guide.",
			ArtifactToolPolicy: "Prefer deterministic tools first.",
		},
		Int: map[string]int{
			"max_turns": 16,
		},
	}

	result, err := optimizer.Optimize(context.Background(), GEPAOptimizeRequest{
		SeedArtifacts: seed,
		TrainingExamples: []AgentExample{
			{ID: "train"},
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
	require.NotNil(t, result.OptimizationState.BestCandidate)
	require.NotNil(t, result.OptimizationState.BestValidationCandidate)
	require.Len(t, result.OptimizationState.PopulationHistory, 4)
	require.Len(t, result.OptimizationState.PopulationHistory[0].Candidates, 1)

	assert.Equal(t, result.OptimizationState.BestCandidate.ID, result.OptimizationState.BestValidationCandidate.ID)
	assert.Equal(t, result.OptimizationState.BestValidationCandidate.ID, result.BestCandidate.ID)
	assert.Equal(t, result.BestCandidate.ID, result.BestValidationEvaluation.Candidate.ID)
	assert.Equal(t, 1, result.ValidationExampleCount)
	assert.InDelta(t, 1.0, result.BestValidationEvaluation.AverageScore, 0.000001)

	require.NotNil(t, result.BestValidationEvaluation.Run)
	require.Len(t, result.BestValidationEvaluation.Run.Results, 1)
	require.NotNil(t, result.BestValidationEvaluation.Run.Results[0].Result)
	require.NotNil(t, result.BestValidationEvaluation.Run.Results[0].Result.SideInfo)
	assert.Equal(t, 1.0, result.BestValidationEvaluation.Run.Results[0].Result.SideInfo.Scores["cached_validation_score"])

	assert.Equal(t, seed.Text[ArtifactSkillPack], result.BestArtifacts.Text[ArtifactSkillPack])
	assert.Contains(t, strings.ToLower(result.BestArtifacts.Text[ArtifactToolPolicy]), "evidence-seeking")
	assert.Equal(t, 12, result.BestArtifacts.Int["max_turns"])

	var intCandidate *optimizers.GEPACandidate
	for _, candidate := range result.OptimizationState.PopulationHistory[3].Candidates {
		if candidate == nil || candidate.ModuleName != intArtifactModuleName("max_turns") {
			continue
		}
		decoded, decodeErr := optimizer.candidateArtifactsWithBase(candidate, seed)
		require.NoError(t, decodeErr)
		if decoded.Int["max_turns"] == 12 {
			intCandidate = candidate
			break
		}
	}
	require.NotNil(t, intCandidate)
	decodedIntCandidate, err := optimizer.candidateArtifactsWithBase(intCandidate, seed)
	require.NoError(t, err)
	assert.Equal(t, 12, decodedIntCandidate.Int["max_turns"])
}

func TestGEPAAgentOptimizer_CachedValidationEvaluation_DoesNotMutateInputCandidate(t *testing.T) {
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
					Diagnostics: map[string]any{},
				},
			}, nil
		}),
		GEPAAdapterConfig{
			MaxGenerations:  1,
			ValidationSplit: 0,
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

	result, err := optimizer.Optimize(context.Background(), GEPAOptimizeRequest{
		SeedArtifacts: seed,
		TrainingExamples: []AgentExample{
			{ID: "training"},
		},
		ValidationExamples: []AgentExample{
			{ID: "validation"},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.OptimizationState)
	require.NotNil(t, result.OptimizationState.BestValidationCandidate)

	inputCandidate := optimizers.CloneCandidate(result.OptimizationState.BestValidationCandidate)
	inputCandidate.Fitness = -123.0

	evaluation, err := optimizer.cachedValidationEvaluation(
		result.OptimizationState,
		inputCandidate,
		seed,
		[]AgentExample{{ID: "validation"}},
	)
	require.NoError(t, err)
	require.NotNil(t, evaluation)
	require.NotNil(t, evaluation.Candidate)

	assert.Equal(t, -123.0, inputCandidate.Fitness)
	assert.NotSame(t, inputCandidate, evaluation.Candidate)
	assert.Equal(t, inputCandidate.ID, evaluation.Candidate.ID)
	assert.InDelta(t, evaluation.Fitness.WeightedScore, evaluation.Candidate.Fitness, 0.000001)
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
					Diagnostics: map[string]any{},
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

func TestGEPAStateBestCandidateForApplication_PrefersBestValidationCandidate(t *testing.T) {
	state := optimizers.NewGEPAState()
	state.BestCandidate = &optimizers.GEPACandidate{ID: "training-best", Fitness: 0.9}
	state.BestFitness = 0.9
	state.BestValidationCandidate = &optimizers.GEPACandidate{ID: "validation-best", Fitness: 0.4}
	state.BestValidationFitness = 0.4

	best, fitness, usingValidation := state.BestCandidateForApplication()
	require.NotNil(t, best)
	assert.Equal(t, "validation-best", best.ID)
	assert.Equal(t, 0.4, fitness)
	assert.True(t, usingValidation)
}

func TestGEPAStateBestCandidateForApplication_FallsBackToBestTrainingCandidate(t *testing.T) {
	state := optimizers.NewGEPAState()
	state.BestCandidate = &optimizers.GEPACandidate{ID: "training-best", Fitness: 0.9}
	state.BestFitness = 0.9

	best, fitness, usingValidation := state.BestCandidateForApplication()
	require.NotNil(t, best)
	assert.Equal(t, "training-best", best.ID)
	assert.Equal(t, 0.9, fitness)
	assert.False(t, usingValidation)
}

func TestGEPAStateBestCandidateForApplication_ReturnsNilWithoutBestCandidate(t *testing.T) {
	state := optimizers.NewGEPAState()
	state.ParetoArchive = []*optimizers.GEPACandidate{
		{ID: "archive-only", Fitness: 0.9},
	}

	best, fitness, usingValidation := state.BestCandidateForApplication()
	assert.Nil(t, best)
	assert.Zero(t, fitness)
	assert.False(t, usingValidation)
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
					Diagnostics: map[string]any{},
				},
			}, nil
		}),
		GEPAAdapterConfig{
			MaxGenerations:  1,
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

type agentMultiArtifactFixtureLLM struct{}

func (f *agentMultiArtifactFixtureLLM) Generate(_ context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	switch {
	case strings.Contains(prompt, "for a skill_pack module."), strings.Contains(prompt, "MODULE: skill_pack"):
		return &core.LLMResponse{Content: "1. Carefully analyze repository state before acting."}, nil
	case strings.Contains(prompt, "for a tool_policy module."), strings.Contains(prompt, "MODULE: tool_policy"):
		return &core.LLMResponse{Content: "1. Use narrow, evidence-seeking tool calls."}, nil
	case strings.Contains(prompt, "for a __int__:max_turns module."), strings.Contains(prompt, "MODULE: __int__:max_turns"):
		return &core.LLMResponse{Content: "1. Set max_turns to 12."}, nil
	default:
		return &core.LLMResponse{Content: "1. Keep the current instruction."}, nil
	}
}

func (f *agentMultiArtifactFixtureLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]any, error) {
	return nil, nil
}

func (f *agentMultiArtifactFixtureLLM) GenerateWithFunctions(context.Context, string, []map[string]any, ...core.GenerateOption) (map[string]any, error) {
	return nil, nil
}

func (f *agentMultiArtifactFixtureLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}

func (f *agentMultiArtifactFixtureLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

func (f *agentMultiArtifactFixtureLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *agentMultiArtifactFixtureLLM) GenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, nil
}

func (f *agentMultiArtifactFixtureLLM) StreamGenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *agentMultiArtifactFixtureLLM) ProviderName() string { return "fixture" }

func (f *agentMultiArtifactFixtureLLM) ModelID() string { return "fixture-agent-multi-artifact" }

func (f *agentMultiArtifactFixtureLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

func setupAgentMultiArtifactGEPAMockLLM(t *testing.T) {
	t.Helper()

	llm := &agentMultiArtifactFixtureLLM{}
	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	core.SetDefaultLLM(llm)
	core.GlobalConfig.TeacherLLM = llm
	t.Cleanup(func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	})
}

// At EvalConcurrency above 1 the optimizer scores through GEPA's concurrent
// entry point, so feedback calls for one candidate overlap.
func TestGEPAAgentOptimizer_Optimize_CallsFeedbackEvaluatorConcurrently(t *testing.T) {
	setupAgentGEPAMockLLM(t)

	for _, tc := range []struct {
		concurrency int
		overlap     bool
	}{{1, false}, {2, true}} {
		var mu sync.Mutex
		active, peak := 0, 0
		arrived := make(chan struct{}, 16)
		feedback := optimizers.GEPAFeedbackEvaluatorFunc(func(context.Context, map[string]any, map[string]any, *optimizers.GEPAFeedbackContext) *optimizers.GEPAFeedback {
			mu.Lock()
			active++
			peak = max(peak, active)
			overlapping := active > 1
			mu.Unlock()
			if overlapping {
				arrived <- struct{}{}
			} else {
				// Give a second call the chance to overlap with this one.
				select {
				case <-arrived:
				case <-time.After(200 * time.Millisecond):
				}
			}
			mu.Lock()
			active--
			mu.Unlock()
			return &optimizers.GEPAFeedback{Feedback: "ok"}
		})

		optimizer := NewGEPAAgentOptimizer(
			newMockOptimizableAgent(),
			agentEvaluatorFunc(func(context.Context, OptimizableAgent, AgentExample) (*EvalResult, error) {
				return &EvalResult{Score: 1.0}, nil
			}),
			GEPAAdapterConfig{
				MaxGenerations:    1,
				EvalConcurrency:   tc.concurrency,
				PassThreshold:     0.5,
				PrimaryArtifact:   ArtifactSkillPack,
				FeedbackEvaluator: feedback,
			},
		)
		_, err := optimizer.Optimize(context.Background(), GEPAOptimizeRequest{
			SeedArtifacts:    AgentArtifacts{Text: map[ArtifactKey]string{ArtifactSkillPack: "Use the guide."}},
			TrainingExamples: []AgentExample{{ID: "train-1"}, {ID: "train-2"}},
		})
		require.NoError(t, err)

		mu.Lock()
		got := peak
		mu.Unlock()
		assert.Equal(t, tc.overlap, got > 1, "EvalConcurrency %d: peak %d", tc.concurrency, got)
	}
}

func TestOptimizationScorerReportsUnreadableScore(t *testing.T) {
	score, err := optimizationScorer().Score(context.Background(), core.Example{}, map[string]any{gepaProgramOutputScoreKey: 0.75})
	require.NoError(t, err)
	assert.Equal(t, 0.75, score)

	_, err = optimizationScorer().Score(context.Background(), core.Example{}, map[string]any{gepaProgramOutputScoreKey: "high"})
	require.Error(t, err)
	_, err = optimizationScorer().Score(context.Background(), core.Example{}, nil)
	require.Error(t, err)
}
