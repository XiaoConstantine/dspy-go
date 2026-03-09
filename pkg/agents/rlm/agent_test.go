package rlm

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	modrlm "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type testLLM struct {
	responses []string
	callCount int
}

func (m *testLLM) Generate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	if m.callCount >= len(m.responses) {
		return nil, fmt.Errorf("no more mock responses")
	}
	response := m.responses[m.callCount]
	m.callCount++
	return &core.LLMResponse{
		Content: response,
		Usage: &core.TokenInfo{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}, nil
}

func (m *testLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *testLLM) GenerateWithFunctions(context.Context, string, []map[string]interface{}, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *testLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	return m.Generate(ctx, "", opts...)
}

func (m *testLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *testLLM) StreamGenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *testLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *testLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *testLLM) ProviderName() string { return "mock" }
func (m *testLLM) ModelID() string      { return "mock-model" }
func (m *testLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

type testAgentEvaluator func(ctx context.Context, agent optimize.OptimizableAgent, ex optimize.AgentExample) (*optimize.EvalResult, error)

func (f testAgentEvaluator) Evaluate(ctx context.Context, agent optimize.OptimizableAgent, ex optimize.AgentExample) (*optimize.EvalResult, error) {
	return f(ctx, agent, ex)
}

type testSubLLMClient struct {
	response string
}

func (c *testSubLLMClient) Query(context.Context, string) (modrlm.QueryResponse, error) {
	return modrlm.QueryResponse{
		Response:         c.response,
		PromptTokens:     25,
		CompletionTokens: 10,
	}, nil
}

func (c *testSubLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]modrlm.QueryResponse, error) {
	results := make([]modrlm.QueryResponse, len(prompts))
	for i := range prompts {
		result, err := c.Query(ctx, prompts[i])
		if err != nil {
			return nil, err
		}
		results[i] = result
	}
	return results, nil
}

func TestAgentExecute_RecordsTraceAndOutput(t *testing.T) {
	module := modrlm.NewFromLLM(&testLLM{
		responses: []string{
			"Reasoning:\nI'll answer directly.\n\nAction:\nfinal\n\nCode:\n\nAnswer:\n42",
		},
	}, modrlm.WithMaxIterations(1))

	agent := NewAgent("adaptive-rlm", module)
	output, err := agent.Execute(context.Background(), map[string]interface{}{
		"context": "ctx",
		"query":   "what is the answer?",
	})
	require.NoError(t, err)
	assert.Equal(t, "42", output["answer"])

	trace := agent.LastExecutionTrace()
	require.NotNil(t, trace)
	assert.Equal(t, "adaptive-rlm", trace.AgentID)
	assert.Equal(t, "rlm", trace.AgentType)
	assert.Equal(t, "what is the answer?", trace.Task)
	assert.Equal(t, agents.TraceStatusSuccess, trace.Status)
	assert.Equal(t, "final_answer", trace.TerminationCause)
	require.Len(t, trace.Steps, 1)
	assert.Equal(t, "final", trace.Steps[0].ActionRaw)
}

func TestAgentExecute_RecordsRLMNativeMetadata(t *testing.T) {
	module := modrlm.New(
		&testLLM{
			responses: []string{
				"Reasoning:\nI'll query the helper and then finish.\n\nAction:\nquery\n\nCode:\nanswer := QueryRaw(\"what is the answer?\")\nFINAL(answer)\n\nAnswer:\n",
			},
		},
		&testSubLLMClient{response: "42"},
		modrlm.WithAdaptiveIteration(),
	)

	agent := NewAgent("adaptive-rlm", module)
	output, err := agent.Execute(context.Background(), map[string]interface{}{
		"context": "ctx",
		"query":   "what is the answer?",
	})
	require.NoError(t, err)
	assert.Equal(t, "42", output["answer"])

	trace := agent.LastExecutionTrace()
	require.NotNil(t, trace)
	assert.Equal(t, 1, trace.ContextMetadata["sub_llm_call_count"])
	assert.Equal(t, 0, trace.ContextMetadata["sub_rlm_call_count"])
	assert.Equal(t, 100, trace.ContextMetadata["root_prompt_mean_tokens"])
	assert.Equal(t, 100, trace.ContextMetadata["root_prompt_max_tokens"])
	assert.Equal(t, 10, trace.ContextMetadata["adaptive_base_iterations"])
	assert.Equal(t, 50, trace.ContextMetadata["adaptive_max_iterations"])
	assert.Equal(t, 1, trace.ContextMetadata["adaptive_confidence_threshold"])
}

func TestAgentExecute_NilReceiverReturnsError(t *testing.T) {
	var agent *Agent
	_, err := agent.Execute(context.Background(), map[string]interface{}{
		"context": "ctx",
		"query":   "what is the answer?",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not initialized")
}

func TestAgentSetArtifacts_AppliesRLMConfig(t *testing.T) {
	module := modrlm.NewFromLLM(&testLLM{
		responses: []string{
			"Reasoning:\nDone.\n\nAction:\nfinal\n\nCode:\n\nAnswer:\ncomplete",
		},
	})
	agent := NewAgent("adaptive-rlm", module)

	err := agent.SetArtifacts(optimize.AgentArtifacts{
		Text: map[optimize.ArtifactKey]string{
			optimize.ArtifactRLMOuterPrompt:     "Custom outer prompt.",
			optimize.ArtifactRLMIterationPrompt: "Custom iteration prompt.",
		},
		Int: map[string]int{
			ArtifactMaxIterations:               7,
			ArtifactMaxTokens:                   900,
			ArtifactAdaptiveBaseIterations:      5,
			ArtifactAdaptiveMaxIterations:       9,
			ArtifactAdaptiveConfidenceThreshold: 3,
		},
		Bool: map[string]bool{
			ArtifactUseIterationDemos:        true,
			ArtifactCompactIterationPrompt:   false,
			ArtifactAdaptiveIterationEnabled: true,
		},
	})
	require.NoError(t, err)

	cfg := module.Config()
	assert.Equal(t, "Custom outer prompt.", cfg.OuterInstruction)
	assert.Equal(t, "Custom iteration prompt.", cfg.IterationInstruction)
	assert.Equal(t, 7, cfg.MaxIterations)
	assert.Equal(t, 900, cfg.MaxTokens)
	assert.True(t, cfg.UseIterationDemos)
	require.NotNil(t, cfg.AdaptiveIteration)
	assert.True(t, cfg.AdaptiveIteration.Enabled)
	assert.Equal(t, 5, cfg.AdaptiveIteration.BaseIterations)
	assert.Equal(t, 9, cfg.AdaptiveIteration.MaxIterations)
	assert.Equal(t, 3, cfg.AdaptiveIteration.ConfidenceThreshold)
}

func TestGEPAAgentOptimizer_Optimize_RLMIterationPrompt(t *testing.T) {
	mutationLLM := &testLLM{
		responses: []string{
			"1. Carefully inspect the context before deciding.\n2. Verify the answer before returning it.\n3. Keep the loop disciplined.",
		},
	}
	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	core.SetDefaultLLM(mutationLLM)
	core.GlobalConfig.TeacherLLM = mutationLLM
	t.Cleanup(func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	})

	baseAgent := NewAgent("adaptive-rlm", modrlm.NewFromLLM(&testLLM{
		responses: []string{
			"Reasoning:\nDone.\n\nAction:\nfinal\n\nCode:\n\nAnswer:\ncomplete",
		},
	}))

	optimizer := optimize.NewGEPAAgentOptimizer(
		baseAgent,
		testAgentEvaluator(func(ctx context.Context, agent optimize.OptimizableAgent, ex optimize.AgentExample) (*optimize.EvalResult, error) {
			prompt := strings.ToLower(agent.GetArtifacts().Text[optimize.ArtifactRLMIterationPrompt])
			score := 0.2
			if strings.Contains(prompt, "carefully") {
				score = 1.0
			}
			return &optimize.EvalResult{
				Score: score,
				SideInfo: &optimize.SideInfo{
					Scores: map[string]float64{"prompt_fit": score},
					Diagnostics: map[string]interface{}{
						"artifact": "rlm_iteration_prompt",
					},
				},
			}, nil
		}),
		optimize.GEPAAdapterConfig{
			PopulationSize:  3,
			MaxGenerations:  1,
			ReflectionFreq:  0,
			ValidationSplit: 0,
			EvalConcurrency: 1,
			PassThreshold:   0.5,
			PrimaryArtifact: optimize.ArtifactRLMIterationPrompt,
		},
	)

	result, err := optimizer.Optimize(context.Background(), optimize.GEPAOptimizeRequest{
		SeedArtifacts: baseAgent.GetArtifacts(),
		TrainingExamples: []optimize.AgentExample{
			{ID: "train-1"},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Contains(t, strings.ToLower(result.BestArtifacts.Text[optimize.ArtifactRLMIterationPrompt]), "carefully")
}
