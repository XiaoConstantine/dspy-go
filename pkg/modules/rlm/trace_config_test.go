package rlm

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRLMCompleteWithTrace_RecordsStructuredSteps(t *testing.T) {
	llm := &mockLLM{
		responses: []string{
			"Reasoning:\nI'll provide the answer directly.\n\nAction:\nfinal\n\nCode:\n\nAnswer:\ntrace answer",
		},
	}

	module := NewFromLLM(llm, WithMaxIterations(1))

	result, trace, err := module.CompleteWithTrace(context.Background(), "ctx", "what is the answer?")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, trace)

	assert.Equal(t, "trace answer", result.Response)
	assert.Equal(t, "final_answer", trace.TerminationCause)
	assert.Equal(t, 1, trace.Iterations)
	assert.Equal(t, "trace answer", trace.Output["answer"])
	require.Len(t, trace.Steps, 1)
	assert.Equal(t, "final", trace.Steps[0].Action)
	assert.Equal(t, "I'll provide the answer directly.", trace.Steps[0].Thought)
}

func TestRLMSetConfig_RebuildsPromptInstructions(t *testing.T) {
	llm := &mockLLM{
		responses: []string{
			"Reasoning:\nDone.\n\nAction:\nfinal\n\nCode:\n\nAnswer:\ncomplete",
		},
	}

	module := NewFromLLM(llm)
	cfg := module.Config()
	cfg.OuterInstruction = "Custom outer instruction."
	cfg.IterationInstruction = "Custom iteration instruction."
	cfg.UseIterationDemos = true
	module.SetConfig(cfg)

	assert.Equal(t, "Custom outer instruction.", module.GetSignature().Instruction)
	require.NotNil(t, module.iterationModule)
	assert.Equal(t, "Custom iteration instruction.", module.iterationModule.GetSignature().Instruction)
	assert.True(t, module.Config().UseIterationDemos)
}

func TestRLMCompleteWithTrace_UsesRequestScopedTokenTracker(t *testing.T) {
	llm := &concurrentTraceLLM{
		response: "Reasoning:\nI'll query once and then finish.\n\nAction:\nquery\n\nCode:\nanswer := QueryRaw(\"what is the answer?\")\nFINAL(answer)\n\nAnswer:\n",
	}
	sub := newBarrierSubLLMClient(2, "42")
	module := New(llm, sub, WithMaxIterations(1))

	type callResult struct {
		trace *RLMTrace
		err   error
	}

	results := make(chan callResult, 2)
	run := func(query string) {
		_, trace, err := module.CompleteWithTrace(context.Background(), "ctx", query)
		results <- callResult{trace: trace, err: err}
	}

	go run("first")
	go run("second")

	for i := 0; i < 2; i++ {
		result := <-results
		require.NoError(t, result.err)
		require.NotNil(t, result.trace)
		assert.Equal(t, 1, result.trace.SubLLMCallCount)
		assert.Equal(t, 0, result.trace.SubRLMCallCount)
		assert.Equal(t, 300, result.trace.Usage.TotalTokens)
	}

	require.NotNil(t, module.GetTokenTracker())
	assert.Equal(t, 300, module.GetTokenTracker().GetTotalUsage().TotalTokens)
}

type concurrentTraceLLM struct {
	mu       sync.Mutex
	response string
}

func (m *concurrentTraceLLM) Generate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	usage := core.TokenInfo{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	}
	return &core.LLMResponse{Content: m.response, Usage: &usage}, nil
}

func (m *concurrentTraceLLM) GenerateWithJSON(ctx context.Context, prompt string, opts ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, fmt.Errorf("JSON generation not implemented in test LLM")
}

func (m *concurrentTraceLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, opts ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, fmt.Errorf("function calling not implemented in test LLM")
}

func (m *concurrentTraceLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	return m.Generate(ctx, "", opts...)
}

func (m *concurrentTraceLLM) StreamGenerate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("streaming not implemented in test LLM")
}

func (m *concurrentTraceLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, opts ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("streaming not implemented in test LLM")
}

func (m *concurrentTraceLLM) CreateEmbedding(ctx context.Context, input string, opts ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, fmt.Errorf("embeddings not implemented in test LLM")
}

func (m *concurrentTraceLLM) CreateEmbeddings(ctx context.Context, inputs []string, opts ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, fmt.Errorf("batch embeddings not implemented in test LLM")
}

func (m *concurrentTraceLLM) ProviderName() string { return "mock" }
func (m *concurrentTraceLLM) ModelID() string      { return "mock-model" }
func (m *concurrentTraceLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

type barrierSubLLMClient struct {
	response string
	ready    chan struct{}
	release  chan struct{}
	once     sync.Once
	mu       sync.Mutex
	waiting  int
	target   int
}

func newBarrierSubLLMClient(target int, response string) *barrierSubLLMClient {
	return &barrierSubLLMClient{
		response: response,
		ready:    make(chan struct{}),
		release:  make(chan struct{}),
		target:   target,
	}
}

func (c *barrierSubLLMClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	c.mu.Lock()
	c.waiting++
	if c.waiting == c.target {
		close(c.ready)
		c.once.Do(func() {
			close(c.release)
		})
	}
	c.mu.Unlock()

	select {
	case <-ctx.Done():
		return QueryResponse{}, ctx.Err()
	case <-c.ready:
	}
	select {
	case <-ctx.Done():
		return QueryResponse{}, ctx.Err()
	case <-c.release:
	}

	return QueryResponse{
		Response:         c.response,
		PromptTokens:     100,
		CompletionTokens: 50,
	}, nil
}

func (c *barrierSubLLMClient) QueryWithContext(ctx context.Context, prompt string, _ []string) (QueryResponse, error) {
	return c.Query(ctx, prompt)
}

func (c *barrierSubLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	responses := make([]QueryResponse, 0, len(prompts))
	for range prompts {
		response, err := c.Query(ctx, "")
		if err != nil {
			return nil, err
		}
		responses = append(responses, response)
	}
	return responses, nil
}
