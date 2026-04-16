package registry

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/require"
)

type testLLM struct {
	*core.BaseLLM
}

func newTestLLM(providerName, modelID string) *testLLM {
	return &testLLM{
		BaseLLM: core.NewBaseLLM(providerName, core.ModelID(modelID), []core.Capability{
			core.CapabilityCompletion,
			core.CapabilityChat,
		}, nil),
	}
}

func (m *testLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{Content: "ok"}, nil
}

func (m *testLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]any, error) {
	return map[string]any{"ok": true}, nil
}

func (m *testLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	return map[string]any{"ok": true}, nil
}

func (m *testLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return &core.EmbeddingResult{Vector: []float32{1}}, nil
}

func (m *testLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return &core.BatchEmbeddingResult{
		Embeddings: []core.EmbeddingResult{{Vector: []float32{1}}},
	}, nil
}

func (m *testLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	chunks := make(chan core.StreamChunk, 1)
	chunks <- core.StreamChunk{Done: true}
	close(chunks)
	return &core.StreamResponse{ChunkChannel: chunks, Cancel: func() {}}, nil
}

func (m *testLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{Content: "ok"}, nil
}

func (m *testLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	chunks := make(chan core.StreamChunk, 1)
	chunks <- core.StreamChunk{Done: true}
	close(chunks)
	return &core.StreamResponse{ChunkChannel: chunks, Cancel: func() {}}, nil
}

func testProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return newTestLLM(config.Name, string(modelID)), nil
}

func TestNewLLMRegistry_CreateLLM(t *testing.T) {
	registry := NewLLMRegistry()
	require.NoError(t, registry.RegisterProvider("mock", testProviderFactory))
	require.NoError(t, registry.LoadFromConfig(context.Background(), map[string]core.ProviderConfig{
		"mock": {
			Name: "mock",
			Models: map[string]core.ModelConfig{
				"test-model": {ID: "test-model"},
			},
		},
	}))

	llm, err := registry.CreateLLM(context.Background(), "test-key", core.ModelID("test-model"))
	require.NoError(t, err)
	require.Equal(t, "mock", llm.ProviderName())
	require.Equal(t, "test-model", llm.ModelID())
}

func TestNewLLMRegistry_GetProviderConfigClones(t *testing.T) {
	registry := NewLLMRegistry()
	require.NoError(t, registry.RegisterProvider("mock", testProviderFactory))
	require.NoError(t, registry.LoadFromConfig(context.Background(), map[string]core.ProviderConfig{
		"mock": {
			Name: "mock",
			Models: map[string]core.ModelConfig{
				"test-model": {
					ID:     "test-model",
					Params: map[string]any{"mode": "original"},
				},
			},
			Params: map[string]any{"top_p": 0.9},
		},
	}))

	config, ok := registry.GetProviderConfig("mock")
	require.True(t, ok)
	config.Params["top_p"] = 0.1
	model := config.Models["test-model"]
	model.Params["mode"] = "mutated"
	config.Models["test-model"] = model

	reloaded, ok := registry.GetProviderConfig("mock")
	require.True(t, ok)
	require.Equal(t, 0.9, reloaded.Params["top_p"])
	require.Equal(t, "original", reloaded.Models["test-model"].Params["mode"])
}
