package registry

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
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
	t.Run("CreateLLM success", func(t *testing.T) {
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
	})

	t.Run("model not supported", func(t *testing.T) {
		registry := NewLLMRegistry()
		require.NoError(t, registry.RegisterProvider("mock", testProviderFactory))

		_, err := registry.CreateLLM(context.Background(), "apikey", core.ModelID("unknown"))

		var e *errors.Error
		require.ErrorAs(t, err, &e)
		require.Equal(t, errors.ModelNotSupported, e.Code())
	})

	t.Run("provider not registered", func(t *testing.T) {
		registry := NewLLMRegistry()
		require.NoError(t, registry.RegisterProvider("mock", testProviderFactory))

		_, err := registry.CreateLLM(context.Background(), "apikey", core.ModelID("gpt-5"))

		var e *errors.Error
		require.ErrorAs(t, err, &e)
		require.Equal(t, errors.ProviderNotFound, e.Code())
	})
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

func TestNewLLMRegistry_RegisterProvider(t *testing.T) {
	t.Run("provider name cannot be empty", func(t *testing.T) {
		registry := NewLLMRegistry()

		err := registry.RegisterProvider("", testProviderFactory)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "name cannot be empty")
	})

	t.Run("factory cannot be nil", func(t *testing.T) {
		registry := NewLLMRegistry()

		err := registry.RegisterProvider("name", nil)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "provider factory cannot be nil")
	})
}

func TestNewLLMRegistry_UnregistryProvider(t *testing.T) {
	registry := NewLLMRegistry()

	require.NoError(t, registry.RegisterProvider("one", testProviderFactory))
	require.NoError(t, registry.RegisterProvider("two", testProviderFactory))

	require.NoError(t, registry.LoadFromConfig(context.Background(), map[string]core.ProviderConfig{
		"one": {
			Name: "one",
			Models: map[string]core.ModelConfig{
				"test-one": {ID: "test-one"},
			},
		},
	}))
	require.NoError(t, registry.LoadFromConfig(context.Background(), map[string]core.ProviderConfig{
		"two": {
			Name: "two",
			Models: map[string]core.ModelConfig{
				"test-two": {ID: "test-two"},
			},
		},
	}))

	require.NoError(t, registry.UnregisterProvider("one"))

	providers := registry.ListProviders()
	require.Equal(t, []string{"two"}, providers)
}

func TestNewLLMRegistry_CreateLLMWithConfig(t *testing.T) {
	t.Run("provider not found", func(t *testing.T) {
		registry := NewLLMRegistry()

		_, err := registry.CreateLLMWithConfig(context.Background(), core.ProviderConfig{Name: "missing"}, core.ModelID("model"))

		var e *errors.Error
		require.ErrorAs(t, err, &e)
		require.Equal(t, errors.ProviderNotFound, e.Code())
	})

	t.Run("uses passed config", func(t *testing.T) {
		var config core.ProviderConfig
		registry := NewLLMRegistry()
		require.NoError(t, registry.RegisterProvider("mock", func(ctx context.Context, input core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
			config = input
			input.Params["top_p"] = 0.1
			return newTestLLM(config.Name, string(modelID)), nil
		}))
		require.NoError(t, registry.LoadFromConfig(context.Background(), map[string]core.ProviderConfig{
			"mock": {Name: "mock", APIKey: "apikey"},
		}))

		passed := core.ProviderConfig{Name: "mock", APIKey: "passed-apikey", Params: map[string]any{
			"top_p": 0.9,
		}}
		_, err := registry.CreateLLMWithConfig(context.Background(), passed, core.ModelID("not-present"))

		require.NoError(t, err)
		require.Equal(t, "passed-apikey", config.APIKey)
		require.Equal(t, 0.9, passed.Params["top_p"])
	})
}

func TestNewLLMRegistry_IsModelSupported(t *testing.T) {
	registry := NewLLMRegistry()
	require.NoError(t, registry.RegisterProvider("mock", testProviderFactory))
	require.NoError(t, registry.LoadFromConfig(context.Background(), map[string]core.ProviderConfig{
		"mock": {Name: "mock", Models: map[string]core.ModelConfig{"test": {ID: "test"}}},
	}))

	tests := []struct {
		name    string
		modelID core.ModelID
		want    bool
	}{
		{"configured", "test", true},
		{"inferred", "gpt-5", true},
		{"unknown", "unknown", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, registry.IsModelSupported(tt.modelID))
		})
	}
}

func TestNewLLMRegistry_GetModelProvider(t *testing.T) {
	registry := NewLLMRegistry()
	require.NoError(t, registry.RegisterProvider("mock", testProviderFactory))
	require.NoError(t, registry.LoadFromConfig(context.Background(), map[string]core.ProviderConfig{
		"mock": {Name: "mock", Models: map[string]core.ModelConfig{
			"test":   {ID: "test"},
			"gpt-4o": {ID: "gpt-4o"},
		}},
	}))

	tests := []struct {
		name         string
		modelID      core.ModelID
		wantProvider string
		wantOK       bool
	}{
		{"configured", "test", "mock", true},
		{"configured preferred", "gpt-4o", "mock", true},
		{"inferred", "gpt-4", "openai", true},
		{"unknown", "unknown", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, ok := registry.GetModelProvider(tt.modelID)
			require.Equal(t, tt.wantOK, ok)
			require.Equal(t, tt.wantProvider, provider)
		})
	}
}

func TestNewLLMRegistry_SetDefaultProvider(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		registry := NewLLMRegistry()
		require.NoError(t, registry.RegisterProvider("mock", testProviderFactory))

		require.NoError(t, registry.SetDefaultProvider("mock"))
	})

	t.Run("provider not found", func(t *testing.T) {
		registry := NewLLMRegistry()
		require.NoError(t, registry.RegisterProvider("mock", testProviderFactory))

		err := registry.SetDefaultProvider("unknown")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "provider not found")
	})
}

func TestNewLLMRegistry_RefreshProvider(t *testing.T) {
	ctx := context.Background()

	t.Run("success", func(t *testing.T) {
		registry := NewLLMRegistry()

		err := registry.RefreshProvider(ctx, "missing", core.ProviderConfig{Name: "missing"})

		var e *errors.Error
		require.ErrorAs(t, err, &e)
		require.Equal(t, errors.ProviderNotFound, e.Code())
	})

	t.Run("keep other models", func(t *testing.T) {
		registry := NewLLMRegistry()
		require.NoError(t, registry.RegisterProvider("one", testProviderFactory))
		require.NoError(t, registry.RegisterProvider("two", testProviderFactory))
		require.NoError(t, registry.LoadFromConfig(ctx, map[string]core.ProviderConfig{
			"one": {Name: "one", Models: map[string]core.ModelConfig{"model-one": {ID: "model-one"}}},
			"two": {Name: "two", Models: map[string]core.ModelConfig{"model-two": {ID: "model-two"}}},
		}))

		require.NoError(t, registry.RefreshProvider(ctx, "one", core.ProviderConfig{Name: "one"}))

		require.False(t, registry.IsModelSupported("model-one"))
		p, ok := registry.GetModelProvider("model-two")
		require.True(t, ok)
		require.Equal(t, "two", p)
	})
}
