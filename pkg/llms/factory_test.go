package llms

import (
	"context"
	"slices"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	llm "github.com/XiaoConstantine/llm-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultFactoryRegistersLLMGoProviders(t *testing.T) {
	resetFactoryForTesting()
	t.Cleanup(func() {
		resetFactoryForTesting()
		require.NoError(t, ensureFactory())
	})

	require.NoError(t, ensureFactory())
	require.Same(t, defaultFactory, core.DefaultFactory)

	providers := core.GetRegistry().ListProviders()
	slices.Sort(providers)
	assert.Equal(t, []string{
		"anthropic",
		"fastchat",
		"google",
		"litellm",
		"llamacpp",
		"localai",
		"ollama",
		"openai",
		"openai-codex",
	}, providers)

	supported := core.GetSupportedModels()
	assert.Contains(t, supported["anthropic"], string(core.ModelAnthropicClaude45Haiku))
	assert.Contains(t, supported["google"], string(core.ModelGoogleGeminiFlash))
	assert.Contains(t, supported["openai"], string(core.ModelOpenAIGPT4o))
	assert.Contains(t, supported["ollama"], string(core.ModelOllamaLlama3_2_3B))
	assert.Contains(t, supported["litellm"], string(core.ModelLiteLLMClaude3))
	assert.Contains(t, supported["localai"], string(core.ModelLocalAICodeLlama))
	assert.Contains(t, supported["fastchat"], string(core.ModelFastChatVicuna))
	assert.Contains(t, supported, "llamacpp")
	assert.Contains(t, supported, "openai-codex")

	// Initialization is idempotent.
	require.NoError(t, ensureFactory())
}

func TestNewLLMUsesLLMGoGenerators(t *testing.T) {
	t.Setenv("ANTHROPIC_OAUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")

	tests := []struct {
		name     string
		apiKey   string
		modelID  core.ModelID
		provider string
		model    string
	}{
		{
			name: "anthropic catalog model", apiKey: "anthropic-key",
			modelID: core.ModelAnthropicHaiku, provider: "anthropic", model: string(core.ModelAnthropicHaiku),
		},
		{
			name: "anthropic alias", apiKey: "anthropic-key",
			modelID: "haiku-4.5", provider: "anthropic", model: string(core.ModelAnthropicHaiku),
		},
		{
			name: "google catalog model", apiKey: "google-key",
			modelID: core.ModelGoogleGeminiFlash, provider: "google", model: string(core.ModelGoogleGeminiFlash),
		},
		{
			name: "openai catalog model", apiKey: "openai-key",
			modelID: core.ModelOpenAIGPT4o, provider: "openai", model: string(core.ModelOpenAIGPT4o),
		},
		{
			name: "OpenAI model shared with a compatible provider", apiKey: "openai-key",
			modelID: core.ModelOpenAIGPT4, provider: "openai", model: string(core.ModelOpenAIGPT4),
		},
		{
			name:    "ollama compatible model",
			modelID: "ollama:llama3.2", provider: "ollama", model: "llama3.2",
		},
		{
			name:    "llama.cpp compatible model",
			modelID: "llamacpp:local", provider: "llamacpp", model: "local",
		},
		{
			name: "LiteLLM registry model", apiKey: "litellm-key",
			modelID: core.ModelLiteLLMClaude3, provider: "litellm", model: string(core.ModelLiteLLMClaude3),
		},
		{
			name:    "LocalAI registry model",
			modelID: core.ModelLocalAICodeLlama, provider: "localai", model: string(core.ModelLocalAICodeLlama),
		},
		{
			name:    "LocalAI model shared with FastChat",
			modelID: core.ModelLocalAILlama2, provider: "localai", model: string(core.ModelLocalAILlama2),
		},
		{
			name:    "FastChat registry model",
			modelID: core.ModelFastChatVicuna, provider: "fastchat", model: string(core.ModelFastChatVicuna),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model, err := NewLLM(test.apiKey, test.modelID)
			require.NoError(t, err)

			adapted, ok := unwrapLLM(model).(interface {
				ProviderName() string
				ModelID() string
				Generator() llm.Generator
			})
			require.True(t, ok)
			assert.Equal(t, test.provider, adapted.ProviderName())
			assert.Equal(t, test.model, adapted.ModelID())
			assert.NotNil(t, adapted.Generator())
		})
	}
}

func TestNewLLMRejectsUnsupportedOrIncompleteModelIDs(t *testing.T) {
	for _, modelID := range []core.ModelID{"not-a-model", "ollama:", "openai-codex:"} {
		t.Run(string(modelID), func(t *testing.T) {
			model, err := NewLLM("key", modelID)
			require.Error(t, err)
			assert.Nil(t, model)
		})
	}
}

func TestOpenAICodexProviderInference(t *testing.T) {
	assert.Equal(t, "openai-codex", core.InferProviderFromModelID("openai-codex:gpt-5.2-codex"))
}

func TestProviderWrappersHaveDistinctRuntimeTypes(t *testing.T) {
	t.Setenv("ANTHROPIC_OAUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	anthropicModel, err := NewAnthropicLLMFromConfig(context.Background(), core.ProviderConfig{
		Name: "anthropic", APIKey: "anthropic-key",
	}, core.ModelAnthropicClaude45Haiku)
	require.NoError(t, err)
	openAIModel, err := NewOpenAI(core.ModelOpenAIGPT4o, "openai-key")
	require.NoError(t, err)

	assert.Equal(t, "anthropic", providerRuntimeType(anthropicModel))
	assert.Equal(t, "openai", providerRuntimeType(openAIModel))
	_, wrongType := any(openAIModel).(*AnthropicLLM)
	assert.False(t, wrongType)
}

func providerRuntimeType(model core.LLM) string {
	switch model.(type) {
	case *AnthropicLLM:
		return "anthropic"
	case *GeminiLLM:
		return "google"
	case *OpenAILLM:
		return "openai"
	case *OpenAICodexLLM:
		return "openai-codex"
	case *OllamaLLM:
		return "ollama"
	case *LlamacppLLM:
		return "llamacpp"
	default:
		return ""
	}
}

func unwrapLLM(model core.LLM) core.LLM {
	for {
		unwrappable, ok := model.(interface{ Unwrap() core.LLM })
		if !ok {
			return model
		}
		model = unwrappable.Unwrap()
	}
}
