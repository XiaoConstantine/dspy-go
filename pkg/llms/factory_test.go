package llms

import (
	"slices"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
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
			name:    "ollama compatible model",
			modelID: "ollama:llama3.2", provider: "ollama", model: "llama3.2",
		},
		{
			name:    "llama.cpp compatible model",
			modelID: "llamacpp:local", provider: "llamacpp", model: "local",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model, err := NewLLM(test.apiKey, test.modelID)
			require.NoError(t, err)

			adapted, ok := unwrapLLM(model).(*GeneratorLLM)
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

func unwrapLLM(model core.LLM) core.LLM {
	for {
		unwrappable, ok := model.(interface{ Unwrap() core.LLM })
		if !ok {
			return model
		}
		model = unwrappable.Unwrap()
	}
}
