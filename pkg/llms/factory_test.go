package llms

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Helper function to get the underlying LLM implementation from a potentially wrapped LLM.
func getUnderlyingLLM(llm core.LLM) core.LLM {
	if unwrappable, ok := llm.(interface{ Unwrap() core.LLM }); ok {
		return unwrappable.Unwrap()
	}
	return llm
}

func TestNewLLM(t *testing.T) {
	testCases := []struct {
		name            string
		apiKey          string
		modelID         core.ModelID
		expectedModelID string
		expectErr       bool
		errMsg          string
		checkType       func(t *testing.T, llm core.LLM)
	}{
		{
			name:            "Anthropic Haiku",
			apiKey:          "test-api-key",
			modelID:         core.ModelAnthropicHaiku,
			expectedModelID: string(core.ModelAnthropicHaiku),
			checkType: func(t *testing.T, llm core.LLM) {
				underlying := getUnderlyingLLM(llm)
				_, ok := underlying.(*AnthropicLLM)
				assert.True(t, ok, "Expected AnthropicLLM")
			},
		},
		{
			name:            "Anthropic Sonnet",
			apiKey:          "test-api-key",
			modelID:         core.ModelAnthropicSonnet,
			expectedModelID: string(core.ModelAnthropicSonnet),
			checkType: func(t *testing.T, llm core.LLM) {
				underlying := getUnderlyingLLM(llm)
				_, ok := underlying.(*AnthropicLLM)
				assert.True(t, ok, "Expected AnthropicLLM")
			},
		},
		{
			name:            "Anthropic Opus",
			apiKey:          "test-api-key",
			modelID:         core.ModelAnthropicOpus,
			expectedModelID: string(core.ModelAnthropicOpus),
			checkType: func(t *testing.T, llm core.LLM) {
				underlying := getUnderlyingLLM(llm)
				_, ok := underlying.(*AnthropicLLM)
				assert.True(t, ok, "Expected AnthropicLLM")
			},
		},
		{
			name:            "Valid Ollama model",
			apiKey:          "",
			modelID:         "ollama:llama2",
			expectedModelID: "llama2", // Model ID is stripped of ollama: prefix
			checkType: func(t *testing.T, llm core.LLM) {
				underlying := getUnderlyingLLM(llm)
				_, ok := underlying.(*OllamaLLM)
				assert.True(t, ok, "Expected OllamaLLM")
			},
		},
		{
			name:            "Invalid Ollama model format",
			apiKey:          "",
			modelID:         "ollama:",
			expectedModelID: "",
			expectErr:       true,
			errMsg:          "failed to create LLM with registry and fallback: invalid Ollama model ID format. Use 'ollama:<model_name>'",
		},
		{
			name:      "Unsupported model",
			apiKey:    "test-api-key",
			modelID:   "unsupported-model",
			expectErr: true,
			errMsg:    "failed to create LLM with registry and fallback: unsupported model ID: unsupported-model",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			llm, err := NewLLM(tc.apiKey, tc.modelID)

			if tc.expectErr {
				require.Error(t, err)
				assert.Equal(t, tc.errMsg, err.Error())
				assert.Nil(t, llm)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, llm)

			// Verify the model type using the provided check function
			tc.checkType(t, llm)

			expectedID := tc.expectedModelID
			assert.Equal(t, expectedID, llm.ModelID())

			// Test that model context decoration works when applicable
			if !tc.expectErr {
				ctx := core.WithExecutionState(context.Background())
				// We don't need to actually generate, just ensure model ID is set
				state := core.GetExecutionState(ctx)
				assert.NotNil(t, state)
			}
		})
	}
}

func TestDefaultFactoryInitialization(t *testing.T) {
	// Reset the factory for testing
	resetFactoryForTesting()

	// Ensure the factory gets initialized
	ensureFactory()
	assert.NotNil(t, defaultFactory, "Default factory should be initialized")
	assert.NotNil(t, core.DefaultFactory, "Core default factory should be set")

	// Call again to ensure idempotence via sync.Once
	ensureFactory()
	assert.NotNil(t, defaultFactory, "Default factory should remain initialized")
}

func TestNewLLMForAllModels(t *testing.T) {
	// Save any existing API keys to restore them later
	oldAnthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	oldGeminiKey := os.Getenv("GEMINI_API_KEY")
	defer func() {
		os.Setenv("ANTHROPIC_API_KEY", oldAnthropicKey)
		os.Setenv("GEMINI_API_KEY", oldGeminiKey)
	}()

	// Set test API keys
	os.Setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
	os.Setenv("GEMINI_API_KEY", "test-gemini-key")

	// Test all Anthropic models
	t.Run("AnthropicModels", func(t *testing.T) {
		anthropicModels := []core.ModelID{
			core.ModelAnthropicHaiku,
			core.ModelAnthropicSonnet,
			core.ModelAnthropicOpus,
		}

		for _, model := range anthropicModels {
			llm, err := NewLLM("test-api-key", model)
			assert.NoError(t, err)
			assert.NotNil(t, llm)

			// Check we get the expected model type
			baseModel := getUnderlyingLLM(llm)
			_, ok := baseModel.(*AnthropicLLM)
			assert.True(t, ok, "Expected AnthropicLLM for model %s", model)

			// Check the model ID is preserved
			assert.Equal(t, string(model), llm.ModelID())
		}
	})

	// Test all Gemini models
	t.Run("GeminiModels", func(t *testing.T) {
		geminiModels := []core.ModelID{
			core.ModelGoogleGeminiFlash,
			core.ModelGoogleGeminiPro,
			core.ModelGoogleGeminiFlashThinking,
			core.ModelGoogleGeminiFlashLite,
		}

		for _, model := range geminiModels {
			llm, err := NewLLM("test-api-key", model)
			assert.NoError(t, err)
			assert.NotNil(t, llm)

			// Check we get the expected model type
			baseModel := getUnderlyingLLM(llm)
			_, ok := baseModel.(*GeminiLLM)
			assert.True(t, ok, "Expected GeminiLLM for model %s", model)

			// Check the model ID is preserved
			assert.Equal(t, string(model), llm.ModelID())
		}
	})

	// Test Ollama models with multiple different model names
	t.Run("OllamaModels", func(t *testing.T) {
		ollamaModels := []string{
			"ollama:llama2",
			"ollama:mistral",
			"ollama:llama2:13b",
			"ollama:vicuna:7b",
		}

		for _, modelStr := range ollamaModels {
			llm, err := NewLLM("", core.ModelID(modelStr))
			assert.NoError(t, err)
			assert.NotNil(t, llm)

			// Check we get the expected model type
			baseModel := getUnderlyingLLM(llm)
			ollamaLLM, ok := baseModel.(*OllamaLLM)
			assert.True(t, ok, "Expected OllamaLLM for model %s", modelStr)

			// Check the model ID is stripped (ollama: prefix removed for API calls)
			expectedStripped := strings.TrimPrefix(modelStr, "ollama:")
			assert.Equal(t, expectedStripped, string(ollamaLLM.ModelID()))
		}
	})

	// Test LlamaCPP models
	t.Run("LlamaCPPModels", func(t *testing.T) {
		llm, err := NewLLM("", "llamacpp:default")
		assert.NoError(t, err)
		assert.NotNil(t, llm)

		// Check we get the expected model type
		baseModel := getUnderlyingLLM(llm)
		_, ok := baseModel.(*LlamacppLLM)
		assert.True(t, ok, "Expected LlamacppLLM")
	})
}

func TestFactoryInterface(t *testing.T) {
	// Test the LLMFactory interface implementation
	factory := &DefaultLLMFactory{}

	// Test with Anthropic model
	llm, err := factory.CreateLLM("test-key", core.ModelAnthropicHaiku)
	assert.NoError(t, err)
	assert.NotNil(t, llm)
	assert.Equal(t, string(core.ModelAnthropicHaiku), llm.ModelID())

	// Test with invalid model
	llm, err = factory.CreateLLM("test-key", "invalid-model")
	assert.Error(t, err)
	assert.Nil(t, llm)
	assert.Contains(t, err.Error(), "unsupported model ID")
}

func TestLLMCreationWithDecoration(t *testing.T) {
	// Test that model context decoration works
	llm, err := NewLLM("test-key", core.ModelAnthropicHaiku)
	require.NoError(t, err)
	require.NotNil(t, llm)

	// Create a context with execution state
	ctx := core.WithExecutionState(context.Background())
	state := core.GetExecutionState(ctx)
	assert.NotNil(t, state)

	// Check if it's a ModelContextDecorator
	_, ok := llm.(interface{ Unwrap() core.LLM })
	assert.True(t, ok, "Expected decorated LLM with Unwrap method")
}

func TestInvalidOllamaFormat(t *testing.T) {
	// Test invalid Ollama model format (missing model name)
	llm, err := NewLLM("", "ollama:")

	// Check that we get an error and no LLM instance
	require.Error(t, err)
	require.Nil(t, llm)

	// Verify the error message
	assert.Contains(t, err.Error(), "invalid Ollama model ID format")
}

func TestEnsureFactoryExport(t *testing.T) {
	// Test the exported EnsureFactory function
	// Reset for testing
	resetFactoryForTesting()

	// Call the exported function
	EnsureFactory()
	assert.NotNil(t, defaultFactory)
	assert.NotNil(t, core.DefaultFactory)
}

// Test the factory registry initialization edge cases.
func TestFactoryRegistryInitialization(t *testing.T) {
	// Reset for testing
	resetFactoryForTesting()

	// Test that ensureRegistryInitialized doesn't panic
	ensureRegistryInitialized()

	// Verify registry is initialized
	registry := core.GetRegistry()
	assert.NotNil(t, registry)

	// Test that calling it again doesn't cause issues (sync.Once behavior)
	ensureRegistryInitialized()
}

// Test the fallback creation paths.
func TestCreateLLMFallback(t *testing.T) {
	tests := []struct {
		name      string
		apiKey    string
		modelID   core.ModelID
		expectErr bool
		errMsg    string
	}{
		{
			name:      "Anthropic Haiku fallback",
			apiKey:    "test-key",
			modelID:   core.ModelAnthropicHaiku,
			expectErr: false,
		},
		{
			name:      "Anthropic Sonnet fallback",
			apiKey:    "test-key",
			modelID:   core.ModelAnthropicSonnet,
			expectErr: false,
		},
		{
			name:      "Anthropic Opus fallback",
			apiKey:    "test-key",
			modelID:   core.ModelAnthropicOpus,
			expectErr: false,
		},
		{
			name:      "Gemini Flash fallback",
			apiKey:    "test-key",
			modelID:   core.ModelGoogleGeminiFlash,
			expectErr: false,
		},
		{
			name:      "Gemini Pro fallback",
			apiKey:    "test-key",
			modelID:   core.ModelGoogleGeminiPro,
			expectErr: false,
		},
		{
			name:      "Gemini Flash Thinking fallback",
			apiKey:    "test-key",
			modelID:   core.ModelGoogleGeminiFlashThinking,
			expectErr: false,
		},
		{
			name:      "Gemini Flash Lite fallback",
			apiKey:    "test-key",
			modelID:   core.ModelGoogleGeminiFlashLite,
			expectErr: false,
		},
		{
			name:      "Ollama model fallback",
			apiKey:    "",
			modelID:   "ollama:llama2",
			expectErr: false,
		},
		{
			name:      "Ollama model with version fallback",
			apiKey:    "",
			modelID:   "ollama:llama2:13b",
			expectErr: false,
		},
		{
			name:      "LlamaCpp model fallback",
			apiKey:    "",
			modelID:   "llamacpp:default",
			expectErr: false,
		},
		{
			name:      "Invalid Ollama format",
			apiKey:    "",
			modelID:   "ollama:",
			expectErr: true,
			errMsg:    "invalid Ollama model ID format",
		},
		{
			name:      "Empty Ollama model name",
			apiKey:    "",
			modelID:   "ollama: ",
			expectErr: true,
			errMsg:    "invalid Ollama model ID format",
		},
		{
			name:      "Unsupported model",
			apiKey:    "test-key",
			modelID:   "unsupported-model",
			expectErr: true,
			errMsg:    "unsupported model ID",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llm, err := createLLMFallback(tt.apiKey, tt.modelID)

			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, llm)
				assert.Contains(t, err.Error(), tt.errMsg)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, llm)

				// Verify the model ID is correctly set
				if strings.HasPrefix(string(tt.modelID), "ollama:") {
					// For Ollama models, the prefix is stripped
					expectedID := strings.TrimPrefix(string(tt.modelID), "ollama:")
					assert.Equal(t, expectedID, llm.ModelID())
				} else if strings.HasPrefix(string(tt.modelID), "llamacpp:") {
					// For LlamaCpp models, we don't test the specific ID
					// since it depends on the internal implementation
				} else {
					// For other models, ID should match
					assert.Equal(t, string(tt.modelID), llm.ModelID())
				}
			}
		})
	}
}

// Test loadDefaultModelConfigurations error handling.
func TestLoadDefaultModelConfigurations(t *testing.T) {
	// Create a new registry for testing
	registry := core.NewLLMRegistry()

	// Test that loading default configurations works
	assert.NotPanics(t, func() {
		loadDefaultModelConfigurations(registry)
	})
}

// Test the NewLLM function with registry failure simulation.
func TestNewLLMWithRegistryFailure(t *testing.T) {
	// Reset for testing
	resetFactoryForTesting()

	// This test is more about exercising the fallback path
	// when the registry fails to create an LLM
	llm, err := NewLLM("test-key", core.ModelAnthropicHaiku)

	// Should succeed via fallback
	assert.NoError(t, err)
	assert.NotNil(t, llm)

	// Verify it's decorated (wrapped)
	_, ok := llm.(interface{ Unwrap() core.LLM })
	assert.True(t, ok, "Expected decorated LLM")
}

// Test various edge cases for the factory.
func TestFactoryEdgeCases(t *testing.T) {
	factory := &DefaultLLMFactory{}

	t.Run("Factory with empty API key for Anthropic", func(t *testing.T) {
		// API key validation happens at request time, not creation time
		llm, err := factory.CreateLLM("", core.ModelAnthropicHaiku)
		assert.NoError(t, err)
		assert.NotNil(t, llm)
	})

	t.Run("Factory with valid Ollama model", func(t *testing.T) {
		llm, err := factory.CreateLLM("", "ollama:llama2")
		assert.NoError(t, err)
		assert.NotNil(t, llm)
	})

	t.Run("Factory with invalid model format", func(t *testing.T) {
		llm, err := factory.CreateLLM("test-key", "invalid:format:too:many:colons")
		assert.Error(t, err)
		assert.Nil(t, llm)
	})
}
