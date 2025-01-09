package llms

import (
	"context"
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
			expectedModelID: "llama2",
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
			checkType: func(t *testing.T, llm core.LLM) {
				underlying := getUnderlyingLLM(llm)
				_, ok := underlying.(*OllamaLLM)
				assert.True(t, ok, "Expected OllamaLLM")
			},
		},
		{
			name:      "Unsupported model",
			apiKey:    "test-api-key",
			modelID:   "unsupported-model",
			expectErr: true,
			errMsg:    "unsupported model ID: unsupported-model",
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
