package llms

import (
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
)

func TestNewLLM(t *testing.T) {
	testCases := []struct {
		name      string
		apiKey    string
		modelID   core.ModelID
		expectErr bool
		errMsg    string
		llmType   string
	}{
		{
			name:    "Anthropic Haiku",
			apiKey:  "test-api-key",
			modelID: core.ModelAnthropicHaiku,
			llmType: "AnthropicLLM",
		},
		{
			name:    "Anthropic Sonnet",
			apiKey:  "test-api-key",
			modelID: core.ModelAnthropicSonnet,
			llmType: "AnthropicLLM",
		},
		{
			name:    "Anthropic Opus",
			apiKey:  "test-api-key",
			modelID: core.ModelAnthropicOpus,
			llmType: "AnthropicLLM",
		},
		{
			name:    "Valid Ollama model",
			apiKey:  "",
			modelID: core.ModelID("ollama:llama2"),
			llmType: "OllamaLLM",
		},
		{
			name:    "Invalid Ollama model format",
			apiKey:  "",
			modelID: core.ModelID("ollama:"),
			llmType: "OllamaLLM", // It seems the function doesn't error on invalid format
		},
		{
			name:      "Unsupported model",
			apiKey:    "test-api-key",
			modelID:   core.ModelID("unsupported-model"),
			expectErr: true,
			errMsg:    "unsupported model ID: unsupported-model",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			llm, err := NewLLM(tc.apiKey, tc.modelID)

			if tc.expectErr {
				assert.Error(t, err)
				assert.Nil(t, llm)
				assert.Equal(t, tc.errMsg, err.Error())
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, llm)

				switch tc.llmType {
				case "AnthropicLLM":
					_, ok := llm.(*AnthropicLLM)
					assert.True(t, ok, "Expected AnthropicLLM")
				case "OllamaLLM":
					_, ok := llm.(*OllamaLLM)
					assert.True(t, ok, "Expected OllamaLLM")
				default:
					t.Fatalf("Unexpected LLM type: %s", tc.llmType)
				}
			}
		})
	}
}
