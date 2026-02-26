package llms

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	stdErr "errors"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestAnthropicLLM_Generate(t *testing.T) {
	prompt := "example prompt"
	ctx := context.Background()
	tests := []struct {
		name        string
		setupMock   func(*testutil.MockLLM)
		options     []core.GenerateOption
		wantResp    *core.LLMResponse
		wantErr     bool
		expectedErr string
	}{
		{
			name: "Successful generation",
			setupMock: func(mockLLM *testutil.MockLLM) {
				mockLLM.On("Generate", mock.Anything, prompt, mock.Anything).Return(&core.LLMResponse{Content: "Generated response"}, nil)
			},
			options:  []core.GenerateOption{core.WithMaxTokens(100), core.WithTemperature(0.7)},
			wantResp: &core.LLMResponse{Content: "Generated response"},
			wantErr:  false,
		},
		{
			name: "API error",
			setupMock: func(mockLLM *testutil.MockLLM) {
				mockLLM.On("Generate", mock.Anything, prompt, mock.Anything).Return(nil, errors.WithFields(
					errors.Wrap(stdErr.New("API error"), errors.LLMGenerationFailed, "failed to generate response"),
					errors.Fields{},
				))
			},
			wantErr:     true,
			expectedErr: "failed to generate response: API error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			mockLLM := new(testutil.MockLLM)
			tt.setupMock(mockLLM)

			resp, err := mockLLM.Generate(ctx, prompt, tt.options...)

			if tt.wantErr {
				assert.Error(t, err)
				if customErr, ok := err.(*errors.Error); ok {
					assert.Equal(t, tt.expectedErr, customErr.Error())
				} else {
					t.Errorf("expected error of type *errors.Error, got %T", err)
				}
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.wantResp, resp)
			}
			mockLLM.AssertExpectations(t)

		})
	}
}

func TestAnthropicLLM_NewClient(t *testing.T) {
	testCases := []struct {
		name      string
		apiKey    string
		model     anthropic.Model
		expectErr bool
	}{
		{
			name:      "Valid API key",
			apiKey:    "test-valid-key",
			model:     anthropic.ModelClaudeOpus4_1_20250805,
			expectErr: false,
		},
		// Note: The Anthropic library doesn't validate API keys at client creation time
		// It will only fail when making API calls, so we've removed the "Empty API key" test case
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			llm, err := NewAnthropicLLM(tc.apiKey, tc.model)

			if tc.expectErr {
				assert.Error(t, err)
				assert.Nil(t, llm)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, llm)
				assert.Equal(t, string(tc.model), llm.ModelID())

				// Check that capabilities are set correctly
				assert.Contains(t, llm.Capabilities(), core.CapabilityCompletion)
				assert.Contains(t, llm.Capabilities(), core.CapabilityChat)
				assert.Contains(t, llm.Capabilities(), core.CapabilityJSON)
				assert.Contains(t, llm.Capabilities(), core.CapabilityStreaming)
				assert.Contains(t, llm.Capabilities(), core.CapabilityToolCalling)
			}
		})
	}
}

func TestAnthropicLLM_GenerateWithJSON(t *testing.T) {
	// Setup
	mockLLM := new(testutil.MockLLM)
	prompt := "Generate JSON: {\"name\": \"John\", \"age\": 30}"
	expectedJSON := map[string]interface{}{
		"name": "John",
		"age":  float64(30),
	}

	ctx := context.Background()

	// Setup the mock to expect GenerateWithJSON directly
	mockLLM.On("GenerateWithJSON", ctx, prompt, mock.Anything).Return(expectedJSON, nil)

	// Call GenerateWithJSON
	result, err := mockLLM.GenerateWithJSON(ctx, prompt)

	// Assertions
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "John", result["name"])
	assert.Equal(t, float64(30), result["age"])

	mockLLM.AssertExpectations(t)

	// Test error case
	mockLLM.On("GenerateWithJSON", ctx, "Invalid JSON prompt", mock.Anything).Return(nil, stdErr.New("Invalid JSON"))

	result, err = mockLLM.GenerateWithJSON(ctx, "Invalid JSON prompt")
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Equal(t, "Invalid JSON", err.Error())
}

func TestAnthropicLLM_StreamGenerate_Cancel(t *testing.T) {
	// Since we can't control the actual Anthropic client's streaming
	// we'll just test that the cancel function doesn't panic

	// Create an actual AnthropicLLM
	llm, err := NewAnthropicLLM("test-key", anthropic.ModelClaudeOpus4_1_20250805)
	require.NoError(t, err)

	// Call StreamGenerate
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Call StreamGenerate but immediately cancel it
	stream, err := llm.StreamGenerate(ctx, "Test prompt")
	require.NoError(t, err)

	// Test the cancel function
	stream.Cancel() // Should not panic
}

func TestAnthropicLLM_GenerateErrorCases(t *testing.T) {
	// Use a mock LLM instead of creating a real client with API key
	mockLLM := new(testutil.MockLLM)

	// Set up the mock to return an error when Generate is called
	errMsg := "API key invalid"
	mockErr := errors.New(errors.LLMGenerationFailed, errMsg)
	mockLLM.On("Generate", mock.Anything, "Test prompt", mock.Anything).Return(nil, mockErr)

	// Call Generate, which should fail with our mocked error
	ctx := context.Background()
	prompt := "Test prompt"
	resp, err := mockLLM.Generate(ctx, prompt)

	// Verify the results
	assert.Error(t, err)
	assert.Nil(t, resp)

	// Check that error is the one we defined
	assert.Contains(t, err.Error(), errMsg)

	// Verify the mock was called as expected
	mockLLM.AssertExpectations(t)
}

func TestAnthropicLLM_GenerateEdgeCases(t *testing.T) {
	// Test nil message case
	t.Run("Nil message response", func(t *testing.T) {
		mockLLM := new(testutil.MockLLM)
		mockLLM.On("Generate", mock.Anything, "test prompt", mock.Anything).
			Return(nil, errors.New(errors.LLMGenerationFailed, "Received nil response from Anthropic API"))

		resp, err := mockLLM.Generate(context.Background(), "test prompt")
		assert.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "Received nil response")
	})

	// Test empty content case
	t.Run("Empty content response", func(t *testing.T) {
		mockLLM := new(testutil.MockLLM)
		mockLLM.On("Generate", mock.Anything, "test prompt", mock.Anything).
			Return(nil, errors.New(errors.LLMGenerationFailed, "Received empty content from Anthropic API"))

		resp, err := mockLLM.Generate(context.Background(), "test prompt")
		assert.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "Received empty content")
	})
}

func TestAnthropicLLM_Embeddings(t *testing.T) {
	// Test the embedding methods which are stubs
	llm, err := NewAnthropicLLM("test-key", anthropic.ModelClaudeOpus4_1_20250805)
	require.NoError(t, err)

	ctx := context.Background()

	// Test CreateEmbedding
	result, err := llm.CreateEmbedding(ctx, "test text")
	assert.Nil(t, result)
	assert.Nil(t, err)

	// Test CreateEmbeddings
	batchResult, err := llm.CreateEmbeddings(ctx, []string{"test1", "test2"})
	assert.Nil(t, batchResult)
	assert.Nil(t, err)
}

func TestAnthropicLLM_GenerateWithFunctions(t *testing.T) {
	var capturedRequest map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "POST", r.Method)
		require.Equal(t, "/v1/messages", r.URL.Path)

		require.NoError(t, json.NewDecoder(r.Body).Decode(&capturedRequest))

		response := map[string]interface{}{
			"id":    "msg_123",
			"type":  "message",
			"role":  "assistant",
			"model": "claude-sonnet-4-5",
			"content": []map[string]interface{}{
				{
					"type": "text",
					"text": "I'll use a tool to check that.",
				},
				{
					"type":  "tool_use",
					"id":    "toolu_123",
					"name":  "get_weather",
					"input": map[string]interface{}{"location": "New York", "unit": "celsius"},
				},
			},
			"stop_reason":   "tool_use",
			"stop_sequence": "",
			"usage": map[string]interface{}{
				"input_tokens":  14,
				"output_tokens": 9,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		require.NoError(t, json.NewEncoder(w).Encode(response))
	}))
	defer server.Close()

	llm, err := NewAnthropicLLMFromConfig(context.Background(), core.ProviderConfig{
		Name:   "anthropic",
		APIKey: "test-key",
		Endpoint: &core.EndpointConfig{
			BaseURL: server.URL,
		},
	}, core.ModelID("claude-sonnet-4-5"))
	require.NoError(t, err)

	functions := []map[string]interface{}{
		{
			"name":        "get_weather",
			"description": "Get weather by city",
			"parameters": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]interface{}{"type": "string"},
					"unit":     map[string]interface{}{"type": "string"},
				},
				"required": []interface{}{"location"},
			},
		},
	}

	result, err := llm.GenerateWithFunctions(context.Background(), "What's the weather?", functions)
	require.NoError(t, err)

	toolCall, ok := result["function_call"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "get_weather", toolCall["name"])

	arguments, ok := toolCall["arguments"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "New York", arguments["location"])
	assert.Equal(t, "celsius", arguments["unit"])

	usage, ok := result["_usage"].(*core.TokenInfo)
	require.True(t, ok)
	assert.Equal(t, 14, usage.PromptTokens)
	assert.Equal(t, 9, usage.CompletionTokens)

	tools, ok := capturedRequest["tools"].([]interface{})
	require.True(t, ok)
	require.Len(t, tools, 1)
	toolDef, ok := tools[0].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "get_weather", toolDef["name"])
	assert.Equal(t, "Get weather by city", toolDef["description"])

	toolSchema, ok := toolDef["input_schema"].(map[string]interface{})
	require.True(t, ok)
	properties, ok := toolSchema["properties"].(map[string]interface{})
	require.True(t, ok)
	_, hasLocation := properties["location"]
	assert.True(t, hasLocation)
	_, hasUnit := properties["unit"]
	assert.True(t, hasUnit)

	toolChoice, ok := capturedRequest["tool_choice"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "auto", toolChoice["type"])
}

func TestAnthropicLLM_GenerateWithFunctions_Validation(t *testing.T) {
	llm, err := NewAnthropicLLM("test-key", anthropic.ModelClaudeOpus4_1_20250805)
	require.NoError(t, err)

	_, err = llm.GenerateWithFunctions(context.Background(), "prompt", []map[string]interface{}{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least one function schema is required")

	_, err = llm.GenerateWithFunctions(context.Background(), "prompt", []map[string]interface{}{
		{
			"name": "bad_tool",
			"parameters": map[string]interface{}{
				"type": "array",
			},
		},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must have type 'object'")
}

func TestAnthropicLLM_StreamGenerate_ChunkHandling(t *testing.T) {
	// Since we can't easily test the streaming behavior directly,
	// we'll test that the stream response has the expected structure

	llm, err := NewAnthropicLLM("test-key", anthropic.ModelClaudeOpus4_1_20250805)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Call StreamGenerate
	stream, err := llm.StreamGenerate(ctx, "Test prompt")
	require.NoError(t, err)

	// Check that the stream has a chunk channel and cancel function
	assert.NotNil(t, stream.ChunkChannel)
	assert.NotNil(t, stream.Cancel)

	// Cancel immediately to prevent actual API calls
	stream.Cancel()
}

func TestAnthropicLLM_Implementation(t *testing.T) {
	// Test basic implementation details
	llm, err := NewAnthropicLLM("test-key", anthropic.ModelClaudeOpus4_1_20250805)
	require.NoError(t, err)

	// Check provider name
	assert.Equal(t, "anthropic", llm.ProviderName())

	// Check model ID
	assert.Equal(t, string(anthropic.ModelClaudeOpus4_1_20250805), llm.ModelID())
}

// Test the actual implementation coverage that was missing.
func TestAnthropicLLM_NewAnthropicLLMFromConfig(t *testing.T) {
	ctx := context.Background()

	t.Run("Valid config with API key", func(t *testing.T) {
		config := core.ProviderConfig{
			Name:   "anthropic",
			APIKey: "test-api-key",
		}

		llm, err := NewAnthropicLLMFromConfig(ctx, config, core.ModelID(anthropic.ModelClaude_3_Haiku_20240307))
		assert.NoError(t, err)
		assert.NotNil(t, llm)
		assert.Equal(t, "anthropic", llm.ProviderName())
		assert.Equal(t, string(anthropic.ModelClaude_3_Haiku_20240307), llm.ModelID())
	})

	t.Run("Config with custom endpoint", func(t *testing.T) {
		config := core.ProviderConfig{
			Name:   "anthropic",
			APIKey: "test-api-key",
			Endpoint: &core.EndpointConfig{
				BaseURL: "https://custom.anthropic.com",
			},
		}

		llm, err := NewAnthropicLLMFromConfig(ctx, config, core.ModelID(anthropic.ModelClaudeSonnet4_5_20250929))
		assert.NoError(t, err)
		assert.NotNil(t, llm)
		assert.Equal(t, "anthropic", llm.ProviderName())
		assert.Equal(t, string(anthropic.ModelClaudeSonnet4_5_20250929), llm.ModelID())
	})

	t.Run("Config with empty API key falls back to env var", func(t *testing.T) {
		// Set environment variable
		oldKey := os.Getenv("ANTHROPIC_API_KEY")
		defer func() {
			if oldKey != "" {
				os.Setenv("ANTHROPIC_API_KEY", oldKey)
			} else {
				os.Unsetenv("ANTHROPIC_API_KEY")
			}
		}()
		os.Setenv("ANTHROPIC_API_KEY", "env-test-key")

		config := core.ProviderConfig{
			Name: "anthropic",
		}

		llm, err := NewAnthropicLLMFromConfig(ctx, config, core.ModelID(anthropic.ModelClaudeOpus4_1_20250805))
		assert.NoError(t, err)
		assert.NotNil(t, llm)
	})

	t.Run("Config with no API key and no env var", func(t *testing.T) {
		// Unset environment variable
		oldKey := os.Getenv("ANTHROPIC_API_KEY")
		defer func() {
			if oldKey != "" {
				os.Setenv("ANTHROPIC_API_KEY", oldKey)
			}
		}()
		os.Unsetenv("ANTHROPIC_API_KEY")

		config := core.ProviderConfig{
			Name: "anthropic",
		}

		llm, err := NewAnthropicLLMFromConfig(ctx, config, core.ModelID(anthropic.ModelClaudeOpus4_1_20250805))
		assert.Error(t, err)
		assert.Nil(t, llm)
		assert.Contains(t, err.Error(), "API key or OAuth token required")
	})

	t.Run("Config with invalid model", func(t *testing.T) {
		config := core.ProviderConfig{
			Name:   "anthropic",
			APIKey: "test-api-key",
		}

		llm, err := NewAnthropicLLMFromConfig(ctx, config, core.ModelID("invalid-model"))
		assert.Error(t, err)
		assert.Nil(t, llm)
		assert.Contains(t, err.Error(), "unsupported Anthropic model")
	})
}

func TestAnthropicLLM_isValidAnthropicModel(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		expected bool
	}{
		{
			name:     "Valid Claude 3 model",
			model:    string(anthropic.ModelClaude_3_Haiku_20240307),
			expected: true,
		},
		{
			name:     "Valid Claude 4.1 model",
			model:    string(anthropic.ModelClaudeOpus4_1_20250805),
			expected: true,
		},
		{
			name:     "Valid Claude Sonnet model",
			model:    string(anthropic.ModelClaudeSonnet4_5_20250929),
			expected: true,
		},
		{
			name:     "Invalid model",
			model:    "invalid-model",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidAnthropicModel(tt.model)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestAnthropicLLM_ModelNameNormalization(t *testing.T) {
	tests := []struct {
		name      string
		oldName   string
		expectNew anthropic.Model
	}{
		{
			name:      "Old Sonnet model name",
			oldName:   "claude-3-sonnet-20240229",
			expectNew: anthropic.ModelClaudeSonnet4_5_20250929,
		},
		{
			name:      "Old Opus model name",
			oldName:   "claude-3-opus-20240229",
			expectNew: anthropic.ModelClaudeOpus4_1_20250805,
		},
		{
			name:      "Old Haiku model name",
			oldName:   "claude-3-haiku-20240307",
			expectNew: anthropic.ModelClaude_3_Haiku_20240307,
		},
		{
			name:      "Already new model name passes through",
			oldName:   string(anthropic.ModelClaudeSonnet4_5_20250929),
			expectNew: anthropic.ModelClaudeSonnet4_5_20250929,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := normalizeModelName(tt.oldName)
			assert.Equal(t, tt.expectNew, result)
		})
	}
}

func TestAnthropicProviderFactory(t *testing.T) {
	ctx := context.Background()

	t.Run("Valid factory creation", func(t *testing.T) {
		config := core.ProviderConfig{
			Name:   "anthropic",
			APIKey: "test-api-key",
		}

		llm, err := AnthropicProviderFactory(ctx, config, core.ModelID(anthropic.ModelClaude_3_Haiku_20240307))
		assert.NoError(t, err)
		assert.NotNil(t, llm)

		// Check that it's an AnthropicLLM
		anthropicLLM, ok := llm.(*AnthropicLLM)
		assert.True(t, ok)
		assert.Equal(t, "anthropic", anthropicLLM.ProviderName())
	})

	t.Run("Factory with invalid config", func(t *testing.T) {
		config := core.ProviderConfig{
			Name: "anthropic",
		}

		llm, err := AnthropicProviderFactory(ctx, config, core.ModelID("invalid-model"))
		assert.Error(t, err)
		assert.Nil(t, llm)
	})
}
