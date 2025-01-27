package llms

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewLlamacppLLM(t *testing.T) {
	tests := []struct {
		name     string
		endpoint string
	}{
		{"Default endpoint", ""},
		{"Custom endpoint", "http://custom:8080"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llm, err := NewLlamacppLLM(tt.endpoint)
			assert.NoError(t, err)
			assert.NotNil(t, llm)
			if tt.endpoint == "" {
				assert.Equal(t, "http://localhost:8080", llm.GetEndpointConfig().BaseURL)
			} else {
				assert.Equal(t, tt.endpoint, llm.GetEndpointConfig().BaseURL)
			}
		})
	}
}

func TestLlamacppLLM_Generate(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse map[string]interface{}
		serverStatus   int
		expectError    bool
		validateFunc   func(*testing.T, *core.LLMResponse)
	}{
		{
			name: "Successful generation",
			serverResponse: map[string]interface{}{
				"index":            0,
				"content":          "Test response content",
				"tokens_predicted": 10,
				"tokens_evaluated": 5,
				"stop":             true,
				"timings": map[string]interface{}{
					"predicted_ms": 100.0,
					"prompt_ms":    50.0,
				},
			},
			serverStatus: http.StatusOK,
			expectError:  false,
			validateFunc: func(t *testing.T, resp *core.LLMResponse) {
				assert.Equal(t, "Test response content", resp.Content)
				require.NotNil(t, resp.Usage)
				assert.Equal(t, 5, resp.Usage.PromptTokens)
				assert.Equal(t, 10, resp.Usage.CompletionTokens)
				assert.Equal(t, 15, resp.Usage.TotalTokens)
			},
		},
		{
			name:           "Server error",
			serverResponse: nil,
			serverStatus:   http.StatusInternalServerError,
			expectError:    true,
		},
		{
			name: "Invalid JSON response",
			serverResponse: map[string]interface{}{
				"content": map[string]interface{}{}, // Invalid content type
			},
			serverStatus: http.StatusOK,
			expectError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/completion", r.URL.Path)
				assert.Equal(t, "POST", r.Method)
				assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

				var reqBody llamacppRequest
				err := json.NewDecoder(r.Body).Decode(&reqBody)
				assert.NoError(t, err)

				w.WriteHeader(tt.serverStatus)
				if tt.serverResponse != nil {
					if tt.name == "Invalid JSON response" {
						if _, err := w.Write([]byte(`{"invalid": json`)); err != nil {
							return
						}
					} else {
						if err := json.NewEncoder(w).Encode(tt.serverResponse); err != nil {
							return
						}
					}
				}
			}))
			defer server.Close()

			llm, err := NewLlamacppLLM(server.URL)
			assert.NoError(t, err)

			response, err := llm.Generate(context.Background(), "Test prompt", core.WithMaxTokens(100), core.WithTemperature(0.7))

			if tt.expectError {
				assert.Error(t, err)
			} else {
				if tt.name == "Invalid JSON response" {
					assert.Error(t, err)
				} else {
					assert.NoError(t, err)
					assert.Equal(t, tt.serverResponse["content"], response.Content)
				}
			}
		})
	}
}

func createFullLlamaCPPResponse() map[string]interface{} {
	return map[string]interface{}{
		"index":            0,
		"content":          "Sample response content",
		"tokens":           []interface{}{},
		"id_slot":          0,
		"stop":             true,
		"model":            "test-model",
		"tokens_predicted": 432,
		"tokens_evaluated": 265,
		"generation_settings": map[string]interface{}{
			"n_predict":         4096,
			"temperature":       0.5,
			"top_k":             40,
			"top_p":             0.95,
			"dynatemp_range":    0.0,
			"dynatemp_exponent": 1.0,
			"min_p":             0.05,
			"mirostat":          0,
			"mirostat_tau":      5.0,
			"mirostat_eta":      0.1,
			"speculative": map[string]interface{}{
				"n_max": 16,
				"n_min": 5,
				"p_min": 0.9,
			},
		},
		"timings": map[string]interface{}{
			"prompt_n":               1,
			"prompt_ms":              4191.321,
			"prompt_per_token_ms":    4191.321,
			"prompt_per_second":      0.238588,
			"predicted_n":            432,
			"predicted_ms":           32496.526,
			"predicted_per_token_ms": 75.223,
			"predicted_per_second":   13.293,
		},
	}
}

func TestLlamaCppLLM_GenerateWithJSON(t *testing.T) {
	tests := []struct {
		name            string
		responseContent string
		expectError     bool
		expectedJSON    map[string]interface{}
	}{
		{
			name:            "valid JSON response",
			responseContent: `{"key": "value"}`,
			expectedJSON:    map[string]interface{}{"key": "value"},
		},
		{
			name:            "invalid JSON response",
			responseContent: "invalid json",
			expectError:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				resp := createFullLlamaCPPResponse()
				resp["content"] = tt.responseContent
				if err := json.NewEncoder(w).Encode(resp); err != nil {
					t.Errorf("Failed to encode response: %v", err)
					http.Error(w, "Failed to encode response", http.StatusInternalServerError)
					return
				}
			}))
			defer server.Close()

			llm, err := NewLlamacppLLM(server.URL)
			require.NoError(t, err)

			response, err := llm.GenerateWithJSON(context.Background(), "test prompt")

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, response)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectedJSON, response)
			}
		})
	}
}
