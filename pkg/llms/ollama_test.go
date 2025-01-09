package llms

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
)

func TestNewOllamaLLM(t *testing.T) {
	tests := []struct {
		name     string
		endpoint string
		model    string
	}{
		{"Default endpoint", "", "test-model"},
		{"Custom endpoint", "http://custom:8080", "test-model"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llm, err := NewOllamaLLM(tt.endpoint, tt.model)
			assert.NoError(t, err)
			assert.NotNil(t, llm)
			if tt.endpoint == "" {
				assert.Equal(t, "http://localhost:11434", llm.GetEndpointConfig().BaseURL)
			} else {
				assert.Equal(t, tt.endpoint, llm.GetEndpointConfig().BaseURL)
			}
			assert.Equal(t, tt.model, llm.ModelID())
		})
	}
}

func TestOllamaLLM_Generate(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse *ollamaResponse
		serverStatus   int
		expectError    bool
	}{
		{
			name: "Successful generation",
			serverResponse: &ollamaResponse{
				Model:     "test-model",
				CreatedAt: "2023-01-01T00:00:00Z",
				Response:  "Generated text",
			},
			serverStatus: http.StatusOK,
			expectError:  false,
		},
		{
			name:           "Server error",
			serverResponse: nil,
			serverStatus:   http.StatusInternalServerError,
			expectError:    true,
		},
		{
			name: "Invalid JSON response",
			serverResponse: &ollamaResponse{
				Model:     "test-model",
				CreatedAt: "2023-01-01T00:00:00Z",
				Response:  "Generated text",
			},
			serverStatus: http.StatusOK,
			expectError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/api/generate", r.URL.Path)
				assert.Equal(t, "POST", r.Method)
				assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

				var reqBody ollamaRequest
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

			llm, err := NewOllamaLLM(server.URL, "test-model")
			assert.NoError(t, err)

			response, err := llm.Generate(context.Background(), "Test prompt", core.WithMaxTokens(100), core.WithTemperature(0.7))

			if tt.expectError {
				assert.Error(t, err)
			} else {
				if tt.name == "Invalid JSON response" {
					assert.Error(t, err)
				} else {
					assert.NoError(t, err)
					assert.Equal(t, tt.serverResponse.Response, response.Content)
				}
			}
		})
	}
}

func TestOllamaLLM_GenerateWithJSON(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse ollamaResponse
		expectError    bool
		expectedJSON   map[string]interface{}
	}{
		{
			name: "Valid JSON response",
			serverResponse: ollamaResponse{
				Model:     "test-model",
				CreatedAt: "2023-01-01T00:00:00Z",
				Response:  `{"key": "value"}`,
			},
			expectError:  false,
			expectedJSON: map[string]interface{}{"key": "value"},
		},
		{
			name: "Invalid JSON response",
			serverResponse: ollamaResponse{
				Model:     "test-model",
				CreatedAt: "2023-01-01T00:00:00Z",
				Response:  `invalid json`,
			},
			expectError:  true,
			expectedJSON: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				if err := json.NewEncoder(w).Encode(tt.serverResponse); err != nil {
					return
				}
			}))
			defer server.Close()

			llm, err := NewOllamaLLM(server.URL, "test-model")
			assert.NoError(t, err)

			response, err := llm.GenerateWithJSON(context.Background(), "Test prompt")

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
