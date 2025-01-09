package llms

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewGeminiLLM(t *testing.T) {
	originalEnv := os.Getenv("GEMINI_API_KEY")
	defer os.Setenv("GEMINI_API_KEY", originalEnv)

	tests := []struct {
		name      string
		apiKey    string
		model     core.ModelID
		envKey    string
		wantError bool
	}{
		{
			name:      "Valid configuration with Pro model",
			apiKey:    "test-api-key",
			model:     core.ModelGoogleGeminiPro,
			wantError: false,
		},
		{
			name:      "Valid configuration with Pro Vision model",
			apiKey:    "test-api-key",
			model:     core.ModelGoogleGeminiFlash,
			wantError: false,
		},
		{
			name:      "Empty API key",
			apiKey:    "",
			envKey:    "",
			model:     core.ModelGoogleGeminiPro,
			wantError: true,
		},
		{
			name:      "Empty API key with env var",
			apiKey:    "",
			envKey:    "env-api-key",
			model:     core.ModelGoogleGeminiPro,
			wantError: false,
		},
		{
			name:      "Default model",
			apiKey:    "test-api-key",
			model:     "",
			wantError: false,
		},
		{
			name:      "Unsupported model",
			apiKey:    "test-api-key",
			model:     "unsupported-model",
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			if tt.envKey != "" {
				os.Setenv("GEMINI_API_KEY", tt.envKey)
			} else {
				os.Setenv("GEMINI_API_KEY", "")
			}
			llm, err := NewGeminiLLM(tt.apiKey, tt.model)
			if tt.wantError {
				assert.Error(t, err)
				assert.Nil(t, llm)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, llm)
				if tt.model == "" {
					assert.Equal(t, "gemini-2.0-flash-exp", llm.ModelID())
				} else {
					assert.Equal(t, tt.model, core.ModelID(llm.ModelID()))
				}
			}
		})
	}
}

func TestGeminiLLM_Generate(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse geminiResponse
		serverStatus   int
		expectError    bool
		expectedTokens *core.TokenInfo
	}{
		{
			name: "Successful generation",
			serverResponse: geminiResponse{
				Candidates: []struct {
					Content struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					} `json:"content"`
				}{
					{
						Content: struct {
							Parts []struct {
								Text string `json:"text"`
							} `json:"parts"`
						}{
							Parts: []struct {
								Text string `json:"text"`
							}{
								{Text: "Generated text"},
							},
						},
					},
				},
				UsageMetadata: struct {
					PromptTokenCount     int `json:"promptTokenCount"`
					CandidatesTokenCount int `json:"candidatesTokenCount"`
					TotalTokenCount      int `json:"totalTokenCount"`
				}{
					PromptTokenCount:     10,
					CandidatesTokenCount: 5,
					TotalTokenCount:      15,
				},
			},
			serverStatus: http.StatusOK,
			expectError:  false,
			expectedTokens: &core.TokenInfo{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		},
		{
			name:           "Server error",
			serverResponse: geminiResponse{},
			serverStatus:   http.StatusInternalServerError,
			expectError:    true,
			expectedTokens: nil,
		},
		{
			name: "Empty candidates",
			serverResponse: geminiResponse{
				Candidates: []struct {
					Content struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					} `json:"content"`
				}{},
			},
			serverStatus:   http.StatusOK,
			expectError:    true,
			expectedTokens: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Verify request
				assert.Equal(t, "POST", r.Method)
				assert.Contains(t, r.URL.String(), "generateContent")
				assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

				// Verify request body structure
				var reqBody geminiRequest
				err := json.NewDecoder(r.Body).Decode(&reqBody)
				require.NoError(t, err)
				assert.NotEmpty(t, reqBody.Contents)

				// Send response
				w.WriteHeader(tt.serverStatus)
				if tt.serverStatus == http.StatusOK {
					err := json.NewEncoder(w).Encode(tt.serverResponse)
					require.NoError(t, err)
				}
			}))
			defer server.Close()

			llm, err := NewGeminiLLM("test-api-key", core.ModelGoogleGeminiFlash)
			require.NoError(t, err)

			llm.endpoint = fmt.Sprintf("%s/v1beta/models/%s:generateContent", server.URL, core.ModelGoogleGeminiFlash)

			response, err := llm.Generate(context.Background(), "Test prompt",
				core.WithMaxTokens(100),
				core.WithTemperature(0.7))

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, response)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, response)
				assert.Equal(t, "Generated text", response.Content)
				assert.Equal(t, tt.expectedTokens, response.Usage)
			}
		})
	}
}

func TestGeminiLLM_GenerateWithJSON(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse geminiResponse
		expectError    bool
		expectedJSON   map[string]interface{}
	}{
		{
			name: "Valid JSON response",
			serverResponse: geminiResponse{
				Candidates: []struct {
					Content struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					} `json:"content"`
				}{
					{
						Content: struct {
							Parts []struct {
								Text string `json:"text"`
							} `json:"parts"`
						}{
							Parts: []struct {
								Text string `json:"text"`
							}{
								{Text: `{"key": "value"}`},
							},
						},
					},
				},
			},
			expectError:  false,
			expectedJSON: map[string]interface{}{"key": "value"},
		},
		{
			name: "Invalid JSON response",
			serverResponse: geminiResponse{
				Candidates: []struct {
					Content struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					} `json:"content"`
				}{
					{
						Content: struct {
							Parts []struct {
								Text string `json:"text"`
							} `json:"parts"`
						}{
							Parts: []struct {
								Text string `json:"text"`
							}{
								{Text: "invalid json"},
							},
						},
					},
				},
			},
			expectError:  true,
			expectedJSON: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create test server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Verify request
				assert.Equal(t, "POST", r.Method)
				assert.Contains(t, r.URL.String(), "generateContent")
				assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

				// Send response
				w.WriteHeader(http.StatusOK)
				err := json.NewEncoder(w).Encode(tt.serverResponse)
				require.NoError(t, err)
			}))
			defer server.Close()

			// Create Gemini client with test server
			llm, err := NewGeminiLLM("test-api-key", core.ModelGoogleGeminiFlash)
			require.NoError(t, err)
			llm.endpoint = fmt.Sprintf("%s/v1beta/models/%s:generateContent", server.URL, core.ModelGoogleGeminiFlash)

			// Make the request
			response, err := llm.GenerateWithJSON(context.Background(), "Test prompt")

			// Verify results
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
