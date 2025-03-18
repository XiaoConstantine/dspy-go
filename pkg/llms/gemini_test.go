package llms

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"

	dspyErrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
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
					assert.Equal(t, "gemini-2.0-flash", llm.ModelID())
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
			// Create GeminiLLM instance with proper initialization
			endpoint := &core.EndpointConfig{
				BaseURL:    server.URL,
				Path:       "/models/gemini-2.0-flash-exp:generateContent",
				Headers:    map[string]string{"Content-Type": "application/json"},
				TimeoutSec: 30,
			}
			llm := &GeminiLLM{
				apiKey: "test-api-key",
				BaseLLM: core.NewBaseLLM(
					"google",
					core.ModelGoogleGeminiFlash,
					[]core.Capability{
						core.CapabilityCompletion,
						core.CapabilityChat,
						core.CapabilityJSON,
					},
					endpoint,
				),
			}

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
			// Create GeminiLLM instance with proper initialization
			endpoint := &core.EndpointConfig{
				BaseURL:    server.URL,
				Path:       "/models/gemini-2.0-flash-exp:generateContent",
				Headers:    map[string]string{"Content-Type": "application/json"},
				TimeoutSec: 30,
			}
			llm := &GeminiLLM{
				apiKey: "test-api-key",
				BaseLLM: core.NewBaseLLM(
					"google",
					core.ModelGoogleGeminiFlash,
					[]core.Capability{
						core.CapabilityCompletion,
						core.CapabilityChat,
						core.CapabilityJSON,
					},
					endpoint,
				),
			}
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
func TestGeminiLLM_StreamGenerate_Cancel(t *testing.T) {
	// Create test server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Set SSE headers
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// Keep connection open without sending data
		flusher, ok := w.(http.Flusher)
		require.True(t, ok)
		flusher.Flush()

		// Wait for context cancellation
		<-r.Context().Done()
	}))
	defer server.Close()

	// Create GeminiLLM with mocked server
	endpoint := &core.EndpointConfig{
		BaseURL:    server.URL,
		Path:       "/models/gemini-2.0-flash:generateContent",
		Headers:    map[string]string{"Content-Type": "application/json"},
		TimeoutSec: 30,
	}
	llm := &GeminiLLM{
		apiKey: "test-api-key",
		BaseLLM: core.NewBaseLLM(
			"google",
			core.ModelGoogleGeminiFlash,
			[]core.Capability{core.CapabilityCompletion, core.CapabilityChat},
			endpoint,
		),
	}

	// Call StreamGenerate
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stream, err := llm.StreamGenerate(ctx, "Test prompt")
	require.NoError(t, err)
	require.NotNil(t, stream)

	// Call Cancel function
	stream.Cancel()

	// Create a channel to signal when we're done reading
	done := make(chan struct{})

	// Start a goroutine to read from the channel until it's closed
	go func() {
		defer close(done)
		for range stream.ChunkChannel {
			// Just drain the channel until it's closed
		}
	}()

	// Wait for the channel to be closed, with a timeout
	select {
	case <-done:
		// Success - channel was closed
	case <-time.After(5 * time.Second):
		t.Fatal("Stream channel did not close after cancellation")
	}
}

func TestGeminiLLM_GenerateErrorCases(t *testing.T) {
	testCases := []struct {
		name      string
		setupMock func(w http.ResponseWriter, r *http.Request)
		expectErr bool
		errType   string
	}{
		{
			name: "Server unavailable",
			setupMock: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusServiceUnavailable)
				if _, err := w.Write([]byte("Service Unavailable")); err != nil {
					t.Fatalf("Failed to write!")
				}
			},
			expectErr: true,
			errType:   "LLMGenerationFailed",
		},
		{
			name: "Malformed response",
			setupMock: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				if _, err := w.Write([]byte("Service Unavailable")); err != nil {
					t.Fatalf("Failed to write!")
				}

			},
			expectErr: true,
			errType:   "InvalidResponse",
		},
		{
			name: "Request failure",
			setupMock: func(w http.ResponseWriter, r *http.Request) {
				// Close connection without responding
				hj, ok := w.(http.Hijacker)
				if !ok {
					http.Error(w, "Hijacking not supported", http.StatusInternalServerError)
					return
				}
				conn, _, err := hj.Hijack()
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}
				conn.Close()
			},
			expectErr: true,
			errType:   "LLMGenerationFailed",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create test server
			server := httptest.NewServer(http.HandlerFunc(tc.setupMock))
			defer server.Close()

			// Create GeminiLLM
			endpoint := &core.EndpointConfig{
				BaseURL:    server.URL,
				Path:       "/models/gemini-2.0-flash:generateContent",
				Headers:    map[string]string{"Content-Type": "application/json"},
				TimeoutSec: 30,
			}
			llm := &GeminiLLM{
				apiKey: "test-api-key",
				BaseLLM: core.NewBaseLLM(
					"google",
					core.ModelGoogleGeminiFlash,
					[]core.Capability{core.CapabilityCompletion},
					endpoint,
				),
			}

			// Make the request
			resp, err := llm.Generate(context.Background(), "Test prompt")

			// Check expectations
			if tc.expectErr {
				assert.Error(t, err)
				assert.Nil(t, resp)
				if tc.errType != "" {
					var dspyErr *dspyErrors.Error
					assert.True(t, errors.As(err, &dspyErr))
					t.Logf("Got: %v, expect: %v", err, tc.errType)

					// Instead of comparing string to numeric code, check if error message contains the expected text
					assert.Contains(t, dspyErr.Error(), tc.errType, "Error message should contain the expected error type")
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, resp)
			}
		})
	}
}

func TestGeminiLLM_Embeddings(t *testing.T) {
	// Tests that both embedding methods work with valid inputs

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path

		// Different response based on which endpoint is called
		if strings.Contains(path, "embedContent") {
			// Single embedding
			w.WriteHeader(http.StatusOK)
			if _, err := w.Write([]byte(`{
				"embedding": {
					"values": [0.1, 0.2, 0.3, 0.4, 0.5],
					"statistics": {
						"truncatedInputTokenCount": 0,
						"tokenCount": 4
					}
				},
				"usageMetadata": {
					"promptTokenCount": 4,
					"candidatesTokenCount": 0,
					"totalTokenCount": 4
				}
			}`)); err != nil {
				t.Fatalf("Failed to write")
			}
		} else if strings.Contains(path, "batchEmbedContents") {
			// Batch embeddings
			w.WriteHeader(http.StatusOK)
			if _, err := w.Write([]byte(`{
				"embeddings": [
					{
						"embedding": {
							"values": [0.1, 0.2, 0.3],
							"statistics": {
								"truncatedInputTokenCount": 0,
								"tokenCount": 2
							}
						},
						"usageMetadata": {
							"promptTokenCount": 2,
							"totalTokenCount": 2
						}
					},
					{
						"embedding": {
							"values": [0.4, 0.5, 0.6],
							"statistics": {
								"truncatedInputTokenCount": 0,
								"tokenCount": 2
							}
						},
						"usageMetadata": {
							"promptTokenCount": 2,
							"totalTokenCount": 2
						}
					}
				]
			}`)); err != nil {
				t.Fatalf("Failed to write")
			}
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Create GeminiLLM with mock server
	endpoint := &core.EndpointConfig{
		BaseURL:    server.URL,
		Path:       "/models/gemini-2.0-flash:generateContent",
		Headers:    map[string]string{"Content-Type": "application/json"},
		TimeoutSec: 30,
	}
	llm := &GeminiLLM{
		apiKey: "test-api-key",
		BaseLLM: core.NewBaseLLM(
			"google",
			core.ModelGoogleGeminiFlash,
			[]core.Capability{
				core.CapabilityCompletion,
				core.CapabilityEmbedding,
			},
			endpoint,
		),
	}

	// Test single embedding
	t.Run("Single embedding", func(t *testing.T) {
		result, err := llm.CreateEmbedding(context.Background(), "Test text")
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Len(t, result.Vector, 5)
		assert.Equal(t, 4, result.TokenCount)
		assert.NotNil(t, result.Metadata)
	})

	// Test batch embeddings
	t.Run("Batch embeddings", func(t *testing.T) {
		result, err := llm.CreateEmbeddings(context.Background(), []string{"Text 1", "Text 2"})
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Len(t, result.Embeddings, 2)
		assert.Nil(t, result.Error)
		assert.Equal(t, -1, result.ErrorIndex)

		// Check first embedding
		assert.Len(t, result.Embeddings[0].Vector, 3)
		assert.Equal(t, 2, result.Embeddings[0].TokenCount)
	})

	// Test with options
	t.Run("With embedding options", func(t *testing.T) {
		result, err := llm.CreateEmbedding(
			context.Background(),
			"Test with options",
			core.WithModel("text-embedding-004"),
		)
		assert.NoError(t, err)
		assert.NotNil(t, result)
	})
}

func TestConstructRequestURL(t *testing.T) {
	tests := []struct {
		name        string
		endpoint    *core.EndpointConfig
		apiKey      string
		expectedURL string
	}{
		{
			name: "Basic URL construction",
			endpoint: &core.EndpointConfig{
				BaseURL: "https://example.com",
				Path:    "api/generate",
			},
			apiKey:      "test-key",
			expectedURL: "https://example.com/api/generate?key=test-key",
		},
		{
			name: "With trailing slash in base URL",
			endpoint: &core.EndpointConfig{
				BaseURL: "https://example.com/",
				Path:    "api/generate",
			},
			apiKey:      "test-key",
			expectedURL: "https://example.com/api/generate?key=test-key",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := constructRequestURL(tc.endpoint, tc.apiKey)
			assert.Equal(t, tc.expectedURL, result)
		})
	}
}

func TestGeminiLLM_GenerateWithFunctions(t *testing.T) {
	// Setup test server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Read request body to verify function declarations
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		// Check if body contains function_declarations
		bodyStr := string(body)
		if strings.Contains(bodyStr, "function_declarations") {
			// Respond with function call
			w.WriteHeader(http.StatusOK)
			if _, err := w.Write([]byte(`{
				"candidates": [{
					"content": {
						"parts": [{
							"function_call": {
								"name": "get_weather",
								"arguments": {
									"location": "New York",
									"unit": "celsius"
								}
							}
						}]
					}
				}],
				"usageMetadata": {
					"promptTokenCount": 15,
					"candidatesTokenCount": 10,
					"totalTokenCount": 25
				}
			}`)); err != nil {
				t.Fatalf("Failed to write")
			}
		} else {
			// Respond with regular text
			w.WriteHeader(http.StatusOK)
			if _, err := w.Write([]byte(`{
				"candidates": [{
					"content": {
						"parts": [{
							"text": "Regular response without function call"
						}]
					}
				}],
				"usageMetadata": {
					"promptTokenCount": 10,
					"candidatesTokenCount": 5,
					"totalTokenCount": 15
				}
			}`)); err != nil {
				t.Fatalf("Failed to write")
			}
		}
	}))
	defer server.Close()

	// Create GeminiLLM
	endpoint := &core.EndpointConfig{
		BaseURL:    server.URL,
		Path:       "/models/gemini-2.0-flash:generateContent",
		Headers:    map[string]string{"Content-Type": "application/json"},
		TimeoutSec: 30,
	}
	llm := &GeminiLLM{
		apiKey: "test-api-key",
		BaseLLM: core.NewBaseLLM(
			"google",
			core.ModelGoogleGeminiFlash,
			[]core.Capability{core.CapabilityCompletion, core.CapabilityJSON},
			endpoint,
		),
	}

	// Define function schema
	functions := []map[string]interface{}{
		{
			"name":        "get_weather",
			"description": "Get weather information for a location",
			"parameters": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]interface{}{
						"type":        "string",
						"description": "The city and state",
					},
					"unit": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"celsius", "fahrenheit"},
						"description": "The temperature unit to use",
					},
				},
				"required": []string{"location"},
			},
		},
	}

	// Call function and verify results
	result, err := llm.GenerateWithFunctions(context.Background(), "What's the weather in New York?", functions)
	assert.NoError(t, err)
	assert.NotNil(t, result)

	// Check function call information
	functionCall, ok := result["function_call"].(map[string]interface{})
	require.True(t, ok, "Expected function_call in result")
	assert.Equal(t, "get_weather", functionCall["name"])

	arguments, ok := functionCall["arguments"].(map[string]interface{})
	require.True(t, ok, "Expected arguments in function_call")
	assert.Equal(t, "New York", arguments["location"])
	assert.Equal(t, "celsius", arguments["unit"])

	// Check usage information
	usage, ok := result["_usage"].(*core.TokenInfo)
	require.True(t, ok, "Expected _usage in result")
	assert.Equal(t, 15, usage.PromptTokens)
	assert.Equal(t, 10, usage.CompletionTokens)
	assert.Equal(t, 25, usage.TotalTokens)
}

func TestGeminiLLM_StreamGenerate_ChunkHandling(t *testing.T) {
	// Create test server that returns SSE responses
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Set response headers for SSE
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		require.True(t, ok, "Flusher interface not supported")

		// Send a series of chunks with delays to simulate streaming
		chunks := []string{
			`data: {"candidates":[{"content":{"parts":[{"text":"Once"}]}}]}`,
			`data: {"candidates":[{"content":{"parts":[{"text":" upon"}]}}]}`,
			`data: {"candidates":[{"content":{"parts":[{"text":" a"}]}}]}`,
			`data: {"candidates":[{"content":{"parts":[{"text":" time"}]}}]}`,
			`data: [DONE]`,
		}

		for _, chunk := range chunks {
			fmt.Fprintln(w, chunk)
			flusher.Flush()
			time.Sleep(50 * time.Millisecond)
		}
	}))
	defer server.Close()

	// Create GeminiLLM
	endpoint := &core.EndpointConfig{
		BaseURL:    server.URL,
		Path:       "/models/gemini-2.0-flash:generateContent",
		Headers:    map[string]string{"Content-Type": "application/json"},
		TimeoutSec: 30,
	}
	llm := &GeminiLLM{
		apiKey: "test-api-key",
		BaseLLM: core.NewBaseLLM(
			"google",
			core.ModelGoogleGeminiFlash,
			[]core.Capability{core.CapabilityCompletion},
			endpoint,
		),
	}

	// Call StreamGenerate
	stream, err := llm.StreamGenerate(context.Background(), "Tell me a story")
	require.NoError(t, err)
	require.NotNil(t, stream)

	// Collect chunks
	var receivedChunks []string
	var done bool

	for !done {
		select {
		case chunk, ok := <-stream.ChunkChannel:
			if !ok {
				// Channel closed
				done = true
				break
			}

			if chunk.Error != nil {
				t.Fatalf("Unexpected error in stream: %v", chunk.Error)
			}

			if chunk.Done {
				// End of stream
				done = true
				break
			}

			if chunk.Content != "" {
				receivedChunks = append(receivedChunks, chunk.Content)
			}

		case <-time.After(2 * time.Second):
			t.Fatal("Timeout waiting for chunks")
		}
	}

	// Verify received chunks
	expectedChunks := []string{"Once", " upon", " a", " time"}
	assert.Equal(t, expectedChunks, receivedChunks)
}

func TestGeminiLLM_EmbeddingErrors(t *testing.T) {
	// Create a dedicated server for this test
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Force internal server error for any request
		w.WriteHeader(http.StatusInternalServerError)
		if _, err := w.Write([]byte(`{"error": {"code": 500, "message": "Internal server error"}}`)); err != nil {
			t.Fatalf("Failed to write")
		}
	}))
	defer server.Close()

	// Create a GeminiLLM with the server's URL - ensure we use the right path to match our handler
	llm := &GeminiLLM{
		apiKey: "test-api-key",
		BaseLLM: core.NewBaseLLM(
			"google",
			core.ModelGoogleGeminiFlash,
			[]core.Capability{core.CapabilityEmbedding},
			&core.EndpointConfig{
				BaseURL: server.URL,
				// The path doesn't actually matter since our test server ignores it
				Path:       "/dummy",
				Headers:    map[string]string{"Content-Type": "application/json"},
				TimeoutSec: 30,
			},
		),
	}

	// Test CreateEmbedding error
	t.Run("Single embedding error", func(t *testing.T) {
		result, err := llm.CreateEmbedding(context.Background(), "Test input")
		assert.Error(t, err, "Expected an error from CreateEmbedding")
		assert.Nil(t, result, "Result should be nil when error occurs")
		assert.Contains(t, err.Error(), "API request failed", "Error should mention API request failure")
	})

	// Test batch embeddings with a separate test function to isolate the panic
	t.Run("Batch embedding error", func(t *testing.T) {
		// Call CreateEmbeddings and ensure we properly handle both return values
		batchResult, err := llm.CreateEmbeddings(context.Background(), []string{"Test 1", "Test 2"})

		t.Logf("err: %v", err)

		// First verify we got an error as expected
		assert.Error(t, err, "Expected an error from CreateEmbeddings")
		assert.Contains(t, err.Error(), "API request failed", "Error should mention API request failure")

		// Then check that batchResult is nil
		assert.Nil(t, batchResult, "Result should be nil when error occurs")
	})
}
func TestGeminiLLM_Implementation(t *testing.T) {
	// Create GeminiLLM
	llm, err := NewGeminiLLM("test-api-key", core.ModelGoogleGeminiFlash)
	require.NoError(t, err)

	// Test core implementation methods
	t.Run("ModelID", func(t *testing.T) {
		assert.Equal(t, string(core.ModelGoogleGeminiFlash), llm.ModelID())
	})

	t.Run("ProviderName", func(t *testing.T) {
		assert.Equal(t, "google", llm.ProviderName())
	})

	t.Run("Capabilities", func(t *testing.T) {
		capabilities := llm.Capabilities()
		assert.Contains(t, capabilities, core.CapabilityCompletion)
		assert.Contains(t, capabilities, core.CapabilityChat)
		assert.Contains(t, capabilities, core.CapabilityJSON)
		assert.Contains(t, capabilities, core.CapabilityEmbedding)
	})

	t.Run("EndpointConfig", func(t *testing.T) {
		config := llm.GetEndpointConfig()
		assert.NotNil(t, config)
		assert.Contains(t, config.Path, "generateContent")
		assert.Contains(t, config.Headers, "Content-Type")
		assert.Equal(t, "application/json", config.Headers["Content-Type"])
	})

	t.Run("HTTPClient", func(t *testing.T) {
		client := llm.GetHTTPClient()
		assert.NotNil(t, client)
	})
}
