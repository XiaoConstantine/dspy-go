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
					assert.Equal(t, "gemini-2.5-flash", llm.ModelID())
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

func TestGeminiLLM_GenerateWithFunctions_ErrorCases(t *testing.T) {
	testCases := []struct {
		name           string
		functions      []map[string]interface{}
		serverStatus   int
		serverResponse string
		expectedErrMsg string
	}{
		{
			name: "Missing name in function schema",
			functions: []map[string]interface{}{
				{
					// Name field missing
					"description": "Get weather information",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type": "string",
							},
						},
					},
				},
			},
			expectedErrMsg: "function schema missing 'name' field",
		},
		{
			name: "Missing parameters in function schema",
			functions: []map[string]interface{}{
				{
					"name":        "get_weather",
					"description": "Get weather information",
					// Parameters field missing
				},
			},
			expectedErrMsg: "function schema missing 'parameters' field",
		},
		{
			name: "Server error",
			functions: []map[string]interface{}{
				{
					"name":        "get_weather",
					"description": "Get weather information",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type": "string",
							},
						},
					},
				},
			},
			serverStatus:   http.StatusInternalServerError,
			serverResponse: `{"error": {"message": "Internal server error"}}`,
			expectedErrMsg: "API request failed with status code 500",
		},
		{
			name: "Invalid response format",
			functions: []map[string]interface{}{
				{
					"name":        "get_weather",
					"description": "Get weather information",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type": "string",
							},
						},
					},
				},
			},
			serverStatus:   http.StatusOK,
			serverResponse: `{invalid-json}`,
			expectedErrMsg: "embedding values missing in response",
		},
		{
			name: "Empty candidates in response",
			functions: []map[string]interface{}{
				{
					"name":        "get_weather",
					"description": "Get weather information",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type": "string",
							},
						},
					},
				},
			},
			serverStatus:   http.StatusOK,
			serverResponse: `{"candidates": []}`,
			expectedErrMsg: "no candidates in response",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create test server if needed
			var server *httptest.Server
			if tc.serverStatus != 0 || tc.serverResponse != "" {
				server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					// Validate request contains expected function info if applicable
					if strings.Contains(tc.name, "Server error") || strings.Contains(tc.name, "Invalid response") {
						var reqBody map[string]interface{}
						err := json.NewDecoder(r.Body).Decode(&reqBody)
						require.NoError(t, err)

						// Verify tools array exists and has expected structure
						tools, ok := reqBody["tools"].([]interface{})
						assert.True(t, ok, "Request should include tools array")

						if ok && len(tools) > 0 {
							tool := tools[0].(map[string]interface{})
							functionDecls, ok := tool["function_declarations"].([]interface{})
							assert.True(t, ok, "Tool should include function_declarations")
							assert.True(t, len(functionDecls) > 0, "Should have at least one function declaration")
						}
					}

					w.WriteHeader(tc.serverStatus)
					if tc.serverResponse != "" {
						if _, err := w.Write([]byte(tc.serverResponse)); err != nil {
							t.Fatalf("Failed to write response")
						}
					}
				}))
				defer server.Close()
			}

			// Create GeminiLLM instance
			var llm *GeminiLLM

			if server != nil {
				// If we have a server, point to it
				endpoint := &core.EndpointConfig{
					BaseURL:    server.URL,
					Path:       "/models/gemini-2.0-flash:generateContent",
					Headers:    map[string]string{"Content-Type": "application/json"},
					TimeoutSec: 30,
				}
				llm = &GeminiLLM{
					apiKey:  "test-api-key",
					BaseLLM: core.NewBaseLLM("google", core.ModelGoogleGeminiFlash, []core.Capability{core.CapabilityCompletion}, endpoint),
				}
			} else {
				// For validation errors, we can use a standard instance
				llm = &GeminiLLM{
					apiKey:  "test-api-key",
					BaseLLM: core.NewBaseLLM("google", core.ModelGoogleGeminiFlash, []core.Capability{core.CapabilityCompletion}, nil),
				}
			}

			// Call function with provided schema
			result, err := llm.GenerateWithFunctions(context.Background(), "Test prompt", tc.functions)

			// Verify error
			assert.Error(t, err, "Expected error case")
			assert.Nil(t, result, "Result should be nil on error")

		})
	}
}

func TestGeminiLLM_GenerateWithFunctions_ResponseVariations(t *testing.T) {
	testCases := []struct {
		name           string
		serverResponse string
		expectTextOnly bool
		expectedText   string
		expectedFunc   string
		expectedArgs   map[string]interface{}
	}{
		{
			name: "Text response with no function call",
			serverResponse: `{
				"candidates": [{
					"content": {
						"parts": [{
							"text": "This is a text-only response"
						}]
					}
				}],
				"usageMetadata": {
					"promptTokenCount": 10,
					"candidatesTokenCount": 5,
					"totalTokenCount": 15
				}
			}`,
			expectTextOnly: true,
			expectedText:   "This is a text-only response",
		},
		{
			name: "Function call with no text",
			serverResponse: `{
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
			}`,
			expectTextOnly: false,
			expectedFunc:   "get_weather",
			expectedArgs: map[string]interface{}{
				"location": "New York",
				"unit":     "celsius",
			},
		},
		{
			name: "Mixed text and function call",
			serverResponse: `{
				"candidates": [{
					"content": {
						"parts": [
							{
								"text": "I'll get the weather for you"
							},
							{
								"function_call": {
									"name": "get_weather",
									"arguments": {
										"location": "London",
										"unit": "fahrenheit"
									}
								}
							}
						]
					}
				}],
				"usageMetadata": {
					"promptTokenCount": 20,
					"candidatesTokenCount": 15,
					"totalTokenCount": 35
				}
			}`,
			expectTextOnly: false,
			expectedText:   "I'll get the weather for you",
			expectedFunc:   "get_weather",
			expectedArgs: map[string]interface{}{
				"location": "London",
				"unit":     "fahrenheit",
			},
		},
		{
			name: "Empty response",
			serverResponse: `{
				"candidates": [{
					"content": {
						"parts": []
					}
				}],
				"usageMetadata": {
					"promptTokenCount": 5,
					"candidatesTokenCount": 0,
					"totalTokenCount": 5
				}
			}`,
			expectTextOnly: true,
			expectedText:   "No content or function call received from model",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create test server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				if _, err := w.Write([]byte(tc.serverResponse)); err != nil {
					t.Fatalf("Failed to write response")
				}
			}))
			defer server.Close()

			// Create GeminiLLM instance
			endpoint := &core.EndpointConfig{
				BaseURL:    server.URL,
				Path:       "/models/gemini-2.0-flash:generateContent",
				Headers:    map[string]string{"Content-Type": "application/json"},
				TimeoutSec: 30,
			}
			llm := &GeminiLLM{
				apiKey:  "test-api-key",
				BaseLLM: core.NewBaseLLM("google", core.ModelGoogleGeminiFlash, []core.Capability{core.CapabilityCompletion}, endpoint),
			}

			// Define standard function schema
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

			// Call function
			result, err := llm.GenerateWithFunctions(context.Background(), "What's the weather?", functions)
			assert.NoError(t, err)
			assert.NotNil(t, result)

			// Check usage info is always present
			usage, ok := result["_usage"].(*core.TokenInfo)
			assert.True(t, ok, "Should have usage info")
			assert.NotNil(t, usage)

			if tc.expectTextOnly {
				// Check for text-only response
				content, ok := result["content"]
				assert.True(t, ok, "Should have content field")
				assert.Equal(t, tc.expectedText, content)

				// No function call should be present
				_, hasFuncCall := result["function_call"]
				assert.False(t, hasFuncCall, "Should not have function_call in text-only response")
			} else {
				// Check function call exists
				funcCall, ok := result["function_call"].(map[string]interface{})
				assert.True(t, ok, "Should have function_call field")

				// Check function name and arguments
				assert.Equal(t, tc.expectedFunc, funcCall["name"])
				args, ok := funcCall["arguments"].(map[string]interface{})
				assert.True(t, ok, "Should have arguments in function call")
				assert.Equal(t, tc.expectedArgs, args)

				// Check text field if expected
				if tc.expectedText != "" {
					content, ok := result["content"]
					assert.True(t, ok, "Should have content field with text")
					assert.Equal(t, tc.expectedText, content)
				}
			}
		})
	}
}

func TestGeminiLLM_GenerateWithFunctions_OptionsHandling(t *testing.T) {
	// Create test server that verifies options are properly passed
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Decode the request to verify options are properly set
		var reqBody geminiRequestWithFunction
		err := json.NewDecoder(r.Body).Decode(&reqBody)
		require.NoError(t, err)

		// Check that generation config reflects the options
		assert.Equal(t, 0.25, reqBody.GenerationConfig.Temperature)
		assert.Equal(t, 0.8, reqBody.GenerationConfig.TopP)
		assert.Equal(t, 300, reqBody.GenerationConfig.MaxOutputTokens)

		// Send a basic response
		w.WriteHeader(http.StatusOK)
		if _, err := w.Write([]byte(`{
			"candidates": [{
				"content": {
					"parts": [{
						"function_call": {
							"name": "test_function",
							"arguments": {"test": "value"}
						}
					}]
				}
			}],
			"usageMetadata": {
				"promptTokenCount": 10,
				"candidatesTokenCount": 5,
				"totalTokenCount": 15
			}
		}`)); err != nil {
			t.Fatalf("Failed to write response")
		}
	}))
	defer server.Close()

	// Create GeminiLLM instance
	endpoint := &core.EndpointConfig{
		BaseURL:    server.URL,
		Path:       "/models/gemini-2.0-flash:generateContent",
		Headers:    map[string]string{"Content-Type": "application/json"},
		TimeoutSec: 30,
	}
	llm := &GeminiLLM{
		apiKey:  "test-api-key",
		BaseLLM: core.NewBaseLLM("google", core.ModelGoogleGeminiFlash, []core.Capability{core.CapabilityCompletion}, endpoint),
	}

	// Define function schema
	functions := []map[string]interface{}{
		{
			"name":        "test_function",
			"description": "Test function",
			"parameters": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"test": map[string]interface{}{
						"type": "string",
					},
				},
			},
		},
	}

	// Call with specific options
	options := []core.GenerateOption{
		core.WithTemperature(0.25),
		core.WithTopP(0.8),
		core.WithMaxTokens(300),
	}

	result, err := llm.GenerateWithFunctions(context.Background(), "Test prompt", functions, options...)
	assert.NoError(t, err)
	assert.NotNil(t, result)

	// Function call should be returned
	funcCall, ok := result["function_call"].(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, "test_function", funcCall["name"])
}

func TestGeminiLLM_CreateEmbedding_Options(t *testing.T) {
	testCases := []struct {
		name         string
		input        string
		options      []core.EmbeddingOption
		expectedReq  map[string]interface{}
		validRequest bool
	}{
		{
			name:    "Default options",
			input:   "Test input",
			options: []core.EmbeddingOption{},
			expectedReq: map[string]interface{}{
				"model": "models/text-embedding-004",
				"content": map[string]interface{}{
					"parts": []interface{}{
						map[string]interface{}{
							"text": "Test input",
						},
					},
				},
			},
			validRequest: true,
		},
		{
			name:  "With custom model",
			input: "Test input",
			options: []core.EmbeddingOption{
				core.WithModel("gemini-embedding-exp-03-07"),
			},
			expectedReq: map[string]interface{}{
				"model": "models/gemini-embedding-exp-03-07",
				"content": map[string]interface{}{
					"parts": []interface{}{
						map[string]interface{}{
							"text": "Test input",
						},
					},
				},
			},
			validRequest: true,
		},
		{
			name:  "With invalid model",
			input: "Test input",
			options: []core.EmbeddingOption{
				core.WithModel("invalid-model"),
			},
			validRequest: false,
		},
		{
			name:  "With task type",
			input: "Test input",
			options: []core.EmbeddingOption{
				core.WithParams(map[string]interface{}{
					"task_type": "retrieval_query",
				}),
			},
			expectedReq: map[string]interface{}{
				"model": "models/text-embedding-004",
				"content": map[string]interface{}{
					"parts": []interface{}{
						map[string]interface{}{
							"text": "Test input",
						},
					},
				},
				"taskType": "retrieval_query",
				"parameters": map[string]interface{}{
					"task_type": "retrieval_query",
				},
			},
			validRequest: true,
		},
		{
			name:  "With additional parameters",
			input: "Test input",
			options: []core.EmbeddingOption{
				core.WithParams(map[string]interface{}{
					"dimension":        768,
					"truncateStrategy": "START",
				}),
			},
			expectedReq: map[string]interface{}{
				"model": "models/text-embedding-004",
				"content": map[string]interface{}{
					"parts": []interface{}{
						map[string]interface{}{
							"text": "Test input",
						},
					},
				},
				"parameters": map[string]interface{}{
					"dimension":        float64(768),
					"truncateStrategy": "START",
				},
			},
			validRequest: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create test server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Check URL contains embedContent
				assert.Contains(t, r.URL.Path, "embedContent")

				// Read and verify request body
				body, err := io.ReadAll(r.Body)
				assert.NoError(t, err)

				var reqMap map[string]interface{}
				err = json.Unmarshal(body, &reqMap)
				assert.NoError(t, err)

				// Check key elements match expectations
				for key, expectedValue := range tc.expectedReq {
					actualValue, exists := reqMap[key]
					assert.True(t, exists, "Expected key '%s' in request", key)

					// For maps, check individual fields
					switch expectedMap := expectedValue.(type) {
					case map[string]interface{}:
						actualMap, ok := actualValue.(map[string]interface{})
						assert.True(t, ok, "Expected %s to be a map", key)

						for subKey, subVal := range expectedMap {
							actualSubVal, exists := actualMap[subKey]
							assert.True(t, exists, "Expected subkey '%s' in %s", subKey, key)
							assert.Equal(t, subVal, actualSubVal)
						}
					default:
						assert.Equal(t, expectedValue, actualValue)
					}
				}

				// Send successful response
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
					t.Fatalf("Failed to write response")
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
					[]core.Capability{core.CapabilityEmbedding},
					endpoint,
				),
			}

			// Call CreateEmbedding with options
			result, err := llm.CreateEmbedding(context.Background(), tc.input, tc.options...)

			if tc.validRequest {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Len(t, result.Vector, 5)
				assert.Equal(t, 4, result.TokenCount)
				assert.NotNil(t, result.Metadata)
			} else {
				assert.Error(t, err)
				assert.Nil(t, result)

				var dspyErr *dspyErrors.Error
				if assert.ErrorAs(t, err, &dspyErr) {
					assert.Equal(t, dspyErrors.InvalidInput, dspyErr.Code())
				}
			}
		})
	}
}

func TestGeminiLLM_CreateEmbeddings_BatchProcessing(t *testing.T) {
	testCases := []struct {
		name          string
		inputs        []string
		batchSize     int
		expectedBatch int
		serverError   bool
	}{
		{
			name:          "Small batch all at once",
			inputs:        []string{"input1", "input2", "input3"},
			batchSize:     5,
			expectedBatch: 3,
			serverError:   false,
		},
		{
			name:          "Multiple batches",
			inputs:        []string{"input1", "input2", "input3", "input4", "input5"},
			batchSize:     2,
			expectedBatch: 2, // First batch size
			serverError:   false,
		},
		{
			name:          "Error on first batch",
			inputs:        []string{"input1", "input2", "input3"},
			batchSize:     3,
			expectedBatch: 3,
			serverError:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Track request count to verify batching
			requestCount := 0

			// Create test server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requestCount++

				// Read request body
				body, err := io.ReadAll(r.Body)
				require.NoError(t, err)

				var reqMap map[string]interface{}
				err = json.Unmarshal(body, &reqMap)
				require.NoError(t, err)

				// Check URL for batch endpoint
				if strings.Contains(r.URL.Path, "batchEmbedContents") {
					// Verify batch size
					requests, ok := reqMap["requests"].([]interface{})
					assert.True(t, ok, "Expected requests array in batch request")

					// For first batch, check exact size
					if requestCount == 1 {
						assert.Len(t, requests, tc.expectedBatch)
					}

					// Simulate error if needed
					if tc.serverError {
						w.WriteHeader(http.StatusInternalServerError)
						if _, err := w.Write([]byte(`{"error": {"message": "Server error"}}`)); err != nil {
							t.Fatalf("Failed to write error response")
						}
						return
					}

					// Create mock batch response with a result for each input
					responseObj := map[string]interface{}{
						"embeddings": make([]interface{}, len(requests)),
					}

					for i := range requests {
						responseObj["embeddings"].([]interface{})[i] = map[string]interface{}{
							"embedding": map[string]interface{}{
								"values": []float64{0.1, 0.2, 0.3},
								"statistics": map[string]interface{}{
									"truncatedInputTokenCount": 0,
									"tokenCount":               2,
								},
							},
							"usageMetadata": map[string]interface{}{
								"promptTokenCount": 2,
								"totalTokenCount":  2,
							},
						}
					}

					respBytes, _ := json.Marshal(responseObj)
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write(respBytes)
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
					[]core.Capability{core.CapabilityEmbedding},
					endpoint,
				),
			}

			// Call CreateEmbeddings with batch size option
			result, err := llm.CreateEmbeddings(
				context.Background(),
				tc.inputs,
				core.WithBatchSize(tc.batchSize),
			)

			if tc.serverError {
				assert.Nil(t, result)
				assert.Error(t, err)

				var dspyErr *dspyErrors.Error
				if assert.ErrorAs(t, err, &dspyErr) {
					assert.Equal(t, dspyErrors.LLMGenerationFailed, dspyErr.Code())
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)

				// Check number of embeddings matches input count
				assert.Len(t, result.Embeddings, len(tc.inputs))

				// Check all embeddings have expected structure
				for _, embedding := range result.Embeddings {
					assert.Len(t, embedding.Vector, 3)
					assert.Equal(t, 2, embedding.TokenCount)
					assert.NotNil(t, embedding.Metadata)

					// Check batch index is tracked
					batchIndex, ok := embedding.Metadata["batch_index"]
					assert.True(t, ok, "Embedding should track batch index")
					assert.True(t, batchIndex.(int) >= 0 && batchIndex.(int) < len(tc.inputs))
				}

				// Verify expected number of requests (based on batch size and input count)
				expectedRequests := (len(tc.inputs) + tc.batchSize - 1) / tc.batchSize
				assert.Equal(t, expectedRequests, requestCount, "Expected %d batch requests", expectedRequests)
			}
		})
	}
}

func TestGeminiLLM_CreateEmbedding_ErrorHandling(t *testing.T) {
	testCases := []struct {
		name           string
		setupServer    func(w http.ResponseWriter, r *http.Request)
		expectedErrMsg string
	}{
		{
			name: "Server error",
			setupServer: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				if _, err := w.Write([]byte(`{"error": {"message": "Internal server error"}}`)); err != nil {
					t.Fatalf("Failed to write response")
				}
			},
			expectedErrMsg: "API request failed with status code 500",
		},
		{
			name: "Invalid response format",
			setupServer: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				if _, err := w.Write([]byte(`{invalid-json}`)); err != nil {
					t.Fatalf("Failed to write response")
				}
			},
			expectedErrMsg: "embedding values missing in response",
		},
		{
			name: "Missing embedding values",
			setupServer: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				if _, err := w.Write([]byte(`{
					"embedding": {
						"statistics": {
							"truncatedInputTokenCount": 0,
							"tokenCount": 4
						}
					},
					"usageMetadata": {
						"promptTokenCount": 4,
						"totalTokenCount": 4
					}
				}`)); err != nil {
					t.Fatalf("Failed to write response")
				}
			},
			expectedErrMsg: "embedding values missing in response",
		},
		{
			name: "Request error",
			setupServer: func(w http.ResponseWriter, r *http.Request) {
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
			expectedErrMsg: "failed to send request",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create test server
			server := httptest.NewServer(http.HandlerFunc(tc.setupServer))
			defer server.Close()

			// Create GeminiLLM with mock server
			endpoint := &core.EndpointConfig{
				BaseURL:    server.URL,
				Path:       "/dummy",
				Headers:    map[string]string{"Content-Type": "application/json"},
				TimeoutSec: 30,
			}
			llm := &GeminiLLM{
				apiKey: "test-api-key",
				BaseLLM: core.NewBaseLLM(
					"google",
					core.ModelGoogleGeminiFlash,
					[]core.Capability{core.CapabilityEmbedding},
					endpoint,
				),
			}

			// Call CreateEmbedding and expect error
			result, err := llm.CreateEmbedding(context.Background(), "Test input")

			// Verify error
			assert.Error(t, err)
			assert.Nil(t, result)
		})
	}
}
