package llms

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewOllamaLLM(t *testing.T) {
	tests := []struct {
		name     string
		endpoint string
		model    string
	}{
		{"Default endpoint", "", "test-model"},
		{"Custom endpoint", "http://custom:8080", "test-model"},
		{"Empty model", "http://custom:8080", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llm, err := NewOllamaLLM(tt.endpoint, tt.model)
			assert.NoError(t, err)
			assert.NotNil(t, llm)

			// Test the endpoint logic
			if tt.endpoint == "" {
				assert.Equal(t, "http://localhost:11434", llm.GetEndpointConfig().BaseURL)
			} else {
				assert.Equal(t, tt.endpoint, llm.GetEndpointConfig().BaseURL)
			}

			// Test the model ID
			assert.Equal(t, tt.model, llm.ModelID())

			// Verify capabilities
			caps := []core.Capability{
				core.CapabilityCompletion,
				core.CapabilityChat,
				core.CapabilityJSON,
			}
			assert.ElementsMatch(t, caps, llm.Capabilities())

			// Verify headers
			assert.Equal(t, "application/json", llm.GetEndpointConfig().Headers["Content-Type"])

			// Verify timeout
			assert.Equal(t, 10*60, llm.GetEndpointConfig().TimeoutSec)
		})
	}
}

func TestOllamaLLM_Generate(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse *ollamaResponse
		serverStatus   int
		options        []core.GenerateOption
		expectError    bool
		errorCheck     func(error) bool
	}{
		{
			name: "Successful generation",
			serverResponse: &ollamaResponse{
				Model:     "test-model",
				CreatedAt: "2023-01-01T00:00:00Z",
				Response:  "Generated text",
			},
			serverStatus: http.StatusOK,
			options:      []core.GenerateOption{core.WithMaxTokens(100), core.WithTemperature(0.7)},
			expectError:  false,
		},
		{
			name:         "Server error",
			serverStatus: http.StatusInternalServerError,
			expectError:  true,
			errorCheck: func(err error) bool {
				return strings.Contains(err.Error(), "API request failed with status code 500")
			},
		},
		{
			name:         "Invalid JSON response",
			serverStatus: http.StatusOK,
			expectError:  true,
			errorCheck: func(err error) bool {
				return strings.Contains(err.Error(), "failed to unmarshal response")
			},
		},
		{
			name:         "Network error simulation",
			serverStatus: -1, // Special flag to trigger network error
			expectError:  true,
			errorCheck: func(err error) bool {
				return strings.Contains(err.Error(), "failed to send request")
			},
		},
		{
			name: "With zero temperature",
			serverResponse: &ollamaResponse{
				Model:     "test-model",
				CreatedAt: "2023-01-01T00:00:00Z",
				Response:  "Generated text",
			},
			serverStatus: http.StatusOK,
			options:      []core.GenerateOption{core.WithTemperature(0.0)},
			expectError:  false,
		},
		{
			name: "With custom max tokens",
			serverResponse: &ollamaResponse{
				Model:     "test-model",
				CreatedAt: "2023-01-01T00:00:00Z",
				Response:  "Generated text",
			},
			serverStatus: http.StatusOK,
			options:      []core.GenerateOption{core.WithMaxTokens(2000)},
			expectError:  false,
		},
		{
			name: "With both custom settings",
			serverResponse: &ollamaResponse{
				Model:     "test-model",
				CreatedAt: "2023-01-01T00:00:00Z",
				Response:  "Generated text",
			},
			serverStatus: http.StatusOK,
			options:      []core.GenerateOption{core.WithMaxTokens(2000), core.WithTemperature(0.2)},
			expectError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var server *httptest.Server

			if tt.serverStatus == -1 {
				// Set up a server that immediately closes the connection to simulate network error
				server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					hj, ok := w.(http.Hijacker)
					if !ok {
						t.Fatal("Server doesn't support hijacking")
					}
					conn, _, err := hj.Hijack()
					if err != nil {
						t.Fatal(err)
					}
					conn.Close()
				}))
			} else {
				server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					assert.Equal(t, "/api/generate", r.URL.Path)
					assert.Equal(t, "POST", r.Method)
					assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

					var reqBody ollamaRequest
					err := json.NewDecoder(r.Body).Decode(&reqBody)
					assert.NoError(t, err)

					// Check if options were properly set
					if len(tt.options) > 0 {
						// Apply all options to a single options object
						opts := core.NewGenerateOptions()
						for _, opt := range tt.options {
							opt(opts)
						}

						// Now compare the expected values with what was sent
						assert.Equal(t, opts.MaxTokens, reqBody.MaxTokens)
						assert.Equal(t, opts.Temperature, reqBody.Temperature)
					}

					assert.Equal(t, "test-model", reqBody.Model)
					assert.Equal(t, "Test prompt", reqBody.Prompt)
					assert.False(t, reqBody.Stream)

					w.WriteHeader(tt.serverStatus)
					if tt.serverResponse != nil {
						if err := json.NewEncoder(w).Encode(tt.serverResponse); err != nil {
							t.Fatalf("Failde to encode response")
						}
					} else if tt.name == "Invalid JSON response" {
						if _, err := w.Write([]byte(`{"invalid json`)); err != nil {
							t.Fatalf("Failed to write")
						}
					}
				}))
			}
			defer server.Close()

			llm, err := NewOllamaLLM(server.URL, "test-model")
			assert.NoError(t, err)

			response, err := llm.Generate(context.Background(), "Test prompt", tt.options...)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorCheck != nil {
					assert.True(t, tt.errorCheck(err), "Error check failed, got: %v", err)
				}
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.serverResponse.Response, response.Content)
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
				Response:  `{"key": "value", "nested": {"inner": 123}}`,
			},
			expectError: false,
			expectedJSON: map[string]interface{}{
				"key": "value",
				"nested": map[string]interface{}{
					"inner": float64(123),
				},
			},
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
		{
			name: "Empty JSON response",
			serverResponse: ollamaResponse{
				Model:     "test-model",
				CreatedAt: "2023-01-01T00:00:00Z",
				Response:  `{}`,
			},
			expectError:  false,
			expectedJSON: map[string]interface{}{},
		},
		{
			name: "JSON array response",
			serverResponse: ollamaResponse{
				Model:     "test-model",
				CreatedAt: "2023-01-01T00:00:00Z",
				Response:  `[1, 2, 3]`,
			},
			expectError:  true, // Should fail because it's not an object
			expectedJSON: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/api/generate", r.URL.Path)
				assert.Equal(t, "POST", r.Method)
				w.WriteHeader(http.StatusOK)
				if err := json.NewEncoder(w).Encode(tt.serverResponse); err != nil {
					t.Fatalf("Failed to encode response")
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

func TestOllamaLLM_StreamGenerate(t *testing.T) {
	tests := []struct {
		name            string
		chunks          []string
		addDoneResponse bool
		expectError     bool
		expectedOutput  string
	}{
		{
			name:            "Successful streaming",
			chunks:          []string{"Hello", " world", "!"},
			addDoneResponse: true,
			expectError:     false,
			expectedOutput:  "Hello world!",
		},
		{
			name:            "Empty streaming",
			chunks:          []string{},
			addDoneResponse: true,
			expectError:     false,
			expectedOutput:  "",
		},
		{
			name:            "Streaming with early context cancellation",
			chunks:          []string{"This", " will", " be", " canceled"},
			addDoneResponse: false,
			expectError:     true,
			expectedOutput:  "This will be canceled",
		},
		{
			name:            "Streaming with invalid JSON responses",
			chunks:          []string{"This is", " valid"},
			addDoneResponse: true,
			expectError:     false,
			expectedOutput:  "This is valid",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/api/generate", r.URL.Path)
				assert.Equal(t, "POST", r.Method)

				// Verify we're requesting a stream
				var reqBody ollamaRequest
				err := json.NewDecoder(r.Body).Decode(&reqBody)
				assert.NoError(t, err)
				assert.True(t, reqBody.Stream)

				// Set headers for streaming
				w.Header().Set("Content-Type", "application/x-ndjson")
				w.WriteHeader(http.StatusOK)

				flusher, ok := w.(http.Flusher)
				require.True(t, ok, "ResponseWriter must be a Flusher")

				// Send chunks as separate JSONL responses
				for _, chunk := range tt.chunks {
					resp := struct {
						Response string `json:"response"`
						Done     bool   `json:"done"`
					}{
						Response: chunk,
						Done:     false,
					}

					if tt.name == "Streaming with invalid JSON responses" {
						// Add some invalid JSON responses that should be skipped
						_, err := w.Write([]byte("invalid json\n"))
						assert.NoError(t, err)
						flusher.Flush()
					}

					jsonData, err := json.Marshal(resp)
					require.NoError(t, err)

					_, err = w.Write(append(jsonData, '\n'))
					assert.NoError(t, err)
					flusher.Flush()

					// Simulate some delay between chunks
					time.Sleep(50 * time.Millisecond)
				}

				if tt.addDoneResponse {
					doneResp := struct {
						Response string `json:"response"`
						Done     bool   `json:"done"`
					}{
						Response: "",
						Done:     true,
					}

					jsonData, err := json.Marshal(doneResp)
					require.NoError(t, err)

					_, err = w.Write(append(jsonData, '\n'))
					assert.NoError(t, err)
					flusher.Flush()
				} else {
					// Just keep the connection open without sending a done message
					time.Sleep(500 * time.Millisecond)
				}
			}))
			defer server.Close()

			llm, err := NewOllamaLLM(server.URL, "test-model")
			assert.NoError(t, err)

			// For the context cancellation test, we'll use a short timeout context
			var streamCtx context.Context
			var streamCancel context.CancelFunc

			if tt.name == "Streaming with early context cancellation" {
				// Use a short context that will expire after receiving a few chunks
				streamCtx, streamCancel = context.WithTimeout(context.Background(), 300*time.Millisecond)
				defer streamCancel()
			} else {
				streamCtx, streamCancel = context.WithTimeout(context.Background(), 2*time.Second)
				defer streamCancel()
			}

			streamResp, err := llm.StreamGenerate(streamCtx, "Test prompt", core.WithMaxTokens(100))
			assert.NoError(t, err)

			// For streaming with early cancellation, we manually cancel after reading some chunks
			readChunks := 0
			manualCancellation := tt.name == "Streaming with early context cancellation"
			var actualOutput strings.Builder

			var output strings.Builder
			var lastErr error

			for chunk := range streamResp.ChunkChannel {
				if chunk.Error != nil {
					lastErr = chunk.Error
					break
				}

				if chunk.Done {
					break
				}

				output.WriteString(chunk.Content)
				actualOutput.WriteString(chunk.Content)

				// For manual cancellation test, cancel after reading a few chunks
				readChunks++
				if manualCancellation && readChunks >= 3 {
					// When testing cancellation, wait longer to receive more chunks
					streamResp.Cancel()
				}
			}

			// If we're testing early cancellation, we have two possibilities:
			// 1. We got an explicit error from the stream
			// 2. We never got an error but the stream was properly terminated after Cancel()
			if tt.name == "Streaming with early context cancellation" {
				if lastErr != nil {
					// Got an error as expected
					assert.Error(t, lastErr)
				} else {
					// No error, but verify we didn't receive all chunks
					assert.True(t, readChunks < len(tt.chunks),
						"Expected stream to end before reading all chunks")
				}
			} else if tt.expectError {
				assert.Error(t, lastErr)
			} else {
				assert.NoError(t, lastErr)
			}

			// For the early cancellation test, check only what we actually got
			if tt.name == "Streaming with early context cancellation" {
				assert.Contains(t, tt.expectedOutput, actualOutput.String(),
					"Streamed content should be a subset of the expected content")
			} else {
				assert.Equal(t, tt.expectedOutput, output.String())
			}
		})
	}
}

func TestOllamaLLM_CreateEmbedding(t *testing.T) {
	tests := []struct {
		name           string
		input          string
		serverResponse ollamaEmbeddingResponse
		serverStatus   int
		options        []core.EmbeddingOption
		expectError    bool
		errorCheck     func(error) bool
	}{
		{
			name:  "Successful embedding creation",
			input: "Test input for embedding",
			serverResponse: ollamaEmbeddingResponse{
				Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5},
				Size:      5,
				Usage: struct {
					Tokens int `json:"tokens"`
				}{
					Tokens: 7,
				},
			},
			serverStatus: http.StatusOK,
			expectError:  false,
		},
		{
			name:         "Server error",
			input:        "Test input",
			serverStatus: http.StatusInternalServerError,
			expectError:  true,
			errorCheck: func(err error) bool {
				return strings.Contains(err.Error(), "API request failed with status code 500")
			},
		},
		{
			name:         "Invalid JSON response",
			input:        "Test input",
			serverStatus: http.StatusOK,
			expectError:  true,
			errorCheck: func(err error) bool {
				return strings.Contains(err.Error(), "failed to unmarshal response")
			},
		},
		{
			name:  "With custom options",
			input: "Test input with options",
			serverResponse: ollamaEmbeddingResponse{
				Embedding: []float32{0.1, 0.2, 0.3},
				Size:      3,
				Usage: struct {
					Tokens int `json:"tokens"`
				}{
					Tokens: 5,
				},
			},
			serverStatus: http.StatusOK,
			options: []core.EmbeddingOption{
				core.WithParams(map[string]interface{}{
					"truncate":  true,
					"normalize": true,
				}),
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/api/embeddings", r.URL.Path)
				assert.Equal(t, "POST", r.Method)
				assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

				var reqBody ollamaEmbeddingRequest
				err := json.NewDecoder(r.Body).Decode(&reqBody)
				assert.NoError(t, err)

				// Verify request data
				assert.Equal(t, "test-model", reqBody.Model)
				assert.Equal(t, tt.input, reqBody.Prompt)

				// Check if options were properly set
				if len(tt.options) > 0 {
					opts := core.NewEmbeddingOptions()
					for _, opt := range tt.options {
						opt(opts)
					}
					assert.Equal(t, opts.Params, reqBody.Options)
				}

				w.WriteHeader(tt.serverStatus)
				if tt.serverStatus == http.StatusOK {
					if tt.name == "Invalid JSON response" {
						if _, err := w.Write([]byte(`{"invalid json`)); err != nil {
							t.Fatalf("Failed to write")
						}
					} else {
						if err := json.NewEncoder(w).Encode(tt.serverResponse); err != nil {
							t.Fatalf("Failed to encode")
						}
					}
				} else {
					if _, err := w.Write([]byte(`{"error": "Server error"}`)); err != nil {
						t.Fatalf("Failed to write")
					}

				}
			}))
			defer server.Close()

			llm, err := NewOllamaLLM(server.URL, "test-model")
			assert.NoError(t, err)

			result, err := llm.CreateEmbedding(context.Background(), tt.input, tt.options...)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorCheck != nil {
					assert.True(t, tt.errorCheck(err), "Error check failed, got: %v", err)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Equal(t, tt.serverResponse.Embedding, result.Vector)
				assert.Equal(t, tt.serverResponse.Usage.Tokens, result.TokenCount)
				assert.Equal(t, tt.serverResponse.Size, result.Metadata["embedding_size"])
				assert.Equal(t, "test-model", result.Metadata["model"])
			}
		})
	}
}

func TestOllamaLLM_CreateEmbeddings(t *testing.T) {
	tests := []struct {
		name            string
		inputs          []string
		batchResponses  []ollamaBatchEmbeddingResponse
		serverStatus    int
		batchSize       int
		expectError     bool
		partialSuccess  bool
		expectedResults int
	}{
		{
			name:   "Successful batch embedding",
			inputs: []string{"Input 1", "Input 2", "Input 3", "Input 4", "Input 5"},
			batchResponses: []ollamaBatchEmbeddingResponse{
				{
					Embeddings: []ollamaEmbeddingResponse{
						{
							Embedding: []float32{0.1, 0.2, 0.3},
							Size:      3,
							Usage: struct {
								Tokens int `json:"tokens"`
							}{Tokens: 2},
						},
						{
							Embedding: []float32{0.4, 0.5, 0.6},
							Size:      3,
							Usage: struct {
								Tokens int `json:"tokens"`
							}{Tokens: 2},
						},
						{
							Embedding: []float32{0.7, 0.8, 0.9},
							Size:      3,
							Usage: struct {
								Tokens int `json:"tokens"`
							}{Tokens: 2},
						},
					},
				},
				{
					Embeddings: []ollamaEmbeddingResponse{
						{
							Embedding: []float32{0.4, 0.5, 0.6},
							Size:      3,
							Usage: struct {
								Tokens int `json:"tokens"`
							}{Tokens: 2},
						},
						{
							Embedding: []float32{0.7, 0.8, 0.9},
							Size:      3,
							Usage: struct {
								Tokens int `json:"tokens"`
							}{Tokens: 2},
						},
					},
				},
			},
			serverStatus:    http.StatusOK,
			batchSize:       3,
			expectError:     false,
			expectedResults: 5,
		},
		{
			name:   "Server error on second batch",
			inputs: []string{"Input 1", "Input 2", "Input 3", "Input 4"},
			batchResponses: []ollamaBatchEmbeddingResponse{
				{
					Embeddings: []ollamaEmbeddingResponse{
						{
							Embedding: []float32{0.1, 0.2, 0.3},
							Size:      3,
							Usage: struct {
								Tokens int `json:"tokens"`
							}{Tokens: 2},
						},
						{
							Embedding: []float32{0.4, 0.5, 0.6},
							Size:      3,
							Usage: struct {
								Tokens int `json:"tokens"`
							}{Tokens: 2},
						},
					},
				},
			},
			serverStatus:    http.StatusInternalServerError,
			batchSize:       2,
			expectError:     true,
			partialSuccess:  true,
			expectedResults: 0,
		},
		{
			name:   "Default batch size",
			inputs: []string{"Single input"},
			batchResponses: []ollamaBatchEmbeddingResponse{
				{
					Embeddings: []ollamaEmbeddingResponse{
						{
							Embedding: []float32{0.1, 0.2, 0.3},
							Size:      3,
							Usage: struct {
								Tokens int `json:"tokens"`
							}{Tokens: 2},
						},
					},
				},
			},
			serverStatus:    http.StatusOK,
			batchSize:       0, // Test default batch size handling
			expectError:     false,
			expectedResults: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup a counter for batch responses
			var batchCounter int
			var mu sync.Mutex

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/api/embeddings/batch", r.URL.Path)
				assert.Equal(t, "POST", r.Method)

				var reqBody ollamaBatchEmbeddingRequest
				err := json.NewDecoder(r.Body).Decode(&reqBody)
				assert.NoError(t, err)

				mu.Lock()
				currentBatch := batchCounter
				batchCounter++
				mu.Unlock()

				// If testing server error on second batch
				if tt.serverStatus != http.StatusOK && currentBatch > 0 {
					w.WriteHeader(tt.serverStatus)
					if _, err := w.Write([]byte(`{"error": "Server error"}`)); err != nil {
						t.Fatalf("Failed to write")
					}

					return
				}

				// Check if we have a response for this batch
				if currentBatch < len(tt.batchResponses) {
					w.WriteHeader(http.StatusOK)
					if err := json.NewEncoder(w).Encode(tt.batchResponses[currentBatch]); err != nil {
						t.Fatalf("Failed to encode")
					}
				} else {
					w.WriteHeader(http.StatusInternalServerError)
					if _, err := w.Write([]byte(`{"error": "Unexpected batch request"}`)); err != nil {
						t.Fatalf("Failed to write")
					}

				}
			}))
			defer server.Close()

			llm, err := NewOllamaLLM(server.URL, "test-model")
			assert.NoError(t, err)

			options := []core.EmbeddingOption{}
			if tt.batchSize > 0 {
				options = append(options, core.WithBatchSize(tt.batchSize))
			}

			result, err := llm.CreateEmbeddings(context.Background(), tt.inputs, options...)

			if tt.expectError {
				assert.Error(t, err)
				if tt.partialSuccess && result != nil {
					assert.Equal(t, err, result.Error)
					assert.NotEqual(t, -1, result.ErrorIndex)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Equal(t, tt.expectedResults, len(result.Embeddings))
				assert.Nil(t, result.Error)
				assert.Equal(t, -1, result.ErrorIndex)

				// Verify the embeddings were properly collected
				for i := 0; i < len(result.Embeddings); i++ {
					assert.NotEmpty(t, result.Embeddings[i].Vector)
					assert.NotZero(t, result.Embeddings[i].TokenCount)
					assert.Equal(t, "test-model", result.Embeddings[i].Metadata["model"])
					assert.NotZero(t, result.Embeddings[i].Metadata["embedding_size"])
					assert.NotNil(t, result.Embeddings[i].Metadata["batch_index"])
				}
			}
		})
	}
}

func TestOllamaLLM_GenerateWithFunctions(t *testing.T) {
	llm, err := NewOllamaLLM("", "test-model")
	assert.NoError(t, err)

	// We expect this to panic since it's not implemented
	assert.Panics(t, func() {
		if _, err := llm.GenerateWithFunctions(context.Background(), "Test prompt", []map[string]interface{}{}); err != nil {
			t.Fatalf("Failed to generate")
		}
	})
}

func TestOllamaLLM_RequestTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate a timeout by sleeping longer than the client timeout
		time.Sleep(500 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		if _, err := w.Write([]byte(`{"model": "test-model", "created_at": "2023-01-01T00:00:00Z", "response": "Generated text"}`)); err != nil {
			t.Fatalf("Failed to write")
		}
	}))
	defer server.Close()

	llm, err := NewOllamaLLM(server.URL, "test-model")
	assert.NoError(t, err)

	// Create a context with a short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// This should time out
	_, err = llm.Generate(ctx, "Test prompt")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "context deadline exceeded")
}

func TestOllamaLLM_ConnectionError(t *testing.T) {
	// For the marshal error test, we'll create a server that simulates the error
	// by returning an error in the initial phase of the request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Immediately close the connection to simulate a client-side error
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("Server doesn't support hijacking")
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			t.Fatal(err)
		}
		conn.Close()
	}))
	defer server.Close()

	// Create a client with a very short timeout
	customClient := &http.Client{
		Timeout: 1 * time.Nanosecond, // Extremely short timeout to ensure error
	}

	// Create a custom transport to intercept request
	transport := &http.Transport{
		ResponseHeaderTimeout: 1 * time.Nanosecond,
	}
	customClient.Transport = transport

	// Create server with invalid URL to trigger different error type
	invalidServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	invalidServerURL := invalidServer.URL
	invalidServer.Close() // Immediately close to make the URL invalid

	// Create LLM with the invalid URL
	llm, err := NewOllamaLLM(invalidServerURL, "test-model")
	assert.NoError(t, err)

	// This should fail with network error
	_, err = llm.Generate(context.Background(), "test")
	assert.Error(t, err)

	// Different error messages depending on OS/platform, so we check partial content
	assert.True(t,
		strings.Contains(err.Error(), "failed to send request") ||
			strings.Contains(err.Error(), "connection refused"),
		"Expected connection error, got: %v", err)
}

func TestOllamaLLM_InvalidEndpoint(t *testing.T) {
	// Test with an invalid endpoint
	llm, err := NewOllamaLLM("http://invalid-endpoint-that-doesnt-exist:12345", "test-model")
	assert.NoError(t, err)

	_, err = llm.Generate(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to send request")
}

func TestOllamaLLM_UnreadableResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Write headers to indicate success but then close the connection
		// to simulate an error reading the response body
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}

		conn, _, _ := w.(http.Hijacker).Hijack()
		conn.Close()
	}))
	defer server.Close()

	llm, err := NewOllamaLLM(server.URL, "test-model")
	assert.NoError(t, err)

	_, err = llm.Generate(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to read response body")
}
