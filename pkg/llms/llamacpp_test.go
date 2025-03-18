package llms

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

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

			// Verify capabilities
			capabilities := llm.Capabilities()
			assert.Contains(t, capabilities, core.CapabilityCompletion)
			assert.Contains(t, capabilities, core.CapabilityChat)
			assert.Contains(t, capabilities, core.CapabilityJSON)

			// Verify provider name
			assert.Equal(t, "llamacpp", llm.ProviderName())
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

				// Verify request parameters
				assert.Equal(t, "Test prompt", reqBody.Prompt)
				assert.Equal(t, 100, reqBody.MaxTokens)
				assert.Equal(t, 0.7, reqBody.Temperature)
				assert.False(t, reqBody.Stream)

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
					if tt.validateFunc != nil {
						tt.validateFunc(t, response)
					}
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

func TestLlamacppLLM_GenerateWithFunctions(t *testing.T) {
	llm, err := NewLlamacppLLM("")
	require.NoError(t, err)

	// Test that the method panics as expected
	assert.Panics(t, func() {
		_, _ = llm.GenerateWithFunctions(context.Background(), "test prompt", nil)
	})
}

func TestLlamacppLLM_CreateEmbedding(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse []interface{}
		serverStatus   int
		expectError    bool
		expectedVector []float32
	}{
		{
			name: "Successful embedding",
			serverResponse: []interface{}{
				map[string]interface{}{
					"index": 0,
					"embedding": []interface{}{
						[]interface{}{float32(0.1), float32(0.2), float32(0.3)},
					},
				},
			},
			serverStatus:   http.StatusOK,
			expectError:    false,
			expectedVector: []float32{0.1, 0.2, 0.3},
		},
		{
			name:           "Server error",
			serverResponse: nil,
			serverStatus:   http.StatusInternalServerError,
			expectError:    true,
		},
		{
			name: "Empty embedding response",
			serverResponse: []interface{}{
				map[string]interface{}{
					"index":     0,
					"embedding": []interface{}{},
				},
			},
			serverStatus: http.StatusOK,
			expectError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/embedding", r.URL.Path)
				assert.Equal(t, "POST", r.Method)
				assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

				var reqBody llamacppEmbeddingRequest
				err := json.NewDecoder(r.Body).Decode(&reqBody)
				assert.NoError(t, err)

				// Verify request parameters
				assert.Equal(t, "test input", reqBody.Input)
				assert.True(t, reqBody.Normalize)

				w.WriteHeader(tt.serverStatus)
				if tt.serverResponse != nil {
					if err := json.NewEncoder(w).Encode(tt.serverResponse); err != nil {
						return
					}
				}
			}))
			defer server.Close()

			llm, err := NewLlamacppLLM(server.URL)
			assert.NoError(t, err)

			result, err := llm.CreateEmbedding(context.Background(), "test input", core.WithModel("test-model"))

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Equal(t, tt.expectedVector, result.Vector)
			}
		})
	}
}

func TestLlamacppLLM_CreateEmbeddings(t *testing.T) {
	tests := []struct {
		name            string
		inputs          []string
		serverResponses [][]interface{}
		serverStatus    int
		expectError     bool
		expectedVectors [][]float32
	}{
		{
			name:   "Successful batch embedding",
			inputs: []string{"input1", "input2"},
			serverResponses: [][]interface{}{
				{
					map[string]interface{}{
						"index": 0,
						"embedding": []interface{}{
							[]interface{}{float32(0.1), float32(0.2), float32(0.3)},
						},
					},
					map[string]interface{}{
						"index": 1,
						"embedding": []interface{}{
							[]interface{}{float32(0.4), float32(0.5), float32(0.6)},
						},
					},
				},
			},
			serverStatus:    http.StatusOK,
			expectError:     false,
			expectedVectors: [][]float32{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}},
		},
		{
			name:            "Server error",
			inputs:          []string{"input1", "input2"},
			serverResponses: nil,
			serverStatus:    http.StatusInternalServerError,
			expectError:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			callCount := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/embeddings", r.URL.Path)
				assert.Equal(t, "POST", r.Method)

				var reqBody llamacppBatchEmbeddingRequest
				err := json.NewDecoder(r.Body).Decode(&reqBody)
				assert.NoError(t, err)

				// For the success case, verify batch request details
				if tt.name == "Successful batch embedding" {
					assert.Equal(t, tt.inputs, reqBody.Inputs)
					assert.True(t, reqBody.Normalize)
				}

				w.WriteHeader(tt.serverStatus)
				if callCount < len(tt.serverResponses) {
					if err := json.NewEncoder(w).Encode(tt.serverResponses[callCount]); err != nil {
						return
					}
					callCount++
				}
			}))
			defer server.Close()

			llm, err := NewLlamacppLLM(server.URL)
			assert.NoError(t, err)

			result, err := llm.CreateEmbeddings(context.Background(), tt.inputs, core.WithBatchSize(32))

			if tt.expectError {
				// For error cases, check first error
				assert.Nil(t, result)

			} else {
				// For success cases, verify embeddings match expectations
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Len(t, result.Embeddings, len(tt.expectedVectors))

				for i, expected := range tt.expectedVectors {
					assert.Equal(t, expected, result.Embeddings[i].Vector)
				}
			}
		})
	}
}

func TestLlamacppLLM_StreamGenerate(t *testing.T) {
	tests := []struct {
		name           string
		streamResponse []string // JSON lines to stream in response
		serverStatus   int
		expectError    bool
		expectedChunks []string
	}{
		{
			name: "Successful streaming",
			streamResponse: []string{
				`{"index":0,"content":"Hello","tokens_predicted":1,"tokens_evaluated":5,"stop":false}`,
				`{"index":1,"content":" world","tokens_predicted":2,"tokens_evaluated":5,"stop":false}`,
				`{"index":2,"content":"!","tokens_predicted":3,"tokens_evaluated":5,"stop":true}`,
			},
			serverStatus:   http.StatusOK,
			expectError:    false,
			expectedChunks: []string{"Hello", " world", "!"},
		},
		{
			name:           "Server error",
			streamResponse: nil,
			serverStatus:   http.StatusInternalServerError,
			expectError:    true,
		},
		{
			name: "Invalid JSON in stream",
			streamResponse: []string{
				`{"index":0,"content":"Start","tokens_predicted":1,"tokens_evaluated":5,"stop":false}`,
				`invalid json line`,
				`{"index":2,"content":"End","tokens_predicted":3,"tokens_evaluated":5,"stop":true}`,
			},
			serverStatus:   http.StatusOK,
			expectError:    false,
			expectedChunks: []string{"Start", "End"}, // Invalid line should be skipped
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/api/completion", r.URL.Path)
				assert.Equal(t, "POST", r.Method)

				// Parse request to verify stream parameter is set
				var reqBody llamacppRequest
				err := json.NewDecoder(r.Body).Decode(&reqBody)
				assert.NoError(t, err)
				assert.True(t, reqBody.Stream)

				w.WriteHeader(tt.serverStatus)

				// Stream the response line by line
				if tt.serverStatus == http.StatusOK && len(tt.streamResponse) > 0 {
					flusher, ok := w.(http.Flusher)
					require.True(t, ok, "ResponseWriter must be a Flusher")

					for _, line := range tt.streamResponse {
						fmt.Fprintln(w, line)
						flusher.Flush()
						// Add small delay to simulate real streaming
						time.Sleep(10 * time.Millisecond)
					}
				}
			}))
			defer server.Close()

			llm, err := NewLlamacppLLM(server.URL)
			require.NoError(t, err)

			stream, err := llm.StreamGenerate(context.Background(), "Test prompt")
			require.NoError(t, err)

			// Wait for and collect all stream chunks
			var chunks []string
			var streamError error

			for chunk := range stream.ChunkChannel {
				if chunk.Error != nil {
					streamError = chunk.Error
					break
				}

				if chunk.Done {
					break
				}

				if chunk.Content != "" {
					chunks = append(chunks, chunk.Content)
				}
			}

			if tt.expectError {
				assert.NotNil(t, streamError)
			} else {
				assert.Nil(t, streamError)
				assert.Equal(t, tt.expectedChunks, chunks)
			}
		})
	}
}

func TestLlamacppLLM_StreamGenerate_Cancellation(t *testing.T) {
	// Set up a server that doesn't immediately respond
	responseChan := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-responseChan // Wait until test signals to respond

		// Stream one chunk of data
		if _, err := w.Write([]byte(`{"index":0,"content":"This will be canceled","tokens_predicted":1,"tokens_evaluated":5,"stop":false}` + "\n")); err != nil {
			t.Fatalf("Failed to write")
		}
		w.(http.Flusher).Flush()
	}))
	defer server.Close()
	defer close(responseChan)

	llm, err := NewLlamacppLLM(server.URL)
	require.NoError(t, err)

	// Set up a context we can cancel
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Request streaming
	stream, err := llm.StreamGenerate(ctx, "Test prompt that will be canceled")
	require.NoError(t, err)

	// Cancel via the context after a short delay
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	// Count chunks we receive - should be zero or at most one
	chunkCount := 0
	for range stream.ChunkChannel {
		chunkCount++
	}

	// We might get 0 or 1 chunks before cancellation takes effect
	assert.LessOrEqual(t, chunkCount, 1, "Should receive at most 1 chunk before cancellation")

	// Now allow the server to finish, otherwise it could hang
	responseChan <- struct{}{}
}

// Test a real world example of streaming parsing.
func TestLlamacppLLM_StreamGenerate_RealWorldExampleParsing(t *testing.T) {
	// Create a more realistic streaming response with various edge cases
	streamResponse := `
{"index":0,"content":"This is chunk 1","tokens_predicted":5,"tokens_evaluated":3,"stop":false}
{"index":1,"content":" This is chunk 2 with a JSON value like {\"key\": \"value\"}","tokens_predicted":15,"tokens_evaluated":3,"stop":false}
{"index":2,"content":"","tokens_predicted":15,"tokens_evaluated":3,"stop":false}
{"index":3,"content":" Final chunk","tokens_predicted":20,"tokens_evaluated":3,"stop":true}
`

	// Set up scanner to read line by line
	scanner := bufio.NewScanner(strings.NewReader(streamResponse))

	// Read and parse each line as we would in the real implementation
	chunks := []string{}

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}

		var streamResp llamacppResponse
		err := json.Unmarshal([]byte(line), &streamResp)
		assert.NoError(t, err, "Failed to unmarshal line: %s", line)

		// Only add non-empty content
		if streamResp.Content != "" {
			chunks = append(chunks, streamResp.Content)
		}

		// Check if the response indicates the end of the stream
		if streamResp.Stop {
			break
		}
	}

	// Verify parsed chunks
	expected := []string{
		"This is chunk 1",
		" This is chunk 2 with a JSON value like {\"key\": \"value\"}",
		" Final chunk",
	}

	assert.Equal(t, expected, chunks)
}
