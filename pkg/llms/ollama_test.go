package llms

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewOllamaLLM_ModernDefaults(t *testing.T) {
	tests := []struct {
		name            string
		modelID         core.ModelID
		options         []OllamaOption
		expectedOpenAI  bool
		expectedBaseURL string
		expectedTimeout int
	}{
		{
			name:            "Default to OpenAI mode",
			modelID:         "llama3:8b",
			options:         nil,
			expectedOpenAI:  true,
			expectedBaseURL: "http://localhost:11434",
			expectedTimeout: 60,
		},
		{
			name:            "Explicit OpenAI mode",
			modelID:         "llama3:8b",
			options:         []OllamaOption{WithOpenAIAPI()},
			expectedOpenAI:  true,
			expectedBaseURL: "http://localhost:11434",
			expectedTimeout: 60,
		},
		{
			name:            "Native mode override",
			modelID:         "llama3:8b",
			options:         []OllamaOption{WithNativeAPI()},
			expectedOpenAI:  false,
			expectedBaseURL: "http://localhost:11434",
			expectedTimeout: 60,
		},
		{
			name:            "Custom base URL",
			modelID:         "llama3:8b",
			options:         []OllamaOption{WithBaseURL("http://custom:11435")},
			expectedOpenAI:  true,
			expectedBaseURL: "http://custom:11435",
			expectedTimeout: 60,
		},
		{
			name:    "Custom timeout",
			modelID: "llama3:8b",
			options: []OllamaOption{WithTimeout(120)},
			expectedOpenAI: true,
			expectedBaseURL: "http://localhost:11434",
			expectedTimeout: 120,
		},
		{
			name:    "Multiple options",
			modelID: "llama3:8b",
			options: []OllamaOption{
				WithNativeAPI(),
				WithBaseURL("http://custom:11435"),
				WithAuth("test-key"),
				WithTimeout(180),
			},
			expectedOpenAI:  false,
			expectedBaseURL: "http://custom:11435",
			expectedTimeout: 180,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llm, err := NewOllamaLLM(tt.modelID, tt.options...)
			assert.NoError(t, err)
			assert.NotNil(t, llm)

			assert.Equal(t, tt.expectedOpenAI, llm.config.UseOpenAIAPI)
			assert.Equal(t, tt.expectedBaseURL, llm.config.BaseURL)
			assert.Equal(t, tt.expectedTimeout, llm.config.Timeout)

			// Verify endpoint config matches mode
			if tt.expectedOpenAI {
				assert.Equal(t, "/v1/chat/completions", llm.GetEndpointConfig().Path)
			} else {
				assert.Equal(t, "/api/generate", llm.GetEndpointConfig().Path)
			}

			// Verify capabilities include embedding in OpenAI mode
			capabilities := llm.Capabilities()
			if tt.expectedOpenAI {
				assert.Contains(t, capabilities, core.CapabilityEmbedding)
			}
		})
	}
}

func TestOllamaLLM_DualModeGeneration(t *testing.T) {
	tests := []struct {
		name         string
		useOpenAI    bool
		expectedPath string
		mockResponse interface{}
		options      []core.GenerateOption
	}{
		{
			name:         "OpenAI mode",
			useOpenAI:    true,
			expectedPath: "/v1/chat/completions",
			mockResponse: &openai.ChatCompletionResponse{
				ID:      "chatcmpl-123",
				Object:  "chat.completion",
				Created: time.Now().Unix(),
				Model:   "llama3:8b",
				Choices: []openai.ChatChoice{{
					Index:   0,
					Message: openai.ChatCompletionMessage{Role: "assistant", Content: "OpenAI response"},
				}},
				Usage: openai.CompletionUsage{
					PromptTokens:     10,
					CompletionTokens: 5,
					TotalTokens:      15,
				},
			},
			options: []core.GenerateOption{core.WithMaxTokens(100), core.WithTemperature(0.7)},
		},
		{
			name:         "Native mode",
			useOpenAI:    false,
			expectedPath: "/api/generate",
			mockResponse: &ollamaResponse{
				Model:     "llama3:8b",
				CreatedAt: "2023-01-01T00:00:00Z",
				Response:  "Native response",
			},
			options: []core.GenerateOption{core.WithMaxTokens(100), core.WithTemperature(0.7)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, tt.expectedPath, r.URL.Path)
				assert.Equal(t, "POST", r.Method)
				assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

				if tt.useOpenAI {
					// Verify OpenAI request format
					var req openai.ChatCompletionRequest
					err := json.NewDecoder(r.Body).Decode(&req)
					assert.NoError(t, err)
					assert.Equal(t, "llama3:8b", req.Model)
					assert.Len(t, req.Messages, 1)
					assert.Equal(t, "user", req.Messages[0].Role)
					assert.Equal(t, "Test prompt", req.Messages[0].Content)
					assert.False(t, req.Stream)
					assert.NotNil(t, req.MaxTokens)
					assert.Equal(t, 100, *req.MaxTokens)
					assert.NotNil(t, req.Temperature)
					assert.Equal(t, 0.7, *req.Temperature)
				} else {
					// Verify native request format
					var req ollamaRequest
					err := json.NewDecoder(r.Body).Decode(&req)
					assert.NoError(t, err)
					assert.Equal(t, "llama3:8b", req.Model)
					assert.Equal(t, "Test prompt", req.Prompt)
					assert.False(t, req.Stream)
					assert.Equal(t, 100, req.MaxTokens)
					assert.Equal(t, 0.7, req.Temperature)
				}

				w.WriteHeader(http.StatusOK)
				if err := json.NewEncoder(w).Encode(tt.mockResponse); err != nil {
					t.Fatalf("Failed to encode response: %v", err)
				}
			}))
			defer server.Close()

			var llm *OllamaLLM
			var err error

			if tt.useOpenAI {
				llm, err = NewOllamaLLM("llama3:8b", WithBaseURL(server.URL), WithOpenAIAPI())
			} else {
				llm, err = NewOllamaLLM("llama3:8b", WithBaseURL(server.URL), WithNativeAPI())
			}
			assert.NoError(t, err)

			response, err := llm.Generate(context.Background(), "Test prompt", tt.options...)
			assert.NoError(t, err)
			assert.NotNil(t, response)

			if tt.useOpenAI {
				assert.Equal(t, "OpenAI response", response.Content)
				assert.NotNil(t, response.Usage)
				assert.Equal(t, 10, response.Usage.PromptTokens)
				assert.Equal(t, "openai", response.Metadata["mode"])
			} else {
				assert.Equal(t, "Native response", response.Content)
				assert.Equal(t, "native", response.Metadata["mode"])
			}
		})
	}
}

func TestOllamaLLM_DualModeStreaming(t *testing.T) {
	tests := []struct {
		name         string
		useOpenAI    bool
		expectedPath string
		chunks       []string
	}{
		{
			name:         "OpenAI streaming",
			useOpenAI:    true,
			expectedPath: "/v1/chat/completions",
			chunks:       []string{"Hello", " world", "!"},
		},
		{
			name:         "Native streaming",
			useOpenAI:    false,
			expectedPath: "/api/generate",
			chunks:       []string{"Hello", " world", "!"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, tt.expectedPath, r.URL.Path)
				assert.Equal(t, "POST", r.Method)

				if tt.useOpenAI {
					// Verify OpenAI streaming request
					var req openai.ChatCompletionRequest
					err := json.NewDecoder(r.Body).Decode(&req)
					assert.NoError(t, err)
					assert.True(t, req.Stream)

					// Set headers for SSE
					w.Header().Set("Content-Type", "text/event-stream")
					w.Header().Set("Cache-Control", "no-cache")
					w.Header().Set("Connection", "keep-alive")
					w.WriteHeader(http.StatusOK)

					flusher, ok := w.(http.Flusher)
					require.True(t, ok)

					// Send SSE chunks
					for _, chunk := range tt.chunks {
						streamResp := openai.ChatCompletionStreamResponse{
							ID:      "chatcmpl-123",
							Object:  "chat.completion.chunk",
							Created: time.Now().Unix(),
							Model:   "llama3:8b",
							Choices: []openai.ChatChoiceStream{{
								Index: 0,
								Delta: openai.ChatCompletionMessage{Content: chunk},
							}},
						}
						jsonData, _ := json.Marshal(streamResp)
						fmt.Fprintf(w, "data: %s\n\n", jsonData)
						flusher.Flush()
						time.Sleep(10 * time.Millisecond)
					}

					fmt.Fprintf(w, "data: [DONE]\n\n")
					flusher.Flush()
				} else {
					// Verify native streaming request
					var req ollamaRequest
					err := json.NewDecoder(r.Body).Decode(&req)
					assert.NoError(t, err)
					assert.True(t, req.Stream)

					// Set headers for JSONL
					w.Header().Set("Content-Type", "application/x-ndjson")
					w.WriteHeader(http.StatusOK)

					flusher, ok := w.(http.Flusher)
					require.True(t, ok)

					// Send JSONL chunks
					for _, chunk := range tt.chunks {
						resp := struct {
							Response string `json:"response"`
							Done     bool   `json:"done"`
						}{
							Response: chunk,
							Done:     false,
						}
						jsonData, _ := json.Marshal(resp)
						_, _ = w.Write(append(jsonData, '\n'))
						flusher.Flush()
						time.Sleep(10 * time.Millisecond)
					}

					// Send done message
					doneResp := struct {
						Response string `json:"response"`
						Done     bool   `json:"done"`
					}{
						Response: "",
						Done:     true,
					}
					jsonData, _ := json.Marshal(doneResp)
					_, _ = w.Write(append(jsonData, '\n'))
					flusher.Flush()
				}
			}))
			defer server.Close()

			var llm *OllamaLLM
			var err error

			if tt.useOpenAI {
				llm, err = NewOllamaLLM("llama3:8b", WithBaseURL(server.URL), WithOpenAIAPI())
			} else {
				llm, err = NewOllamaLLM("llama3:8b", WithBaseURL(server.URL), WithNativeAPI())
			}
			assert.NoError(t, err)

			streamResp, err := llm.StreamGenerate(context.Background(), "Test prompt")
			assert.NoError(t, err)
			assert.NotNil(t, streamResp)

			var output strings.Builder
			for chunk := range streamResp.ChunkChannel {
				if chunk.Error != nil {
					t.Fatalf("Stream error: %v", chunk.Error)
				}
				if chunk.Done {
					break
				}
				output.WriteString(chunk.Content)
			}

			expectedOutput := strings.Join(tt.chunks, "")
			assert.Equal(t, expectedOutput, output.String())
		})
	}
}

func TestOllamaLLM_EmbeddingDualMode(t *testing.T) {
	tests := []struct {
		name         string
		useOpenAI    bool
		modelID      string
		expectedPath string
		shouldError  bool
		errorMsg     string
	}{
		{
			name:         "OpenAI embeddings",
			useOpenAI:    true,
			modelID:      "llama3:8b",
			expectedPath: "/v1/embeddings",
			shouldError:  false,
		},
		{
			name:         "Native embeddings with embedding model",
			useOpenAI:    false,
			modelID:      "nomic-embed-text",
			expectedPath: "/api/embeddings",
			shouldError:  false,
		},
		{
			name:        "Native embeddings with non-embedding model",
			useOpenAI:   false,
			modelID:     "llama3:8b",
			shouldError: true,
			errorMsg:    "embeddings require OpenAI API mode or embedding model",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.shouldError {
				// Test error case
				llm, err := NewOllamaLLM(core.ModelID(tt.modelID), WithNativeAPI())
				assert.NoError(t, err)

				_, err = llm.CreateEmbedding(context.Background(), "test input")
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
				return
			}

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, tt.expectedPath, r.URL.Path)
				assert.Equal(t, "POST", r.Method)

				w.WriteHeader(http.StatusOK)

				if tt.useOpenAI {
					// OpenAI embedding response
					resp := &openai.EmbeddingResponse{
						Object: "list",
						Data: []openai.EmbeddingData{{
							Object:    "embedding",
							Embedding: []float64{0.1, 0.2, 0.3, 0.4, 0.5},
							Index:     0,
						}},
						Model: tt.modelID,
						Usage: openai.CompletionUsage{TotalTokens: 10},
					}
					if err := json.NewEncoder(w).Encode(resp); err != nil {
						t.Fatalf("Failed to encode response: %v", err)
					}
				} else {
					// Native embedding response
					resp := &ollamaEmbeddingResponse{
						Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5},
						Size:      5,
						Usage: struct {
							Tokens int `json:"tokens"`
						}{Tokens: 10},
					}
					if err := json.NewEncoder(w).Encode(resp); err != nil {
						t.Fatalf("Failed to encode response: %v", err)
					}
				}
			}))
			defer server.Close()

			var llm *OllamaLLM
			var err error

			if tt.useOpenAI {
				llm, err = NewOllamaLLM(core.ModelID(tt.modelID), WithBaseURL(server.URL), WithOpenAIAPI())
			} else {
				llm, err = NewOllamaLLM(core.ModelID(tt.modelID), WithBaseURL(server.URL), WithNativeAPI())
			}
			assert.NoError(t, err)

			result, err := llm.CreateEmbedding(context.Background(), "test input")
			assert.NoError(t, err)
			assert.NotNil(t, result)
			assert.Len(t, result.Vector, 5)
			assert.Equal(t, []float32{0.1, 0.2, 0.3, 0.4, 0.5}, result.Vector)
			assert.Equal(t, 10, result.TokenCount)

			if tt.useOpenAI {
				assert.Equal(t, "openai", result.Metadata["mode"])
			} else {
				assert.Equal(t, "native", result.Metadata["mode"])
			}
		})
	}
}

func TestOllamaLLM_ConfigurationParsing(t *testing.T) {
	tests := []struct {
		name           string
		config         core.ProviderConfig
		expectedOpenAI bool
		expectedURL    string
		expectedKey    string
		expectedTimeout int
	}{
		{
			name: "Default modern config",
			config: core.ProviderConfig{
				BaseURL: "http://test:11434",
			},
			expectedOpenAI:  true,
			expectedURL:     "http://test:11434",
			expectedTimeout: 60,
		},
		{
			name: "Explicit native mode",
			config: core.ProviderConfig{
				Params: map[string]interface{}{
					"use_openai_api": false,
				},
			},
			expectedOpenAI:  false,
			expectedURL:     "http://localhost:11434",
			expectedTimeout: 60,
		},
		{
			name: "Custom settings",
			config: core.ProviderConfig{
				BaseURL: "http://custom:11435",
				APIKey:  "test-key",
				Params: map[string]interface{}{
					"use_openai_api": true,
					"timeout":        120,
				},
			},
			expectedOpenAI:  true,
			expectedURL:     "http://custom:11435",
			expectedKey:     "test-key",
			expectedTimeout: 120,
		},
		{
			name: "Endpoint override",
			config: core.ProviderConfig{
				BaseURL: "http://base:11434",
				Endpoint: &core.EndpointConfig{
					BaseURL:    "http://endpoint:11435",
					TimeoutSec: 180,
				},
			},
			expectedOpenAI:  true,
			expectedURL:     "http://endpoint:11435",
			expectedTimeout: 180,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llm, err := NewOllamaLLMFromConfig(context.Background(), tt.config, "llama3:8b")
			assert.NoError(t, err)
			assert.NotNil(t, llm)

			assert.Equal(t, tt.expectedOpenAI, llm.config.UseOpenAIAPI)
			assert.Equal(t, tt.expectedURL, llm.config.BaseURL)
			assert.Equal(t, tt.expectedKey, llm.config.APIKey)
			assert.Equal(t, tt.expectedTimeout, llm.config.Timeout)
		})
	}
}

func TestOllamaLLM_BackwardCompatibility(t *testing.T) {
	// Test that existing configurations still work without breaking changes
	t.Run("Legacy NewOllamaLLM call", func(t *testing.T) {
		// This simulates the old API usage pattern
		llm, err := NewOllamaLLM("llama2:7b")
		assert.NoError(t, err)
		assert.NotNil(t, llm)
		assert.True(t, llm.config.UseOpenAIAPI) // Should default to modern mode
		assert.Equal(t, "llama2:7b", llm.ModelID())
	})

	t.Run("Legacy config format", func(t *testing.T) {
		config := core.ProviderConfig{
			BaseURL: "http://legacy:11434",
			// No explicit OpenAI setting - should default to modern
		}
		llm, err := NewOllamaLLMFromConfig(context.Background(), config, "llama2:7b")
		assert.NoError(t, err)
		assert.True(t, llm.config.UseOpenAIAPI) // Should default to modern
	})

	t.Run("Explicit native mode for backward compatibility", func(t *testing.T) {
		llm, err := NewOllamaLLM("llama2:7b", WithNativeAPI())
		assert.NoError(t, err)
		assert.False(t, llm.config.UseOpenAIAPI)
		assert.Equal(t, "/api/generate", llm.GetEndpointConfig().Path)
	})
}

func TestOllamaLLM_ErrorHandling(t *testing.T) {
	tests := []struct {
		name           string
		serverStatus   int
		serverResponse string
		useOpenAI      bool
		expectedError  string
	}{
		{
			name:          "OpenAI server error",
			serverStatus:  http.StatusInternalServerError,
			useOpenAI:     true,
			expectedError: "API request failed with status 500",
		},
		{
			name:          "Native server error",
			serverStatus:  http.StatusInternalServerError,
			useOpenAI:     false,
			expectedError: "API request failed with status code 500",
		},
		{
			name:           "OpenAI invalid JSON",
			serverStatus:   http.StatusOK,
			serverResponse: "{invalid json",
			useOpenAI:      true,
			expectedError:  "failed to unmarshal response",
		},
		{
			name:           "Native invalid JSON",
			serverStatus:   http.StatusOK,
			serverResponse: "{invalid json",
			useOpenAI:      false,
			expectedError:  "failed to unmarshal response",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.serverStatus)
				if tt.serverResponse != "" {
					_, _ = w.Write([]byte(tt.serverResponse))
				}
			}))
			defer server.Close()

			var llm *OllamaLLM
			var err error

			if tt.useOpenAI {
				llm, err = NewOllamaLLM("llama3:8b", WithBaseURL(server.URL), WithOpenAIAPI())
			} else {
				llm, err = NewOllamaLLM("llama3:8b", WithBaseURL(server.URL), WithNativeAPI())
			}
			assert.NoError(t, err)

			_, err = llm.Generate(context.Background(), "test")
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedError)
		})
	}
}

func TestOllamaLLM_JSON_Generation(t *testing.T) {
	tests := []struct {
		name      string
		response  string
		useOpenAI bool
		expectErr bool
	}{
		{
			name:      "Valid JSON OpenAI",
			response:  `{"key": "value", "number": 42}`,
			useOpenAI: true,
			expectErr: false,
		},
		{
			name:      "Valid JSON Native",
			response:  `{"key": "value", "number": 42}`,
			useOpenAI: false,
			expectErr: false,
		},
		{
			name:      "Invalid JSON",
			response:  `invalid json`,
			useOpenAI: true,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)

				if tt.useOpenAI {
					resp := &openai.ChatCompletionResponse{
						Choices: []openai.ChatChoice{{
							Message: openai.ChatCompletionMessage{Content: tt.response},
						}},
						Usage: openai.CompletionUsage{},
					}
					if err := json.NewEncoder(w).Encode(resp); err != nil {
						t.Fatalf("Failed to encode response: %v", err)
					}
				} else {
					resp := &ollamaResponse{
						Response: tt.response,
					}
					if err := json.NewEncoder(w).Encode(resp); err != nil {
						t.Fatalf("Failed to encode response: %v", err)
					}
				}
			}))
			defer server.Close()

			var llm *OllamaLLM
			var err error

			if tt.useOpenAI {
				llm, err = NewOllamaLLM("llama3:8b", WithBaseURL(server.URL), WithOpenAIAPI())
			} else {
				llm, err = NewOllamaLLM("llama3:8b", WithBaseURL(server.URL), WithNativeAPI())
			}
			assert.NoError(t, err)

			result, err := llm.GenerateWithJSON(context.Background(), "Generate JSON")

			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Equal(t, "value", result["key"])
				assert.Equal(t, float64(42), result["number"])
			}
		})
	}
}

func TestOllamaLLM_MultimodalContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		resp := &openai.ChatCompletionResponse{
			Choices: []openai.ChatChoice{{
				Message: openai.ChatCompletionMessage{Content: "Processed text content"},
			}},
			Usage: openai.CompletionUsage{},
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("Failed to encode response: %v", err)
		}
	}))
	defer server.Close()

	llm, err := NewOllamaLLM("llama3:8b", WithBaseURL(server.URL))
	assert.NoError(t, err)

	t.Run("Text content only", func(t *testing.T) {
		content := []core.ContentBlock{
			core.NewTextBlock("Hello world"),
			core.NewTextBlock("Second text block"),
		}

		result, err := llm.GenerateWithContent(context.Background(), content)
		assert.NoError(t, err)
		assert.Equal(t, "Processed text content", result.Content)
	})

	t.Run("Non-text content should fail", func(t *testing.T) {
		content := []core.ContentBlock{
			core.NewImageBlock([]byte("fake image data"), "image/jpeg"),
		}

		_, err := llm.GenerateWithContent(context.Background(), content)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "multimodal content not yet supported")
	})

	t.Run("Streaming with content", func(t *testing.T) {
		content := []core.ContentBlock{
			core.NewTextBlock("Hello streaming world"),
		}

		// Mock streaming server
		streamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			flusher := w.(http.Flusher)
			streamResp := openai.ChatCompletionStreamResponse{
				Choices: []openai.ChatChoiceStream{{
					Delta: openai.ChatCompletionMessage{Content: "Streamed response"},
				}},
			}
			jsonData, _ := json.Marshal(streamResp)
			fmt.Fprintf(w, "data: %s\n\n", jsonData)
			flusher.Flush()

			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		}))
		defer streamServer.Close()

		streamLLM, err := NewOllamaLLM("llama3:8b", WithBaseURL(streamServer.URL))
		assert.NoError(t, err)

		streamResp, err := streamLLM.StreamGenerateWithContent(context.Background(), content)
		assert.NoError(t, err)

		var output strings.Builder
		for chunk := range streamResp.ChunkChannel {
			if chunk.Error != nil {
				t.Fatalf("Stream error: %v", chunk.Error)
			}
			if chunk.Done {
				break
			}
			output.WriteString(chunk.Content)
		}

		assert.Equal(t, "Streamed response", output.String())
	})
}

func TestOllamaLLM_Functions(t *testing.T) {
	llm, err := NewOllamaLLM("llama3:8b")
	assert.NoError(t, err)

	_, err = llm.GenerateWithFunctions(context.Background(), "test", []map[string]interface{}{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "function calling not yet implemented")
}

func TestOllamaLLM_BatchEmbeddings(t *testing.T) {
	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		w.WriteHeader(http.StatusOK)

		resp := &openai.EmbeddingResponse{
			Object: "list",
			Data: []openai.EmbeddingData{{
				Object:    "embedding",
				Embedding: []float64{0.1, 0.2, 0.3},
				Index:     0,
			}},
			Usage: openai.CompletionUsage{TotalTokens: 5},
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("Failed to encode response: %v", err)
		}
	}))
	defer server.Close()

	llm, err := NewOllamaLLM("llama3:8b", WithBaseURL(server.URL))
	assert.NoError(t, err)

	inputs := []string{"input1", "input2", "input3"}
	result, err := llm.CreateEmbeddings(context.Background(), inputs)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result.Embeddings, 3)
	assert.Nil(t, result.Error)
	assert.Equal(t, -1, result.ErrorIndex)

	// Verify each embedding has batch index
	for i, embedding := range result.Embeddings {
		assert.Equal(t, i, embedding.Metadata["batch_index"])
	}
}

func TestOllamaLLM_CapabilityDetection(t *testing.T) {
	tests := []struct {
		name               string
		modelID            string
		useOpenAI          bool
		expectedEmbedding  bool
		expectedStreaming  bool
	}{
		{
			name:              "Regular model OpenAI mode",
			modelID:           "llama3:8b",
			useOpenAI:         true,
			expectedEmbedding: true,  // OpenAI mode supports embeddings
			expectedStreaming: true,
		},
		{
			name:              "Regular model native mode",
			modelID:           "llama3:8b",
			useOpenAI:         false,
			expectedEmbedding: false, // Native mode requires embedding model
			expectedStreaming: true,
		},
		{
			name:              "Embedding model native mode",
			modelID:           "nomic-embed-text",
			useOpenAI:         false,
			expectedEmbedding: true,  // Embedding model in native mode
			expectedStreaming: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var llm *OllamaLLM
			var err error

			if tt.useOpenAI {
				llm, err = NewOllamaLLM(core.ModelID(tt.modelID), WithOpenAIAPI())
			} else {
				llm, err = NewOllamaLLM(core.ModelID(tt.modelID), WithNativeAPI())
			}
			assert.NoError(t, err)

			capabilities := llm.Capabilities()

			if tt.expectedEmbedding {
				assert.Contains(t, capabilities, core.CapabilityEmbedding)
			} else {
				assert.NotContains(t, capabilities, core.CapabilityEmbedding)
			}

			if tt.expectedStreaming {
				assert.Contains(t, capabilities, core.CapabilityStreaming)
			} else {
				assert.NotContains(t, capabilities, core.CapabilityStreaming)
			}

			// All should have basic capabilities
			assert.Contains(t, capabilities, core.CapabilityCompletion)
			assert.Contains(t, capabilities, core.CapabilityChat)
			assert.Contains(t, capabilities, core.CapabilityJSON)
		})
	}
}

func TestOllamaLLM_ProviderFactory(t *testing.T) {
	config := core.ProviderConfig{
		BaseURL: "http://test:11434",
		Params: map[string]interface{}{
			"use_openai_api": true,
		},
	}

	llm, err := OllamaProviderFactory(context.Background(), config, "llama3:8b")
	assert.NoError(t, err)
	assert.NotNil(t, llm)

	ollamaLLM, ok := llm.(*OllamaLLM)
	assert.True(t, ok)
	assert.True(t, ollamaLLM.config.UseOpenAIAPI)
	assert.Equal(t, "http://test:11434", ollamaLLM.config.BaseURL)
}

func TestOllamaLLM_ModelIDParsing(t *testing.T) {
	tests := []struct {
		name           string
		input          core.ModelID
		expectedOutput string
	}{
		{
			name:           "Plain model ID",
			input:          "llama3:8b",
			expectedOutput: "llama3:8b",
		},
		{
			name:           "Model ID with ollama prefix",
			input:          "ollama:llama3:8b",
			expectedOutput: "llama3:8b",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llm, err := NewOllamaLLM(tt.input)
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedOutput, llm.ModelID())
		})
	}
}

func TestOllamaLLM_ConcurrentStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher := w.(http.Flusher)
		for i := 0; i < 5; i++ {
			streamResp := openai.ChatCompletionStreamResponse{
				Choices: []openai.ChatChoiceStream{{
					Delta: openai.ChatCompletionMessage{Content: fmt.Sprintf("chunk_%d ", i)},
				}},
			}
			jsonData, _ := json.Marshal(streamResp)
			fmt.Fprintf(w, "data: %s\n\n", jsonData)
			flusher.Flush()
			time.Sleep(10 * time.Millisecond)
		}

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	llm, err := NewOllamaLLM("llama3:8b", WithBaseURL(server.URL))
	assert.NoError(t, err)

	// Test concurrent streaming
	const numStreams = 3
	var wg sync.WaitGroup
	results := make([]string, numStreams)

	for i := 0; i < numStreams; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			streamResp, err := llm.StreamGenerate(context.Background(), fmt.Sprintf("prompt_%d", idx))
			assert.NoError(t, err)

			var output strings.Builder
			for chunk := range streamResp.ChunkChannel {
				if chunk.Error != nil {
					t.Errorf("Stream %d error: %v", idx, chunk.Error)
					return
				}
				if chunk.Done {
					break
				}
				output.WriteString(chunk.Content)
			}

			results[idx] = output.String()
		}(i)
	}

	wg.Wait()

	// All streams should complete successfully
	for i, result := range results {
		assert.Equal(t, "chunk_0 chunk_1 chunk_2 chunk_3 chunk_4 ", result, "Stream %d failed", i)
	}
}