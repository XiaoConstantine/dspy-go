package llms

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/llms/openai"
)

func TestNewOpenAILLM(t *testing.T) {
	tests := []struct {
		name        string
		apiKey      string
		modelID     core.ModelID
		expectError bool
		errorType   errors.ErrorCode
	}{
		{
			name:        "valid api key and model",
			apiKey:      "test-api-key",
			modelID:     core.ModelOpenAIGPT4,
			expectError: false,
		},
		{
			name:        "empty api key",
			apiKey:      "",
			modelID:     core.ModelOpenAIGPT4,
			expectError: true,
			errorType:   errors.InvalidInput,
		},
		{
			name:        "different model",
			apiKey:      "test-api-key",
			modelID:     core.ModelOpenAIGPT4oMini,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// For empty api key test, temporarily unset environment variable
			var oldKey string
			if tt.apiKey == "" {
				oldKey = os.Getenv("OPENAI_API_KEY")
				os.Unsetenv("OPENAI_API_KEY")
				defer func() {
					if oldKey != "" {
						os.Setenv("OPENAI_API_KEY", oldKey)
					}
				}()
			}

			llm, err := NewOpenAI(tt.modelID, tt.apiKey)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
					return
				}
				if dsyErr, ok := err.(*errors.Error); ok {
					if dsyErr.Code() != tt.errorType {
						t.Errorf("expected error type %v, got %v", tt.errorType, dsyErr.Code())
					}
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if llm == nil {
				t.Errorf("expected non-nil LLM")
				return
			}

			if llm.ProviderName() != "openai" {
				t.Errorf("expected provider name 'openai', got %s", llm.ProviderName())
			}

			if llm.ModelID() != string(tt.modelID) {
				t.Errorf("expected model ID %s, got %s", tt.modelID, llm.ModelID())
			}

			// Check capabilities
			capabilities := llm.Capabilities()
			expectedCapabilities := []core.Capability{
				core.CapabilityCompletion,
				core.CapabilityChat,
				core.CapabilityJSON,
				core.CapabilityStreaming,
				core.CapabilityEmbedding,
				core.CapabilityToolCalling,
			}

			if len(capabilities) != len(expectedCapabilities) {
				t.Errorf("expected %d capabilities, got %d", len(expectedCapabilities), len(capabilities))
			}
		})
	}
}

func TestNewOpenAILLMFromConfig(t *testing.T) {
	tests := []struct {
		name        string
		config      core.ProviderConfig
		modelID     core.ModelID
		expectError bool
		errorType   errors.ErrorCode
		setupFunc   func()
	}{
		{
			name: "valid config",
			config: core.ProviderConfig{
				Name:   "openai",
				APIKey: "test-api-key",
			},
			modelID:     core.ModelOpenAIGPT4,
			expectError: false,
		},
		{
			name: "config without api key",
			config: core.ProviderConfig{
				Name: "openai",
			},
			modelID:     core.ModelOpenAIGPT4,
			expectError: true,
			errorType:   errors.InvalidInput,
			setupFunc:   func() { os.Unsetenv("OPENAI_API_KEY") },
		},
		{
			name: "invalid model id",
			config: core.ProviderConfig{
				Name:   "openai",
				APIKey: "test-api-key",
			},
			modelID:     "invalid-model",
			expectError: true,
			errorType:   errors.InvalidInput,
		},
		{
			name: "custom base url",
			config: core.ProviderConfig{
				Name:    "openai",
				APIKey:  "test-api-key",
				BaseURL: "https://custom.openai.com",
			},
			modelID:     core.ModelOpenAIGPT4,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setupFunc != nil {
				tt.setupFunc()
			}
			ctx := context.Background()
			llm, err := NewOpenAILLMFromConfig(ctx, tt.config, tt.modelID)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
					return
				}
				if dsyErr, ok := err.(*errors.Error); ok {
					if dsyErr.Code() != tt.errorType {
						t.Errorf("expected error type %v, got %v", tt.errorType, dsyErr.Code())
					}
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if llm == nil {
				t.Errorf("expected non-nil LLM")
				return
			}

			// Check custom base URL if provided
			if tt.config.BaseURL != "" {
				endpoint := llm.GetEndpointConfig()
				if endpoint.BaseURL != tt.config.BaseURL {
					t.Errorf("expected base URL %s, got %s", tt.config.BaseURL, endpoint.BaseURL)
				}
			}
		})
	}
}

func TestOpenAIProviderFactory(t *testing.T) {
	config := core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
	}
	ctx := context.Background()

	llm, err := OpenAIProviderFactory(ctx, config, core.ModelOpenAIGPT4)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}

	if llm == nil {
		t.Errorf("expected non-nil LLM")
		return
	}

	if llm.ProviderName() != "openai" {
		t.Errorf("expected provider name 'openai', got %s", llm.ProviderName())
	}
}

func TestIsValidOpenAIModel(t *testing.T) {
	tests := []struct {
		model    core.ModelID
		expected bool
	}{
		{core.ModelOpenAIGPT4, true},
		{core.ModelOpenAIGPT4Turbo, true},
		{core.ModelOpenAIGPT35Turbo, true},
		{core.ModelOpenAIGPT4o, true},
		{core.ModelOpenAIGPT4oMini, true},
		{"invalid-model", false},
		{core.ModelAnthropicSonnet, false},
		{core.ModelGoogleGeminiFlash, false},
	}

	for _, tt := range tests {
		t.Run(string(tt.model), func(t *testing.T) {
			result := isValidOpenAIModel(tt.model)
			if result != tt.expected {
				t.Errorf("expected %v for model %s, got %v", tt.expected, tt.model, result)
			}
		})
	}
}

func TestOpenAILLM_Generate(t *testing.T) {
	// Create a mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request method and path
		if r.Method != "POST" {
			t.Errorf("expected POST request, got %s", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("expected path /v1/chat/completions, got %s", r.URL.Path)
		}

		// Verify headers
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}
		if !strings.HasPrefix(r.Header.Get("Authorization"), "Bearer ") {
			t.Errorf("expected Authorization header with Bearer token")
		}

		// Parse and verify request body
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("failed to read request body: %v", err)
		}

		var req openai.ChatCompletionRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("failed to parse request body: %v", err)
		}

		if req.Model != "gpt-4" {
			t.Errorf("expected model gpt-4, got %s", req.Model)
		}
		if len(req.Messages) != 1 {
			t.Errorf("expected 1 message, got %d", len(req.Messages))
		}
		if req.Messages[0].Role != "user" {
			t.Errorf("expected role user, got %s", req.Messages[0].Role)
		}
		if req.Messages[0].Content != "Hello, world!" {
			t.Errorf("expected content 'Hello, world!', got %s", req.Messages[0].Content)
		}

		// Send mock response
		response := openai.ChatCompletionResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "gpt-4",
			Choices: []openai.ChatChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: "Hello! How can I help you today?",
					},
					FinishReason: "stop",
				},
			},
			Usage: openai.CompletionUsage{
				PromptTokens:     10,
				CompletionTokens: 15,
				TotalTokens:      25,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			t.Errorf("failed to encode response: %v", err)
		}
	}))
	defer server.Close()

	// Create LLM with custom endpoint
	config := core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}

	ctx := context.Background()
	llm, err := NewOpenAILLMFromConfig(ctx, config, core.ModelOpenAIGPT4)
	if err != nil {
		t.Fatalf("failed to create LLM: %v", err)
	}

	// Test generation
	response, err := llm.Generate(ctx, "Hello, world!")
	if err != nil {
		t.Fatalf("failed to generate: %v", err)
	}

	if response == nil {
		t.Fatalf("expected non-nil response")
	}

	if response.Content != "Hello! How can I help you today?" {
		t.Errorf("expected content 'Hello! How can I help you today?', got %s", response.Content)
	}

	if response.Usage == nil {
		t.Fatalf("expected non-nil usage")
	}

	if response.Usage.PromptTokens != 10 {
		t.Errorf("expected 10 prompt tokens, got %d", response.Usage.PromptTokens)
	}
	if response.Usage.CompletionTokens != 15 {
		t.Errorf("expected 15 completion tokens, got %d", response.Usage.CompletionTokens)
	}
	if response.Usage.TotalTokens != 25 {
		t.Errorf("expected 25 total tokens, got %d", response.Usage.TotalTokens)
	}
}

func TestOpenAILLM_GenerateWithJSON(t *testing.T) {
	// Create a mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Parse request body to verify response_format
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("failed to read request body: %v", err)
		}

		var req openai.ChatCompletionRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("failed to parse request body: %v", err)
		}

		if req.ResponseFormat == nil || req.ResponseFormat.Type != "json_object" {
			t.Errorf("expected response_format type json_object")
		}

		// Send mock JSON response
		response := openai.ChatCompletionResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "gpt-4",
			Choices: []openai.ChatChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: `{"result": "success", "data": {"key": "value"}}`,
					},
					FinishReason: "stop",
				},
			},
			Usage: openai.CompletionUsage{
				PromptTokens:     10,
				CompletionTokens: 15,
				TotalTokens:      25,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			t.Errorf("failed to encode response: %v", err)
		}
	}))
	defer server.Close()

	// Create LLM with custom endpoint
	config := core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}

	ctx := context.Background()
	llm, err := NewOpenAILLMFromConfig(ctx, config, core.ModelOpenAIGPT4)
	if err != nil {
		t.Fatalf("failed to create LLM: %v", err)
	}

	// Test JSON generation
	response, err := llm.GenerateWithJSON(ctx, "Generate JSON response")
	if err != nil {
		t.Fatalf("failed to generate JSON: %v", err)
	}

	if response == nil {
		t.Fatalf("expected non-nil response")
	}

	// Verify the parsed JSON structure
	if result, ok := response["result"]; !ok || result != "success" {
		t.Errorf("expected result field with value 'success'")
	}

	if data, ok := response["data"]; !ok {
		t.Errorf("expected data field")
	} else if dataMap, ok := data.(map[string]interface{}); !ok {
		t.Errorf("expected data to be a map")
	} else if key, ok := dataMap["key"]; !ok || key != "value" {
		t.Errorf("expected data.key field with value 'value'")
	}
}

func TestOpenAILLM_CreateEmbedding(t *testing.T) {
	// Create a mock server for embeddings
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request method and path
		if r.Method != "POST" {
			t.Errorf("expected POST request, got %s", r.Method)
		}
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("expected path /v1/embeddings, got %s", r.URL.Path)
		}

		// Parse and verify request body
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("failed to read request body: %v", err)
		}

		var req openai.EmbeddingRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("failed to parse request body: %v", err)
		}

		if req.Input != "test input" {
			t.Errorf("expected input 'test input', got %v", req.Input)
		}

		// Send mock response
		response := openai.EmbeddingResponse{
			Object: "list",
			Model:  "text-embedding-3-small",
			Data: []openai.EmbeddingData{
				{
					Object:    "embedding",
					Index:     0,
					Embedding: []float64{0.1, 0.2, 0.3, 0.4, 0.5},
				},
			},
			Usage: openai.CompletionUsage{
				PromptTokens: 5,
				TotalTokens:  5,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			t.Errorf("failed to encode response: %v", err)
		}
	}))
	defer server.Close()

	// Create LLM with custom endpoint
	config := core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}

	ctx := context.Background()
	llm, err := NewOpenAILLMFromConfig(ctx, config, core.ModelOpenAIGPT4)
	if err != nil {
		t.Fatalf("failed to create LLM: %v", err)
	}

	// Test embedding creation
	result, err := llm.CreateEmbedding(ctx, "test input")
	if err != nil {
		t.Fatalf("failed to create embedding: %v", err)
	}

	if result == nil {
		t.Fatalf("expected non-nil result")
	}

	if len(result.Vector) != 5 {
		t.Errorf("expected vector length 5, got %d", len(result.Vector))
	}

	expectedVector := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	for i, expected := range expectedVector {
		if result.Vector[i] != expected {
			t.Errorf("expected vector[%d] = %f, got %f", i, expected, result.Vector[i])
		}
	}

	if result.TokenCount != 5 {
		t.Errorf("expected token count 5, got %d", result.TokenCount)
	}
}

func TestOpenAILLM_CreateEmbeddings(t *testing.T) {
	// Create a mock server for batch embeddings
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Parse and verify request body
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("failed to read request body: %v", err)
		}

		var req openai.EmbeddingRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("failed to parse request body: %v", err)
		}

		// Verify input is an array
		inputs, ok := req.Input.([]interface{})
		if !ok {
			t.Errorf("expected input to be an array")
		}
		if len(inputs) != 2 {
			t.Errorf("expected 2 inputs, got %d", len(inputs))
		}

		// Send mock response with multiple embeddings
		response := openai.EmbeddingResponse{
			Object: "list",
			Model:  "text-embedding-3-small",
			Data: []openai.EmbeddingData{
				{
					Object:    "embedding",
					Index:     0,
					Embedding: []float64{0.1, 0.2, 0.3},
				},
				{
					Object:    "embedding",
					Index:     1,
					Embedding: []float64{0.4, 0.5, 0.6},
				},
			},
			Usage: openai.CompletionUsage{
				PromptTokens: 10,
				TotalTokens:  10,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			t.Errorf("failed to encode response: %v", err)
		}
	}))
	defer server.Close()

	// Create LLM with custom endpoint
	config := core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}

	ctx := context.Background()
	llm, err := NewOpenAILLMFromConfig(ctx, config, core.ModelOpenAIGPT4)
	if err != nil {
		t.Fatalf("failed to create LLM: %v", err)
	}

	// Test batch embedding creation
	inputs := []string{"input 1", "input 2"}
	result, err := llm.CreateEmbeddings(ctx, inputs)
	if err != nil {
		t.Fatalf("failed to create embeddings: %v", err)
	}

	if result == nil {
		t.Fatalf("expected non-nil result")
	}

	if result.Error != nil {
		t.Fatalf("expected no error, got %v", result.Error)
	}

	if len(result.Embeddings) != 2 {
		t.Errorf("expected 2 embeddings, got %d", len(result.Embeddings))
	}

	// Check first embedding
	if len(result.Embeddings[0].Vector) != 3 {
		t.Errorf("expected vector length 3, got %d", len(result.Embeddings[0].Vector))
	}
	expectedVector1 := []float32{0.1, 0.2, 0.3}
	for i, expected := range expectedVector1 {
		if result.Embeddings[0].Vector[i] != expected {
			t.Errorf("expected vector[0][%d] = %f, got %f", i, expected, result.Embeddings[0].Vector[i])
		}
	}

	// Check second embedding
	if len(result.Embeddings[1].Vector) != 3 {
		t.Errorf("expected vector length 3, got %d", len(result.Embeddings[1].Vector))
	}
	expectedVector2 := []float32{0.4, 0.5, 0.6}
	for i, expected := range expectedVector2 {
		if result.Embeddings[1].Vector[i] != expected {
			t.Errorf("expected vector[1][%d] = %f, got %f", i, expected, result.Embeddings[1].Vector[i])
		}
	}
}

func TestOpenAILLM_StreamGenerate(t *testing.T) {
	// Create a mock server for streaming
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Parse request body to verify streaming is enabled
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("failed to read request body: %v", err)
		}

		var req openai.ChatCompletionRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("failed to parse request body: %v", err)
		}

		if !req.Stream {
			t.Errorf("expected stream to be true")
		}

		// Set SSE headers
		w.Header().Set("Content-Type", "text/plain")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// Send streaming chunks
		chunks := []string{
			`data: {"id":"test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
			`data: {"id":"test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
			`data: {"id":"test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}`,
			"data: [DONE]",
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Errorf("expected flusher interface")
			return
		}

		for _, chunk := range chunks {
			fmt.Fprintf(w, "%s\n\n", chunk)
			flusher.Flush()
		}
	}))
	defer server.Close()

	// Create LLM with custom endpoint
	config := core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}

	ctx := context.Background()
	llm, err := NewOpenAILLMFromConfig(ctx, config, core.ModelOpenAIGPT4)
	if err != nil {
		t.Fatalf("failed to create LLM: %v", err)
	}

	// Test streaming generation
	stream, err := llm.StreamGenerate(ctx, "Hello")
	if err != nil {
		t.Fatalf("failed to start streaming: %v", err)
	}

	if stream == nil {
		t.Fatalf("expected non-nil stream")
	}

	var chunks []string
	var done bool

	// Read all chunks
	for chunk := range stream.ChunkChannel {
		if chunk.Error != nil {
			t.Fatalf("unexpected error in stream: %v", chunk.Error)
		}
		if chunk.Done {
			done = true
			break
		}
		if chunk.Content != "" {
			chunks = append(chunks, chunk.Content)
		}
	}

	// Verify chunks
	expectedChunks := []string{"Hello", " world", "!"}
	if len(chunks) != len(expectedChunks) {
		t.Errorf("expected %d chunks, got %d", len(expectedChunks), len(chunks))
	}

	for i, expected := range expectedChunks {
		if i < len(chunks) && chunks[i] != expected {
			t.Errorf("expected chunk[%d] = %s, got %s", i, expected, chunks[i])
		}
	}

	if !done {
		t.Errorf("expected stream to be done")
	}
}

func TestOpenAILLM_ErrorHandling(t *testing.T) {
	// Create a mock server that returns errors
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return API error
		w.WriteHeader(http.StatusBadRequest)
		errorResp := openai.ErrorResponse{
			Error: openai.APIError{
				Message: "Invalid request",
				Type:    "invalid_request_error",
				Code:    "invalid_api_key",
			},
		}
		if err := json.NewEncoder(w).Encode(errorResp); err != nil {
			t.Errorf("failed to encode error response: %v", err)
		}
	}))
	defer server.Close()

	// Create LLM with custom endpoint
	config := core.ProviderConfig{
		Name:   "openai",
		APIKey: "invalid-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}

	ctx := context.Background()
	llm, err := NewOpenAILLMFromConfig(ctx, config, core.ModelOpenAIGPT4)
	if err != nil {
		t.Fatalf("failed to create LLM: %v", err)
	}

	// Test error handling in Generate
	_, err = llm.Generate(ctx, "Hello")
	if err == nil {
		t.Errorf("expected error but got none")
	}

	// Verify error details
	if dsyErr, ok := err.(*errors.Error); ok {
		if dsyErr.Code() != errors.LLMGenerationFailed {
			t.Errorf("expected LLMGenerationFailed error, got %v", dsyErr.Code())
		}
	}
}

func TestOpenAILLM_GenerateWithOptions(t *testing.T) {
	// Create a mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Parse request body to verify options
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("failed to read request body: %v", err)
		}

		var req openai.ChatCompletionRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("failed to parse request body: %v", err)
		}

		// Verify options
		if req.MaxTokens == nil || *req.MaxTokens != 100 {
			t.Errorf("expected max_tokens 100, got %v", req.MaxTokens)
		}
		if req.Temperature == nil || *req.Temperature != 0.8 {
			t.Errorf("expected temperature 0.8, got %v", req.Temperature)
		}
		if req.TopP == nil || *req.TopP != 0.9 {
			t.Errorf("expected top_p 0.9, got %v", req.TopP)
		}
		if len(req.Stop) != 2 || req.Stop[0] != "\\n" || req.Stop[1] != "END" {
			t.Errorf("expected stop sequences [\\n, END], got %v", req.Stop)
		}

		// Send mock response
		response := openai.ChatCompletionResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "gpt-4",
			Choices: []openai.ChatChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: "Test response",
					},
					FinishReason: "stop",
				},
			},
			Usage: openai.CompletionUsage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			t.Errorf("failed to encode response: %v", err)
		}
	}))
	defer server.Close()

	// Create LLM with custom endpoint
	config := core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}

	ctx := context.Background()
	llm, err := NewOpenAILLMFromConfig(ctx, config, core.ModelOpenAIGPT4)
	if err != nil {
		t.Fatalf("failed to create LLM: %v", err)
	}

	// Test generation with options
	response, err := llm.Generate(ctx, "Hello",
		core.WithMaxTokens(100),
		core.WithTemperature(0.8),
		core.WithTopP(0.9),
		core.WithStopSequences("\\n", "END"),
	)
	if err != nil {
		t.Fatalf("failed to generate: %v", err)
	}

	if response == nil {
		t.Fatalf("expected non-nil response")
	}

	if response.Content != "Test response" {
		t.Errorf("expected content 'Test response', got %s", response.Content)
	}
}

func TestOpenAILLM_GenerateWithFunctions(t *testing.T) {
	var capturedRequest openai.ChatCompletionRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("expected /v1/chat/completions, got %s", r.URL.Path)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}
		if err := json.Unmarshal(body, &capturedRequest); err != nil {
			t.Fatalf("failed to parse request body: %v", err)
		}

		response := openai.ChatCompletionResponse{
			ID:      "chatcmpl-functions",
			Object:  "chat.completion",
			Created: 1234567890,
			Model:   "gpt-4o",
			Choices: []openai.ChatChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role: "assistant",
						ToolCalls: []openai.ChatCompletionToolCall{
							{
								ID:   "call_123",
								Type: "function",
								Function: openai.ChatCompletionFunctionCallDelta{
									Name:      "get_weather",
									Arguments: `{"location":"New York","unit":"celsius"}`,
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
			Usage: openai.CompletionUsage{
				PromptTokens:     11,
				CompletionTokens: 7,
				TotalTokens:      18,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			t.Fatalf("failed to encode response: %v", err)
		}
	}))
	defer server.Close()

	llm, err := NewOpenAILLM(
		core.ModelOpenAIGPT4o,
		WithAPIKey("test-api-key"),
		WithOpenAIBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("failed to create LLM: %v", err)
	}

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
				"required": []string{"location"},
			},
		},
	}

	result, err := llm.GenerateWithFunctions(context.Background(), "What's the weather?", functions)
	if err != nil {
		t.Fatalf("GenerateWithFunctions failed: %v", err)
	}

	toolCall, ok := result["function_call"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected function_call map in result, got %#v", result["function_call"])
	}

	if toolCall["name"] != "get_weather" {
		t.Errorf("expected tool name get_weather, got %v", toolCall["name"])
	}

	arguments, ok := toolCall["arguments"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected arguments map, got %#v", toolCall["arguments"])
	}
	if arguments["location"] != "New York" {
		t.Errorf("expected location New York, got %v", arguments["location"])
	}
	if arguments["unit"] != "celsius" {
		t.Errorf("expected unit celsius, got %v", arguments["unit"])
	}

	usage, ok := result["_usage"].(*core.TokenInfo)
	if !ok {
		t.Fatalf("expected token usage metadata in result")
	}
	if usage.TotalTokens != 18 {
		t.Errorf("expected 18 total tokens, got %d", usage.TotalTokens)
	}

	if len(capturedRequest.Tools) != 1 {
		t.Fatalf("expected 1 tool in request, got %d", len(capturedRequest.Tools))
	}
	if capturedRequest.Tools[0].Function.Name != "get_weather" {
		t.Errorf("expected tool name get_weather in request, got %s", capturedRequest.Tools[0].Function.Name)
	}
	if capturedRequest.ToolChoice != "auto" {
		t.Errorf("expected tool_choice auto, got %v", capturedRequest.ToolChoice)
	}
}

// Tests for new functional options pattern

func TestNewOpenAILLM_Options(t *testing.T) {
	tests := []struct {
		name        string
		options     []OpenAIOption
		expectError bool
		validateFn  func(*testing.T, *OpenAILLM)
	}{
		{
			name:    "default configuration",
			options: []OpenAIOption{WithAPIKey("test-key")},
			validateFn: func(t *testing.T, llm *OpenAILLM) {
				if llm.GetEndpointConfig().BaseURL != "https://api.openai.com" {
					t.Errorf("expected base URL https://api.openai.com, got %s", llm.GetEndpointConfig().BaseURL)
				}
				if llm.GetEndpointConfig().Path != "/v1/chat/completions" {
					t.Errorf("expected path /v1/chat/completions, got %s", llm.GetEndpointConfig().Path)
				}
			},
		},
		{
			name: "custom base URL",
			options: []OpenAIOption{
				WithAPIKey("test-key"),
				WithOpenAIBaseURL("http://localhost:4000"),
			},
			validateFn: func(t *testing.T, llm *OpenAILLM) {
				if llm.GetEndpointConfig().BaseURL != "http://localhost:4000" {
					t.Errorf("expected base URL http://localhost:4000, got %s", llm.GetEndpointConfig().BaseURL)
				}
			},
		},
		{
			name: "custom path",
			options: []OpenAIOption{
				WithAPIKey("test-key"),
				WithOpenAIPath("/chat/completions"),
			},
			validateFn: func(t *testing.T, llm *OpenAILLM) {
				if llm.GetEndpointConfig().Path != "/chat/completions" {
					t.Errorf("expected path /chat/completions, got %s", llm.GetEndpointConfig().Path)
				}
			},
		},
		{
			name: "custom headers",
			options: []OpenAIOption{
				WithAPIKey("test-key"),
				WithHeader("Custom-Header", "custom-value"),
			},
			validateFn: func(t *testing.T, llm *OpenAILLM) {
				if llm.GetEndpointConfig().Headers["Custom-Header"] != "custom-value" {
					t.Errorf("expected Custom-Header value custom-value, got %s", llm.GetEndpointConfig().Headers["Custom-Header"])
				}
			},
		},
		{
			name: "custom timeout",
			options: []OpenAIOption{
				WithAPIKey("test-key"),
				WithOpenAITimeout(90 * time.Second),
			},
			validateFn: func(t *testing.T, llm *OpenAILLM) {
				if llm.GetEndpointConfig().TimeoutSec != 90 {
					t.Errorf("expected timeout 90 seconds, got %d", llm.GetEndpointConfig().TimeoutSec)
				}
			},
		},
		{
			name:        "missing API key",
			options:     []OpenAIOption{},
			expectError: true,
		},
		{
			name:    "environment variable fallback",
			options: []OpenAIOption{
				// No explicit API key, should fall back to environment
			},
			expectError: false, // Will only pass if OPENAI_API_KEY is set
			validateFn: func(t *testing.T, llm *OpenAILLM) {
				// This test might fail if OPENAI_API_KEY is not set, but that's expected
				envKey := os.Getenv("OPENAI_API_KEY")
				if envKey != "" && llm.apiKey != envKey {
					t.Errorf("expected API key from environment %s, got %s", envKey, llm.apiKey)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Handle environment variable for specific tests
			var oldKey string
			switch tt.name {
			case "environment variable fallback":
				oldKey = os.Getenv("OPENAI_API_KEY")
				os.Setenv("OPENAI_API_KEY", "test-env-key")
				defer func() {
					if oldKey == "" {
						os.Unsetenv("OPENAI_API_KEY")
					} else {
						os.Setenv("OPENAI_API_KEY", oldKey)
					}
				}()
			case "missing API key":
				// For missing API key test, temporarily unset environment variable
				oldKey = os.Getenv("OPENAI_API_KEY")
				os.Unsetenv("OPENAI_API_KEY")
				defer func() {
					if oldKey != "" {
						os.Setenv("OPENAI_API_KEY", oldKey)
					}
				}()
			}

			llm, err := NewOpenAILLM(core.ModelOpenAIGPT4, tt.options...)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
					return
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if llm == nil {
				t.Errorf("expected non-nil LLM")
				return
			}

			if tt.validateFn != nil {
				tt.validateFn(t, llm)
			}
		})
	}
}

func TestConvenienceConstructors(t *testing.T) {
	t.Run("NewLiteLLM", func(t *testing.T) {
		llm, err := NewLiteLLM(core.ModelOpenAIGPT4, "test-key")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			return
		}
		if llm.ProviderName() != "litellm" {
			t.Errorf("expected provider name litellm, got %s", llm.ProviderName())
		}
		if llm.GetEndpointConfig().BaseURL != "http://localhost:4000" {
			t.Errorf("expected base URL http://localhost:4000, got %s", llm.GetEndpointConfig().BaseURL)
		}
	})

	t.Run("NewLocalAI", func(t *testing.T) {
		llm, err := NewLocalAI(core.ModelOpenAIGPT4, "http://localhost:8080")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			return
		}
		if llm.ProviderName() != "localai" {
			t.Errorf("expected provider name localai, got %s", llm.ProviderName())
		}
		if llm.GetEndpointConfig().BaseURL != "http://localhost:8080" {
			t.Errorf("expected base URL http://localhost:8080, got %s", llm.GetEndpointConfig().BaseURL)
		}
	})

	t.Run("NewFastChat", func(t *testing.T) {
		llm, err := NewFastChat(core.ModelOpenAIGPT4, "http://localhost:8000")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			return
		}
		if llm.ProviderName() != "fastchat" {
			t.Errorf("expected provider name fastchat, got %s", llm.ProviderName())
		}
		if llm.GetEndpointConfig().BaseURL != "http://localhost:8000" {
			t.Errorf("expected base URL http://localhost:8000, got %s", llm.GetEndpointConfig().BaseURL)
		}
	})

	t.Run("NewOpenAICompatible", func(t *testing.T) {
		llm, err := NewOpenAICompatible("custom-provider", core.ModelOpenAIGPT4, "http://custom.ai", WithAPIKey("test-key"))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			return
		}
		if llm.ProviderName() != "custom-provider" {
			t.Errorf("expected provider name custom-provider, got %s", llm.ProviderName())
		}
		if llm.GetEndpointConfig().BaseURL != "http://custom.ai" {
			t.Errorf("expected base URL http://custom.ai, got %s", llm.GetEndpointConfig().BaseURL)
		}
	})
}

func TestProviderFactories(t *testing.T) {
	ctx := context.Background()

	t.Run("LiteLLMProviderFactory", func(t *testing.T) {
		config := core.ProviderConfig{
			Name:   "litellm",
			APIKey: "test-key",
		}
		llm, err := LiteLLMProviderFactory(ctx, config, core.ModelOpenAIGPT4)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			return
		}
		if llm.ProviderName() != "litellm" {
			t.Errorf("expected provider name litellm, got %s", llm.ProviderName())
		}
	})

	t.Run("LocalAIProviderFactory", func(t *testing.T) {
		config := core.ProviderConfig{
			Name:    "localai",
			BaseURL: "http://test.local",
		}
		llm, err := LocalAIProviderFactory(ctx, config, core.ModelOpenAIGPT4)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			return
		}
		if llm.ProviderName() != "localai" {
			t.Errorf("expected provider name localai, got %s", llm.ProviderName())
		}
	})

	t.Run("FastChatProviderFactory", func(t *testing.T) {
		config := core.ProviderConfig{
			Name:    "fastchat",
			BaseURL: "http://fastchat.local",
		}
		llm, err := FastChatProviderFactory(ctx, config, core.ModelOpenAIGPT4)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			return
		}
		if llm.ProviderName() != "fastchat" {
			t.Errorf("expected provider name fastchat, got %s", llm.ProviderName())
		}
	})
}

func TestOptionsChaining(t *testing.T) {
	// Test that multiple options can be chained together properly
	llm, err := NewOpenAILLM(core.ModelOpenAIGPT4,
		WithAPIKey("test-key"),
		WithOpenAIBaseURL("http://localhost:4000"),
		WithOpenAIPath("/custom/completions"),
		WithOpenAITimeout(120*time.Second),
		WithHeader("X-Custom", "value1"),
		WithHeader("X-Another", "value2"),
	)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	endpoint := llm.GetEndpointConfig()

	if endpoint.BaseURL != "http://localhost:4000" {
		t.Errorf("expected base URL http://localhost:4000, got %s", endpoint.BaseURL)
	}

	if endpoint.Path != "/custom/completions" {
		t.Errorf("expected path /custom/completions, got %s", endpoint.Path)
	}

	if endpoint.TimeoutSec != 120 {
		t.Errorf("expected timeout 120 seconds, got %d", endpoint.TimeoutSec)
	}

	if endpoint.Headers["X-Custom"] != "value1" {
		t.Errorf("expected X-Custom header value1, got %s", endpoint.Headers["X-Custom"])
	}

	if endpoint.Headers["X-Another"] != "value2" {
		t.Errorf("expected X-Another header value2, got %s", endpoint.Headers["X-Another"])
	}

	// Verify standard headers are still present
	if !strings.HasPrefix(endpoint.Headers["Authorization"], "Bearer ") {
		t.Errorf("expected Authorization header with Bearer token")
	}

	if endpoint.Headers["Content-Type"] != "application/json" {
		t.Errorf("expected Content-Type application/json, got %s", endpoint.Headers["Content-Type"])
	}
}
