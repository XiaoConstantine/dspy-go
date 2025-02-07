package llms

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// OllamaLLM implements the core.LLM interface for Ollama-hosted models.
type OllamaLLM struct {
	*core.BaseLLM
}

// NewOllamaLLM creates a new OllamaLLM instance.
func NewOllamaLLM(endpoint, model string) (*OllamaLLM, error) {
	if endpoint == "" {
		endpoint = "http://localhost:11434" // Default Ollama endpoint
	}
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
	}
	endpointCfg := &core.EndpointConfig{
		BaseURL: endpoint,
		Path:    "api/generate", // Ollama's generation endpoint
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		TimeoutSec: 10 * 60, // Default timeout
	}

	return &OllamaLLM{
		BaseLLM: core.NewBaseLLM("ollama", core.ModelID(model), capabilities, endpointCfg),
	}, nil
}

type ollamaRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Stream      bool    `json:"stream"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
}

type ollamaResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Response  string `json:"response"`
}

// Define request/response structures for Ollama embeddings
type ollamaEmbeddingRequest struct {
	Model   string                 `json:"model"`   // Required model name
	Prompt  string                 `json:"prompt"`  // Input text for embedding
	Options map[string]interface{} `json:"options"` // Additional model-specific options
}

type ollamaBatchEmbeddingRequest struct {
	Model   string                 `json:"model"`   // Required model name
	Prompts []string               `json:"prompts"` // List of inputs for batch embedding
	Options map[string]interface{} `json:"options"` // Additional model-specific options
}

type ollamaEmbeddingResponse struct {
	Embedding []float32 `json:"embedding"` // The embedding vector
	Size      int       `json:"size"`      // Dimension of the embedding
	Usage     struct {
		Tokens int `json:"tokens"` // Number of tokens processed
	} `json:"usage"`
}

type ollamaBatchEmbeddingResponse struct {
	Embeddings []ollamaEmbeddingResponse `json:"embeddings"`
}

// Generate implements the core.LLM interface.
func (o *OllamaLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := ollamaRequest{
		Model:       o.ModelID(),
		Prompt:      prompt,
		Stream:      false,
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.GetEndpointConfig().BaseURL+"/api/generate", bytes.NewBuffer(jsonData))
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to create request: %w", err)
	}

	for key, value := range o.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	resp, err := o.GetHTTPClient().Do(req)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return &core.LLMResponse{}, fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(body))
	}

	var ollamaResp ollamaResponse
	err = json.Unmarshal(body, &ollamaResp)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// TODO: add token usage
	return &core.LLMResponse{Content: ollamaResp.Response}, nil
}

// GenerateWithJSON implements the core.LLM interface.
func (o *OllamaLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	response, err := o.Generate(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	return utils.ParseJSONResponse(response.Content)
}

// CreateEmbedding generates embeddings for a single input
func (o *OllamaLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	// Apply the provided options
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Create the request body
	reqBody := ollamaEmbeddingRequest{
		Model:   o.ModelID(), // Use the model ID from the LLM instance
		Prompt:  input,
		Options: opts.Params,
	}

	// Marshal the request body
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request: %w", err)
	}

	// Create the HTTP request
	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		fmt.Sprintf("%s/api/embeddings", o.GetEndpointConfig().BaseURL),
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	for key, value := range o.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	// Execute the request
	resp, err := o.GetHTTPClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check for non-200 status codes
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(body))
	}

	// Parse the response
	var ollamaResp ollamaEmbeddingResponse
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Convert to our standard EmbeddingResult format
	result := &core.EmbeddingResult{
		Vector:     ollamaResp.Embedding,
		TokenCount: ollamaResp.Usage.Tokens,
		Metadata: map[string]interface{}{
			"model":          o.ModelID(),
			"embedding_size": ollamaResp.Size,
		},
	}

	return result, nil
}

// CreateEmbeddings generates embeddings for multiple inputs in batches
func (o *OllamaLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	// Apply options
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Use default batch size if not specified
	if opts.BatchSize <= 0 {
		opts.BatchSize = 32
	}

	// Process inputs in batches
	var allResults []core.EmbeddingResult
	var firstError error
	var errorIndex int = -1

	// Process each batch
	for i := 0; i < len(inputs); i += opts.BatchSize {
		end := i + opts.BatchSize
		if end > len(inputs) {
			end = len(inputs)
		}

		// Create batch request
		batch := inputs[i:end]
		reqBody := ollamaBatchEmbeddingRequest{
			Model:   o.ModelID(),
			Prompts: batch,
			Options: opts.Params,
		}

		// Marshal request body
		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("failed to marshal batch request: %w", err)
				errorIndex = i
			}
			continue
		}

		// Create HTTP request
		req, err := http.NewRequestWithContext(
			ctx,
			"POST",
			fmt.Sprintf("%s/api/embeddings/batch", o.GetEndpointConfig().BaseURL),
			bytes.NewBuffer(jsonData),
		)
		if err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("failed to create batch request: %w", err)
				errorIndex = i
			}
			continue
		}

		// Set headers
		for key, value := range o.GetEndpointConfig().Headers {
			req.Header.Set(key, value)
		}

		// Execute request
		resp, err := o.GetHTTPClient().Do(req)
		if err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("failed to send batch request: %w", err)
				errorIndex = i
			}
			continue
		}

		// Read and parse response
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("failed to read batch response: %w", err)
				errorIndex = i
			}
			continue
		}

		if resp.StatusCode != http.StatusOK {
			if firstError == nil {
				firstError = fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(body))
				errorIndex = i
			}
			continue
		}

		var batchResp ollamaBatchEmbeddingResponse
		if err := json.Unmarshal(body, &batchResp); err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("failed to unmarshal batch response: %w", err)
				errorIndex = i
			}
			continue
		}

		// Convert batch results
		for j, embedding := range batchResp.Embeddings {
			result := core.EmbeddingResult{
				Vector:     embedding.Embedding,
				TokenCount: embedding.Usage.Tokens,
				Metadata: map[string]interface{}{
					"model":          o.ModelID(),
					"embedding_size": embedding.Size,
					"batch_index":    i + j,
				},
			}
			allResults = append(allResults, result)
		}
	}

	// Return the combined results
	return &core.BatchEmbeddingResult{
		Embeddings: allResults,
		Error:      firstError,
		ErrorIndex: errorIndex,
	}, nil
}
