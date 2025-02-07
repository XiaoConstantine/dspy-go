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

// LlamacppLLM implements the core.LLM interface for Llamacpp-hosted models.
type LlamacppLLM struct {
	*core.BaseLLM
}

// NewLlamacppLLM creates a new LlamacppLLM instance.
func NewLlamacppLLM(endpoint string) (*LlamacppLLM, error) {
	if endpoint == "" {
		endpoint = "http://localhost:8080" // Default llamacpp endpoint
	}
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
	}
	endpointCfg := &core.EndpointConfig{
		BaseURL: endpoint,
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		TimeoutSec: 10 * 60, // Default timeout
	}

	return &LlamacppLLM{
		BaseLLM: core.NewBaseLLM("llamacpp", "", capabilities, endpointCfg),
	}, nil
}

type llamacppRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Stream      bool    `json:"stream"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
}

type llamacppResponse struct {
	Index           int    `json:"index"`            // Response index number
	Content         string `json:"content"`          // The actual generated text response
	Tokens          []any  `json:"tokens"`           // Token information (if requested)
	IDSlot          int    `json:"id_slot"`          // Slot ID in the server
	Stop            bool   `json:"stop"`             // Whether generation stopped naturally
	Model           string `json:"model"`            // Model identifier
	TokensPredicted int    `json:"tokens_predicted"` // Number of tokens generated
	TokensEvaluated int    `json:"tokens_evaluated"` // Number of tokens processed

	// Input and processing metadata
	Prompt       string `json:"prompt"`        // Original input prompt
	HasNewLine   bool   `json:"has_new_line"`  // Whether response has a newline
	Truncated    bool   `json:"truncated"`     // Whether response was truncated
	StopType     string `json:"stop_type"`     // Type of stop condition met
	StoppingWord string `json:"stopping_word"` // Word that triggered stopping
	TokensCached int    `json:"tokens_cached"` // Number of tokens cached
	// Performance timing information
	Timings struct {
		PromptN             int     `json:"prompt_n"`               // Number of prompt tokens
		PromptMS            float64 `json:"prompt_ms"`              // Time spent on prompt processing
		PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`    // Average time per prompt token
		PromptPerSecond     float64 `json:"prompt_per_second"`      // Tokens per second for prompt
		PredictedN          int     `json:"predicted_n"`            // Number of predicted tokens
		PredictedMS         float64 `json:"predicted_ms"`           // Time spent on prediction
		PredictedPerTokenMS float64 `json:"predicted_per_token_ms"` // Average time per predicted token
		PredictedPerSecond  float64 `json:"predicted_per_second"`   // Tokens per second for prediction
	} `json:"timings"`
}

type llamacppEmbeddingRequest struct {
	// Basic parameters
	Input     string `json:"input"`
	Model     string `json:"model,omitempty"`
	Normalize bool   `json:"normalize,omitempty"` // Whether to L2-normalize embeddings
	// Additional parameters from options
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type llamacppBatchEmbeddingRequest struct {
	// Multiple inputs for batch processing
	Inputs     []string               `json:"inputs"`
	Model      string                 `json:"model,omitempty"`
	Normalize  bool                   `json:"normalize,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type llamacppEmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
	// Token usage information
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
	Model string `json:"model"`
}

type llamacppBatchEmbeddingResponse struct {
	Embeddings []llamacppEmbeddingResponse `json:"embeddings"`
}

// Generate implements the core.LLM interface.
func (o *LlamacppLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := llamacppRequest{
		Prompt:      prompt,
		Stream:      false,
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.GetEndpointConfig().BaseURL+"/completion", bytes.NewBuffer(jsonData))
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
	var llamacppResp llamacppResponse
	err = json.Unmarshal(body, &llamacppResp)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to unmarshal response: %w", err)
	}
	// Create token info if available
	tokenInfo := &core.TokenInfo{
		PromptTokens:     llamacppResp.TokensEvaluated,
		CompletionTokens: llamacppResp.TokensPredicted,
		TotalTokens:      llamacppResp.TokensEvaluated + llamacppResp.TokensPredicted,
	}
	return &core.LLMResponse{Content: llamacppResp.Content, Usage: tokenInfo}, nil
}

// GenerateWithJSON implements the core.LLM interface.
func (o *LlamacppLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	response, err := o.Generate(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	return utils.ParseJSONResponse(response.Content)
}

func (o *LlamacppLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	// Apply options
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Prepare request body
	reqBody := llamacppEmbeddingRequest{
		Input:      input,
		Model:      opts.Model,
		Normalize:  true, // Default to normalized embeddings
		Parameters: opts.Params,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request: %w", err)
	}

	// Create request with timeout
	req, err := http.NewRequestWithContext(ctx, "POST", o.GetEndpointConfig().BaseURL+"/embedding", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding request: %w", err)
	}

	// Set headers
	for key, value := range o.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	// Execute request
	resp, err := o.GetHTTPClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send embedding request: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read embedding response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var llamacppResp llamacppEmbeddingResponse
	if err := json.Unmarshal(body, &llamacppResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal embedding response: %w", err)
	}

	// Convert to standard EmbeddingResult format
	result := &core.EmbeddingResult{
		Vector:     llamacppResp.Embedding,
		TokenCount: llamacppResp.Usage.TotalTokens,
		Metadata: map[string]interface{}{
			"model":         llamacppResp.Model,
			"prompt_tokens": llamacppResp.Usage.PromptTokens,
		},
	}

	return result, nil
}

// CreateEmbeddings implements batch embedding creation
func (o *LlamacppLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	// Apply options
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Validate batch size
	if opts.BatchSize <= 0 {
		opts.BatchSize = 32 // Default batch size
	}

	// Process in batches
	var allResults []core.EmbeddingResult
	var firstError error
	var errorIndex int = -1

	for i := 0; i < len(inputs); i += opts.BatchSize {
		end := i + opts.BatchSize
		if end > len(inputs) {
			end = len(inputs)
		}

		batch := inputs[i:end]
		reqBody := llamacppBatchEmbeddingRequest{
			Inputs:     batch,
			Model:      opts.Model,
			Normalize:  true,
			Parameters: opts.Params,
		}

		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("failed to marshal batch request: %w", err)
				errorIndex = i
			}
			continue
		}

		// Create request
		req, err := http.NewRequestWithContext(ctx, "POST", o.GetEndpointConfig().BaseURL+"/embeddings", bytes.NewBuffer(jsonData))
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

		var batchResp llamacppBatchEmbeddingResponse
		if err := json.Unmarshal(body, &batchResp); err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("failed to unmarshal batch response: %w", err)
				errorIndex = i
			}
			continue
		}

		// Convert batch results
		for _, embedding := range batchResp.Embeddings {
			result := core.EmbeddingResult{
				Vector:     embedding.Embedding,
				TokenCount: embedding.Usage.TotalTokens,
				Metadata: map[string]interface{}{
					"model":         embedding.Model,
					"prompt_tokens": embedding.Usage.PromptTokens,
				},
			}
			allResults = append(allResults, result)
		}
	}

	return &core.BatchEmbeddingResult{
		Embeddings: allResults,
		Error:      firstError,
		ErrorIndex: errorIndex,
	}, nil
}
