package llms

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
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
		return &core.LLMResponse{}, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to marshal request body"),
			errors.Fields{
				"model": o.ModelID(),
			})
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.GetEndpointConfig().BaseURL+"/completion", bytes.NewBuffer(jsonData))
	if err != nil {
		return &core.LLMResponse{}, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to create request"),
			errors.Fields{
				"model": o.ModelID(),
			})
	}

	for key, value := range o.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	resp, err := o.GetHTTPClient().Do(req)
	if err != nil {
		return &core.LLMResponse{}, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to send request"),
			errors.Fields{
				"model": o.ModelID(),
			})
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return &core.LLMResponse{}, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to read response body"),
			errors.Fields{
				"model": o.ModelID(),
			})

	}

	if resp.StatusCode != http.StatusOK {
		return &core.LLMResponse{}, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("API request failed with status code %d", resp.StatusCode)),
			errors.Fields{
				"model":         o.ModelID(),
				"status_code":   resp.StatusCode,
				"response_body": string(body),
			})
	}
	var llamacppResp llamacppResponse
	err = json.Unmarshal(body, &llamacppResp)
	if err != nil {
		return &core.LLMResponse{}, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal response"),
			errors.Fields{
				"resp":  body[:50],
				"model": o.ModelID(),
			})

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

func (o *LlamacppLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	panic("Not implemented")
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
	var rawResp []struct {
		Index     int         `json:"index"`
		Embedding [][]float32 `json:"embedding"`
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(body))
	}
	if err := json.Unmarshal(body, &rawResp); err != nil {
		return nil, fmt.Errorf("failed to parse raw JSON response: %w", err)
	}

	// Validate the response
	if len(rawResp) == 0 || len(rawResp[0].Embedding) == 0 {
		return nil, fmt.Errorf("received empty embedding response")
	}

	// Extract the embedding vector and create our result
	vector := rawResp[0].Embedding[0]

	// Convert to standard EmbeddingResult format
	result := &core.EmbeddingResult{
		Vector: vector,
	}

	return result, nil
}

// CreateEmbeddings implements batch embedding creation.
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
	var errorIndex = -1

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
		var rawResp []struct {
			Index     int         `json:"index"`
			Embedding [][]float32 `json:"embedding"`
		}

		if err := json.Unmarshal(body, &rawResp); err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("failed to unmarshal batch response: %w", err)
				errorIndex = i
			}
			continue
		}

		// Convert batch results
		for _, item := range rawResp {
			if len(item.Embedding) == 0 || len(item.Embedding[0]) == 0 {
				if firstError == nil {
					firstError = fmt.Errorf("received empty embedding at index %d", item.Index)
					errorIndex = i + item.Index
				}
				continue
			}

			// Create the embedding result with metadata
			result := core.EmbeddingResult{
				Vector: item.Embedding[0],
				Metadata: map[string]interface{}{
					"index":        item.Index,
					"batch_offset": i,
					"model":        opts.Model,
					"vector_size":  len(item.Embedding[0]),
				},
			}

			// Ensure results are ordered correctly using the index
			for len(allResults) <= i+item.Index {
				allResults = append(allResults, core.EmbeddingResult{})
			}
			allResults[i+item.Index] = result
		}
	}
	if firstError != nil {
		return nil, firstError
	}

	return &core.BatchEmbeddingResult{
		Embeddings: allResults,
		Error:      firstError,
		ErrorIndex: errorIndex,
	}, nil
}

// StreamGenerate implements streaming for LlamaCPP.
func (o *LlamacppLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Create request body with streaming enabled
	reqBody := llamacppRequest{
		Model:       o.ModelID(),
		Prompt:      prompt,
		Stream:      true, // Enable streaming
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to marshal request body"),
			errors.Fields{
				"model": o.ModelID(),
			})
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, "POST",
		o.GetEndpointConfig().BaseURL+"/completion", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to create request"),
			errors.Fields{
				"model": o.ModelID(),
			})
	}

	for key, value := range o.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	// Create stream context and cancellation function
	streamCtx, cancelFunc := context.WithCancel(ctx)

	// Create channel for stream chunks
	chunkChan := make(chan core.StreamChunk)

	// Create stream response object
	response := &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelFunc,
	}

	// Start goroutine to handle streaming
	go func() {
		defer close(chunkChan)
		defer cancelFunc() // Ensure context is cancelled when goroutine exits

		// Send HTTP request
		resp, err := o.GetHTTPClient().Do(req)
		if err != nil {
			chunkChan <- core.StreamChunk{
				Error: errors.WithFields(
					errors.Wrap(err, errors.LLMGenerationFailed, "failed to send request"),
					errors.Fields{
						"model": o.ModelID(),
					}),
			}
			return
		}
		defer resp.Body.Close()

		// Check response status
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			chunkChan <- core.StreamChunk{
				Error: errors.WithFields(
					errors.New(errors.LLMGenerationFailed, fmt.Sprintf(
						"API request failed with status code %d", resp.StatusCode)),
					errors.Fields{
						"model":         o.ModelID(),
						"status_code":   resp.StatusCode,
						"response_body": string(body),
					}),
			}
			return
		}

		// LlamaCPP returns newline-delimited JSON objects for streaming
		scanner := bufio.NewScanner(resp.Body)
		var tokenCount int

		for scanner.Scan() {
			// Check if context was cancelled
			select {
			case <-streamCtx.Done():
				return
			default:
				// Process next chunk
			}

			line := scanner.Text()
			if strings.TrimSpace(line) == "" {
				continue
			}

			// Parse the streaming response
			var streamResp llamacppResponse
			if err := json.Unmarshal([]byte(line), &streamResp); err != nil {
				// Skip lines that aren't valid JSON
				continue
			}

			// Only send non-empty content
			if streamResp.Content != "" {
				// Create token info for this chunk
				tokenCount += len(strings.Split(streamResp.Content, " ")) // Rough approximation
				tokenInfo := &core.TokenInfo{
					PromptTokens:     streamResp.TokensEvaluated,
					CompletionTokens: tokenCount,
					TotalTokens:      streamResp.TokensEvaluated + tokenCount,
				}

				// Send the chunk
				chunkChan <- core.StreamChunk{
					Content: streamResp.Content,
					Usage:   tokenInfo,
				}
			}

			// Check if we've reached the end of the stream
			if streamResp.Stop {
				// Send completion signal
				chunkChan <- core.StreamChunk{Done: true}
				return
			}
		}

		// Handle scanner errors
		if err := scanner.Err(); err != nil {
			chunkChan <- core.StreamChunk{
				Error: errors.WithFields(
					errors.Wrap(err, errors.LLMGenerationFailed, "error reading stream"),
					errors.Fields{
						"model": o.ModelID(),
					}),
			}
			return
		}

		// If we get here without seeing a 'stop' flag, still signal completion
		chunkChan <- core.StreamChunk{Done: true}
	}()

	return response, nil
}
