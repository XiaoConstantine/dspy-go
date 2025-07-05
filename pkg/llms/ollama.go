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

// NewOllamaLLMFromConfig creates a new OllamaLLM instance from configuration.
func NewOllamaLLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*OllamaLLM, error) {
	// Extract model name from modelID (remove "ollama:" prefix if present)
	modelName := strings.TrimPrefix(string(modelID), "ollama:")

	if modelName == "" {
		return nil, errors.New(errors.InvalidInput, "model name is required")
	}

	// Default endpoint
	endpoint := "http://localhost:11434"
	if config.BaseURL != "" {
		endpoint = config.BaseURL
	}

	// Create endpoint configuration
	endpointCfg := &core.EndpointConfig{
		BaseURL: endpoint,
		Path:    "api/generate",
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		TimeoutSec: 10 * 60,
	}

	// Override with config endpoint if provided
	if config.Endpoint != nil {
		if config.Endpoint.BaseURL != "" {
			endpointCfg.BaseURL = config.Endpoint.BaseURL
		}
		if config.Endpoint.Path != "" {
			endpointCfg.Path = config.Endpoint.Path
		}
		if config.Endpoint.TimeoutSec > 0 {
			endpointCfg.TimeoutSec = config.Endpoint.TimeoutSec
		}
		for k, v := range config.Endpoint.Headers {
			endpointCfg.Headers[k] = v
		}
	}

	// Set capabilities
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
	}

	// Check if streaming is supported (most Ollama models support it)
	if supportsOllamaStreaming(modelName) {
		capabilities = append(capabilities, core.CapabilityStreaming)
	}

	// Check if embedding is supported
	if supportsOllamaEmbedding(modelName) {
		capabilities = append(capabilities, core.CapabilityEmbedding)
	}

	return &OllamaLLM{
		BaseLLM: core.NewBaseLLM("ollama", modelID, capabilities, endpointCfg),
	}, nil
}

// OllamaProviderFactory creates OllamaLLM instances.
func OllamaProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewOllamaLLMFromConfig(ctx, config, modelID)
}

// supportsOllamaStreaming checks if the model supports streaming.
func supportsOllamaStreaming(modelName string) bool {
	// Most Ollama models support streaming
	return true
}

// supportsOllamaEmbedding checks if the model supports embedding.
func supportsOllamaEmbedding(modelName string) bool {
	// Common embedding models in Ollama
	embeddingModels := []string{
		"nomic-embed-text",
		"mxbai-embed-large",
		"snowflake-arctic-embed",
		"all-minilm",
	}

	for _, embeddingModel := range embeddingModels {
		if strings.Contains(strings.ToLower(modelName), embeddingModel) {
			return true
		}
	}

	// Check for common embedding model patterns
	if strings.Contains(strings.ToLower(modelName), "embed") {
		return true
	}

	return false
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

// Define request/response structures for Ollama embeddings.
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
		return &core.LLMResponse{}, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to marshal request body"),
			errors.Fields{
				"model": o.ModelID(),
			})
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.GetEndpointConfig().BaseURL+"/api/generate", bytes.NewBuffer(jsonData))
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

	var ollamaResp ollamaResponse
	err = json.Unmarshal(body, &ollamaResp)
	if err != nil {
		return &core.LLMResponse{}, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal response"),
			errors.Fields{
				"resp":  body[:50],
				"model": o.ModelID(),
			})

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

func (o *OllamaLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	panic("Not implemented")
}

// CreateEmbedding generates embeddings for a single input.
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

// CreateEmbeddings generates embeddings for multiple inputs in batches.
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
	var errorIndex = -1

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
	if firstError != nil {
		return nil, firstError
	}

	// Return the combined results
	return &core.BatchEmbeddingResult{
		Embeddings: allResults,
		Error:      firstError,
		ErrorIndex: errorIndex,
	}, nil
}

// StreamGenerate for Ollama.
func (o *OllamaLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := ollamaRequest{
		Model:       o.ModelID(),
		Prompt:      prompt,
		Stream:      true,
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "failed to marshal request")
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, "POST",
		o.GetEndpointConfig().BaseURL+"/api/generate", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "failed to create request")
	}

	for key, value := range o.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	// Create channel and response
	chunkChan := make(chan core.StreamChunk)
	streamCtx, cancelStream := context.WithCancel(ctx)

	response := &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelStream,
	}

	go func() {
		defer close(chunkChan)

		resp, err := o.GetHTTPClient().Do(req)
		if err != nil {
			chunkChan <- core.StreamChunk{Error: err}
			return
		}
		defer resp.Body.Close()

		// Ollama returns JSONL stream
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			select {
			case <-streamCtx.Done():
				return
			default:
				line := scanner.Text()
				if line == "" {
					continue
				}

				var response struct {
					Response string `json:"response"`
					Done     bool   `json:"done"`
				}

				if err := json.Unmarshal([]byte(line), &response); err != nil {
					continue
				}

				// Send the chunk
				chunkChan <- core.StreamChunk{Content: response.Response}

				// Check if we're done
				if response.Done {
					chunkChan <- core.StreamChunk{Done: true}
					return
				}
			}
		}
	}()

	return response, nil
}
