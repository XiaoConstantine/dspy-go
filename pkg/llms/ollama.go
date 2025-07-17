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
	"github.com/XiaoConstantine/dspy-go/pkg/llms/openai"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// OllamaConfig holds configuration for Ollama provider.
type OllamaConfig struct {
	UseOpenAIAPI bool   `yaml:"use_openai_api" json:"use_openai_api"` // Default: true (modern Ollama)
	BaseURL      string `yaml:"base_url" json:"base_url"`             // Default: http://localhost:11434
	APIKey       string `yaml:"api_key" json:"api_key"`               // Optional for auth
	Timeout      int    `yaml:"timeout" json:"timeout"`               // Default: 60
}

// OllamaLLM implements the core.LLM interface for Ollama-hosted models with dual-mode support.
type OllamaLLM struct {
	*core.BaseLLM
	config    OllamaConfig
	nativeAPI *ollamaNativeAPI // For backward compatibility
}

// ollamaNativeAPI handles Ollama's native API calls.
type ollamaNativeAPI struct {
	baseURL string
	client  *http.Client
}

// Option pattern for flexible configuration.
type OllamaOption func(*OllamaConfig)

// WithNativeAPI configures Ollama to use native API mode.
func WithNativeAPI() OllamaOption {
	return func(c *OllamaConfig) { c.UseOpenAIAPI = false }
}

// WithOpenAIAPI configures Ollama to use OpenAI-compatible API mode.
func WithOpenAIAPI() OllamaOption {
	return func(c *OllamaConfig) { c.UseOpenAIAPI = true }
}

// WithBaseURL sets the base URL for Ollama.
func WithBaseURL(url string) OllamaOption {
	return func(c *OllamaConfig) { c.BaseURL = url }
}

// WithAuth sets authentication for Ollama (some deployments require it).
func WithAuth(apiKey string) OllamaOption {
	return func(c *OllamaConfig) { c.APIKey = apiKey }
}

// WithTimeout sets the timeout for requests.
func WithTimeout(timeout int) OllamaOption {
	return func(c *OllamaConfig) { c.Timeout = timeout }
}

// NewOllamaLLM creates a new OllamaLLM instance with modern defaults.
func NewOllamaLLM(modelID core.ModelID, options ...OllamaOption) (*OllamaLLM, error) {
	config := OllamaConfig{
		UseOpenAIAPI: true,  // Default to modern OpenAI-compatible mode
		BaseURL:     "http://localhost:11434",
		Timeout:     60,
	}
	
	// Apply options to override defaults
	for _, option := range options {
		option(&config)
	}
	
	return newOllamaLLMWithConfig(config, modelID)
}

// NewOllamaLLMFromConfig creates a new OllamaLLM instance from configuration.
func NewOllamaLLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*OllamaLLM, error) {
	ollamaConfig, err := parseOllamaConfig(config)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to parse Ollama configuration"),
			errors.Fields{"model": modelID})
	}
	
	return newOllamaLLMWithConfig(ollamaConfig, modelID)
}

// newOllamaLLMWithConfig creates OllamaLLM with the given configuration.
func newOllamaLLMWithConfig(config OllamaConfig, modelID core.ModelID) (*OllamaLLM, error) {
	// Extract model name from modelID (remove "ollama:" prefix if present)
	modelName := strings.TrimPrefix(string(modelID), "ollama:")
	if modelName == "" {
		return nil, errors.New(errors.InvalidInput, "model name is required")
	}

	// Set up endpoint configuration based on mode
	var endpointCfg *core.EndpointConfig
	if config.UseOpenAIAPI {
		// OpenAI-compatible mode
		headers := map[string]string{
			"Content-Type": "application/json",
		}
		if config.APIKey != "" {
			headers["Authorization"] = "Bearer " + config.APIKey
		}
		
		endpointCfg = &core.EndpointConfig{
			BaseURL:    config.BaseURL,
			Path:       "/v1/chat/completions",
			Headers:    headers,
			TimeoutSec: config.Timeout,
		}
	} else {
		// Native API mode
		endpointCfg = &core.EndpointConfig{
			BaseURL: config.BaseURL,
			Path:    "/api/generate",
			Headers: map[string]string{
				"Content-Type": "application/json",
			},
			TimeoutSec: config.Timeout,
		}
	}

	// Set capabilities based on model and mode
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
	}

	// Most Ollama models support streaming
	if supportsOllamaStreaming(modelName) {
		capabilities = append(capabilities, core.CapabilityStreaming)
	}

	// Embedding support - only in OpenAI mode or for specific embedding models
	if config.UseOpenAIAPI || supportsOllamaEmbedding(modelName) {
		capabilities = append(capabilities, core.CapabilityEmbedding)
	}

	// Create native API helper for backward compatibility
	nativeAPI := &ollamaNativeAPI{
		baseURL: config.BaseURL,
		client:  &http.Client{},
	}

	return &OllamaLLM{
		BaseLLM:   core.NewBaseLLM("ollama", core.ModelID(modelName), capabilities, endpointCfg),
		config:    config,
		nativeAPI: nativeAPI,
	}, nil
}

// parseOllamaConfig parses configuration supporting both legacy and modern formats.
func parseOllamaConfig(config core.ProviderConfig) (OllamaConfig, error) {
	result := OllamaConfig{
		UseOpenAIAPI: true, // Default to modern mode
		BaseURL:     "http://localhost:11434",
		Timeout:     60,
	}

	// Parse base URL
	if config.BaseURL != "" {
		result.BaseURL = config.BaseURL
	}
	if config.Endpoint != nil && config.Endpoint.BaseURL != "" {
		result.BaseURL = config.Endpoint.BaseURL
	}

	// Parse API key
	if config.APIKey != "" {
		result.APIKey = config.APIKey
	}

	// Parse timeout
	if config.Endpoint != nil && config.Endpoint.TimeoutSec > 0 {
		result.Timeout = config.Endpoint.TimeoutSec
	}

	// Parse mode setting from params
	if config.Params != nil {
		if useOpenAI, ok := config.Params["use_openai_api"].(bool); ok {
			result.UseOpenAIAPI = useOpenAI
		}
		if timeout, ok := config.Params["timeout"].(int); ok {
			result.Timeout = timeout
		}
	}

	return result, nil
}

// Generate implements the core.LLM interface with dual-mode support.
func (o *OllamaLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	if o.config.UseOpenAIAPI {
		return o.generateOpenAI(ctx, prompt, options...)
	}
	return o.generateNative(ctx, prompt, options...)
}

// generateOpenAI uses OpenAI-compatible API.
func (o *OllamaLLM) generateOpenAI(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Create OpenAI-compatible request
	req := &openai.ChatCompletionRequest{
		Model:    string(o.ModelID()),
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: prompt}},
		Stream:   false,
	}

	// Apply generate options
	o.applyGenerateOptions(req, opts)

	// Make HTTP request
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to marshal request"),
			errors.Fields{"model": o.ModelID()})
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", 
		o.GetEndpointConfig().BaseURL+"/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to create request"),
			errors.Fields{"model": o.ModelID()})
	}

	// Set headers
	for key, value := range o.GetEndpointConfig().Headers {
		httpReq.Header.Set(key, value)
	}

	resp, err := o.GetHTTPClient().Do(httpReq)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to send request"),
			errors.Fields{"model": o.ModelID()})
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to read response"),
			errors.Fields{"model": o.ModelID()})
	}

	if resp.StatusCode != http.StatusOK {
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("API request failed with status %d", resp.StatusCode)),
			errors.Fields{
				"model":         o.ModelID(),
				"status_code":   resp.StatusCode,
				"response_body": string(body),
			})
	}

	var openaiResp openai.ChatCompletionResponse
	if err := json.Unmarshal(body, &openaiResp); err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal response"),
			errors.Fields{
				"model": o.ModelID(),
				"body":  string(body[:min(len(body), 100)]),
			})
	}

	// Transform to LLMResponse
	if len(openaiResp.Choices) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no choices in response"),
			errors.Fields{"model": o.ModelID()})
	}

	return &core.LLMResponse{
		Content: openaiResp.Choices[0].Message.Content,
		Usage: &core.TokenInfo{
			PromptTokens:     openaiResp.Usage.PromptTokens,
			CompletionTokens: openaiResp.Usage.CompletionTokens,
			TotalTokens:      openaiResp.Usage.TotalTokens,
		},
		Metadata: map[string]interface{}{
			"model":  openaiResp.Model,
			"mode":   "openai",
		},
	}, nil
}

// generateNative uses Ollama's native API for backward compatibility.
func (o *OllamaLLM) generateNative(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
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
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to marshal request body"),
			errors.Fields{"model": o.ModelID()})
	}

	req, err := http.NewRequestWithContext(ctx, "POST", 
		o.GetEndpointConfig().BaseURL+"/api/generate", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to create request"),
			errors.Fields{"model": o.ModelID()})
	}

	for key, value := range o.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	resp, err := o.GetHTTPClient().Do(req)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to send request"),
			errors.Fields{"model": o.ModelID()})
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to read response body"),
			errors.Fields{"model": o.ModelID()})
	}

	if resp.StatusCode != http.StatusOK {
		return nil, errors.WithFields(
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
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal response"),
			errors.Fields{
				"model": o.ModelID(),
				"body":  string(body[:min(len(body), 50)]),
			})
	}

	return &core.LLMResponse{
		Content: ollamaResp.Response,
		Metadata: map[string]interface{}{
			"model": ollamaResp.Model,
			"mode":  "native",
		},
	}, nil
}

// StreamGenerate implements streaming with dual-mode support.
func (o *OllamaLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	if o.config.UseOpenAIAPI {
		return o.streamGenerateOpenAI(ctx, prompt, options...)
	}
	return o.streamGenerateNative(ctx, prompt, options...)
}

// streamGenerateOpenAI uses OpenAI-compatible streaming.
func (o *OllamaLLM) streamGenerateOpenAI(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	req := &openai.ChatCompletionRequest{
		Model:    string(o.ModelID()),
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: prompt}},
		Stream:   true,
	}

	o.applyGenerateOptions(req, opts)

	chunkChan := make(chan core.StreamChunk, 100)
	streamCtx, cancelStream := context.WithCancel(ctx)

	response := &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelStream,
	}

	go func() {
		defer close(chunkChan)
		defer cancelStream()

		// Make streaming request
		jsonData, err := json.Marshal(req)
		if err != nil {
			chunkChan <- core.StreamChunk{Error: err}
			return
		}

		httpReq, err := http.NewRequestWithContext(streamCtx, "POST",
			o.GetEndpointConfig().BaseURL+"/v1/chat/completions", bytes.NewBuffer(jsonData))
		if err != nil {
			chunkChan <- core.StreamChunk{Error: err}
			return
		}

		for key, value := range o.GetEndpointConfig().Headers {
			httpReq.Header.Set(key, value)
		}

		resp, err := o.GetHTTPClient().Do(httpReq)
		if err != nil {
			chunkChan <- core.StreamChunk{Error: err}
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			chunkChan <- core.StreamChunk{Error: fmt.Errorf("HTTP %d", resp.StatusCode)}
			return
		}

		// Parse SSE format
		o.parseOpenAIStreamResponse(resp.Body, chunkChan)
	}()

	return response, nil
}

// streamGenerateNative uses Ollama's native streaming.
func (o *OllamaLLM) streamGenerateNative(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
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

	req, err := http.NewRequestWithContext(ctx, "POST",
		o.GetEndpointConfig().BaseURL+"/api/generate", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "failed to create request")
	}

	for key, value := range o.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

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

				chunkChan <- core.StreamChunk{Content: response.Response}

				if response.Done {
					chunkChan <- core.StreamChunk{Done: true}
					return
				}
			}
		}
	}()

	return response, nil
}

// CreateEmbedding implements embedding generation with OpenAI-compatible mode support.
func (o *OllamaLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	if !o.config.UseOpenAIAPI && !supportsOllamaEmbedding(o.ModelID()) {
		return nil, errors.WithFields(
			errors.New(errors.UnsupportedOperation, "embeddings require OpenAI API mode or embedding model"),
			errors.Fields{
				"provider":       "ollama",
				"use_openai_api": o.config.UseOpenAIAPI,
				"model":          o.ModelID(),
			})
	}

	if o.config.UseOpenAIAPI {
		return o.createEmbeddingOpenAI(ctx, input, options...)
	}
	return o.createEmbeddingNative(ctx, input, options...)
}

// createEmbeddingOpenAI uses OpenAI-compatible embeddings API.
func (o *OllamaLLM) createEmbeddingOpenAI(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	req := &openai.EmbeddingRequest{
		Input: input,
		Model: string(o.ModelID()),
	}

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		o.GetEndpointConfig().BaseURL+"/v1/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	for key, value := range o.GetEndpointConfig().Headers {
		httpReq.Header.Set(key, value)
	}

	resp, err := o.GetHTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var embeddingResp openai.EmbeddingResponse
	if err := json.Unmarshal(body, &embeddingResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(embeddingResp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings in response")
	}

	// Convert float64 to float32
	embedding32 := make([]float32, len(embeddingResp.Data[0].Embedding))
	for i, v := range embeddingResp.Data[0].Embedding {
		embedding32[i] = float32(v)
	}

	return &core.EmbeddingResult{
		Vector:     embedding32,
		TokenCount: embeddingResp.Usage.TotalTokens,
		Metadata: map[string]interface{}{
			"model": embeddingResp.Model,
			"mode":  "openai",
		},
	}, nil
}

// createEmbeddingNative uses Ollama's native embedding API.
func (o *OllamaLLM) createEmbeddingNative(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := ollamaEmbeddingRequest{
		Model:   o.ModelID(),
		Prompt:  input,
		Options: opts.Params,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST",
		fmt.Sprintf("%s/api/embeddings", o.GetEndpointConfig().BaseURL), bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	for key, value := range o.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	resp, err := o.GetHTTPClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(body))
	}

	var ollamaResp ollamaEmbeddingResponse
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &core.EmbeddingResult{
		Vector:     ollamaResp.Embedding,
		TokenCount: ollamaResp.Usage.Tokens,
		Metadata: map[string]interface{}{
			"model":          o.ModelID(),
			"embedding_size": ollamaResp.Size,
			"mode":           "native",
		},
	}, nil
}

// GenerateWithJSON implements JSON mode generation.
func (o *OllamaLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	response, err := o.Generate(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	return utils.ParseJSONResponse(response.Content)
}

// GenerateWithFunctions is not yet implemented for Ollama.
func (o *OllamaLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, errors.WithFields(
		errors.New(errors.UnsupportedOperation, "function calling not yet implemented for Ollama"),
		errors.Fields{
			"provider": "ollama",
			"model":    o.ModelID(),
		})
}

// GenerateWithContent implements multimodal content generation.
func (o *OllamaLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	// For now, extract text content and fall back to text generation
	// Future versions could support vision models in Ollama
	var textContent string
	for _, block := range content {
		if block.Type == core.FieldTypeText {
			textContent += block.Text + "\n"
		}
	}
	
	if textContent == "" {
		return nil, errors.WithFields(
			errors.New(errors.UnsupportedOperation, "multimodal content not yet supported for Ollama"),
			errors.Fields{
				"provider": "ollama",
				"model":    o.ModelID(),
			})
	}
	
	return o.Generate(ctx, strings.TrimSpace(textContent), options...)
}

// StreamGenerateWithContent implements multimodal streaming content generation.
func (o *OllamaLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	// For now, extract text content and fall back to text streaming
	var textContent string
	for _, block := range content {
		if block.Type == core.FieldTypeText {
			textContent += block.Text + "\n"
		}
	}
	
	if textContent == "" {
		return nil, errors.WithFields(
			errors.New(errors.UnsupportedOperation, "multimodal streaming not yet supported for Ollama"),
			errors.Fields{
				"provider": "ollama",
				"model":    o.ModelID(),
			})
	}
	
	return o.StreamGenerate(ctx, strings.TrimSpace(textContent), options...)
}

// CreateEmbeddings generates embeddings for multiple inputs.
func (o *OllamaLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	if opts.BatchSize <= 0 {
		opts.BatchSize = 32
	}

	var allResults []core.EmbeddingResult
	var firstError error
	var errorIndex = -1

	for i := 0; i < len(inputs); i += opts.BatchSize {
		end := i + opts.BatchSize
		if end > len(inputs) {
			end = len(inputs)
		}

		batch := inputs[i:end]
		for j, input := range batch {
			result, err := o.CreateEmbedding(ctx, input, options...)
			if err != nil {
				if firstError == nil {
					firstError = err
					errorIndex = i + j
				}
				continue
			}
			
			// Add batch index to metadata
			if result.Metadata == nil {
				result.Metadata = make(map[string]interface{})
			}
			result.Metadata["batch_index"] = i + j
			
			allResults = append(allResults, *result)
		}
		
		if firstError != nil {
			break
		}
	}

	return &core.BatchEmbeddingResult{
		Embeddings: allResults,
		Error:      firstError,
		ErrorIndex: errorIndex,
	}, firstError
}

// Helper functions

// applyGenerateOptions applies generation options to OpenAI request.
func (o *OllamaLLM) applyGenerateOptions(req *openai.ChatCompletionRequest, opts *core.GenerateOptions) {
	if opts.MaxTokens > 0 {
		req.MaxTokens = &opts.MaxTokens
	}
	if opts.Temperature >= 0 {
		req.Temperature = &opts.Temperature
	}
	if opts.TopP > 0 {
		req.TopP = &opts.TopP
	}
	if opts.FrequencyPenalty != 0 {
		req.FrequencyPenalty = &opts.FrequencyPenalty
	}
	if opts.PresencePenalty != 0 {
		req.PresencePenalty = &opts.PresencePenalty
	}
	if len(opts.Stop) > 0 {
		req.Stop = opts.Stop
	}
}

// parseOpenAIStreamResponse parses OpenAI SSE format.
func (o *OllamaLLM) parseOpenAIStreamResponse(body io.Reader, chunkChan chan<- core.StreamChunk) {
	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")

			if data == "[DONE]" {
				chunkChan <- core.StreamChunk{Done: true}
				return
			}

			var streamResp openai.ChatCompletionStreamResponse
			if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
				continue
			}

			if len(streamResp.Choices) > 0 && streamResp.Choices[0].Delta.Content != "" {
				chunkChan <- core.StreamChunk{
					Content: streamResp.Choices[0].Delta.Content,
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		chunkChan <- core.StreamChunk{Error: err}
	}
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

	modelLower := strings.ToLower(modelName)
	for _, embeddingModel := range embeddingModels {
		if strings.Contains(modelLower, embeddingModel) {
			return true
		}
	}

	// Check for common embedding model patterns
	if strings.Contains(modelLower, "embed") {
		return true
	}

	return false
}

// OllamaProviderFactory creates OllamaLLM instances.
func OllamaProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewOllamaLLMFromConfig(ctx, config, modelID)
}

// min returns the smaller of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Legacy types for backward compatibility (native API mode)

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

type ollamaEmbeddingRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	Options map[string]interface{} `json:"options"`
}

type ollamaEmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
	Size      int       `json:"size"`
	Usage     struct {
		Tokens int `json:"tokens"`
	} `json:"usage"`
}

// ollamaBatchEmbeddingRequest is reserved for future native batch embedding support.
// Currently unused but kept for potential future Ollama batch API support.
//
//nolint:unused // Reserved for future use
type ollamaBatchEmbeddingRequest struct {
	Model   string                 `json:"model"`
	Prompts []string               `json:"prompts"`
	Options map[string]interface{} `json:"options"`
}

// ollamaBatchEmbeddingResponse is reserved for future native batch embedding support.
// Currently unused but kept for potential future Ollama batch API support.
//
//nolint:unused // Reserved for future use
type ollamaBatchEmbeddingResponse struct {
	Embeddings []ollamaEmbeddingResponse `json:"embeddings"`
}