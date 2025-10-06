package llms

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/llms/openai"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// OpenAILLM implements the core.LLM interface for OpenAI's models.
type OpenAILLM struct {
	*core.BaseLLM
	apiKey string
}

// OpenAIOption is a functional option for configuring OpenAI provider.
type OpenAIOption func(*OpenAIConfig)

// OpenAIConfig holds configuration for OpenAI provider.
type OpenAIConfig struct {
	baseURL    string
	path       string
	apiKey     string
	headers    map[string]string
	timeout    time.Duration
	httpClient *http.Client
}

// NewOpenAILLM creates a new OpenAILLM instance with functional options.
func NewOpenAILLM(modelID core.ModelID, opts ...OpenAIOption) (*OpenAILLM, error) {
	config := &OpenAIConfig{
		baseURL: "https://api.openai.com", // default
		path:    "/v1/chat/completions",
		timeout: 60 * time.Second,
		headers: make(map[string]string),
	}

	for _, opt := range opts {
		opt(config)
	}

	// Environment variable fallback for API key
	if config.apiKey == "" {
		config.apiKey = os.Getenv("OPENAI_API_KEY")
	}

	// API key validation - required for official OpenAI API endpoint
	if config.apiKey == "" && config.baseURL == "https://api.openai.com" {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "OpenAI API key is required for api.openai.com"),
			errors.Fields{"env_var": "OPENAI_API_KEY"})
	}

	// Build endpoint configuration
	endpointCfg := &core.EndpointConfig{
		BaseURL:    config.baseURL,
		Path:       config.path,
		Headers:    config.headers,
		TimeoutSec: int(config.timeout.Seconds()),
	}

	// Set authorization header only if API key is provided
	if config.apiKey != "" {
		endpointCfg.Headers["Authorization"] = "Bearer " + config.apiKey
	}
	endpointCfg.Headers["Content-Type"] = "application/json"

	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
		core.CapabilityStreaming,
		core.CapabilityEmbedding,
	}

	baseLLM := core.NewBaseLLM("openai", modelID, capabilities, endpointCfg)

	return &OpenAILLM{
		BaseLLM: baseLLM,
		apiKey:  config.apiKey,
	}, nil
}

// Option functions for OpenAI configuration

// WithAPIKey sets the API key.
func WithAPIKey(apiKey string) OpenAIOption {
	return func(c *OpenAIConfig) { c.apiKey = apiKey }
}

// WithOpenAIBaseURL sets the base URL.
func WithOpenAIBaseURL(baseURL string) OpenAIOption {
	return func(c *OpenAIConfig) { c.baseURL = baseURL }
}

// WithOpenAIPath sets the endpoint path.
func WithOpenAIPath(path string) OpenAIOption {
	return func(c *OpenAIConfig) { c.path = path }
}

// WithOpenAITimeout sets the request timeout.
func WithOpenAITimeout(timeout time.Duration) OpenAIOption {
	return func(c *OpenAIConfig) { c.timeout = timeout }
}

// WithHeader sets a custom header.
func WithHeader(key, value string) OpenAIOption {
	return func(c *OpenAIConfig) {
		if c.headers == nil {
			c.headers = make(map[string]string)
		}
		c.headers[key] = value
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) OpenAIOption {
	return func(c *OpenAIConfig) { c.httpClient = client }
}

// Convenience constructor for standard OpenAI.
func NewOpenAI(modelID core.ModelID, apiKey string) (*OpenAILLM, error) {
	if apiKey == "" {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "OpenAI API key is required"),
			errors.Fields{"env_var": "OPENAI_API_KEY"})
	}
	return NewOpenAILLM(modelID, WithAPIKey(apiKey))
}

// NewOpenAILLMFromConfig creates a new OpenAILLM instance from configuration.
func NewOpenAILLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*OpenAILLM, error) {
	opts := []OpenAIOption{}

	// Parse configuration into options
	if config.APIKey != "" {
		opts = append(opts, WithAPIKey(config.APIKey))
	}

	// Set base URL from config.BaseURL or config.Endpoint.BaseURL (endpoint takes priority)
	baseURL := config.BaseURL
	if config.Endpoint != nil && config.Endpoint.BaseURL != "" {
		baseURL = config.Endpoint.BaseURL
	}
	if baseURL != "" {
		opts = append(opts, WithOpenAIBaseURL(baseURL))
	}

	// Validate model ID only for the official OpenAI API endpoint to allow custom models with compatible APIs.
	// An empty baseURL defaults to the official OpenAI API.
	if (baseURL == "" || baseURL == "https://api.openai.com") && !isValidOpenAIModel(modelID) {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "unsupported model for official OpenAI API"),
			errors.Fields{"model": modelID})
	}

	if config.Endpoint != nil && config.Endpoint.Path != "" {
		opts = append(opts, WithOpenAIPath(config.Endpoint.Path))
	}

	if config.Endpoint != nil && config.Endpoint.TimeoutSec > 0 {
		opts = append(opts, WithOpenAITimeout(time.Duration(config.Endpoint.TimeoutSec)*time.Second))
	}

	if config.Endpoint != nil {
		for key, value := range config.Endpoint.Headers {
			opts = append(opts, WithHeader(key, value))
		}
	}

	return NewOpenAILLM(modelID, opts...)
}

// OpenAIProviderFactory creates OpenAILLM instances.
func OpenAIProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewOpenAILLMFromConfig(ctx, config, modelID)
}

// isValidOpenAIModel checks if the model is a valid OpenAI model.
func isValidOpenAIModel(modelID core.ModelID) bool {
	validModels := []core.ModelID{
		core.ModelOpenAIGPT4,
		core.ModelOpenAIGPT4Turbo,
		core.ModelOpenAIGPT35Turbo,
		core.ModelOpenAIGPT4o,
		core.ModelOpenAIGPT4oMini,
		core.ModelOpenAIGPT5,
		core.ModelOpenAIGPT5Mini,
		core.ModelOpenAIGPT5Nano,
	}

	for _, validModel := range validModels {
		if modelID == validModel {
			return true
		}
	}
	return false
}

// Generate implements the core.LLM interface.
func (o *OpenAILLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	request := &openai.ChatCompletionRequest{
		Model: o.ModelID(),
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens:   &opts.MaxTokens,
		Temperature: &opts.Temperature,
	}

	if opts.TopP > 0 {
		request.TopP = &opts.TopP
	}
	if opts.FrequencyPenalty != 0 {
		request.FrequencyPenalty = &opts.FrequencyPenalty
	}
	if opts.PresencePenalty != 0 {
		request.PresencePenalty = &opts.PresencePenalty
	}
	if len(opts.Stop) > 0 {
		request.Stop = opts.Stop
	}

	response, err := o.makeRequest(ctx, request)
	if err != nil {
		return nil, err
	}

	if len(response.Choices) == 0 {
		return nil, errors.New(errors.InvalidResponse, "no choices returned from OpenAI API")
	}

	usage := &core.TokenInfo{
		PromptTokens:     response.Usage.PromptTokens,
		CompletionTokens: response.Usage.CompletionTokens,
		TotalTokens:      response.Usage.TotalTokens,
	}

	return &core.LLMResponse{
		Content: response.Choices[0].Message.Content,
		Usage:   usage,
		Metadata: map[string]interface{}{
			"finish_reason": response.Choices[0].FinishReason,
			"id":            response.ID,
			"model":         response.Model,
		},
	}, nil
}

// GenerateWithJSON implements the core.LLM interface.
func (o *OpenAILLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	request := &openai.ChatCompletionRequest{
		Model: o.ModelID(),
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens:   &opts.MaxTokens,
		Temperature: &opts.Temperature,
		ResponseFormat: &openai.ResponseFormat{
			Type: "json_object",
		},
	}

	if opts.TopP > 0 {
		request.TopP = &opts.TopP
	}
	if opts.FrequencyPenalty != 0 {
		request.FrequencyPenalty = &opts.FrequencyPenalty
	}
	if opts.PresencePenalty != 0 {
		request.PresencePenalty = &opts.PresencePenalty
	}
	if len(opts.Stop) > 0 {
		request.Stop = opts.Stop
	}

	response, err := o.makeRequest(ctx, request)
	if err != nil {
		return nil, err
	}

	if len(response.Choices) == 0 {
		return nil, errors.New(errors.InvalidResponse, "no choices returned from OpenAI API")
	}

	return utils.ParseJSONResponse(response.Choices[0].Message.Content)
}

// GenerateWithFunctions implements the core.LLM interface.
func (o *OpenAILLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, errors.New(errors.UnsupportedOperation, "function calling not yet implemented for OpenAI provider")
}

// CreateEmbedding implements the core.LLM interface.
func (o *OpenAILLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	model := "text-embedding-3-small"
	if opts.Model != "" {
		model = opts.Model
	}

	request := &openai.EmbeddingRequest{
		Input:          input,
		Model:          model,
		EncodingFormat: "float",
	}

	response, err := o.makeEmbeddingRequest(ctx, request)
	if err != nil {
		return nil, err
	}

	if len(response.Data) == 0 {
		return nil, errors.New(errors.InvalidResponse, "no embeddings returned from OpenAI API")
	}

	// Convert []float64 to []float32
	embedding := make([]float32, len(response.Data[0].Embedding))
	for i, v := range response.Data[0].Embedding {
		embedding[i] = float32(v)
	}

	return &core.EmbeddingResult{
		Vector:     embedding,
		TokenCount: response.Usage.TotalTokens,
		Metadata: map[string]interface{}{
			"model": response.Model,
		},
	}, nil
}

// CreateEmbeddings implements the core.LLM interface.
func (o *OpenAILLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	model := "text-embedding-3-small"
	if opts.Model != "" {
		model = opts.Model
	}

	request := &openai.EmbeddingRequest{
		Input:          inputs,
		Model:          model,
		EncodingFormat: "float",
	}

	response, err := o.makeEmbeddingRequest(ctx, request)
	if err != nil {
		return &core.BatchEmbeddingResult{Error: err}, nil
	}

	results := make([]core.EmbeddingResult, len(response.Data))
	for i, data := range response.Data {
		// Convert []float64 to []float32
		embedding := make([]float32, len(data.Embedding))
		for j, v := range data.Embedding {
			embedding[j] = float32(v)
		}

		results[i] = core.EmbeddingResult{
			Vector:     embedding,
			TokenCount: response.Usage.TotalTokens / len(inputs), // Approximate per input
			Metadata: map[string]interface{}{
				"model": response.Model,
				"index": data.Index,
			},
		}
	}

	return &core.BatchEmbeddingResult{
		Embeddings: results,
	}, nil
}

// StreamGenerate implements the core.LLM interface.
func (o *OpenAILLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	logger := logging.GetLogger()
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	request := &openai.ChatCompletionRequest{
		Model: o.ModelID(),
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens:   &opts.MaxTokens,
		Temperature: &opts.Temperature,
		Stream:      true,
	}

	if opts.TopP > 0 {
		request.TopP = &opts.TopP
	}
	if opts.FrequencyPenalty != 0 {
		request.FrequencyPenalty = &opts.FrequencyPenalty
	}
	if opts.PresencePenalty != 0 {
		request.PresencePenalty = &opts.PresencePenalty
	}
	if len(opts.Stop) > 0 {
		request.Stop = opts.Stop
	}

	// Create a channel for the stream chunks
	chunkChan := make(chan core.StreamChunk)

	// Create a cancellable context
	streamCtx, cancelFunc := context.WithCancel(ctx)

	// Start a goroutine to handle the streaming request
	go func() {
		defer close(chunkChan)
		defer cancelFunc()

		resp, err := o.makeStreamingRequest(streamCtx, request)
		if err != nil {
			chunkChan <- core.StreamChunk{
				Error: errors.Wrap(err, errors.LLMGenerationFailed, "streaming request failed"),
			}
			return
		}
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			select {
			case <-streamCtx.Done():
				return
			default:
			}

			line := scanner.Text()
			line = strings.TrimSpace(line)

			if line == "" {
				continue
			}

			// Skip lines that don't start with "data: "
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			// Remove "data: " prefix
			data := strings.TrimPrefix(line, "data: ")

			// Check for end marker
			if data == "[DONE]" {
				chunkChan <- core.StreamChunk{Done: true}
				return
			}

			// Parse the JSON response
			var streamResponse openai.ChatCompletionStreamResponse
			if err := json.Unmarshal([]byte(data), &streamResponse); err != nil {
				logger.Debug(ctx, "Error parsing stream chunk: %v", err)
				continue
			}

			// Process the response
			if len(streamResponse.Choices) > 0 {
				choice := streamResponse.Choices[0]
				content := choice.Delta.Content

				if content != "" {
					chunkChan <- core.StreamChunk{Content: content}
				}

				// Check for finish reason
				if choice.FinishReason != nil && *choice.FinishReason != "" {
					chunkChan <- core.StreamChunk{Done: true}
					return
				}
			}
		}

		if err := scanner.Err(); err != nil {
			chunkChan <- core.StreamChunk{
				Error: errors.Wrap(err, errors.LLMGenerationFailed, "error reading stream"),
			}
		}
	}()

	return &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelFunc,
	}, nil
}

// makeRequest sends a chat completion request to the OpenAI API.
func (o *OpenAILLM) makeRequest(ctx context.Context, request *openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "failed to marshal request")
	}

	endpoint := o.GetEndpointConfig()
	url := endpoint.BaseURL + endpoint.Path

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, errors.Wrap(err, errors.Unknown, "failed to create request")
	}

	// Set headers
	for key, value := range endpoint.Headers {
		req.Header.Set(key, value)
	}

	resp, err := o.GetHTTPClient().Do(req)
	if err != nil {
		return nil, errors.Wrap(err, errors.LLMGenerationFailed, "request failed")
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, errors.LLMGenerationFailed, "failed to read response")
	}

	if resp.StatusCode != http.StatusOK {
		var errorResp openai.ErrorResponse
		if err := json.Unmarshal(body, &errorResp); err != nil {
			return nil, errors.WithFields(
				errors.New(errors.LLMGenerationFailed, "API request failed"),
				errors.Fields{"status": resp.StatusCode, "body": string(body)})
		}
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, errorResp.Error.Message),
			errors.Fields{"type": errorResp.Error.Type, "code": errorResp.Error.Code})
	}

	var response openai.ChatCompletionResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, errors.Wrap(err, errors.InvalidResponse, "failed to parse response")
	}

	return &response, nil
}

// makeEmbeddingRequest sends an embedding request to the OpenAI API.
func (o *OpenAILLM) makeEmbeddingRequest(ctx context.Context, request *openai.EmbeddingRequest) (*openai.EmbeddingResponse, error) {
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "failed to marshal request")
	}

	endpoint := o.GetEndpointConfig()
	url := endpoint.BaseURL + "/v1/embeddings"

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, errors.Wrap(err, errors.Unknown, "failed to create request")
	}

	// Set headers
	for key, value := range endpoint.Headers {
		req.Header.Set(key, value)
	}

	resp, err := o.GetHTTPClient().Do(req)
	if err != nil {
		return nil, errors.Wrap(err, errors.LLMGenerationFailed, "request failed")
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, errors.LLMGenerationFailed, "failed to read response")
	}

	if resp.StatusCode != http.StatusOK {
		var errorResp openai.ErrorResponse
		if err := json.Unmarshal(body, &errorResp); err != nil {
			return nil, errors.WithFields(
				errors.New(errors.LLMGenerationFailed, "API request failed"),
				errors.Fields{"status": resp.StatusCode, "body": string(body)})
		}
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, errorResp.Error.Message),
			errors.Fields{"type": errorResp.Error.Type, "code": errorResp.Error.Code})
	}

	var response openai.EmbeddingResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, errors.Wrap(err, errors.InvalidResponse, "failed to parse response")
	}

	return &response, nil
}

// makeStreamingRequest sends a streaming chat completion request to the OpenAI API.
func (o *OpenAILLM) makeStreamingRequest(ctx context.Context, request *openai.ChatCompletionRequest) (*http.Response, error) {
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "failed to marshal request")
	}

	endpoint := o.GetEndpointConfig()
	url := endpoint.BaseURL + endpoint.Path

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, errors.Wrap(err, errors.Unknown, "failed to create request")
	}

	// Set headers
	for key, value := range endpoint.Headers {
		req.Header.Set(key, value)
	}

	resp, err := o.GetHTTPClient().Do(req)
	if err != nil {
		return nil, errors.Wrap(err, errors.LLMGenerationFailed, "request failed")
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		var errorResp openai.ErrorResponse
		if err := json.Unmarshal(body, &errorResp); err != nil {
			return nil, errors.WithFields(
				errors.New(errors.LLMGenerationFailed, "API request failed"),
				errors.Fields{"status": resp.StatusCode, "body": string(body)})
		}
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, errorResp.Error.Message),
			errors.Fields{"type": errorResp.Error.Type, "code": errorResp.Error.Code})
	}

	return resp, nil
}
