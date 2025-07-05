package llms

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// GeminiLLM implements the core.LLM interface for Google's Gemini model.
type GeminiLLM struct {
	*core.BaseLLM
	apiKey string
}

// GeminiRequest represents the request structure for Gemini API.
type geminiRequest struct {
	Contents         []geminiContent        `json:"contents"`
	GenerationConfig geminiGenerationConfig `json:"generationConfig,omitempty"`
}

// Add this to your existing geminiRequest struct or create a new one for function calling.
type geminiRequestWithFunction struct {
	Contents         []geminiContent        `json:"contents"`
	Tools            []geminiTool           `json:"tools,omitempty"`
	GenerationConfig geminiGenerationConfig `json:"generationConfig,omitempty"`
}

// Add these new types to support function calling.
type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"function_declarations"`
}

type geminiFunctionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters"`
}
type geminiContent struct {
	Parts []geminiPart `json:"parts"`
	Role  string       `json:"role,omitempty"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiGenerationConfig struct {
	Temperature     float64 `json:"temperature,omitempty"`
	MaxOutputTokens int     `json:"maxOutputTokens,omitempty"`
	TopP            float64 `json:"topP,omitempty"`
}

// GeminiResponse represents the response structure from Gemini API.
type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

type geminiFunctionResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text         string              `json:"text,omitempty"`
				FunctionCall *geminiFunctionCall `json:"function_call,omitempty"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}
type geminiFunctionCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// Request and response structures for Gemini embeddings.
type geminiEmbeddingRequest struct {
	Model   string `json:"model"`
	Content struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"content"`
	// Task type helps the model generate appropriate embeddings
	TaskType string `json:"taskType,omitempty"`
	// Additional configuration
	Title      string                 `json:"title,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type geminiBatchEmbeddingRequest struct {
	Model    string `json:"model"`
	Requests []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"requests"`
	TaskType   string                 `json:"taskType,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type geminiEmbeddingResponse struct {
	Embedding struct {
		Values []float32 `json:"values"`
		// Statistics about the generated embedding
		Statistics struct {
			TruncatedInputTokenCount int `json:"truncatedInputTokenCount"`
			TokenCount               int `json:"tokenCount"`
		} `json:"statistics"`
	} `json:"embedding"`
	// Usage information
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

type geminiBatchEmbeddingResponse struct {
	Embeddings []geminiEmbeddingResponse `json:"embeddings"`
}

// NewGeminiLLM creates a new GeminiLLM instance.
func NewGeminiLLM(apiKey string, model core.ModelID) (*GeminiLLM, error) {
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY") // or whatever env var name you prefer
		if apiKey == "" {
			return nil, errors.New(errors.InvalidInput, "API key is required")
		}

	}

	if model == "" {
		model = core.ModelGoogleGeminiFlash // Default model
	}
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
		core.CapabilityEmbedding,
	}
	// Validate model ID
	switch model {
	case core.ModelGoogleGeminiPro, core.ModelGoogleGeminiFlash, core.ModelGoogleGeminiFlashThinking, core.ModelGoogleGeminiFlashLite:
		break
	default:
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, fmt.Sprintf("unsupported Gemini model: %s", model)),
			errors.Fields{"model": model})
	}
	endpoint := &core.EndpointConfig{
		BaseURL: "https://generativelanguage.googleapis.com/v1beta",
		Path:    fmt.Sprintf("/models/%s:generateContent", model),
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		TimeoutSec: 10 * 60,
	}

	return &GeminiLLM{
		apiKey:  apiKey,
		BaseLLM: core.NewBaseLLM("google", model, capabilities, endpoint),
	}, nil
}

// NewGeminiLLMFromConfig creates a new GeminiLLM instance from configuration.
func NewGeminiLLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*GeminiLLM, error) {
	// Get API key from config or environment
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New(errors.InvalidInput, "API key is required")
	}

	// Use default model if none specified
	if modelID == "" {
		modelID = core.ModelGoogleGeminiFlash
	}

	// Validate model ID
	if !isValidGeminiModel(modelID) {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "unsupported Gemini model"),
			errors.Fields{"model": modelID})
	}

	// Create endpoint configuration
	baseURL := "https://generativelanguage.googleapis.com/v1beta"
	if config.BaseURL != "" {
		baseURL = config.BaseURL
	}

	endpoint := &core.EndpointConfig{
		BaseURL: baseURL,
		Path:    fmt.Sprintf("/models/%s:generateContent", modelID),
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		TimeoutSec: 10 * 60,
	}

	// Override with config endpoint if provided
	if config.Endpoint != nil {
		if config.Endpoint.BaseURL != "" {
			endpoint.BaseURL = config.Endpoint.BaseURL
		}
		if config.Endpoint.TimeoutSec > 0 {
			endpoint.TimeoutSec = config.Endpoint.TimeoutSec
		}
		for k, v := range config.Endpoint.Headers {
			endpoint.Headers[k] = v
		}
	}

	// Set capabilities
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
		core.CapabilityEmbedding,
	}

	// Check if streaming is supported
	if supportsGeminiStreaming(modelID) {
		capabilities = append(capabilities, core.CapabilityStreaming)
	}

	// Check if function calling is supported
	if supportsGeminiFunctionCalling(modelID) {
		capabilities = append(capabilities, core.CapabilityToolCalling)
	}

	return &GeminiLLM{
		apiKey:  apiKey,
		BaseLLM: core.NewBaseLLM("google", modelID, capabilities, endpoint),
	}, nil
}

// GeminiProviderFactory creates GeminiLLM instances.
func GeminiProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewGeminiLLMFromConfig(ctx, config, modelID)
}

// isValidGeminiModel checks if the model is a valid Gemini model.
func isValidGeminiModel(modelID core.ModelID) bool {
	validModels := []core.ModelID{
		core.ModelGoogleGeminiPro,
		core.ModelGoogleGeminiFlash,
		core.ModelGoogleGeminiFlashThinking,
		core.ModelGoogleGeminiFlashLite,
	}

	for _, validModel := range validModels {
		if modelID == validModel {
			return true
		}
	}
	return false
}

// supportsGeminiStreaming checks if the model supports streaming.
func supportsGeminiStreaming(modelID core.ModelID) bool {
	// Most Gemini models support streaming
	return true
}

// supportsGeminiFunctionCalling checks if the model supports function calling.
func supportsGeminiFunctionCalling(modelID core.ModelID) bool {
	// Most Gemini models support function calling
	return true
}

// Generate implements the core.LLM interface.
func (g *GeminiLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{Text: prompt},
				},
			},
		},
		GenerationConfig: geminiGenerationConfig{
			Temperature:     opts.Temperature,
			MaxOutputTokens: opts.MaxTokens,
			TopP:            opts.TopP,
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to marshal request body"),
			errors.Fields{
				"prompt": prompt,
				"model":  g.ModelID(),
			})
	}

	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		constructRequestURL(g.GetEndpointConfig(), g.apiKey),
		bytes.NewBuffer(jsonData),
	)

	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to create request"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}
	// TODO: make basellm make request to dry this up
	for key, value := range g.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	resp, err := g.GetHTTPClient().Do(req)
	if err != nil {
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("LLMGenerationFailed: failed to send request: %v", err)),
			errors.Fields{
				"model": g.ModelID(),
			})
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("LLMGenerationFailed: failed to read response body: %v", err)),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	if resp.StatusCode != http.StatusOK {
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("LLMGenerationFailed: API request failed with status code %d: %s", resp.StatusCode, string(body))),
			errors.Fields{
				"model":      g.ModelID(),
				"statusCode": resp.StatusCode,
			})
	}

	var geminiResp geminiResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, fmt.Sprintf("InvalidResponse: failed to unmarshal response: %v", err)),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	if len(geminiResp.Candidates) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "InvalidResponse: no candidates in response"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	content := geminiResp.Candidates[0].Content.Parts[0].Text
	usage := &core.TokenInfo{
		PromptTokens:     geminiResp.UsageMetadata.PromptTokenCount,
		CompletionTokens: geminiResp.UsageMetadata.CandidatesTokenCount,
		TotalTokens:      geminiResp.UsageMetadata.TotalTokenCount,
	}

	return &core.LLMResponse{
		Content: content,
		Usage:   usage,
	}, nil
}

// GenerateWithJSON implements the core.LLM interface.
func (g *GeminiLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	response, err := g.Generate(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	return utils.ParseJSONResponse(response.Content)
}

// Implement the GenerateWithFunctions method for GeminiLLM.
func (g *GeminiLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Convert the generic function schemas to Gemini's format
	functionDeclarations := make([]geminiFunctionDeclaration, 0, len(functions))
	for _, function := range functions {
		name, ok := function["name"].(string)
		if !ok {
			return nil, errors.WithFields(
				errors.New(errors.InvalidInput, "function schema missing 'name' field"),
				errors.Fields{
					"function": function,
				})
		}

		description := ""
		if desc, ok := function["description"].(string); ok {
			description = desc
		}

		parameters, ok := function["parameters"].(map[string]interface{})
		if !ok {
			return nil, errors.WithFields(
				errors.New(errors.InvalidInput, "function schema missing 'parameters' field"),
				errors.Fields{
					"function": function,
				})
		}

		functionDeclarations = append(functionDeclarations, geminiFunctionDeclaration{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		})
	}

	// Create the request body with functions
	reqBody := geminiRequestWithFunction{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{Text: prompt},
				},
				Role: "user",
			},
		},
		Tools: []geminiTool{
			{
				FunctionDeclarations: functionDeclarations,
			},
		},
		GenerationConfig: geminiGenerationConfig{
			Temperature:     opts.Temperature,
			MaxOutputTokens: opts.MaxTokens,
			TopP:            opts.TopP,
		},
	}
	requestJSON, _ := json.MarshalIndent(reqBody, "", "  ")
	logger.Debug(ctx, "Function call request JSON: %s", string(requestJSON))

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to marshal request body"),
			errors.Fields{
				"prompt": prompt,
				"model":  g.ModelID(),
			})
	}

	// Create the request URL
	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		constructRequestURL(g.GetEndpointConfig(), g.apiKey),
		bytes.NewBuffer(jsonData),
	)

	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to create request"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	// Set headers
	for key, value := range g.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	// Send the request
	resp, err := g.GetHTTPClient().Do(req)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to send request"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}
	defer resp.Body.Close()

	// Read and process the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to read response body"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	if resp.StatusCode != http.StatusOK {
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("API request failed with status code %d: %s", resp.StatusCode, string(body))),
			errors.Fields{
				"model":      g.ModelID(),
				"statusCode": resp.StatusCode,
			})
	}

	logger.Debug(ctx, "Raw Gemini response: %s", string(body))

	// Parse the response
	var geminiResp geminiFunctionResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal response"),
			errors.Fields{
				"model": g.ModelID(),
				"body":  string(body),
			})
	}

	if len(geminiResp.Candidates) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no candidates in response"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	// Extract usage information
	usage := &core.TokenInfo{
		PromptTokens:     geminiResp.UsageMetadata.PromptTokenCount,
		CompletionTokens: geminiResp.UsageMetadata.CandidatesTokenCount,
		TotalTokens:      geminiResp.UsageMetadata.TotalTokenCount,
	}

	// Process the response to extract function call if present
	result := make(map[string]interface{})

	// Check if there are any parts in the response
	if len(geminiResp.Candidates[0].Content.Parts) > 0 {
		// Process all parts
		var textContent string
		var functionCall *geminiFunctionCall

		for _, part := range geminiResp.Candidates[0].Content.Parts {
			// Collect text content
			if part.Text != "" {
				if textContent != "" {
					textContent += " "
				}
				textContent += part.Text
			}

			// Get function call if present
			if part.FunctionCall != nil {
				functionCall = part.FunctionCall
			}
		}

		// Add text content if available
		if textContent != "" {
			result["content"] = textContent
		}

		// Add function call if available
		if functionCall != nil {
			result["function_call"] = map[string]interface{}{
				"name":      functionCall.Name,
				"arguments": functionCall.Arguments,
			}
		}
	}

	// If no content or function call was found, add a default message
	if len(result) == 0 {
		result["content"] = "No content or function call received from model"
	}

	// Add token usage information
	result["_usage"] = usage

	return result, nil
}

// CreateEmbedding implements the embedding generation for a single input.
func (g *GeminiLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	// Apply options
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}
	if opts.Model == "" {
		opts.Model = "text-embedding-004"
	} else if !isValidGeminiEmbeddingModel(opts.Model) {
		return nil, errors.New(errors.InvalidInput, fmt.Sprintf("invalid Gemini embedding model: %s", opts.Model))
	}

	// Prepare the request body
	reqBody := geminiEmbeddingRequest{
		Model: fmt.Sprintf("models/%s", opts.Model),
	}
	reqBody.Content.Parts = []struct {
		Text string `json:"text"`
	}{{Text: input}}

	// Add task type if specified in options
	if taskType, ok := opts.Params["task_type"].(string); ok {
		reqBody.TaskType = taskType
	}

	// Add any additional parameters
	reqBody.Parameters = opts.Params

	// Marshal request
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to marshal request body"),
			errors.Fields{
				"model":        opts.Model,
				"input_length": len(input),
			})
	}

	// Create request
	url := fmt.Sprintf("%s/models/%s:embedContent?key=%s",
		g.GetEndpointConfig().BaseURL,
		opts.Model,
		g.apiKey)
	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		url,
		bytes.NewBuffer(jsonData),
	)

	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to create request"),
			errors.Fields{
				"model": opts.Model,
			})
	}
	// Set headers
	for key, value := range g.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	// Execute request
	resp, err := g.GetHTTPClient().Do(req)

	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to send request"),
			errors.Fields{
				"model": opts.Model,
			})
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to read response body"),
			errors.Fields{
				"model": opts.Model,
			})
	}

	if resp.StatusCode != http.StatusOK {
		truncatedBody := string(body)
		if len(truncatedBody) > 500 { // Example: truncate to 500 characters
			truncatedBody = truncatedBody[:500] + "... (truncated)"
		}
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("API request failed with status code %d: %s", resp.StatusCode, string(truncatedBody))),
			errors.Fields{
				"model":      opts.Model,
				"statusCode": resp.StatusCode,
			})
	}

	var geminiResp geminiEmbeddingResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal response"),
			errors.Fields{
				"model": opts.Model,
			})
	}

	// Check if the embedding values exist
	if len(geminiResp.Embedding.Values) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "embedding values missing in response"),
			errors.Fields{
				"model": opts.Model,
			})
	}

	// Convert to standard format
	result := &core.EmbeddingResult{
		Vector:     geminiResp.Embedding.Values,
		TokenCount: geminiResp.UsageMetadata.TotalTokenCount,
		Metadata: map[string]interface{}{

			"model":            opts.Model,
			"prompt_tokens":    geminiResp.UsageMetadata.PromptTokenCount,
			"truncated_tokens": geminiResp.Embedding.Statistics.TruncatedInputTokenCount,
			"embedding_tokens": geminiResp.Embedding.Statistics.TokenCount,
		},
	}

	return result, nil
}

// CreateEmbeddings implements batch embedding generation.
func (g *GeminiLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	// Apply options
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Use default batch size if not specified
	if opts.BatchSize <= 0 {
		opts.BatchSize = 32
	}

	var allResults []core.EmbeddingResult
	var firstError error
	var errorIndex = -1

	// Process in batches
	for i := 0; i < len(inputs); i += opts.BatchSize {
		end := i + opts.BatchSize
		if end > len(inputs) {
			end = len(inputs)
		}

		batch := inputs[i:end]
		// Prepare batch request
		reqBody := geminiBatchEmbeddingRequest{
			Model: "text-embedding-004",
		}

		// Add each input to the batch request
		reqBody.Requests = make([]struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		}, len(batch))

		for j, input := range batch {
			reqBody.Requests[j].Content.Parts = []struct {
				Text string `json:"text"`
			}{{Text: input}}
		}

		// Add task type if specified
		if taskType, ok := opts.Params["task_type"].(string); ok {
			reqBody.TaskType = taskType
		}
		reqBody.Parameters = opts.Params

		// Marshal request
		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.Wrap(err, errors.InvalidInput, "failed to marshal batch request"),
					errors.Fields{
						"model":      "text-embedding-004",
						"batch_size": len(batch),
					})
				errorIndex = i
			}
			continue
		}

		url := fmt.Sprintf("%s/models/text-embedding-004:batchEmbedContents?key=%s",
			g.GetEndpointConfig().BaseURL,
			g.apiKey)

		// Create request
		req, err := http.NewRequestWithContext(
			ctx,
			"POST",
			url,
			bytes.NewBuffer(jsonData),
		)
		if err != nil {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.Wrap(err, errors.InvalidInput, "failed to create batch request"),
					errors.Fields{
						"model": "text-embedding-004",
					})
				errorIndex = i
			}
			continue
		}

		// Set headers
		for key, value := range g.GetEndpointConfig().Headers {
			req.Header.Set(key, value)
		}

		// Execute request
		resp, err := g.GetHTTPClient().Do(req)
		if err != nil {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.Wrap(err, errors.LLMGenerationFailed, "failed to send batch request"),
					errors.Fields{
						"model": "text-embedding-004",
					})
				errorIndex = i
			}
			continue
		}

		// Read response
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.Wrap(err, errors.LLMGenerationFailed, "failed to read batch response"),
					errors.Fields{
						"model": "text-embedding-004",
					})
				errorIndex = i
			}
			continue
		}

		if resp.StatusCode != http.StatusOK {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.New(errors.LLMGenerationFailed, fmt.Sprintf("API request failed with status code %d: %s", resp.StatusCode, string(body))),
					errors.Fields{
						"model":      "text-embedding-004",
						"statusCode": resp.StatusCode,
					})
				errorIndex = i
			}
			continue
		}

		var batchResp geminiBatchEmbeddingResponse
		if err := json.Unmarshal(body, &batchResp); err != nil {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal batch response"),
					errors.Fields{
						"model": "text-embedding-004",
					})
				errorIndex = i
			}
			continue
		}

		// Process batch results
		for j, embedding := range batchResp.Embeddings {
			result := core.EmbeddingResult{
				Vector:     embedding.Embedding.Values,
				TokenCount: embedding.UsageMetadata.TotalTokenCount,
				Metadata: map[string]interface{}{
					"model":            "text-embedding-004",
					"prompt_tokens":    embedding.UsageMetadata.PromptTokenCount,
					"truncated_tokens": embedding.Embedding.Statistics.TruncatedInputTokenCount,
					"embedding_tokens": embedding.Embedding.Statistics.TokenCount,
					"batch_index":      i + j,
				},
			}
			allResults = append(allResults, result)
		}
	}
	// If we had errors but still got some results, return what we have
	if firstError != nil && len(allResults) == 0 {
		return nil, firstError
	}

	return &core.BatchEmbeddingResult{
		Embeddings: allResults,
		Error:      firstError,
		ErrorIndex: errorIndex,
	}, nil
}

// StreamGenerate for Gemini.
func (g *GeminiLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{Text: prompt},
				},
			},
		},
		GenerationConfig: geminiGenerationConfig{
			Temperature:     opts.Temperature,
			MaxOutputTokens: opts.MaxTokens,
			TopP:            opts.TopP,
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, fmt.Sprintf("InvalidInput: failed to marshal request body: %v", err)),
			errors.Fields{"prompt": prompt, "model": g.ModelID()})
	}

	// Add streaming parameter
	streamURL := constructRequestURL(g.GetEndpointConfig(), g.apiKey) + "&alt=sse"

	req, err := http.NewRequestWithContext(ctx, "POST", streamURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, fmt.Sprintf("InvalidInput: failed to create request: %v", err)),
			errors.Fields{"model": g.ModelID()})
	}

	for key, value := range g.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}
	req.Header.Set("Accept", "text/event-stream")

	// Create channels and response
	chunkChan := make(chan core.StreamChunk)
	streamCtx, cancelStream := context.WithCancel(ctx)

	// Used to protect against multiple closes
	var channelClosed sync.Once

	// Create a safe way to close the channel
	safeCloseChannel := func() {
		channelClosed.Do(func() {
			close(chunkChan)
		})
	}

	response := &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel: func() {
			// Cancel the context first
			cancelStream()
		},
	}

	// Start streaming goroutine
	go func() {
		// Always close the channel when done
		defer safeCloseChannel()

		// Create a client that will properly cancel on context cancellation
		client := g.GetHTTPClient()
		resp, err := client.Do(req)
		if err != nil {
			// Check if context is already cancelled
			if streamCtx.Err() != nil {
				return
			}
			chunkChan <- core.StreamChunk{
				Error: errors.New(errors.LLMGenerationFailed, fmt.Sprintf("LLMGenerationFailed: request failed: %v", err)),
			}
			return
		}
		defer resp.Body.Close()

		reader := bufio.NewReader(resp.Body)

		// Process stream until we're done or cancelled
		for {
			// Check if context is cancelled first
			select {
			case <-streamCtx.Done():
				return
			default:
				// Continue processing
			}

			// Create a timeout for reading
			readCtx, cancel := context.WithTimeout(streamCtx, 500*time.Millisecond)

			// Read with a timeout
			ch := make(chan struct {
				line string
				err  error
			}, 1)

			go func() {
				line, err := reader.ReadString('\n')
				ch <- struct {
					line string
					err  error
				}{line, err}
			}()

			// Wait for read completion or timeout
			var line string
			var readErr error

			select {
			case result := <-ch:
				line = result.line
				readErr = result.err
				cancel() // Cancel the read context
			case <-readCtx.Done():
				cancel() // Explicit cancel
				// Check if main context is cancelled
				if streamCtx.Err() != nil {
					return
				}
				// Just timed out, try again
				continue
			}

			// Process read results
			if readErr != nil {
				// Check for end of stream or cancelled context
				if readErr == io.EOF || streamCtx.Err() != nil {
					return
				}

				// Only send error if context not cancelled
				if streamCtx.Err() == nil {
					chunkChan <- core.StreamChunk{
						Error: errors.New(errors.LLMGenerationFailed, fmt.Sprintf("LLMGenerationFailed: stream read error: %v", readErr)),
					}
				}
				return
			}

			// Process the line if context not cancelled
			if streamCtx.Err() != nil {
				return
			}

			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "data: ") {
				data := strings.TrimPrefix(line, "data: ")

				// Check for end marker
				if data == "[DONE]" {
					return
				}

				// Parse the response
				var chunk geminiResponse
				if err := json.Unmarshal([]byte(data), &chunk); err != nil {
					continue // Skip invalid chunks
				}

				if len(chunk.Candidates) > 0 && len(chunk.Candidates[0].Content.Parts) > 0 {
					content := chunk.Candidates[0].Content.Parts[0].Text
					// Only send if context not cancelled
					if streamCtx.Err() == nil {
						chunkChan <- core.StreamChunk{Content: content}
					}
				}
			}
		}
	}()

	return response, nil
}

func isValidGeminiEmbeddingModel(s string) bool {
	validModels := []string{
		"gemini-embedding-exp-03-07",
		"text-embedding-004",
		"gemini-embedding-004",
		"embedding-001",
		"embedding-latest",
		"embedding-gecko",
		"embedding-gecko-001",
		"text-embedding-gecko-001",
	}

	for _, model := range validModels {
		if s == model {
			return true
		}
	}

	return false
}

func constructRequestURL(endpoint *core.EndpointConfig, apiKey string) string {
	// Remove any trailing slashes from base URL and leading slashes from path
	baseURL := strings.TrimRight(endpoint.BaseURL, "/")
	path := strings.TrimLeft(endpoint.Path, "/")

	// Join them with a single slash
	fullEndpoint := fmt.Sprintf("%s/%s", baseURL, path)

	// Add the API key as query parameter
	return fmt.Sprintf("%s?key=%s", fullEndpoint, apiKey)
}
