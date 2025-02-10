package llms

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
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
	case core.ModelGoogleGeminiPro, core.ModelGoogleGeminiFlash, core.ModelGoogleGeminiFlashThinking:
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
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to send request"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}
	defer resp.Body.Close()

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

	var geminiResp geminiResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal response"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	if len(geminiResp.Candidates) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no candidates in response"),
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

func constructRequestURL(endpoint *core.EndpointConfig, apiKey string) string {
	// Remove any trailing slashes from base URL and leading slashes from path
	baseURL := strings.TrimRight(endpoint.BaseURL, "/")
	path := strings.TrimLeft(endpoint.Path, "/")

	// Join them with a single slash
	fullEndpoint := fmt.Sprintf("%s/%s", baseURL, path)

	// Add the API key as query parameter
	return fmt.Sprintf("%s?key=%s", fullEndpoint, apiKey)
}

// CreateEmbedding implements the embedding generation for a single input.
func (g *GeminiLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	// Apply options
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Prepare the request body
	reqBody := geminiEmbeddingRequest{
		Model: "models/text-embedding-004",
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
				"model":        "text-embedding-004",
				"input_length": len(input),
			})
	}

	// Create request
	url := fmt.Sprintf("%s/models/text-embedding-004:embedContent?key=%s",
		g.GetEndpointConfig().BaseURL,
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
				"model": "text-embedding-004",
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

				"model": "text-embedding-004",
			})
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to read response body"),
			errors.Fields{

				"model": "text-embedding-004",
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

	var geminiResp geminiEmbeddingResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal response"),
			errors.Fields{
				"model": "text-embedding-004",
			})
	}

	// Convert to standard format
	result := &core.EmbeddingResult{
		Vector:     geminiResp.Embedding.Values,
		TokenCount: geminiResp.UsageMetadata.TotalTokenCount,
		Metadata: map[string]interface{}{

			"model":            "text-embedding-004",
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
	var errorIndex int = -1

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

	return &core.BatchEmbeddingResult{
		Embeddings: allResults,
		Error:      firstError,
		ErrorIndex: errorIndex,
	}, nil
}
