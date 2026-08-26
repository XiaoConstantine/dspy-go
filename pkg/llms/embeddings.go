package llms

import (
	"bytes"
	"context"
	jsonv1 "encoding/json"
	jsonv2 "encoding/json/v2"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"google.golang.org/genai"
)

const maxEmbeddingResponseBytes = 128 << 20

type embeddingClient interface {
	core.Embedder
	core.BatchEmbedder
}

type openAIEmbeddingClient struct {
	endpoint     string
	headers      map[string]string
	client       *http.Client
	defaultModel string
}

type openAIEmbeddingRequest struct {
	Input          any    `json:"input"`
	Model          string `json:"model"`
	EncodingFormat string `json:"encoding_format"`
}

type openAIEmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

func newOpenAIEmbeddingClient(baseURL string, headers map[string]string, client *http.Client, defaultModel string) *openAIEmbeddingClient {
	baseURL = strings.TrimRight(baseURL, "/")
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}
	if !strings.HasSuffix(baseURL, "/v1") {
		baseURL += "/v1"
	}
	if defaultModel == "" {
		defaultModel = "text-embedding-3-small"
	}
	headerCopy := make(map[string]string, len(headers))
	for key, value := range headers {
		headerCopy[key] = value
	}
	return &openAIEmbeddingClient{
		endpoint:     baseURL + "/embeddings",
		headers:      headerCopy,
		client:       client,
		defaultModel: defaultModel,
	}
}

func (c *openAIEmbeddingClient) CreateEmbedding(
	ctx context.Context,
	input string,
	options ...core.EmbeddingOption,
) (*core.EmbeddingResult, error) {
	opts := embeddingOptions(options)
	model := c.defaultModel
	if opts.Model != "" {
		model = opts.Model
	}
	response, err := c.request(ctx, input, model)
	if err != nil {
		return nil, err
	}
	if len(response.Data) == 0 {
		return nil, errors.New(errors.InvalidResponse, "embedding values missing in OpenAI-compatible response")
	}
	return &core.EmbeddingResult{
		Vector:     response.Data[0].Embedding,
		TokenCount: response.Usage.TotalTokens,
		Metadata: map[string]any{
			"model": response.Model,
			"index": response.Data[0].Index,
		},
	}, nil
}

func (c *openAIEmbeddingClient) CreateEmbeddings(
	ctx context.Context,
	inputs []string,
	options ...core.EmbeddingOption,
) (*core.BatchEmbeddingResult, error) {
	if len(inputs) == 0 {
		return &core.BatchEmbeddingResult{ErrorIndex: -1}, nil
	}
	opts := embeddingOptions(options)
	model := c.defaultModel
	if opts.Model != "" {
		model = opts.Model
	}
	response, err := c.request(ctx, inputs, model)
	if err != nil {
		return &core.BatchEmbeddingResult{Error: err, ErrorIndex: 0}, nil
	}
	if len(response.Data) != len(inputs) {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "OpenAI-compatible embedding count does not match input count"),
			errors.Fields{"expected": len(inputs), "actual": len(response.Data)},
		)
	}
	results := make([]core.EmbeddingResult, len(inputs))
	seen := make([]bool, len(inputs))
	perInputTokens := response.Usage.TotalTokens / len(inputs)
	for _, embedding := range response.Data {
		if embedding.Index < 0 || embedding.Index >= len(inputs) {
			return nil, errors.WithFields(
				errors.New(errors.InvalidResponse, "OpenAI-compatible embedding index is out of range"),
				errors.Fields{"index": embedding.Index, "input_count": len(inputs)},
			)
		}
		if seen[embedding.Index] {
			return nil, errors.WithFields(
				errors.New(errors.InvalidResponse, "OpenAI-compatible embedding index is duplicated"),
				errors.Fields{"index": embedding.Index},
			)
		}
		seen[embedding.Index] = true
		results[embedding.Index] = core.EmbeddingResult{
			Vector:     embedding.Embedding,
			TokenCount: perInputTokens,
			Metadata: map[string]any{
				"model": response.Model,
				"index": embedding.Index,
			},
		}
	}
	return &core.BatchEmbeddingResult{Embeddings: results, ErrorIndex: -1}, nil
}

func (c *openAIEmbeddingClient) request(ctx context.Context, input any, model string) (*openAIEmbeddingResponse, error) {
	payload, err := jsonv2.Marshal(openAIEmbeddingRequest{
		Input:          input,
		Model:          model,
		EncodingFormat: "float",
	}, jsonv1.DefaultOptionsV1())
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "failed to marshal embedding request")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "failed to create embedding request")
	}
	for key, value := range c.headers {
		req.Header.Set(key, value)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to send embedding request"),
			errors.Fields{"model": model},
		)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxEmbeddingResponseBytes+1))
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidResponse, "failed to read embedding response")
	}
	if len(body) > maxEmbeddingResponseBytes {
		return nil, errors.New(errors.InvalidResponse, "embedding response exceeds size limit")
	}
	if resp.StatusCode != http.StatusOK {
		message := strings.TrimSpace(string(body))
		if len(message) > 500 {
			message = message[:500] + "... (truncated)"
		}
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("embedding request failed with status %d: %s", resp.StatusCode, message)),
			errors.Fields{"model": model, "status_code": resp.StatusCode},
		)
	}

	var response openAIEmbeddingResponse
	if err := jsonv2.Unmarshal(body, &response); err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to decode embedding response"),
			errors.Fields{"model": model},
		)
	}
	return &response, nil
}

type geminiEmbeddingClient struct {
	client *genai.Client
}

func newGeminiEmbeddingClient(
	ctx context.Context,
	apiKey string,
	baseURL string,
	headers map[string]string,
	client *http.Client,
) (*geminiEmbeddingClient, error) {
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com/"
	}
	header := make(http.Header, len(headers))
	for key, value := range headers {
		header.Set(key, value)
	}
	sdkClient, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:     apiKey,
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: client,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    strings.TrimRight(baseURL, "/") + "/",
			APIVersion: "v1beta",
			Headers:    header,
		},
	})
	if err != nil {
		return nil, errors.Wrap(err, errors.ConfigurationError, "failed to configure Gemini embedding client")
	}
	return &geminiEmbeddingClient{client: sdkClient}, nil
}

func (c *geminiEmbeddingClient) CreateEmbedding(
	ctx context.Context,
	input string,
	options ...core.EmbeddingOption,
) (*core.EmbeddingResult, error) {
	batch, err := c.CreateEmbeddings(ctx, []string{input}, options...)
	if err != nil {
		return nil, err
	}
	if batch.Error != nil {
		return nil, batch.Error
	}
	if len(batch.Embeddings) == 0 {
		return nil, errors.New(errors.InvalidResponse, "embedding values missing in Gemini response")
	}
	return &batch.Embeddings[0], nil
}

func (c *geminiEmbeddingClient) CreateEmbeddings(
	ctx context.Context,
	inputs []string,
	options ...core.EmbeddingOption,
) (*core.BatchEmbeddingResult, error) {
	opts := embeddingOptions(options)
	model := opts.Model
	if model == "" {
		model = string(core.ModelGoogleGeminiEmbedding2)
	}
	if model != string(core.ModelGoogleGeminiEmbedding2) && model != "gemini-embedding-001" {
		return nil, errors.New(errors.InvalidInput, fmt.Sprintf("invalid Gemini embedding model: %s", model))
	}
	if len(inputs) == 0 {
		return &core.BatchEmbeddingResult{ErrorIndex: -1}, nil
	}

	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = 32
	}
	config := geminiEmbeddingConfig(opts)
	results := make([]core.EmbeddingResult, 0, len(inputs))
	var firstError error
	errorIndex := -1
	for start := 0; start < len(inputs); start += batchSize {
		end := min(start+batchSize, len(inputs))
		contents := make([]*genai.Content, end-start)
		for i, input := range inputs[start:end] {
			contents[i] = &genai.Content{Parts: []*genai.Part{genai.NewPartFromText(input)}}
		}
		response, err := c.client.Models.EmbedContent(ctx, model, contents, config)
		if err != nil {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.Wrap(err, errors.LLMGenerationFailed, "Gemini embedding request failed"),
					errors.Fields{"model": model},
				)
				errorIndex = start
			}
			continue
		}
		if len(response.Embeddings) != len(contents) {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.New(errors.InvalidResponse, "Gemini embedding count does not match input count"),
					errors.Fields{"model": model, "expected": len(contents), "actual": len(response.Embeddings)},
				)
				errorIndex = start
			}
			continue
		}
		for i, embedding := range response.Embeddings {
			if embedding == nil || len(embedding.Values) == 0 {
				if firstError == nil {
					firstError = errors.WithFields(
						errors.New(errors.InvalidResponse, "embedding values missing in Gemini response"),
						errors.Fields{"model": model},
					)
					errorIndex = start + i
				}
				continue
			}
			result := core.EmbeddingResult{
				Vector: embedding.Values,
				Metadata: map[string]any{
					"model":       model,
					"batch_index": start + i,
				},
			}
			if embedding.Statistics != nil {
				result.TokenCount = int(embedding.Statistics.TokenCount)
				result.Metadata["embedding_tokens"] = embedding.Statistics.TokenCount
				result.Metadata["truncated"] = embedding.Statistics.Truncated
			}
			results = append(results, result)
		}
	}
	if firstError != nil && len(results) == 0 {
		return nil, firstError
	}
	return &core.BatchEmbeddingResult{
		Embeddings: results,
		Error:      firstError,
		ErrorIndex: errorIndex,
	}, nil
}

func embeddingOptions(options []core.EmbeddingOption) *core.EmbeddingOptions {
	opts := core.NewEmbeddingOptions()
	for _, option := range options {
		option(opts)
	}
	return opts
}

func geminiEmbeddingConfig(options *core.EmbeddingOptions) *genai.EmbedContentConfig {
	config := &genai.EmbedContentConfig{}
	if taskType, ok := options.Params["task_type"].(string); ok {
		config.TaskType = taskType
	}
	if title, ok := options.Params["title"].(string); ok {
		config.Title = title
	}
	if dimensions, ok := options.Params["output_dimensionality"].(int); ok && dimensions > 0 {
		value := int32(dimensions)
		config.OutputDimensionality = &value
	}
	return config
}
