package llms

import (
	"context"
	"reflect"

	"github.com/XiaoConstantine/anthropic-go/anthropic"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// AnthropicLLM implements the core.LLM interface for Anthropic's models.
type AnthropicLLM struct {
	client *anthropic.Client
	*core.BaseLLM
}

// NewAnthropicLLM creates a new AnthropicLLM instance.
func NewAnthropicLLM(apiKey string, model anthropic.ModelID) (*AnthropicLLM, error) {
	client, err := anthropic.NewClient(anthropic.WithAPIKey(apiKey))
	if err != nil {
		return nil, errors.Wrap(err, errors.Unknown, "failed to create Anthropic client")
	}
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
	}

	return &AnthropicLLM{
		client:  client,
		BaseLLM: core.NewBaseLLM("anthropic", core.ModelID(model), capabilities, nil),
	}, nil
}

// Generate implements the core.LLM interface.
func (a *AnthropicLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	params := &anthropic.MessageParams{
		Model: string(a.ModelID()),
		Messages: []anthropic.MessageParam{
			{
				Role: "user",
				Content: []anthropic.ContentBlock{
					{
						Type: "text",
						Text: prompt,
					},
				},
			},
		},
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
	}

	message, err := a.client.Messages().Create(ctx, params)
	if message == nil {
		return nil, errors.New(errors.LLMGenerationFailed, "Received nil response from Anthropic API")
	}

	if len(message.Content) == 0 {
		return nil, errors.New(errors.LLMGenerationFailed, "Received empty content from Anthropic API")
	}

	usage := &core.TokenInfo{}
	logger := logging.GetLogger()
	logger.Info(ctx, "Message: %v", message)

	if !reflect.ValueOf(message.Usage).IsZero() {
		usage = &core.TokenInfo{
			PromptTokens:     message.Usage.InputTokens,
			CompletionTokens: message.Usage.OutputTokens,
			TotalTokens:      message.Usage.InputTokens + message.Usage.OutputTokens,
		}
	}

	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to generate response"),
			errors.Fields{
				"model":      string(a.ModelID()),
				"max_tokens": opts.MaxTokens,
			})
	}

	return &core.LLMResponse{Content: message.Content[0].Text, Usage: usage}, nil
}

// GenerateWithJSON implements the core.LLM interface.
func (a *AnthropicLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	// This method is not directly supported by Anthropic's API
	// We'll generate a response and attempt to parse it as JSON
	response, err := a.Generate(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	return utils.ParseJSONResponse(response.Content)
}

func (a *AnthropicLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	// Apply options
	// opts := core.NewEmbeddingOptions()
	//
	// for _, opt := range options {
	// 	opt(opts)
	// }
	//
	// // Create the request parameters
	// params := &anthropic.EmbeddingParams{
	// 	Model:   string(a.ModelID()),
	// 	Input:   input,
	// 	Options: make(map[string]interface{}),
	// }
	//
	// // Apply any additional parameters from options
	// for k, v := range opts.Params {
	// 	params.Options[k] = v
	// }
	//
	// // Call Anthropic's embedding API
	// embedding, err := a.client.Embeddings().Create(ctx, params)
	// if err != nil {
	// 	return nil, errors.WithFields(
	// 		errors.Wrap(err, errors.LLMGenerationFailed, "failed to create embedding"),
	// 		errors.Fields{
	// 			"model":        a.ModelID(),
	// 			"input_length": len(input),
	// 		})
	// }
	//
	// // Convert the response to our standard format
	// result := &core.EmbeddingResult{
	// 	Vector:     embedding.Embedding,
	// 	TokenCount: embedding.TokenCount,
	// 	Metadata: map[string]interface{}{
	// 		"model": embedding.Model,
	// 	},
	// }
	//
	// return result, nil
	return nil, nil
}

func (a *AnthropicLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	// Apply options
	// opts := core.NewEmbeddingOptions()
	// for _, opt := range options {
	// 	opt(opts)
	// }
	//
	// // Validate batch size
	// if opts.BatchSize <= 0 {
	// 	opts.BatchSize = 32 // Default batch size
	// }
	//
	// // Process in batches
	// var allResults []core.EmbeddingResult
	// var firstError error
	// var errorIndex int
	//
	// for i := 0; i < len(inputs); i += opts.BatchSize {
	// 	end := i + opts.BatchSize
	// 	if end > len(inputs) {
	// 		end = len(inputs)
	// 	}
	//
	// 	batch := inputs[i:end]
	// 	params := &anthropic.BatchEmbeddingParams{
	// 		Model:   string(a.ModelID()),
	// 		Inputs:  batch,
	// 		Options: opts.Params,
	// 	}
	//
	// 	// Call Anthropic's batch embedding API
	// 	batchResults, err := a.client.Embeddings().CreateBatch(ctx, params)
	// 	if err != nil {
	// 		if firstError == nil {
	// 			firstError = err
	// 			errorIndex = i
	// 		}
	// 		continue
	// 	}
	//
	// 	// Convert batch results
	// 	for _, embedding := range batchResults.Embeddings {
	// 		result := core.EmbeddingResult{
	// 			Vector:     embedding.Embedding,
	// 			TokenCount: embedding.TokenCount,
	// 			Metadata: map[string]interface{}{
	// 				"model": embedding.Model,
	// 			},
	// 		}
	// 		allResults = append(allResults, result)
	// 	}
	// }
	//
	// return &core.BatchEmbeddingResult{
	// 	Embeddings: allResults,
	// 	Error:      firstError,
	// 	ErrorIndex: errorIndex,
	// }, nil
	return nil, nil
}
