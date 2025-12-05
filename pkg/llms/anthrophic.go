package llms

import (
	"context"
	"encoding/base64"
	"errors"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	errs "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// AnthropicLLM implements the core.LLM interface for Anthropic's models.
type AnthropicLLM struct {
	client *anthropic.Client
	*core.BaseLLM
}

// Model name compatibility layer for backwards compatibility.
var modelNameMapping = map[string]anthropic.Model{
	// Old Claude 3 names to new equivalents
	"claude-3-opus-20240229":     anthropic.ModelClaudeOpus4_1_20250805,
	"claude-3-sonnet-20240229":   anthropic.ModelClaudeSonnet4_5_20250929,
	"claude-3-haiku-20240307":    anthropic.ModelClaude_3_Haiku_20240307,
	"claude-3.5-sonnet-20241022": anthropic.ModelClaudeSonnet4_5_20250929,
	"claude-3-5-sonnet-20240620": anthropic.ModelClaudeSonnet4_5_20250929,
	"claude-3.5-sonnet-20250929": anthropic.ModelClaudeSonnet4_5_20250929,
	"claude-3-opus":              anthropic.ModelClaudeOpus4_1_20250805,
	"claude-3-sonnet":            anthropic.ModelClaudeSonnet4_5_20250929,
	"claude-3-haiku":             anthropic.ModelClaude_3_Haiku_20240307,
}

// normalizeModelName maps old model names to new official ones.
func normalizeModelName(name string) anthropic.Model {
	if normalized, ok := modelNameMapping[name]; ok {
		return normalized
	}
	// Return as-is if not in mapping (allows new models to work automatically)
	return anthropic.Model(name)
}

// NewAnthropicLLM creates a new AnthropicLLM instance.
func NewAnthropicLLM(apiKey string, model anthropic.Model) (*AnthropicLLM, error) {
	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
	)

	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
		core.CapabilityStreaming,
	}

	return &AnthropicLLM{
		client:  &client,
		BaseLLM: core.NewBaseLLM("anthropic", core.ModelID(model), capabilities, nil),
	}, nil
}

// NewAnthropicLLMFromConfig creates a new AnthropicLLM instance from configuration.
func NewAnthropicLLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*AnthropicLLM, error) {
	// Get API key from config or environment
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey == "" {
		return nil, errs.New(errs.InvalidInput, "API key is required")
	}

	// Normalize model ID and validate
	normalizedModelID := normalizeModelName(string(modelID))
	if !isValidAnthropicModel(string(normalizedModelID)) {
		return nil, errs.WithFields(
			errs.New(errs.InvalidInput, "unsupported Anthropic model"),
			errs.Fields{"model": modelID})
	}

	// Create client with optional configuration
	clientOpts := []option.RequestOption{option.WithAPIKey(apiKey)}

	// Apply endpoint configuration if provided
	if config.Endpoint != nil && config.Endpoint.BaseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(config.Endpoint.BaseURL))
	}

	client := anthropic.NewClient(clientOpts...)

	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
		core.CapabilityStreaming,
	}

	return &AnthropicLLM{
		client:  &client,
		BaseLLM: core.NewBaseLLM("anthropic", modelID, capabilities, config.Endpoint),
	}, nil
}

// AnthropicProviderFactory creates AnthropicLLM instances.
func AnthropicProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewAnthropicLLMFromConfig(ctx, config, modelID)
}

// isValidAnthropicModel checks if the model is a valid Anthropic model.
func isValidAnthropicModel(model string) bool {
	validPrefixes := []string{
		"claude-3",
		"claude-4",
		"claude-haiku",    // claude-haiku-4-5, etc.
		"claude-sonnet",   // claude-sonnet-4, etc.
		"claude-opus",     // claude-opus-4, etc.
	}

	for _, prefix := range validPrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

// Generate implements the core.LLM interface.
func (a *AnthropicLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	logger := logging.GetLogger()
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	normalizedModelID := normalizeModelName(a.ModelID())

	message, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model: normalizedModelID,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewTextBlock(prompt),
			),
		},
		MaxTokens:   int64(opts.MaxTokens),
		Temperature: anthropic.Float(opts.Temperature),
	})

	if err != nil {
		var apiErr *anthropic.Error
		if errors.As(err, &apiErr) {
			logger.Error(ctx, "Anthropic API error: status code %d", apiErr.StatusCode)
		}
		return nil, errs.WithFields(
			errs.Wrap(err, errs.LLMGenerationFailed, "failed to generate response"),
			errs.Fields{
				"model":      string(normalizedModelID),
				"max_tokens": opts.MaxTokens,
			})
	}

	if message == nil {
		return nil, errs.New(errs.LLMGenerationFailed, "Received nil response from Anthropic API")
	}

	if len(message.Content) == 0 {
		return nil, errs.New(errs.LLMGenerationFailed, "Received empty content from Anthropic API")
	}

	// Extract text from response using union type methods
	var responseText string
	if block := message.Content[0]; block.Type == "text" {
		responseText = block.Text
	}

	usage := &core.TokenInfo{
		PromptTokens:     int(message.Usage.InputTokens),
		CompletionTokens: int(message.Usage.OutputTokens),
		TotalTokens:      int(message.Usage.InputTokens + message.Usage.OutputTokens),
	}

	logger.Debug(ctx, "Anthropic response: %d prompt tokens, %d completion tokens", message.Usage.InputTokens, message.Usage.OutputTokens)

	return &core.LLMResponse{Content: responseText, Usage: usage}, nil
}

// GenerateWithJSON implements the core.LLM interface.
func (a *AnthropicLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	// Generate a response and attempt to parse it as JSON
	response, err := a.Generate(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	return utils.ParseJSONResponse(response.Content)
}

func (a *AnthropicLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	panic("Not implemented")
}

func (a *AnthropicLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	// Anthropic does not provide embedding api directly, use voyage or other providers
	return nil, nil
}

func (a *AnthropicLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

// StreamGenerate implements streaming text generation using the official SDK's iterator pattern.
func (a *AnthropicLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	logger := logging.GetLogger()
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Create a channel for the stream chunks
	chunkChan := make(chan core.StreamChunk)

	// Create a cancellable context
	streamCtx, cancelFunc := context.WithCancel(ctx)

	normalizedModelID := normalizeModelName(a.ModelID())

	// Start a goroutine to handle the streaming request
	go func() {
		defer close(chunkChan)
		defer cancelFunc()

		stream := a.client.Messages.NewStreaming(streamCtx, anthropic.MessageNewParams{
			Model: normalizedModelID,
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(
					anthropic.NewTextBlock(prompt),
				),
			},
			MaxTokens:   int64(opts.MaxTokens),
			Temperature: anthropic.Float(opts.Temperature),
			TopP:        anthropic.Float(opts.TopP),
		})

		defer stream.Close()

		var tokenInfo core.TokenInfo

		for stream.Next() {
			event := stream.Current()

			// Process event using AsAny() for union type switch
			switch variant := event.AsAny().(type) {
			case anthropic.ContentBlockDeltaEvent:
				// Handle text delta
				if textDelta := variant.Delta.AsTextDelta(); textDelta.Text != "" {
					chunkChan <- core.StreamChunk{Content: textDelta.Text}
				}

			case anthropic.MessageStartEvent:
				// Message beginning
				tokenInfo.PromptTokens = int(variant.Message.Usage.InputTokens)

			case anthropic.MessageDeltaEvent:
				// Message delta with usage update
				tokenInfo.CompletionTokens = int(variant.Usage.OutputTokens)
				tokenInfo.TotalTokens = tokenInfo.PromptTokens + tokenInfo.CompletionTokens

				chunkChan <- core.StreamChunk{
					Usage: &tokenInfo,
				}

			case anthropic.MessageStopEvent:
				// End of message
				chunkChan <- core.StreamChunk{Done: true}

			case anthropic.ContentBlockStartEvent:
				// Beginning of a content block, nothing to do

			default:
				// Handle any other event types gracefully
				logger.Debug(streamCtx, "Received event type: %T", event)
			}
		}

		if err := stream.Err(); err != nil {
			var apiErr *anthropic.Error
			if errors.As(err, &apiErr) {
				logger.Error(streamCtx, "Anthropic streaming error: status code %d", apiErr.StatusCode)
			}
			chunkChan <- core.StreamChunk{
				Error: errs.Wrap(err, errs.LLMGenerationFailed, "streaming failed"),
			}
		}
	}()

	return &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelFunc,
	}, nil
}

// GenerateWithContent generates a response with multimodal content.
func (a *AnthropicLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	logger := logging.GetLogger()
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Convert core.ContentBlock to anthropic message params
	messages := convertContentBlocksToMessages(content)
	if len(messages) == 0 {
		return nil, errs.New(errs.InvalidInput, "no content provided")
	}

	normalizedModelID := normalizeModelName(a.ModelID())

	message, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:       normalizedModelID,
		Messages:    messages,
		MaxTokens:   int64(opts.MaxTokens),
		Temperature: anthropic.Float(opts.Temperature),
	})

	if err != nil {
		var apiErr *anthropic.Error
		if errors.As(err, &apiErr) {
			logger.Error(ctx, "Anthropic API error: status code %d", apiErr.StatusCode)
		}
		return nil, errs.Wrap(err, errs.LLMGenerationFailed, "failed to generate response with content")
	}

	if message == nil || len(message.Content) == 0 {
		return nil, errs.New(errs.LLMGenerationFailed, "Received empty response from Anthropic API")
	}

	// Extract text from response
	var responseText string
	if block := message.Content[0]; block.Type == "text" {
		responseText = block.Text
	}

	usage := &core.TokenInfo{
		PromptTokens:     int(message.Usage.InputTokens),
		CompletionTokens: int(message.Usage.OutputTokens),
		TotalTokens:      int(message.Usage.InputTokens + message.Usage.OutputTokens),
	}

	return &core.LLMResponse{Content: responseText, Usage: usage}, nil
}

// StreamGenerateWithContent generates a streaming response with multimodal content.
func (a *AnthropicLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	logger := logging.GetLogger()
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	chunkChan := make(chan core.StreamChunk)
	streamCtx, cancelFunc := context.WithCancel(ctx)

	messages := convertContentBlocksToMessages(content)
	if len(messages) == 0 {
		cancelFunc()
		close(chunkChan)
		return nil, errs.New(errs.InvalidInput, "no content provided")
	}

	normalizedModelID := normalizeModelName(a.ModelID())

	go func() {
		defer close(chunkChan)
		defer cancelFunc()

		stream := a.client.Messages.NewStreaming(streamCtx, anthropic.MessageNewParams{
			Model:       normalizedModelID,
			Messages:    messages,
			MaxTokens:   int64(opts.MaxTokens),
			Temperature: anthropic.Float(opts.Temperature),
			TopP:        anthropic.Float(opts.TopP),
		})

		defer stream.Close()

		var tokenInfo core.TokenInfo

		for stream.Next() {
			event := stream.Current()

			switch variant := event.AsAny().(type) {
			case anthropic.ContentBlockDeltaEvent:
				if textDelta := variant.Delta.AsTextDelta(); textDelta.Text != "" {
					chunkChan <- core.StreamChunk{Content: textDelta.Text}
				}

			case anthropic.MessageStopEvent:
				chunkChan <- core.StreamChunk{Done: true}

			case anthropic.MessageDeltaEvent:
				tokenInfo.CompletionTokens = int(variant.Usage.OutputTokens)
				tokenInfo.TotalTokens = tokenInfo.PromptTokens + tokenInfo.CompletionTokens

				chunkChan <- core.StreamChunk{
					Usage: &tokenInfo,
				}
			}
		}

		if err := stream.Err(); err != nil {
			var apiErr *anthropic.Error
			if errors.As(err, &apiErr) {
				logger.Error(streamCtx, "Anthropic streaming error: status code %d", apiErr.StatusCode)
			}
			chunkChan <- core.StreamChunk{
				Error: errs.Wrap(err, errs.LLMGenerationFailed, "streaming with content failed"),
			}
		}
	}()

	return &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelFunc,
	}, nil
}

// convertContentBlocksToMessages converts core.ContentBlock to anthropic MessageParam.
func convertContentBlocksToMessages(blocks []core.ContentBlock) []anthropic.MessageParam {
	var messages []anthropic.MessageParam

	// Group text and media blocks
	var contentBlockUnions []anthropic.ContentBlockParamUnion

	for _, block := range blocks {
		switch block.Type {
		case core.FieldTypeText:
			contentBlockUnions = append(contentBlockUnions, anthropic.ContentBlockParamUnion{
				OfText: &anthropic.TextBlockParam{
					Text: block.Text,
				},
			})

		case core.FieldTypeImage:
			if len(block.Data) > 0 {
				// Create image block - Data field requires base64-encoded string
				contentBlockUnions = append(contentBlockUnions, anthropic.ContentBlockParamUnion{
					OfImage: &anthropic.ImageBlockParam{
						Source: anthropic.ImageBlockParamSourceUnion{
							OfBase64: &anthropic.Base64ImageSourceParam{
								Data:      base64.StdEncoding.EncodeToString(block.Data),
								MediaType: anthropic.Base64ImageSourceMediaType(block.MimeType),
							},
						},
					},
				})
			}

		case core.FieldTypeAudio:
			// Handle audio content if supported in future
			// For now, add as text description
			if block.Text != "" {
				contentBlockUnions = append(contentBlockUnions, anthropic.ContentBlockParamUnion{
					OfText: &anthropic.TextBlockParam{
						Text: "[Audio: " + block.MimeType + "]",
					},
				})
			}
		}
	}

	// Create a single user message with all content blocks
	if len(contentBlockUnions) > 0 {
		messages = append(messages, anthropic.MessageParam{
			Content: contentBlockUnions,
			Role:    anthropic.MessageParamRoleUser,
		})
	}

	return messages
}
