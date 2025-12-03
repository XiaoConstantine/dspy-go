package llms

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"strings"

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

// NewAnthropicLLMFromConfig creates a new AnthropicLLM instance from configuration.
func NewAnthropicLLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*AnthropicLLM, error) {
	// Get API key from config or environment
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New(errors.InvalidInput, "API key is required")
	}

	// Validate model ID
	model := anthropic.ModelID(modelID)
	if !isValidAnthropicModel(model) {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "unsupported Anthropic model"),
			errors.Fields{"model": modelID})
	}

	// Create client with optional configuration
	clientOpts := []anthropic.ClientOption{anthropic.WithAPIKey(apiKey)}

	// Apply endpoint configuration if provided
	if config.Endpoint != nil && config.Endpoint.BaseURL != "" {
		clientOpts = append(clientOpts, anthropic.WithBaseURL(config.Endpoint.BaseURL))
	}

	client, err := anthropic.NewClient(clientOpts...)
	if err != nil {
		return nil, errors.Wrap(err, errors.Unknown, "failed to create Anthropic client")
	}

	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
	}

	// Check if streaming is supported
	if supportsStreaming(model) {
		capabilities = append(capabilities, core.CapabilityStreaming)
	}

	return &AnthropicLLM{
		client:  client,
		BaseLLM: core.NewBaseLLM("anthropic", modelID, capabilities, config.Endpoint),
	}, nil
}

// AnthropicProviderFactory creates AnthropicLLM instances.
func AnthropicProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewAnthropicLLMFromConfig(ctx, config, modelID)
}

// isValidAnthropicModel checks if the model is a valid Anthropic model.
func isValidAnthropicModel(model anthropic.ModelID) bool {
	validModels := []anthropic.ModelID{
		// Claude 3.x series
		anthropic.ModelHaiku,
		anthropic.ModelSonnet,
		anthropic.ModelOpus,
		// Claude 4.x series
		anthropic.ModelClaude4Opus,
		anthropic.ModelClaude4Sonnet,
		anthropic.ModelClaude45Sonnet,
	}

	for _, validModel := range validModels {
		if model == validModel {
			return true
		}
	}
	return false
}

// supportsStreaming checks if the model supports streaming.
func supportsStreaming(model anthropic.ModelID) bool {
	// All current Anthropic models support streaming
	return true
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

func (a *AnthropicLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	panic("Not implemented")
}

func (a *AnthropicLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	// Anthropic does not provide embedding api directly, but go through voyage
	return nil, nil
}

func (a *AnthropicLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

// StreamGenerate for Anthropic adapts the callback-based API to our channel-based approach.
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

	// Prepare the message parameters with streaming
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
		TopP:        opts.TopP,
	}

	// Track accumulated content for token counting
	var contentBuffer strings.Builder
	var tokenInfo core.TokenInfo

	// Define the StreamFunc that will receive streaming chunks
	params.StreamFunc = func(ctx context.Context, chunk []byte) error {
		// Check if streaming has been cancelled
		select {
		case <-streamCtx.Done():
			return streamCtx.Err()
		default:
			// Continue processing
		}

		// Parse the streaming chunk
		var streamData struct {
			Type  string `json:"type"`
			Delta struct {
				Text string `json:"text"`
			} `json:"delta"`
			Usage *struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
			} `json:"usage,omitempty"`
		}

		if err := json.Unmarshal(chunk, &streamData); err != nil {
			logger.Debug(ctx, "Error parsing stream chunk: %v", err)
			return nil // Skip unparseable chunks
		}

		// Handle different message types
		switch streamData.Type {
		case "content_block_delta":
			// Text content
			if streamData.Delta.Text != "" {
				contentBuffer.WriteString(streamData.Delta.Text)
				chunkChan <- core.StreamChunk{Content: streamData.Delta.Text}
			}

		case "message_stop":
			// End of message
			chunkChan <- core.StreamChunk{Done: true}

		case "error":
			// Error in the stream
			chunkChan <- core.StreamChunk{
				Error: errors.New(errors.LLMGenerationFailed, "Error from Anthropic API"),
			}

		case "message_start":
			// Message beginning, nothing to do

		case "content_block_start":
			// Beginning of a content block, nothing to do

		case "ping":
			// Keepalive ping, nothing to do
		}

		// Update token info if usage data is provided
		if streamData.Usage != nil {
			tokenInfo.PromptTokens = streamData.Usage.InputTokens
			tokenInfo.CompletionTokens = streamData.Usage.OutputTokens
			tokenInfo.TotalTokens = tokenInfo.PromptTokens + tokenInfo.CompletionTokens

			// Add token info to the latest chunk
			chunkChan <- core.StreamChunk{
				Usage: &tokenInfo,
			}
		}

		return nil
	}

	// Start a goroutine to handle the streaming request
	go func() {
		defer close(chunkChan)
		defer cancelFunc() // Ensure context is cancelled when done

		_, err := a.client.Messages().Create(streamCtx, params)
		if err != nil {
			chunkChan <- core.StreamChunk{
				Error: errors.Wrap(err, errors.LLMGenerationFailed, "streaming failed"),
			}
			return
		}
	}()

	return &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelFunc,
	}, nil
}
