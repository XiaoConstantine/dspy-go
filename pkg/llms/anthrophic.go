package llms

import (
	"context"

	"github.com/XiaoConstantine/anthropic-go/anthropic"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// AnthropicLLM implements the core.LLM interface for Anthropic's models.
type AnthropicLLM struct {
	client *anthropic.Client
	model  anthropic.ModelID
}

// NewAnthropicLLM creates a new AnthropicLLM instance.
func NewAnthropicLLM(apiKey string, model anthropic.ModelID) (*AnthropicLLM, error) {
	client, err := anthropic.NewClient(anthropic.WithAPIKey(apiKey))
	if err != nil {
		return nil, errors.Wrap(err, errors.Unknown, "failed to create Anthropic client")
	}

	return &AnthropicLLM{
		client: client,
		model:  model,
	}, nil
}

// Generate implements the core.LLM interface.
func (a *AnthropicLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	params := &anthropic.MessageParams{
		Model: string(a.model),
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
	usage := &core.TokenInfo{
		PromptTokens:     message.Usage.InputTokens,
		CompletionTokens: message.Usage.OutputTokens,
		TotalTokens:      message.Usage.InputTokens + message.Usage.OutputTokens,
	}

	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to generate response"),
			errors.Fields{
				"model":      string(a.model),
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
