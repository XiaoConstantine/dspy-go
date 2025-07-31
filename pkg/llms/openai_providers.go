package llms

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// LiteLLM convenience constructor.
func NewLiteLLM(modelID core.ModelID, apiKey string, opts ...OpenAIOption) (*OpenAILLM, error) {
	if apiKey == "" {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "LiteLLM API key is required"),
			errors.Fields{"provider": "litellm"})
	}

	// Default LiteLLM configuration
	defaultOpts := []OpenAIOption{
		WithAPIKey(apiKey),
		WithOpenAIBaseURL("http://localhost:4000"),
		WithOpenAIPath("/chat/completions"),
	}

	// Merge with user options (user options take precedence)
	allOpts := append(defaultOpts, opts...)

	llm, err := NewOpenAILLM(modelID, allOpts...)
	if err != nil {
		return nil, err
	}

	// Override provider name for clarity by creating a new BaseLLM with the correct provider name
	endpointCfg := llm.GetEndpointConfig()
	capabilities := llm.Capabilities()
	newBaseLLM := core.NewBaseLLM("litellm", modelID, capabilities, endpointCfg)

	return &OpenAILLM{
		BaseLLM: newBaseLLM,
		apiKey:  llm.apiKey,
	}, nil
}

// LocalAI convenience constructor.
func NewLocalAI(modelID core.ModelID, baseURL string, opts ...OpenAIOption) (*OpenAILLM, error) {
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}

	defaultOpts := []OpenAIOption{
		WithOpenAIBaseURL(baseURL),
		WithOpenAIPath("/v1/chat/completions"),
		// LocalAI typically doesn't require auth
	}

	allOpts := append(defaultOpts, opts...)

	llm, err := NewOpenAILLM(modelID, allOpts...)
	if err != nil {
		return nil, err
	}

	// Override provider name for clarity
	endpointCfg := llm.GetEndpointConfig()
	capabilities := llm.Capabilities()
	newBaseLLM := core.NewBaseLLM("localai", modelID, capabilities, endpointCfg)

	return &OpenAILLM{
		BaseLLM: newBaseLLM,
		apiKey:  llm.apiKey,
	}, nil
}

// FastChat convenience constructor.
func NewFastChat(modelID core.ModelID, baseURL string, opts ...OpenAIOption) (*OpenAILLM, error) {
	if baseURL == "" {
		baseURL = "http://localhost:8000"
	}

	defaultOpts := []OpenAIOption{
		WithOpenAIBaseURL(baseURL),
		WithOpenAIPath("/v1/chat/completions"),
		// FastChat typically doesn't require auth
	}

	allOpts := append(defaultOpts, opts...)

	llm, err := NewOpenAILLM(modelID, allOpts...)
	if err != nil {
		return nil, err
	}

	// Override provider name for clarity
	endpointCfg := llm.GetEndpointConfig()
	capabilities := llm.Capabilities()
	newBaseLLM := core.NewBaseLLM("fastchat", modelID, capabilities, endpointCfg)

	return &OpenAILLM{
		BaseLLM: newBaseLLM,
		apiKey:  llm.apiKey,
	}, nil
}

// Generic OpenAI-compatible constructor.
func NewOpenAICompatible(provider string, modelID core.ModelID, baseURL string, opts ...OpenAIOption) (*OpenAILLM, error) {
	defaultOpts := []OpenAIOption{
		WithOpenAIBaseURL(baseURL),
		WithOpenAIPath("/v1/chat/completions"),
	}

	allOpts := append(defaultOpts, opts...)

	llm, err := NewOpenAILLM(modelID, allOpts...)
	if err != nil {
		return nil, err
	}

	// Override provider name for clarity
	endpointCfg := llm.GetEndpointConfig()
	capabilities := llm.Capabilities()
	newBaseLLM := core.NewBaseLLM(provider, modelID, capabilities, endpointCfg)

	return &OpenAILLM{
		BaseLLM: newBaseLLM,
		apiKey:  llm.apiKey,
	}, nil
}

// Factory functions for use with the registry system

// LiteLLMProviderFactory creates LiteLLM instances from provider config.
func LiteLLMProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:4000"
	}

	return NewLiteLLM(modelID, config.APIKey, WithOpenAIBaseURL(baseURL))
}

// LocalAIProviderFactory creates LocalAI instances from provider config.
func LocalAIProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}

	return NewLocalAI(modelID, baseURL)
}

// FastChatProviderFactory creates FastChat instances from provider config.
func FastChatProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:8000"
	}

	return NewFastChat(modelID, baseURL)
}
