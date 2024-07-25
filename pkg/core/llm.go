package core

import (
	"context"
	"errors"
)

// LLM represents an interface for language models
type LLM interface {
	// Generate produces text completions based on the given prompt
	Generate(ctx context.Context, prompt string, options ...GenerateOption) (string, error)

	// GenerateWithJSON produces structured JSON output based on the given prompt
	GenerateWithJSON(ctx context.Context, prompt string, options ...GenerateOption) (map[string]interface{}, error)
}

// GenerateOption represents an option for text generation
type GenerateOption func(*GenerateOptions)

// GenerateOptions holds configuration for text generation
type GenerateOptions struct {
	MaxTokens        int
	Temperature      float64
	TopP             float64
	PresencePenalty  float64
	FrequencyPenalty float64
	Stop             []string
}

// WithMaxTokens sets the maximum number of tokens to generate
func WithMaxTokens(n int) GenerateOption {
	return func(o *GenerateOptions) {
		o.MaxTokens = n
	}
}

// WithTemperature sets the sampling temperature
func WithTemperature(t float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.Temperature = t
	}
}

// WithTopP sets the nucleus sampling probability
func WithTopP(p float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.TopP = p
	}
}

// WithPresencePenalty sets the presence penalty
func WithPresencePenalty(p float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.PresencePenalty = p
	}
}

// WithFrequencyPenalty sets the frequency penalty
func WithFrequencyPenalty(p float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.FrequencyPenalty = p
	}
}

// WithStopSequences sets the stop sequences
func WithStopSequences(sequences ...string) GenerateOption {
	return func(o *GenerateOptions) {
		o.Stop = sequences
	}
}

// BaseLLM provides a base implementation of the LLM interface
type BaseLLM struct {
	Name string
}

// Generate is a placeholder implementation and should be overridden by specific LLM implementations
func (b *BaseLLM) Generate(ctx context.Context, prompt string, options ...GenerateOption) (string, error) {
	return "", errors.New("Generate method not implemented")
}

// GenerateWithJSON is a placeholder implementation and should be overridden by specific LLM implementations
func (b *BaseLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...GenerateOption) (map[string]interface{}, error) {
	return nil, errors.New("GenerateWithJSON method not implemented")
}

// LLMFactory is a function type for creating LLM instances
type LLMFactory func() (LLM, error)

// LLMRegistry maintains a registry of available LLM implementations
type LLMRegistry struct {
	factories map[string]LLMFactory
}

// NewLLMRegistry creates a new LLMRegistry
func NewLLMRegistry() *LLMRegistry {
	return &LLMRegistry{
		factories: make(map[string]LLMFactory),
	}
}

// Register adds a new LLM factory to the registry
func (r *LLMRegistry) Register(name string, factory LLMFactory) {
	r.factories[name] = factory
}

// Create instantiates a new LLM based on the given name
func (r *LLMRegistry) Create(name string) (LLM, error) {
	factory, exists := r.factories[name]
	if !exists {
		return nil, errors.New("unknown LLM type: " + name)
	}
	return factory()
}

// DefaultLLM represents the default LLM to be used when none is specified
var DefaultLLM LLM

// SetDefaultLLM sets the default LLM
func SetDefaultLLM(llm LLM) {
	DefaultLLM = llm
}
