package core

import (
	"context"
	"net/http"
	"time"

	"github.com/XiaoConstantine/anthropic-go/anthropic"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

type TokenInfo struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

type LLMResponse struct {
	Content string
	Usage   *TokenInfo
}

type Capability string

const (
	// Core capabilities.
	CapabilityCompletion Capability = "completion"
	CapabilityChat       Capability = "chat"
	CapabilityEmbedding  Capability = "embedding"

	// Advanced capabilities.
	CapabilityJSON        Capability = "json"
	CapabilityStreaming   Capability = "streaming"
	CapabilityToolCalling Capability = "tool-calling"
)

// LLM represents an interface for language models.
type LLM interface {
	// Generate produces text completions based on the given prompt
	Generate(ctx context.Context, prompt string, options ...GenerateOption) (*LLMResponse, error)

	// GenerateWithJSON produces structured JSON output based on the given prompt
	GenerateWithJSON(ctx context.Context, prompt string, options ...GenerateOption) (map[string]interface{}, error)

	ProviderName() string
	ModelID() string
	Capabilities() []Capability
}

// GenerateOption represents an option for text generation.
type GenerateOption func(*GenerateOptions)

// GenerateOptions holds configuration for text generation.
type GenerateOptions struct {
	MaxTokens        int
	Temperature      float64
	TopP             float64
	PresencePenalty  float64
	FrequencyPenalty float64
	Stop             []string
}

// NewGenerateOptions creates a new GenerateOptions with default values.
func NewGenerateOptions() *GenerateOptions {
	return &GenerateOptions{
		MaxTokens:   4096, // Default max tokens
		Temperature: 0.5,  // Default temperature
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) GenerateOption {
	return func(o *GenerateOptions) {
		o.MaxTokens = n
	}
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.Temperature = t
	}
}

// WithTopP sets the nucleus sampling probability.
func WithTopP(p float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.TopP = p
	}
}

// WithPresencePenalty sets the presence penalty.
func WithPresencePenalty(p float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.PresencePenalty = p
	}
}

// WithFrequencyPenalty sets the frequency penalty.
func WithFrequencyPenalty(p float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.FrequencyPenalty = p
	}
}

// WithStopSequences sets the stop sequences.
func WithStopSequences(sequences ...string) GenerateOption {
	return func(o *GenerateOptions) {
		o.Stop = sequences
	}
}

type EndpointConfig struct {
	BaseURL    string            // Base API URL
	Path       string            // Specific endpoint path
	Headers    map[string]string // Common headers
	TimeoutSec int               // Request timeout in seconds
}

// BaseLLM provides a base implementation of the LLM interface.
type BaseLLM struct {
	providerName string
	modelID      ModelID
	capabilities []Capability

	endpoint *EndpointConfig // Optional endpoint configuration
	client   *http.Client    // Common HTTP client
}

// ProviderName implements LLM interface.
func (b *BaseLLM) ProviderName() string {
	return b.providerName
}

// ModelID implements LLM interface.
func (b *BaseLLM) ModelID() string {
	return string(b.modelID)
}

// Capabilities implements LLM interface.
func (b *BaseLLM) Capabilities() []Capability {
	return b.capabilities
}

func NewBaseLLM(providerName string, modelID ModelID, capabilities []Capability, endpoint *EndpointConfig) *BaseLLM {

	var timeout time.Duration
	if endpoint != nil && endpoint.TimeoutSec >= 0 {
		timeout = time.Duration(endpoint.TimeoutSec) * time.Second
	} else {
		timeout = 30 * time.Second
	}

	client := &http.Client{
		Timeout: timeout,
	}
	return &BaseLLM{
		providerName: providerName,
		modelID:      modelID,
		capabilities: capabilities,
		endpoint:     endpoint,
		client:       client,
	}
}

func ValidateEndpointConfig(cfg *EndpointConfig) error {
	if cfg == nil {
		return nil // Valid to have no endpoint config
	}

	if cfg.BaseURL == "" {
		return errors.New(errors.InvalidInput, "base URL required in endpoint configuration")
	}

	if cfg.TimeoutSec <= 0 {
		cfg.TimeoutSec = 30 // Default timeout
	}

	return nil
}

// Generate is a placeholder implementation and should be overridden by specific LLM implementations.
func (b *BaseLLM) Generate(ctx context.Context, prompt string, options ...GenerateOption) (string, error) {
	return "", errors.New(errors.Unknown, "Generate method not implemented")
}

// GenerateWithJSON is a placeholder implementation and should be overridden by specific LLM implementations.
func (b *BaseLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...GenerateOption) (map[string]interface{}, error) {
	return nil, errors.New(errors.Unknown, "GenerateWithJSON method not implemented")
}

// GetEndpointConfig returns the current endpoint configuration.
func (b *BaseLLM) GetEndpointConfig() *EndpointConfig {
	return b.endpoint
}

// GetHTTPClient returns the HTTP client.
func (b *BaseLLM) GetHTTPClient() *http.Client {
	return b.client
}

// LLMFactory is a function type for creating LLM instances.
type LLMFactory func() (LLM, error)

// LLMRegistry maintains a registry of available LLM implementations.
type LLMRegistry struct {
	factories map[string]LLMFactory
}

// NewLLMRegistry creates a new LLMRegistry.
func NewLLMRegistry() *LLMRegistry {
	return &LLMRegistry{
		factories: make(map[string]LLMFactory),
	}
}

// Register adds a new LLM factory to the registry.
func (r *LLMRegistry) Register(name string, factory LLMFactory) {
	r.factories[name] = factory
}

// Create instantiates a new LLM based on the given name.
func (r *LLMRegistry) Create(name string) (LLM, error) {
	factory, exists := r.factories[name]
	if !exists {
		return nil, errors.New(errors.Unknown, "unknown LLM type: "+name)
	}
	return factory()
}

// DefaultLLM represents the default LLM to be used when none is specified.
var DefaultLLM LLM

// SetDefaultLLM sets the default LLM.
func SetDefaultLLM(llm LLM) {
	DefaultLLM = llm
}

// ModelID represents the available model IDs.
type ModelID string

const (
	// Anthropic models.
	ModelAnthropicHaiku    ModelID = ModelID(anthropic.ModelHaiku)
	ModelAnthropicSonnet   ModelID = ModelID(anthropic.ModelSonnet)
	ModelAnthropicOpus     ModelID = ModelID(anthropic.ModelOpus)
	ModelGoogleGeminiFlash ModelID = "gemini-2.0-flash-exp"
	ModelGoogleGeminiPro   ModelID = "gemini-1.5-pro"
)
