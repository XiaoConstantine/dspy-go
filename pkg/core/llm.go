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

type StreamChunk struct {
	Content string     // The text content of this chunk
	Done    bool       // Indicates if this is the final chunk
	Error   error      // Any error that occurred during streaming
	Usage   *TokenInfo // Optional token usage information (may be nil)
}

// StreamResponse encapsulates a streaming response.
type StreamResponse struct {
	ChunkChannel <-chan StreamChunk // Channel receiving response chunks
	Cancel       func()             // Function to cancel the stream
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

	GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...GenerateOption) (map[string]interface{}, error)
	CreateEmbedding(ctx context.Context, input string, options ...EmbeddingOption) (*EmbeddingResult, error)
	CreateEmbeddings(ctx context.Context, inputs []string, options ...EmbeddingOption) (*BatchEmbeddingResult, error)

	StreamGenerate(ctx context.Context, prompt string, options ...GenerateOption) (*StreamResponse, error)

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

type EmbeddingOptions struct {
	// Model-specific options for embedding
	Model string
	// Optional batch size for bulk embeddings
	BatchSize int
	// Additional model-specific parameters
	Params map[string]interface{}
}

// EmbeddingResult represents the result of embedding generation.
type EmbeddingResult struct {
	// The generated embedding vector
	Vector []float32
	// Token count and other metadata
	TokenCount int
	// Any model-specific metadata
	Metadata map[string]interface{}
}

// BatchEmbeddingResult represents results for multiple inputs.
type BatchEmbeddingResult struct {
	// Embeddings for each input
	Embeddings []EmbeddingResult
	// Any error that occurred during processing
	Error error
	// Input index that caused the error (if applicable)
	ErrorIndex int
}

// EmbeddingOption allows for optional parameters.
type EmbeddingOption func(*EmbeddingOptions)

// NewGenerateOptions creates a new GenerateOptions with default values.
func NewGenerateOptions() *GenerateOptions {
	return &GenerateOptions{
		MaxTokens:   8192, // Default max tokens
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

func WithModel(model string) EmbeddingOption {
	return func(o *EmbeddingOptions) {
		o.Model = model
	}
}

func WithBatchSize(size int) EmbeddingOption {
	return func(o *EmbeddingOptions) {
		o.BatchSize = size
	}
}

func WithParams(params map[string]interface{}) EmbeddingOption {
	return func(o *EmbeddingOptions) {
		if o.Params == nil {
			o.Params = make(map[string]interface{})
		}
		for k, v := range params {
			o.Params[k] = v
		}
	}
}

// Default options for embeddings.
func NewEmbeddingOptions() *EmbeddingOptions {
	return &EmbeddingOptions{
		BatchSize: 32, // Default batch size
		Params:    make(map[string]interface{}),
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

// GetEndpointConfig returns the current endpoint configuration.
func (b *BaseLLM) GetEndpointConfig() *EndpointConfig {
	return b.endpoint
}

// GetHTTPClient returns the HTTP client.
func (b *BaseLLM) GetHTTPClient() *http.Client {
	return b.client
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
	ModelAnthropicHaiku            ModelID = ModelID(anthropic.ModelHaiku)
	ModelAnthropicSonnet           ModelID = ModelID(anthropic.ModelSonnet)
	ModelAnthropicOpus             ModelID = ModelID(anthropic.ModelOpus)
	ModelGoogleGeminiFlash         ModelID = "gemini-2.0-flash"
	ModelGoogleGeminiPro           ModelID = "gemini-2.5-pro-exp-03-25"
	ModelGoogleGeminiFlashThinking ModelID = "gemini-2.0-flash-thinking-exp"
	ModelGoogleGeminiFlashLite     ModelID = "gemini-2.0-flash-lite"
)

var ProviderModels = map[string][]ModelID{
	"anthropic": {ModelAnthropicSonnet, ModelAnthropicHaiku, ModelAnthropicOpus},
	"google":    {ModelGoogleGeminiFlash, ModelGoogleGeminiPro, ModelGoogleGeminiFlashThinking, ModelGoogleGeminiFlashLite},
	"ollama":    {},
	"llamacpp":  {},
}
