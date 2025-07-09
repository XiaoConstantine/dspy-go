package core

import (
	"context"
	"fmt"
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
	// New multimodal capabilities.
	CapabilityMultimodal Capability = "multimodal"
	CapabilityVision     Capability = "vision"
	CapabilityAudio      Capability = "audio"
)

// It's provider-agnostic - each LLM provider handles its own format conversion.
type ContentBlock struct {
	Type     FieldType `json:"type"`
	Text     string    `json:"text,omitempty"`
	Data     []byte    `json:"-"` // Raw binary data for images/audio
	MimeType string    `json:"mime_type,omitempty"`
	// Optional metadata for extensibility
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// NewTextBlock creates a text content block.
func NewTextBlock(text string) ContentBlock {
	return ContentBlock{
		Type: FieldTypeText,
		Text: text,
	}
}

// NewImageBlock creates an image content block.
func NewImageBlock(data []byte, mimeType string) ContentBlock {
	return ContentBlock{
		Type:     FieldTypeImage,
		Data:     data,
		MimeType: mimeType,
	}
}

// NewAudioBlock creates an audio content block.
func NewAudioBlock(data []byte, mimeType string) ContentBlock {
	return ContentBlock{
		Type:     FieldTypeAudio,
		Data:     data,
		MimeType: mimeType,
	}
}

// String returns a string representation of the content block.
func (cb ContentBlock) String() string {
	switch cb.Type {
	case FieldTypeText:
		return cb.Text
	case FieldTypeImage:
		return fmt.Sprintf("[Image: %s, %d bytes]", cb.MimeType, len(cb.Data))
	case FieldTypeAudio:
		return fmt.Sprintf("[Audio: %s, %d bytes]", cb.MimeType, len(cb.Data))
	default:
		return fmt.Sprintf("[Unknown content type: %s]", cb.Type)
	}
}

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

	// Multimodal methods - new additions that don't break existing interface
	GenerateWithContent(ctx context.Context, content []ContentBlock, options ...GenerateOption) (*LLMResponse, error)
	StreamGenerateWithContent(ctx context.Context, content []ContentBlock, options ...GenerateOption) (*StreamResponse, error)

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

// Default implementations for multimodal methods
// These provide fallback behavior for LLM implementations that don't support multimodal content

// Default implementations for multimodal methods that can be embedded in concrete LLM implementations

// Concrete LLM implementations should override this if they support multimodal content.
func (b *BaseLLM) GenerateWithContent(ctx context.Context, content []ContentBlock, options ...GenerateOption) (*LLMResponse, error) {
	return nil, errors.New(errors.UnsupportedOperation, "multimodal content not supported by this LLM provider")
}

// Concrete LLM implementations should override this if they support multimodal streaming.
func (b *BaseLLM) StreamGenerateWithContent(ctx context.Context, content []ContentBlock, options ...GenerateOption) (*StreamResponse, error) {
	return nil, errors.New(errors.UnsupportedOperation, "multimodal streaming not supported by this LLM provider")
}

// Helper functions for converting between map[string]any and ContentBlock formats
// These maintain backward compatibility with existing code

// This enables backward compatibility while supporting multimodal content.
func ConvertInputsToContentBlocks(signature Signature, inputs map[string]any) []ContentBlock {
	var blocks []ContentBlock

	for _, field := range signature.Inputs {
		value, exists := inputs[field.Name]
		if !exists {
			continue
		}

		// Type detection based on field type and value type
		switch field.Type {
		case FieldTypeText:
			// Handle text content
			if str, ok := value.(string); ok {
				blocks = append(blocks, NewTextBlock(str))
			} else {
				// Convert any type to string as fallback
				blocks = append(blocks, NewTextBlock(fmt.Sprintf("%v", value)))
			}
		case FieldTypeImage:
			// Handle image content - check if it's already a ContentBlock or needs conversion
			if block, ok := value.(ContentBlock); ok && block.Type == FieldTypeImage {
				blocks = append(blocks, block)
			} else if str, ok := value.(string); ok {
				// Assume it's a text description of an image for now
				blocks = append(blocks, NewTextBlock(fmt.Sprintf("[Image: %s]", str)))
			}
		case FieldTypeAudio:
			// Handle audio content
			if block, ok := value.(ContentBlock); ok && block.Type == FieldTypeAudio {
				blocks = append(blocks, block)
			} else if str, ok := value.(string); ok {
				// Assume it's a text description of audio for now
				blocks = append(blocks, NewTextBlock(fmt.Sprintf("[Audio: %s]", str)))
			}
		default:
			// Default to text for unknown types
			blocks = append(blocks, NewTextBlock(fmt.Sprintf("%v", value)))
		}
	}

	return blocks
}

// IsMultimodalContent checks if the inputs contain any non-text content.
func IsMultimodalContent(signature Signature, inputs map[string]any) bool {
	for _, field := range signature.Inputs {
		if field.Type != FieldTypeText {
			if value, exists := inputs[field.Name]; exists {
				// Check if it's a ContentBlock with non-text type
				if block, ok := value.(ContentBlock); ok && block.Type != FieldTypeText {
					return true
				}
			}
		}
	}
	return false
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
