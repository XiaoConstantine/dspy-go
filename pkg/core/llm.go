package core

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

type TokenInfo struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

type LLMResponse struct {
	Content  string
	Usage    *TokenInfo
	Metadata map[string]interface{}
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

// TransportConfig configures HTTP connection pooling behavior for LLM requests.
// Tuning these values can significantly improve performance for parallel workloads.
type TransportConfig struct {
	MaxIdleConns        int           // Total idle connections across all hosts (default: 100)
	MaxIdleConnsPerHost int           // Idle connections per host (default: 100)
	MaxConnsPerHost     int           // Max concurrent connections per host (default: 100)
	IdleConnTimeout     time.Duration // How long idle connections stay open (default: 90s)
	TLSHandshakeTimeout time.Duration // TLS handshake timeout (default: 10s)
}

// DefaultTransportConfig returns optimized defaults for parallel LLM workloads.
// These values support up to 100 concurrent requests to the same API endpoint.
func DefaultTransportConfig() TransportConfig {
	return TransportConfig{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		MaxConnsPerHost:     100,
		IdleConnTimeout:     90 * time.Second,
		TLSHandshakeTimeout: 10 * time.Second,
	}
}

// ToTransport converts the config to an http.Transport.
func (tc TransportConfig) ToTransport() *http.Transport {
	return &http.Transport{
		MaxIdleConns:        tc.MaxIdleConns,
		MaxIdleConnsPerHost: tc.MaxIdleConnsPerHost,
		MaxConnsPerHost:     tc.MaxConnsPerHost,
		IdleConnTimeout:     tc.IdleConnTimeout,
		TLSHandshakeTimeout: tc.TLSHandshakeTimeout,
	}
}

// WorkloadType indicates whether LLM calls are I/O-bound or CPU-bound.
// This affects the optimal number of parallel workers.
type WorkloadType int

const (
	// WorkloadIOBound indicates remote API calls (Gemini, OpenAI, Anthropic).
	// I/O-bound workloads benefit from many concurrent workers (default: 100).
	WorkloadIOBound WorkloadType = iota

	// WorkloadCPUBound indicates local model inference (Ollama, llama.cpp).
	// CPU-bound workloads should use NumCPU() workers to avoid thrashing.
	WorkloadCPUBound
)

// BaseLLMOption configures BaseLLM behavior.
type BaseLLMOption func(*BaseLLM)

// WithTransportConfig sets custom HTTP transport configuration.
// Use this for high-throughput scenarios that need more concurrent connections.
func WithTransportConfig(config TransportConfig) BaseLLMOption {
	return func(b *BaseLLM) {
		b.client.Transport = config.ToTransport()
	}
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

// IsLocal returns true if this LLM runs locally (CPU-bound inference).
func (b *BaseLLM) IsLocal() bool {
	switch b.providerName {
	case "ollama", "llamacpp":
		return true
	default:
		return false
	}
}

// WorkloadType returns the workload classification based on provider.
// Remote APIs are I/O-bound (network latency), local models are CPU-bound.
func (b *BaseLLM) WorkloadType() WorkloadType {
	if b.IsLocal() {
		return WorkloadCPUBound
	}
	return WorkloadIOBound
}

func NewBaseLLM(providerName string, modelID ModelID, capabilities []Capability, endpoint *EndpointConfig, opts ...BaseLLMOption) *BaseLLM {
	var timeout time.Duration
	if endpoint != nil && endpoint.TimeoutSec >= 0 {
		timeout = time.Duration(endpoint.TimeoutSec) * time.Second
	} else {
		timeout = 30 * time.Second
	}

	// Use default transport config optimized for parallel LLM workloads
	client := &http.Client{
		Timeout:   timeout,
		Transport: DefaultTransportConfig().ToTransport(),
	}

	llm := &BaseLLM{
		providerName: providerName,
		modelID:      modelID,
		capabilities: capabilities,
		endpoint:     endpoint,
		client:       client,
	}

	// Apply custom options (e.g., WithTransportConfig)
	for _, opt := range opts {
		opt(llm)
	}

	return llm
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
// It creates a structured prompt that includes field instructions and prefix information.
func ConvertInputsToContentBlocks(signature Signature, inputs map[string]any) []ContentBlock {
	var blocks []ContentBlock

	// Add instruction block that explains the expected format
	instructionText := fmt.Sprintf("Given the fields '%s', produce the fields '%s'.\n\n",
		JoinFieldNames(InputFieldsToFields(signature.Inputs)),
		JoinFieldNames(OutputFieldsToFields(signature.Outputs)),
	)

	// Add prefix instructions for output fields
	for _, field := range signature.Outputs {
		if field.Prefix != "" {
			instructionText += fmt.Sprintf("The %s field should start with '%s' followed by the content on new lines.\n",
				field.Name, field.Prefix)
		}
		if field.Description != "" {
			instructionText += fmt.Sprintf(" %s", field.Description)
			instructionText += "\n"
		}
	}
	instructionText += "\n"

	// Add the main signature instruction if present
	if signature.Instruction != "" {
		instructionText += signature.Instruction + "\n\n"
	}

	// Add conversation context if present
	if context, ok := inputs["conversation_context"].(string); ok && context != "" {
		instructionText += "===== CONVERSATION HISTORY =====\n"
		instructionText += context
		instructionText += "\n===== END HISTORY =====\n\n"
	}

	// Add the instruction block
	blocks = append(blocks, NewTextBlock(instructionText))

	// Add input field separator
	blocks = append(blocks, NewTextBlock("---\n"))

	// Process input fields in signature order
	for _, field := range signature.Inputs {
		value, exists := inputs[field.Name]
		if !exists {
			continue
		}

		// Add field label
		blocks = append(blocks, NewTextBlock(fmt.Sprintf("%s: ", field.Name)))

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

		// Add newline after each field
		blocks = append(blocks, NewTextBlock("\n"))
	}

	return blocks
}

// Helper functions for field processing in core package

// JoinFieldNames joins field names with commas.
func JoinFieldNames(fields []Field) string {
	names := make([]string, len(fields))
	for i, field := range fields {
		names[i] = field.Name
	}
	return strings.Join(names, ", ")
}

// InputFieldsToFields converts InputField slice to Field slice.
func InputFieldsToFields(inputs []InputField) []Field {
	fields := make([]Field, len(inputs))
	for i, input := range inputs {
		fields[i] = input.Field
	}
	return fields
}

// OutputFieldsToFields converts OutputField slice to Field slice.
func OutputFieldsToFields(outputs []OutputField) []Field {
	fields := make([]Field, len(outputs))
	for i, output := range outputs {
		fields[i] = output.Field
	}
	return fields
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

// SetDefaultLLM sets the default LLM.
func SetDefaultLLM(llm LLM) {
	GlobalConfig.DefaultLLM = llm
}

// ModelID represents the available model IDs.
type ModelID string

const (
	// Anthropic models - Claude 3.x and newer series.
	ModelAnthropicHaiku               ModelID = ModelID(anthropic.ModelClaude_3_Haiku_20240307)
	ModelAnthropicSonnet              ModelID = ModelID(anthropic.ModelClaudeSonnet4_5_20250929) // For backwards compatibility
	ModelAnthropicSonnet35            ModelID = ModelID(anthropic.ModelClaudeSonnet4_5_20250929)
	ModelAnthropicOpus                ModelID = ModelID(anthropic.ModelClaudeOpus4_1_20250805)
	ModelAnthropicClaude4Opus         ModelID = ModelID(anthropic.ModelClaudeOpus4_1_20250805)
	ModelAnthropicClaude4Sonnet       ModelID = ModelID(anthropic.ModelClaudeSonnet4_5_20250929)
	ModelAnthropicClaude45Sonnet      ModelID = ModelID(anthropic.ModelClaudeSonnet4_5_20250929)
	ModelAnthropicClaude45Opus        ModelID = ModelID(anthropic.ModelClaudeOpus4_5_20251101)

	// Google Gemini models (existing).
	ModelGoogleGeminiFlash     ModelID = "gemini-2.5-flash"
	ModelGoogleGeminiPro       ModelID = "gemini-2.5-pro"
	ModelGoogleGeminiFlashLite ModelID = "gemini-2.5-flash-lite"
	// Google Gemini 3 series (new).
	ModelGoogleGemini3ProPreview   ModelID = "gemini-3-pro-preview"
	ModelGoogleGemini3FlashPreview ModelID = "gemini-3-flash-preview"
	// Google Gemini 2.0 series (new).
	ModelGoogleGemini20Flash     ModelID = "gemini-2.0-flash"
	ModelGoogleGemini20FlashLite ModelID = "gemini-2.0-flash-lite"

	// OpenAI models - GPT-4 series.
	ModelOpenAIGPT4       ModelID = "gpt-4"
	ModelOpenAIGPT4Turbo  ModelID = "gpt-4-turbo"
	ModelOpenAIGPT35Turbo ModelID = "gpt-3.5-turbo"
	ModelOpenAIGPT4o      ModelID = "gpt-4o"
	ModelOpenAIGPT4oMini  ModelID = "gpt-4o-mini"
	// OpenAI models - GPT-4.1 series.
	ModelOpenAIGPT41     ModelID = "gpt-4.1"
	ModelOpenAIGPT41Mini ModelID = "gpt-4.1-mini"
	ModelOpenAIGPT41Nano ModelID = "gpt-4.1-nano"
	// OpenAI models - o1 reasoning series.
	ModelOpenAIO1     ModelID = "o1"
	ModelOpenAIO1Pro  ModelID = "o1-pro"
	ModelOpenAIO1Mini ModelID = "o1-mini"
	// OpenAI models - o3 reasoning series.
	ModelOpenAIO3     ModelID = "o3"
	ModelOpenAIO3Mini ModelID = "o3-mini"
	// OpenAI models - GPT-5 series.
	ModelOpenAIGPT5     ModelID = "gpt-5"
	ModelOpenAIGPT5Mini ModelID = "gpt-5-mini"
	ModelOpenAIGPT5Nano ModelID = "gpt-5-nano"

	// LiteLLM models (can proxy to any provider).
	ModelLiteLLMGPT4    ModelID = "gpt-4"
	ModelLiteLLMClaude3 ModelID = "claude-3-sonnet-20240229"
	ModelLiteLLMLlama2  ModelID = "llama-2-70b-chat"
	ModelLiteLLMGemini  ModelID = "gemini-pro"

	// Ollama models (OpenAI-compatible mode).
	ModelOllamaLlama3_8B    ModelID = "llama3:8b"
	ModelOllamaLlama3_70B   ModelID = "llama3:70b"
	ModelOllamaLlama3_1_8B  ModelID = "llama3.1:8b"
	ModelOllamaLlama3_1_70B ModelID = "llama3.1:70b"
	ModelOllamaLlama3_2_3B  ModelID = "llama3.2:3b"
	ModelOllamaCodeLlama13B ModelID = "codellama:13b"
	ModelOllamaCodeLlama34B ModelID = "codellama:34b"
	ModelOllamaMistral7B    ModelID = "mistral:7b"
	ModelOllamaGemma2B      ModelID = "gemma:2b"
	ModelOllamaGemma7B      ModelID = "gemma:7b"
	ModelOllamaQwen2_5_7B   ModelID = "qwen2.5:7b"
	ModelOllamaQwen2_5_14B  ModelID = "qwen2.5:14b"

	// Ollama embedding models.
	ModelOllamaNomicEmbed ModelID = "nomic-embed-text"
	ModelOllamaMxbaiEmbed ModelID = "mxbai-embed-large"
	ModelOllamaAllMiniLM  ModelID = "all-minilm"

	// LocalAI models.
	ModelLocalAILlama2    ModelID = "llama-2-7b-chat"
	ModelLocalAICodeLlama ModelID = "codellama-13b-instruct"
	ModelLocalAIAlpaca    ModelID = "alpaca-7b"
	ModelLocalAIVicuna    ModelID = "vicuna-7b"

	// FastChat models.
	ModelFastChatVicuna    ModelID = "vicuna-7b-v1.5"
	ModelFastChatAlpaca    ModelID = "alpaca-13b"
	ModelFastChatCodeLlama ModelID = "codellama-7b-instruct"
	ModelFastChatLlama2    ModelID = "llama-2-7b-chat"

	// Generic OpenAI-compatible models.
	ModelCompatibleGPT4   ModelID = "gpt-4"
	ModelCompatibleGPT35  ModelID = "gpt-3.5-turbo"
	ModelCompatibleClaude ModelID = "claude-3-sonnet"
	ModelCompatibleLlama2 ModelID = "llama-2-chat"
)

var ProviderModels = map[string][]ModelID{
	"anthropic": {
		ModelAnthropicSonnet, ModelAnthropicHaiku, ModelAnthropicOpus,
		ModelAnthropicClaude4Opus, ModelAnthropicClaude4Sonnet, ModelAnthropicClaude45Sonnet,
		ModelAnthropicClaude45Opus,
	},
	"google": {
		ModelGoogleGeminiFlash, ModelGoogleGeminiPro, ModelGoogleGeminiFlashLite,
		ModelGoogleGemini3ProPreview, ModelGoogleGemini3FlashPreview,
		ModelGoogleGemini20Flash, ModelGoogleGemini20FlashLite,
	},
	"openai": {
		ModelOpenAIGPT4, ModelOpenAIGPT4Turbo, ModelOpenAIGPT35Turbo, ModelOpenAIGPT4o, ModelOpenAIGPT4oMini,
		ModelOpenAIGPT41, ModelOpenAIGPT41Mini, ModelOpenAIGPT41Nano,
		ModelOpenAIO1, ModelOpenAIO1Pro, ModelOpenAIO1Mini,
		ModelOpenAIO3, ModelOpenAIO3Mini,
		ModelOpenAIGPT5, ModelOpenAIGPT5Mini, ModelOpenAIGPT5Nano,
	},
	"ollama": {
		ModelOllamaLlama3_8B, ModelOllamaLlama3_70B, ModelOllamaLlama3_1_8B, ModelOllamaLlama3_1_70B,
		ModelOllamaLlama3_2_3B, ModelOllamaCodeLlama13B, ModelOllamaCodeLlama34B, ModelOllamaMistral7B,
		ModelOllamaGemma2B, ModelOllamaGemma7B, ModelOllamaQwen2_5_7B, ModelOllamaQwen2_5_14B,
		ModelOllamaNomicEmbed, ModelOllamaMxbaiEmbed, ModelOllamaAllMiniLM,
	},
	"llamacpp":          {},
	"litellm":           {ModelLiteLLMGPT4, ModelLiteLLMClaude3, ModelLiteLLMLlama2, ModelLiteLLMGemini},
	"localai":           {ModelLocalAILlama2, ModelLocalAICodeLlama, ModelLocalAIAlpaca, ModelLocalAIVicuna},
	"fastchat":          {ModelFastChatVicuna, ModelFastChatAlpaca, ModelFastChatCodeLlama, ModelFastChatLlama2},
	"openai_compatible": {ModelCompatibleGPT4, ModelCompatibleGPT35, ModelCompatibleClaude, ModelCompatibleLlama2},
}
