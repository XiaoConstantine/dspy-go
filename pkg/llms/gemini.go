package llms

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// GeminiLLM implements the core.LLM interface for Google's Gemini model.
type GeminiLLM struct {
	*core.BaseLLM
	apiKey string
}

// GeminiRequest represents the request structure for Gemini API.
type geminiRequest struct {
	Contents         []geminiContent        `json:"contents"`
	GenerationConfig geminiGenerationConfig `json:"generationConfig,omitempty"`
	Tools            []geminiTool           `json:"tools,omitempty"`
	ToolConfig       *geminiToolConfig      `json:"toolConfig,omitempty"`
}

// Add these new types to support function calling.
type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations"`
}

type geminiToolConfig struct {
	FunctionCallingConfig *geminiFunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type geminiFunctionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type geminiFunctionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters"`
}
type geminiContent struct {
	Parts []geminiPart `json:"parts"`
	Role  string       `json:"role,omitempty"`
}

type geminiPart struct {
	Text             string                      `json:"text,omitempty"`
	InlineData       *geminiInlineData           `json:"inlineData,omitempty"`
	FileData         *geminiFileData             `json:"fileData,omitempty"`
	FunctionCall     *geminiFunctionCall         `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponsePart `json:"functionResponse,omitempty"`
	Thought          bool                        `json:"thought,omitempty"`
	ThoughtSignature string                      `json:"thoughtSignature,omitempty"`
}

type geminiFunctionResponsePart struct {
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response"`
}

// geminiInlineData represents inline binary data (base64 encoded).
type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64 encoded
}

// geminiFileData represents a file uploaded to Gemini (for large files).
type geminiFileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

type geminiGenerationConfig struct {
	Temperature     float64               `json:"temperature,omitempty"`
	MaxOutputTokens int                   `json:"maxOutputTokens,omitempty"`
	TopP            float64               `json:"topP,omitempty"`
	ThinkingConfig  *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

type geminiThinkingConfig struct {
	IncludeThoughts bool `json:"includeThoughts,omitempty"`
}

// GeminiResponse represents the response structure from Gemini API.
type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []geminiPart `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason,omitempty"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
		ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
	} `json:"usageMetadata"`
}

type geminiFunctionCallResponse struct {
	Candidates []struct {
		Content struct {
			Parts []geminiPart `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason,omitempty"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
		ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
	} `json:"usageMetadata"`
	PromptFeedback map[string]any `json:"promptFeedback,omitempty"`
}
type geminiFunctionCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"args"`
}

// Request and response structures for Gemini embeddings.
type geminiEmbeddingRequest struct {
	Model   string `json:"model"`
	Content struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"content"`
	// Task type helps the model generate appropriate embeddings
	TaskType string `json:"taskType,omitempty"`
	// Additional configuration
	Title      string                 `json:"title,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type geminiBatchEmbeddingRequest struct {
	Model    string `json:"model"`
	Requests []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"requests"`
	TaskType   string                 `json:"taskType,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type geminiEmbeddingResponse struct {
	Embedding struct {
		Values []float32 `json:"values"`
		// Statistics about the generated embedding
		Statistics struct {
			TruncatedInputTokenCount int `json:"truncatedInputTokenCount"`
			TokenCount               int `json:"tokenCount"`
		} `json:"statistics"`
	} `json:"embedding"`
	// Usage information
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

type geminiBatchEmbeddingResponse struct {
	Embeddings []geminiEmbeddingResponse `json:"embeddings"`
}

// NewGeminiLLM creates a new GeminiLLM instance.
func NewGeminiLLM(apiKey string, model core.ModelID) (*GeminiLLM, error) {
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY") // or whatever env var name you prefer
		if apiKey == "" {
			return nil, errors.New(errors.InvalidInput, "API key is required")
		}

	}

	if model == "" {
		model = core.ModelGoogleGeminiFlash // Default model
	}
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
		core.CapabilityEmbedding,
		core.CapabilityMultimodal,
		core.CapabilityVision,
		core.CapabilityAudio,
	}
	// Validate model ID
	switch model {
	case core.ModelGoogleGeminiPro, core.ModelGoogleGeminiFlash, core.ModelGoogleGeminiFlashLite,
		core.ModelGoogleGemini3ProPreview, core.ModelGoogleGemini3FlashPreview,
		core.ModelGoogleGemini20Flash, core.ModelGoogleGemini20FlashLite:
		break
	default:
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, fmt.Sprintf("unsupported Gemini model: %s", model)),
			errors.Fields{"model": model})
	}
	endpoint := &core.EndpointConfig{
		BaseURL: "https://generativelanguage.googleapis.com/v1beta",
		Path:    fmt.Sprintf("/models/%s:generateContent", model),
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		TimeoutSec: 10 * 60,
	}

	return &GeminiLLM{
		apiKey:  apiKey,
		BaseLLM: core.NewBaseLLM("google", model, capabilities, endpoint),
	}, nil
}

// NewGeminiLLMFromConfig creates a new GeminiLLM instance from configuration.
func NewGeminiLLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*GeminiLLM, error) {
	// Get API key from config or environment
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New(errors.InvalidInput, "API key is required")
	}

	// Use default model if none specified
	if modelID == "" {
		modelID = core.ModelGoogleGeminiFlash
	}

	// Validate model ID
	if !isValidGeminiModel(modelID) {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "unsupported Gemini model"),
			errors.Fields{"model": modelID})
	}

	// Create endpoint configuration
	baseURL := "https://generativelanguage.googleapis.com/v1beta"
	if config.BaseURL != "" {
		baseURL = config.BaseURL
	}

	endpoint := &core.EndpointConfig{
		BaseURL: baseURL,
		Path:    fmt.Sprintf("/models/%s:generateContent", modelID),
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		TimeoutSec: 10 * 60,
	}

	// Override with config endpoint if provided
	if config.Endpoint != nil {
		if config.Endpoint.BaseURL != "" {
			endpoint.BaseURL = config.Endpoint.BaseURL
		}
		if config.Endpoint.TimeoutSec > 0 {
			endpoint.TimeoutSec = config.Endpoint.TimeoutSec
		}
		for k, v := range config.Endpoint.Headers {
			endpoint.Headers[k] = v
		}
	}

	// Set capabilities
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
		core.CapabilityEmbedding,
		core.CapabilityMultimodal,
		core.CapabilityVision,
		core.CapabilityAudio,
	}

	// Check if streaming is supported
	if supportsGeminiStreaming(modelID) {
		capabilities = append(capabilities, core.CapabilityStreaming)
	}

	// Check if function calling is supported
	if supportsGeminiFunctionCalling(modelID) {
		capabilities = append(capabilities, core.CapabilityToolCalling)
	}

	return &GeminiLLM{
		apiKey:  apiKey,
		BaseLLM: core.NewBaseLLM("google", modelID, capabilities, endpoint),
	}, nil
}

// GeminiProviderFactory creates GeminiLLM instances.
func GeminiProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewGeminiLLMFromConfig(ctx, config, modelID)
}

// validGeminiModels is the list of supported Gemini model IDs.
var validGeminiModels = []core.ModelID{
	// Gemini 2.5 series (existing)
	core.ModelGoogleGeminiFlash,     // gemini-2.5-flash
	core.ModelGoogleGeminiPro,       // gemini-2.5-pro
	core.ModelGoogleGeminiFlashLite, // gemini-2.5-flash-lite
	// Gemini 3 series (new)
	core.ModelGoogleGemini3ProPreview,   // gemini-3-pro-preview
	core.ModelGoogleGemini3FlashPreview, // gemini-3-flash-preview
	// Gemini 2.0 series (new)
	core.ModelGoogleGemini20Flash,     // gemini-2.0-flash
	core.ModelGoogleGemini20FlashLite, // gemini-2.0-flash-lite
}

// isValidGeminiModel checks if the model is a valid Gemini model.
func isValidGeminiModel(modelID core.ModelID) bool {
	return isValidModelInList(modelID, validGeminiModels)
}

// supportsGeminiStreaming checks if the model supports streaming.
// Currently all Gemini models support streaming, but this function is kept
// as a placeholder for future model-specific capability checks.
// The modelID parameter is unused but kept for API consistency.
func supportsGeminiStreaming(_ core.ModelID) bool {
	return true
}

// supportsGeminiFunctionCalling checks if the model supports function calling.
// Currently all Gemini models support function calling, but this function is kept
// as a placeholder for future model-specific capability checks.
// The modelID parameter is unused but kept for API consistency.
func supportsGeminiFunctionCalling(_ core.ModelID) bool {
	return true
}

const (
	geminiThoughtMetadataKey          = "gemini_thought"
	geminiThoughtSignatureMetadataKey = "gemini_thought_signature"
)

func supportsGeminiThoughtSignatures(modelID core.ModelID) bool {
	model := strings.ToLower(strings.TrimSpace(string(modelID)))
	return strings.HasPrefix(model, "gemini-3")
}

// Gemini 3 thought signatures must be round-tripped when tool calling is active.
// We currently enable thoughts automatically for Gemini 3 family models so the
// caller receives the signatures needed for subsequent function-response turns.
func (g *GeminiLLM) buildGenerationConfig(opts *core.GenerateOptions) geminiGenerationConfig {
	cfg := geminiGenerationConfig{
		Temperature:     opts.Temperature,
		MaxOutputTokens: opts.MaxTokens,
		TopP:            opts.TopP,
	}
	if supportsGeminiThoughtSignatures(core.ModelID(g.ModelID())) {
		cfg.ThinkingConfig = &geminiThinkingConfig{IncludeThoughts: true}
	}
	return cfg
}

func geminiUsageToTokenInfo(metadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
	ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
}) *core.TokenInfo {
	return &core.TokenInfo{
		PromptTokens:     metadata.PromptTokenCount,
		CompletionTokens: metadata.CandidatesTokenCount,
		TotalTokens:      metadata.TotalTokenCount,
	}
}

func geminiUsageMetadataMap(metadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
	ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
}) map[string]any {
	result := map[string]any{}
	if metadata.ThoughtsTokenCount > 0 {
		result["thoughts_token_count"] = metadata.ThoughtsTokenCount
	}
	return result
}

func buildGeminiContentResponse(parts []geminiPart) (string, []core.ContentBlock, []core.ToolCall, map[string]any) {
	var (
		textParts     []string
		contentBlocks []core.ContentBlock
		toolCalls     []core.ToolCall
		thoughtBlocks []core.ContentBlock
	)

	for idx, part := range parts {
		if part.Text != "" {
			block := core.NewTextBlock(part.Text)
			if part.Thought {
				if block.Metadata == nil {
					block.Metadata = map[string]any{}
				}
				block.Metadata[geminiThoughtMetadataKey] = true
				if part.ThoughtSignature != "" {
					block.Metadata[geminiThoughtSignatureMetadataKey] = part.ThoughtSignature
				}
				thoughtBlocks = append(thoughtBlocks, block)
			} else {
				textParts = append(textParts, part.Text)
				contentBlocks = append(contentBlocks, block)
			}
		}

		if part.FunctionCall != nil {
			call := core.ToolCall{
				ID:        fmt.Sprintf("gemini-call-%d", idx),
				Name:      part.FunctionCall.Name,
				Arguments: part.FunctionCall.Arguments,
			}
			if part.Thought || part.ThoughtSignature != "" {
				call.Metadata = map[string]any{}
				if part.Thought {
					call.Metadata[geminiThoughtMetadataKey] = true
				}
				if part.ThoughtSignature != "" {
					call.Metadata[geminiThoughtSignatureMetadataKey] = part.ThoughtSignature
				}
			}
			toolCalls = append(toolCalls, call)
		}
	}

	metadata := map[string]any{}
	if len(thoughtBlocks) > 0 {
		metadata["thought_blocks"] = thoughtBlocks
	}
	if len(toolCalls) > 0 {
		metadata["tool_call_count"] = len(toolCalls)
	}

	return strings.Join(textParts, " "), contentBlocks, toolCalls, metadata
}

func (g *GeminiLLM) chatMessagesToGeminiContents(messages []core.ChatMessage) []geminiContent {
	contents := make([]geminiContent, 0, len(messages))
	for _, msg := range messages {
		// Gemini expects tool results as function_response parts rather than a
		// standalone "tool" role. If we somehow receive a tool-role message
		// without a tool result payload, drop it instead of emitting a malformed
		// synthetic user turn.
		if msg.Role == "tool" && msg.ToolResult == nil {
			continue
		}

		content := geminiContent{
			Role:  geminiRoleForChatMessage(msg.Role),
			Parts: make([]geminiPart, 0, len(msg.Content)+len(msg.ToolCalls)+1),
		}

		if msg.ToolResult != nil {
			response := map[string]any{
				"content": contentBlocksToText(msg.ToolResult.Content),
			}
			content.Parts = append(content.Parts, geminiPart{
				FunctionResponse: &geminiFunctionResponsePart{
					Name:     msg.ToolResult.Name,
					Response: response,
				},
			})
		} else {
			content.Parts = append(content.Parts, g.convertToGeminiParts(msg.Content)...)
		}

		for _, toolCall := range msg.ToolCalls {
			part := geminiPart{
				FunctionCall: &geminiFunctionCall{
					Name:      toolCall.Name,
					Arguments: toolCall.Arguments,
				},
			}
			if toolCall.Metadata != nil {
				if thought, ok := toolCall.Metadata[geminiThoughtMetadataKey].(bool); ok {
					part.Thought = thought
				}
				if signature, ok := toolCall.Metadata[geminiThoughtSignatureMetadataKey].(string); ok {
					part.ThoughtSignature = signature
				}
			}
			content.Parts = append(content.Parts, part)
		}

		if len(content.Parts) == 0 {
			continue
		}
		contents = append(contents, content)
	}
	return contents
}

func geminiRoleForChatMessage(role string) string {
	switch role {
	case "assistant":
		return "model"
	default:
		return "user"
	}
}

func (g *GeminiLLM) doGeminiRequest(ctx context.Context, reqBody any, dest any) error {
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return errors.WithFields(
			errors.New(errors.InvalidInput, fmt.Sprintf("failed to marshal Gemini request body: %v", err)),
			errors.Fields{"model": g.ModelID()},
		)
	}

	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		constructRequestURL(g.GetEndpointConfig(), g.apiKey),
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return errors.WithFields(
			errors.New(errors.InvalidInput, fmt.Sprintf("failed to create Gemini request: %v", err)),
			errors.Fields{"model": g.ModelID()},
		)
	}
	for key, value := range g.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	resp, err := g.GetHTTPClient().Do(req)
	if err != nil {
		return errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("LLMGenerationFailed: failed to send Gemini request: %v", err)),
			errors.Fields{"model": g.ModelID()},
		)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("LLMGenerationFailed: failed to read Gemini response body: %v", err)),
			errors.Fields{"model": g.ModelID()},
		)
	}
	if resp.StatusCode != http.StatusOK {
		return errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("LLMGenerationFailed: API request failed with status code %d: %s", resp.StatusCode, string(body))),
			errors.Fields{"model": g.ModelID(), "statusCode": resp.StatusCode},
		)
	}
	if err := json.Unmarshal(body, dest); err != nil {
		return errors.WithFields(
			errors.New(errors.InvalidResponse, fmt.Sprintf("InvalidResponse: failed to unmarshal Gemini response: %v", err)),
			errors.Fields{"model": g.ModelID(), "body": string(body)},
		)
	}
	return nil
}

func contentBlocksToText(blocks []core.ContentBlock) string {
	parts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		if block.Type == core.FieldTypeText && block.Text != "" {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func toolSchemasToDeclarations(tools []map[string]interface{}) ([]geminiFunctionDeclaration, error) {
	functionDeclarations := make([]geminiFunctionDeclaration, 0, len(tools))
	for _, tool := range tools {
		name, ok := tool["name"].(string)
		if !ok {
			return nil, errors.WithFields(
				errors.New(errors.InvalidInput, "tool schema missing 'name' field"),
				errors.Fields{"tool": tool},
			)
		}
		description, _ := tool["description"].(string)
		parameters, ok := tool["parameters"].(map[string]interface{})
		if !ok {
			return nil, errors.WithFields(
				errors.New(errors.InvalidInput, "tool schema missing 'parameters' field"),
				errors.Fields{"tool": tool},
			)
		}

		functionDeclarations = append(functionDeclarations, geminiFunctionDeclaration{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		})
	}
	return functionDeclarations, nil
}

// Generate implements the core.LLM interface.
func (g *GeminiLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{Text: prompt},
				},
			},
		},
		GenerationConfig: g.buildGenerationConfig(opts),
	}
	var geminiResp geminiResponse
	if err := g.doGeminiRequest(ctx, reqBody, &geminiResp); err != nil {
		return nil, errors.WithFields(
			err,
			errors.Fields{
				"prompt": prompt,
				"model":  g.ModelID(),
			})
	}

	if len(geminiResp.Candidates) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "InvalidResponse: no candidates in response"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	if len(geminiResp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "InvalidResponse: no parts in response candidate"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	content, _, _, responseMetadata := buildGeminiContentResponse(geminiResp.Candidates[0].Content.Parts)
	usage := geminiUsageToTokenInfo(geminiResp.UsageMetadata)
	metadata := geminiUsageMetadataMap(geminiResp.UsageMetadata)
	for key, value := range responseMetadata {
		metadata[key] = value
	}

	return &core.LLMResponse{
		Content:  content,
		Usage:    usage,
		Metadata: metadata,
	}, nil
}

// GenerateWithJSON implements the core.LLM interface.
func (g *GeminiLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	response, err := g.Generate(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	return utils.ParseJSONResponse(response.Content)
}

// Implement the GenerateWithFunctions method for GeminiLLM.
func (g *GeminiLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	functionDeclarations, err := toolSchemasToDeclarations(functions)
	if err != nil {
		return nil, err
	}

	// Create the request body with functions
	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{Text: prompt},
				},
				Role: "user",
			},
		},
		Tools: []geminiTool{
			{
				FunctionDeclarations: functionDeclarations,
			},
		},
		ToolConfig:       requiredGeminiToolConfig(functionDeclarations),
		GenerationConfig: g.buildGenerationConfig(opts),
	}
	requestJSON, _ := json.MarshalIndent(reqBody, "", "  ")
	logger.Debug(ctx, "Function call request JSON: %s", string(requestJSON))

	// Parse the response
	var geminiResp geminiFunctionCallResponse
	if err := g.doGeminiRequest(ctx, reqBody, &geminiResp); err != nil {
		return nil, errors.WithFields(
			err,
			errors.Fields{
				"prompt": prompt,
				"model":  g.ModelID(),
			})
	}

	if len(geminiResp.Candidates) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no candidates in response"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	// Extract usage information
	usage := geminiUsageToTokenInfo(geminiResp.UsageMetadata)

	// Process the response to extract function call if present
	result := make(map[string]interface{})

	// Check if there are any parts in the response
	if len(geminiResp.Candidates[0].Content.Parts) > 0 {
		textContent, contentBlocks, toolCalls, responseMetadata := buildGeminiContentResponse(geminiResp.Candidates[0].Content.Parts)

		// Add text content if available
		if textContent != "" {
			result["content"] = textContent
		}

		if len(contentBlocks) > 0 {
			result["content_blocks"] = contentBlocks
		}

		if len(toolCalls) > 0 {
			result["tool_calls"] = toolCalls
			call := toolCalls[0]
			result["function_call"] = map[string]interface{}{
				"name":      call.Name,
				"arguments": call.Arguments,
			}
			if len(call.Metadata) > 0 {
				result["function_call"].(map[string]interface{})["metadata"] = call.Metadata
			}
		}

		for key, value := range responseMetadata {
			result[key] = value
		}
	}

	// If no content or function call was found, add a default message and stamp
	// provider diagnostics so callers can distinguish an actually empty Gemini
	// tool-call response from a normal plain-text reply.
	if len(result) == 0 {
		result["content"] = "No content or function call received from model"
		result["provider_diagnostic"] = geminiEmptyToolResponseDiagnostic(
			"functions",
			geminiResp.Candidates[0].FinishReason,
			len(geminiResp.Candidates),
			len(geminiResp.Candidates[0].Content.Parts),
			geminiResp.PromptFeedback,
		)
	}

	// Add token usage information
	result["_usage"] = usage

	return result, nil
}

// GenerateWithTools implements native multi-turn tool calling for Gemini.
func (g *GeminiLLM) GenerateWithTools(ctx context.Context, messages []core.ChatMessage, tools []map[string]any, options ...core.GenerateOption) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	functionDeclarations, err := toolSchemasToDeclarations(tools)
	if err != nil {
		return nil, err
	}

	reqBody := geminiRequest{
		Contents:         g.chatMessagesToGeminiContents(messages),
		GenerationConfig: g.buildGenerationConfig(opts),
	}
	if len(functionDeclarations) > 0 {
		reqBody.Tools = []geminiTool{{FunctionDeclarations: functionDeclarations}}
		reqBody.ToolConfig = requiredGeminiToolConfig(functionDeclarations)
	}

	requestJSON, _ := json.MarshalIndent(reqBody, "", "  ")
	logger.Debug(ctx, "Gemini tool request JSON: %s", string(requestJSON))

	var geminiResp geminiFunctionCallResponse
	if err := g.doGeminiRequest(ctx, reqBody, &geminiResp); err != nil {
		return nil, errors.WithFields(
			err,
			errors.Fields{"model": g.ModelID()},
		)
	}
	if len(geminiResp.Candidates) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no candidates in Gemini tool response"),
			errors.Fields{"model": g.ModelID()},
		)
	}

	textContent, contentBlocks, toolCalls, responseMetadata := buildGeminiContentResponse(geminiResp.Candidates[0].Content.Parts)
	result := map[string]any{
		"_usage": geminiUsageToTokenInfo(geminiResp.UsageMetadata),
	}
	if textContent != "" {
		result["content"] = textContent
	}
	if len(contentBlocks) > 0 {
		result["content_blocks"] = contentBlocks
	}
	if len(toolCalls) > 0 {
		result["tool_calls"] = toolCalls
		first := toolCalls[0]
		functionCall := map[string]any{
			"name":      first.Name,
			"arguments": first.Arguments,
		}
		if len(first.Metadata) > 0 {
			functionCall["metadata"] = first.Metadata
		}
		result["function_call"] = functionCall
	}
	for key, value := range geminiUsageMetadataMap(geminiResp.UsageMetadata) {
		result[key] = value
	}
	for key, value := range responseMetadata {
		result[key] = value
	}
	if _, hasContent := result["content"]; !hasContent && len(toolCalls) == 0 {
		result["content"] = "No content or function call received from model"
		result["provider_diagnostic"] = geminiEmptyToolResponseDiagnostic(
			"tools",
			geminiResp.Candidates[0].FinishReason,
			len(geminiResp.Candidates),
			len(geminiResp.Candidates[0].Content.Parts),
			geminiResp.PromptFeedback,
		)
	}

	return result, nil
}

func requiredGeminiToolConfig(functions []geminiFunctionDeclaration) *geminiToolConfig {
	if len(functions) == 0 {
		return nil
	}

	names := make([]string, 0, len(functions))
	for _, function := range functions {
		if strings.TrimSpace(function.Name) != "" {
			names = append(names, function.Name)
		}
	}
	if len(names) == 0 {
		return nil
	}

	return &geminiToolConfig{
		FunctionCallingConfig: &geminiFunctionCallingConfig{
			Mode:                 "ANY",
			AllowedFunctionNames: names,
		},
	}
}

func geminiEmptyToolResponseDiagnostic(mode, finishReason string, candidateCount, partCount int, promptFeedback map[string]any) map[string]any {
	diagnostic := map[string]any{
		"provider":        "google",
		"provider_mode":   mode,
		"reason":          "empty_content_and_function_call",
		"candidate_count": candidateCount,
		"part_count":      partCount,
	}
	if strings.TrimSpace(finishReason) != "" {
		diagnostic["finish_reason"] = finishReason
	}
	if len(promptFeedback) > 0 {
		diagnostic["prompt_feedback"] = promptFeedback
	}
	return diagnostic
}

// CreateEmbedding implements the embedding generation for a single input.
func (g *GeminiLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	// Apply options
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}
	if opts.Model == "" {
		opts.Model = "text-embedding-004"
	} else if !isValidGeminiEmbeddingModel(opts.Model) {
		return nil, errors.New(errors.InvalidInput, fmt.Sprintf("invalid Gemini embedding model: %s", opts.Model))
	}

	// Prepare the request body
	reqBody := geminiEmbeddingRequest{
		Model: fmt.Sprintf("models/%s", opts.Model),
	}
	reqBody.Content.Parts = []struct {
		Text string `json:"text"`
	}{{Text: input}}

	// Add task type if specified in options
	if taskType, ok := opts.Params["task_type"].(string); ok {
		reqBody.TaskType = taskType
	}

	// Add any additional parameters
	reqBody.Parameters = opts.Params

	// Marshal request
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to marshal request body"),
			errors.Fields{
				"model":        opts.Model,
				"input_length": len(input),
			})
	}

	// Create request
	url := fmt.Sprintf("%s/models/%s:embedContent?key=%s",
		g.GetEndpointConfig().BaseURL,
		opts.Model,
		g.apiKey)
	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		url,
		bytes.NewBuffer(jsonData),
	)

	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to create request"),
			errors.Fields{
				"model": opts.Model,
			})
	}
	// Set headers
	for key, value := range g.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	// Execute request
	resp, err := g.GetHTTPClient().Do(req)

	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to send request"),
			errors.Fields{
				"model": opts.Model,
			})
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to read response body"),
			errors.Fields{
				"model": opts.Model,
			})
	}

	if resp.StatusCode != http.StatusOK {
		truncatedBody := string(body)
		if len(truncatedBody) > 500 { // Example: truncate to 500 characters
			truncatedBody = truncatedBody[:500] + "... (truncated)"
		}
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("API request failed with status code %d: %s", resp.StatusCode, string(truncatedBody))),
			errors.Fields{
				"model":      opts.Model,
				"statusCode": resp.StatusCode,
			})
	}

	var geminiResp geminiEmbeddingResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal response"),
			errors.Fields{
				"model": opts.Model,
			})
	}

	// Check if the embedding values exist
	if len(geminiResp.Embedding.Values) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "embedding values missing in response"),
			errors.Fields{
				"model": opts.Model,
			})
	}

	// Convert to standard format
	result := &core.EmbeddingResult{
		Vector:     geminiResp.Embedding.Values,
		TokenCount: geminiResp.UsageMetadata.TotalTokenCount,
		Metadata: map[string]interface{}{

			"model":            opts.Model,
			"prompt_tokens":    geminiResp.UsageMetadata.PromptTokenCount,
			"truncated_tokens": geminiResp.Embedding.Statistics.TruncatedInputTokenCount,
			"embedding_tokens": geminiResp.Embedding.Statistics.TokenCount,
		},
	}

	return result, nil
}

// CreateEmbeddings implements batch embedding generation.
func (g *GeminiLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	// Apply options
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Use default batch size if not specified
	if opts.BatchSize <= 0 {
		opts.BatchSize = 32
	}

	var allResults []core.EmbeddingResult
	var firstError error
	var errorIndex = -1

	// Process in batches
	for i := 0; i < len(inputs); i += opts.BatchSize {
		end := i + opts.BatchSize
		if end > len(inputs) {
			end = len(inputs)
		}

		batch := inputs[i:end]
		// Prepare batch request
		reqBody := geminiBatchEmbeddingRequest{
			Model: "text-embedding-004",
		}

		// Add each input to the batch request
		reqBody.Requests = make([]struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		}, len(batch))

		for j, input := range batch {
			reqBody.Requests[j].Content.Parts = []struct {
				Text string `json:"text"`
			}{{Text: input}}
		}

		// Add task type if specified
		if taskType, ok := opts.Params["task_type"].(string); ok {
			reqBody.TaskType = taskType
		}
		reqBody.Parameters = opts.Params

		// Marshal request
		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.Wrap(err, errors.InvalidInput, "failed to marshal batch request"),
					errors.Fields{
						"model":      "text-embedding-004",
						"batch_size": len(batch),
					})
				errorIndex = i
			}
			continue
		}

		url := fmt.Sprintf("%s/models/text-embedding-004:batchEmbedContents?key=%s",
			g.GetEndpointConfig().BaseURL,
			g.apiKey)

		// Create request
		req, err := http.NewRequestWithContext(
			ctx,
			"POST",
			url,
			bytes.NewBuffer(jsonData),
		)
		if err != nil {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.Wrap(err, errors.InvalidInput, "failed to create batch request"),
					errors.Fields{
						"model": "text-embedding-004",
					})
				errorIndex = i
			}
			continue
		}

		// Set headers
		for key, value := range g.GetEndpointConfig().Headers {
			req.Header.Set(key, value)
		}

		// Execute request
		resp, err := g.GetHTTPClient().Do(req)
		if err != nil {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.Wrap(err, errors.LLMGenerationFailed, "failed to send batch request"),
					errors.Fields{
						"model": "text-embedding-004",
					})
				errorIndex = i
			}
			continue
		}

		// Read response
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.Wrap(err, errors.LLMGenerationFailed, "failed to read batch response"),
					errors.Fields{
						"model": "text-embedding-004",
					})
				errorIndex = i
			}
			continue
		}

		if resp.StatusCode != http.StatusOK {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.New(errors.LLMGenerationFailed, fmt.Sprintf("API request failed with status code %d: %s", resp.StatusCode, string(body))),
					errors.Fields{
						"model":      "text-embedding-004",
						"statusCode": resp.StatusCode,
					})
				errorIndex = i
			}
			continue
		}

		var batchResp geminiBatchEmbeddingResponse
		if err := json.Unmarshal(body, &batchResp); err != nil {
			if firstError == nil {
				firstError = errors.WithFields(
					errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal batch response"),
					errors.Fields{
						"model": "text-embedding-004",
					})
				errorIndex = i
			}
			continue
		}

		// Process batch results
		for j, embedding := range batchResp.Embeddings {
			result := core.EmbeddingResult{
				Vector:     embedding.Embedding.Values,
				TokenCount: embedding.UsageMetadata.TotalTokenCount,
				Metadata: map[string]interface{}{
					"model":            "text-embedding-004",
					"prompt_tokens":    embedding.UsageMetadata.PromptTokenCount,
					"truncated_tokens": embedding.Embedding.Statistics.TruncatedInputTokenCount,
					"embedding_tokens": embedding.Embedding.Statistics.TokenCount,
					"batch_index":      i + j,
				},
			}
			allResults = append(allResults, result)
		}
	}
	// If we had errors but still got some results, return what we have
	if firstError != nil && len(allResults) == 0 {
		return nil, firstError
	}

	return &core.BatchEmbeddingResult{
		Embeddings: allResults,
		Error:      firstError,
		ErrorIndex: errorIndex,
	}, nil
}

// streamRequest handles the common streaming logic for both StreamGenerate and StreamGenerateWithContent.
func (g *GeminiLLM) streamRequest(ctx context.Context, reqBody interface{}) (*core.StreamResponse, error) {
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, fmt.Sprintf("failed to marshal request body: %v", err)),
			errors.Fields{"model": g.ModelID()})
	}

	// Add streaming parameter
	streamURL := constructRequestURL(g.GetEndpointConfig(), g.apiKey) + "&alt=sse"

	req, err := http.NewRequestWithContext(ctx, "POST", streamURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, fmt.Sprintf("failed to create request: %v", err)),
			errors.Fields{"model": g.ModelID()})
	}

	for key, value := range g.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}
	req.Header.Set("Accept", "text/event-stream")

	// Create channels and response
	chunkChan := make(chan core.StreamChunk)
	streamCtx, cancelStream := context.WithCancel(ctx)

	// Used to protect against multiple closes
	var channelClosed sync.Once

	// Create a safe way to close the channel
	safeCloseChannel := func() {
		channelClosed.Do(func() {
			close(chunkChan)
		})
	}

	response := &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel: func() {
			cancelStream()
		},
	}

	// Start streaming goroutine
	go func() {
		defer safeCloseChannel()

		client := g.GetHTTPClient()
		resp, err := client.Do(req)
		if err != nil {
			if streamCtx.Err() != nil {
				return
			}
			chunkChan <- core.StreamChunk{
				Error: errors.New(errors.LLMGenerationFailed, fmt.Sprintf("request failed: %v", err)),
			}
			return
		}
		defer resp.Body.Close()

		reader := bufio.NewReader(resp.Body)

		for {
			select {
			case <-streamCtx.Done():
				return
			default:
			}

			readCtx, cancel := context.WithTimeout(streamCtx, 500*time.Millisecond)

			ch := make(chan struct {
				line string
				err  error
			}, 1)

			go func() {
				line, err := reader.ReadString('\n')
				ch <- struct {
					line string
					err  error
				}{line, err}
			}()

			var line string
			var readErr error

			select {
			case result := <-ch:
				line = result.line
				readErr = result.err
				cancel()
			case <-readCtx.Done():
				cancel()
				if streamCtx.Err() != nil {
					return
				}
				continue
			}

			if readErr != nil {
				if readErr == io.EOF || streamCtx.Err() != nil {
					return
				}
				if streamCtx.Err() == nil {
					chunkChan <- core.StreamChunk{
						Error: errors.New(errors.LLMGenerationFailed, fmt.Sprintf("stream read error: %v", readErr)),
					}
				}
				return
			}

			if streamCtx.Err() != nil {
				return
			}

			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "data: ") {
				data := strings.TrimPrefix(line, "data: ")

				if data == "[DONE]" {
					return
				}

				var chunk geminiResponse
				if err := json.Unmarshal([]byte(data), &chunk); err != nil {
					continue
				}

				if len(chunk.Candidates) > 0 && len(chunk.Candidates[0].Content.Parts) > 0 {
					content := chunk.Candidates[0].Content.Parts[0].Text
					if streamCtx.Err() == nil {
						chunkChan <- core.StreamChunk{Content: content}
					}
				}
			}
		}
	}()

	return response, nil
}

// StreamGenerate for Gemini.
func (g *GeminiLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{Text: prompt},
				},
			},
		},
		GenerationConfig: geminiGenerationConfig{
			Temperature:     opts.Temperature,
			MaxOutputTokens: opts.MaxTokens,
			TopP:            opts.TopP,
		},
	}

	return g.streamRequest(ctx, reqBody)
}

func isValidGeminiEmbeddingModel(s string) bool {
	validModels := []string{
		"gemini-embedding-exp-03-07",
		"text-embedding-004",
		"gemini-embedding-004",
		"embedding-001",
		"embedding-latest",
		"embedding-gecko",
		"embedding-gecko-001",
		"text-embedding-gecko-001",
	}

	for _, model := range validModels {
		if s == model {
			return true
		}
	}

	return false
}

func constructRequestURL(endpoint *core.EndpointConfig, apiKey string) string {
	// Remove any trailing slashes from base URL and leading slashes from path
	baseURL := strings.TrimRight(endpoint.BaseURL, "/")
	path := strings.TrimLeft(endpoint.Path, "/")

	// Join them with a single slash
	fullEndpoint := fmt.Sprintf("%s/%s", baseURL, path)

	// Add the API key as query parameter
	return fmt.Sprintf("%s?key=%s", fullEndpoint, apiKey)
}

// GenerateWithContent implements multimodal content generation for Gemini.
func (g *GeminiLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Convert ContentBlocks to Gemini's format
	geminiParts := g.convertToGeminiParts(content)

	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: geminiParts,
			},
		},
		GenerationConfig: g.buildGenerationConfig(opts),
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to marshal request body"),
			errors.Fields{
				"content_blocks": len(content),
				"model":          g.ModelID(),
			})
	}

	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		constructRequestURL(g.GetEndpointConfig(), g.apiKey),
		bytes.NewBuffer(jsonData),
	)

	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to create request"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	for key, value := range g.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	resp, err := g.GetHTTPClient().Do(req)
	if err != nil {
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("failed to send request: %v", err)),
			errors.Fields{
				"model": g.ModelID(),
			})
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("failed to read response body: %v", err)),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	if resp.StatusCode != http.StatusOK {
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("API request failed with status code %d: %s", resp.StatusCode, string(body))),
			errors.Fields{
				"model":      g.ModelID(),
				"statusCode": resp.StatusCode,
			})
	}

	var geminiResp geminiResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, fmt.Sprintf("failed to unmarshal response: %v", err)),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	if len(geminiResp.Candidates) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no candidates in response"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	if len(geminiResp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no parts in response candidate"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	contentText, _, _, responseMetadata := buildGeminiContentResponse(geminiResp.Candidates[0].Content.Parts)
	usage := geminiUsageToTokenInfo(geminiResp.UsageMetadata)
	metadata := geminiUsageMetadataMap(geminiResp.UsageMetadata)
	for key, value := range responseMetadata {
		metadata[key] = value
	}

	return &core.LLMResponse{
		Content:  contentText,
		Usage:    usage,
		Metadata: metadata,
	}, nil
}

// StreamGenerateWithContent implements multimodal streaming for Gemini.
func (g *GeminiLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Convert ContentBlocks to Gemini's format
	geminiParts := g.convertToGeminiParts(content)

	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: geminiParts,
			},
		},
		GenerationConfig: g.buildGenerationConfig(opts),
	}

	return g.streamRequest(ctx, reqBody)
}

// convertToGeminiParts converts ContentBlocks to Gemini's format.
func (g *GeminiLLM) convertToGeminiParts(blocks []core.ContentBlock) []geminiPart {
	var parts []geminiPart

	for _, block := range blocks {
		switch block.Type {
		case core.FieldTypeText:
			part := geminiPart{
				Text: block.Text,
			}
			if block.Metadata != nil {
				if thought, ok := block.Metadata[geminiThoughtMetadataKey].(bool); ok {
					part.Thought = thought
				}
				if signature, ok := block.Metadata[geminiThoughtSignatureMetadataKey].(string); ok {
					part.ThoughtSignature = signature
				}
			}
			parts = append(parts, part)
		case core.FieldTypeImage:
			parts = append(parts, geminiPart{
				InlineData: &geminiInlineData{
					MimeType: block.MimeType,
					Data:     base64.StdEncoding.EncodeToString(block.Data),
				},
			})
		case core.FieldTypeAudio:
			parts = append(parts, geminiPart{
				InlineData: &geminiInlineData{
					MimeType: block.MimeType,
					Data:     base64.StdEncoding.EncodeToString(block.Data),
				},
			})
		default:
			// Fallback to text
			parts = append(parts, geminiPart{
				Text: fmt.Sprintf("[Unsupported content type: %s]", block.Type),
			})
		}
	}

	return parts
}
