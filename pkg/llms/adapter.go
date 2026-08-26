package llms

import (
	"context"
	jsonv1 "encoding/json"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
	llm "github.com/XiaoConstantine/llm-go"
)

// GeneratorLLM adapts llm-go's provider-neutral Generator to dspy-go's legacy
// convenience interface. Provider protocol behavior remains owned by llm-go.
type GeneratorLLM struct {
	*core.BaseLLM
	generator llm.Generator
	embedder  embeddingClient
}

// Adapt makes an llm-go Generator usable anywhere dspy-go accepts a core.LLM.
func Adapt(generator llm.Generator) (*GeneratorLLM, error) {
	return adapt(generator, nil, nil, nil)
}

func adapt(generator llm.Generator, endpoint *core.EndpointConfig, client *http.Client, embedder embeddingClient) (*GeneratorLLM, error) {
	if generator == nil {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "llm-go generator is required")
	}
	info := generator.Info()
	capabilities := coreCapabilities(info.Capabilities)
	if embedder != nil {
		capabilities = append(capabilities, core.CapabilityEmbedding)
	}
	base := core.NewBaseLLM(
		info.Provider,
		core.ModelID(info.Model),
		capabilities,
		endpoint,
		core.WithHTTPClient(client),
	)
	return &GeneratorLLM{BaseLLM: base, generator: generator, embedder: embedder}, nil
}

// Generator returns the underlying llm-go Generator.
func (l *GeneratorLLM) Generator() llm.Generator {
	if l == nil {
		return nil
	}
	return l.generator
}

func (l *GeneratorLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	response, err := l.generate(ctx, []llm.Message{userMessage([]llm.Part{{Text: prompt}})}, llm.ResponseFormatText, nil, options)
	if err != nil {
		return nil, err
	}
	return legacyResponse(response)
}

func (l *GeneratorLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]any, error) {
	format := llm.ResponseFormatJSON
	if !hasCapability(l.generator.Info().Capabilities, llm.CapabilityJSON) {
		format = llm.ResponseFormatText
		prompt += "\n\nReturn only a valid JSON object."
	}
	response, err := l.generate(ctx, []llm.Message{userMessage([]llm.Part{{Text: prompt}})}, format, nil, options)
	if err != nil {
		return nil, err
	}
	return utils.ParseJSONResponse(response.Text())
}

func (l *GeneratorLLM) GenerateWithFunctions(
	ctx context.Context,
	prompt string,
	functions []map[string]any,
	options ...core.GenerateOption,
) (map[string]any, error) {
	return l.GenerateWithTools(ctx, []core.ChatMessage{{
		Role:    "user",
		Content: []core.ContentBlock{core.NewTextBlock(prompt)},
	}}, functions, options...)
}

// GenerateWithTools preserves native multi-turn tool calling through llm-go's
// canonical Message and Tool types.
func (l *GeneratorLLM) GenerateWithTools(
	ctx context.Context,
	messages []core.ChatMessage,
	tools []map[string]any,
	options ...core.GenerateOption,
) (map[string]any, error) {
	if len(tools) == 0 {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "at least one tool schema is required")
	}
	canonicalMessages, err := canonicalMessages(messages)
	if err != nil {
		return nil, err
	}
	canonicalTools, err := canonicalTools(tools)
	if err != nil {
		return nil, err
	}
	response, err := l.generate(ctx, canonicalMessages, llm.ResponseFormatText, canonicalTools, options)
	if err != nil {
		return nil, err
	}
	return legacyToolResponse(response)
}

func (l *GeneratorLLM) GenerateWithContent(
	ctx context.Context,
	content []core.ContentBlock,
	options ...core.GenerateOption,
) (*core.LLMResponse, error) {
	parts, err := canonicalParts(content)
	if err != nil {
		return nil, err
	}
	response, err := l.generate(ctx, []llm.Message{userMessage(parts)}, llm.ResponseFormatText, nil, options)
	if err != nil {
		return nil, err
	}
	return legacyResponse(response)
}

func (l *GeneratorLLM) StreamGenerate(
	ctx context.Context,
	prompt string,
	options ...core.GenerateOption,
) (*core.StreamResponse, error) {
	return l.stream(ctx, []llm.Message{userMessage([]llm.Part{{Text: prompt}})}, options)
}

func (l *GeneratorLLM) StreamGenerateWithContent(
	ctx context.Context,
	content []core.ContentBlock,
	options ...core.GenerateOption,
) (*core.StreamResponse, error) {
	parts, err := canonicalParts(content)
	if err != nil {
		return nil, err
	}
	return l.stream(ctx, []llm.Message{userMessage(parts)}, options)
}

func (l *GeneratorLLM) CreateEmbedding(
	ctx context.Context,
	input string,
	options ...core.EmbeddingOption,
) (*core.EmbeddingResult, error) {
	if l.embedder == nil {
		return nil, dspyerrors.New(dspyerrors.UnsupportedOperation, "this provider does not support embeddings")
	}
	return l.embedder.CreateEmbedding(ctx, input, options...)
}

func (l *GeneratorLLM) CreateEmbeddings(
	ctx context.Context,
	inputs []string,
	options ...core.EmbeddingOption,
) (*core.BatchEmbeddingResult, error) {
	if l.embedder == nil {
		return nil, dspyerrors.New(dspyerrors.UnsupportedOperation, "this provider does not support embeddings")
	}
	return l.embedder.CreateEmbeddings(ctx, inputs, options...)
}

func (l *GeneratorLLM) generate(
	ctx context.Context,
	messages []llm.Message,
	format llm.ResponseFormat,
	tools []llm.Tool,
	options []core.GenerateOption,
) (*llm.Response, error) {
	request := canonicalRequest(messages, options, l.generator.Info())
	request.ResponseFormat = format
	request.Tools = tools
	response, err := l.generator.Generate(ctx, request)
	if err != nil {
		return nil, err
	}
	if response == nil {
		return nil, dspyerrors.New(dspyerrors.InvalidResponse, "llm-go returned a nil response")
	}
	return response, nil
}

func (l *GeneratorLLM) stream(
	ctx context.Context,
	messages []llm.Message,
	options []core.GenerateOption,
) (*core.StreamResponse, error) {
	streamCtx, cancel := context.WithCancel(ctx)
	stream, err := l.generator.Stream(streamCtx, canonicalRequest(messages, options, l.generator.Info()))
	if err != nil {
		cancel()
		return nil, err
	}

	chunks := make(chan core.StreamChunk)
	userDone := make(chan struct{})
	var cancelOnce sync.Once
	cancelResponse := func() {
		cancelOnce.Do(func() {
			close(userDone)
			cancel()
		})
		_ = stream.Close()
	}

	go func() {
		defer close(chunks)
		defer cancel()

		var usage *core.TokenInfo
		for {
			chunk, recvErr := stream.Recv()
			if recvErr != nil {
				closeErr := stream.Close()
				if recvErr == io.EOF {
					recvErr = closeErr
				} else if closeErr != nil {
					recvErr = errors.Join(recvErr, closeErr)
				}
				terminal := core.StreamChunk{Done: true, Error: recvErr, Usage: usage}
				select {
				case chunks <- terminal:
				case <-userDone:
				}
				return
			}

			if chunk.Usage != nil {
				usage = tokenInfo(chunk.Usage)
			}
			content := canonicalText(chunk.Content)
			if content == "" {
				continue
			}
			select {
			case chunks <- core.StreamChunk{Content: content}:
			case <-streamCtx.Done():
				// Recv will expose parent-context cancellation on the next pass.
			case <-userDone:
				_ = stream.Close()
				return
			}
		}
	}()

	return &core.StreamResponse{ChunkChannel: chunks, Cancel: cancelResponse}, nil
}

func canonicalRequest(messages []llm.Message, options []core.GenerateOption, info llm.ModelInfo) llm.Request {
	defaults := core.NewGenerateOptions()
	configured := core.NewGenerateOptions()
	for _, option := range options {
		option(configured)
	}
	request := llm.Request{
		Messages: messages,
		Stop:     append([]string(nil), configured.Stop...),
	}
	// Do not turn dspy-go's implicit defaults into unsupported explicit options.
	// Non-default values still reach llm-go so it can validate the caller's choice.
	if info.Provider != "openai-codex" || configured.MaxTokens != defaults.MaxTokens {
		request.MaxOutputTokens = configured.MaxTokens
	}
	temperatureDisabled := info.Compatibility != nil &&
		info.Compatibility.Anthropic != nil &&
		info.Compatibility.Anthropic.Temperature == llm.CompatibilityDisabled
	if !temperatureDisabled || configured.Temperature != defaults.Temperature {
		temperature := float64(configured.Temperature)
		request.Temperature = &temperature
	}
	if configured.TopP != 0 {
		topP := float64(configured.TopP)
		request.TopP = &topP
	}
	if configured.PresencePenalty != 0 {
		penalty := float64(configured.PresencePenalty)
		request.PresencePenalty = &penalty
	}
	if configured.FrequencyPenalty != 0 {
		penalty := float64(configured.FrequencyPenalty)
		request.FrequencyPenalty = &penalty
	}
	return request
}

func userMessage(parts []llm.Part) llm.Message {
	return llm.Message{Role: llm.RoleUser, Content: parts}
}

func canonicalMessages(messages []core.ChatMessage) ([]llm.Message, error) {
	converted := make([]llm.Message, 0, len(messages))
	for index, message := range messages {
		role, err := canonicalRole(message.Role)
		if err != nil {
			return nil, fmt.Errorf("messages[%d]: %w", index, err)
		}
		content, err := canonicalParts(message.Content)
		if err != nil {
			return nil, fmt.Errorf("messages[%d]: %w", index, err)
		}
		canonical := llm.Message{Role: role, Content: content}
		if len(message.ToolCalls) != 0 {
			canonical.ToolCalls = make([]llm.ToolCall, 0, len(message.ToolCalls))
			for _, call := range message.ToolCalls {
				arguments, err := jsonv2.Marshal(call.Arguments, jsonv1.DefaultOptionsV1())
				if err != nil {
					return nil, fmt.Errorf("messages[%d] tool %q arguments: %w", index, call.Name, err)
				}
				canonical.ToolCalls = append(canonical.ToolCalls, llm.ToolCall{
					ID: call.ID, Name: call.Name, Arguments: arguments,
				})
			}
		}
		if message.ToolResult != nil {
			resultContent, err := canonicalParts(message.ToolResult.Content)
			if err != nil {
				return nil, fmt.Errorf("messages[%d] tool result: %w", index, err)
			}
			canonical.Content = nil
			canonical.ToolResults = []llm.ToolResult{{
				CallID:  message.ToolResult.ToolCallID,
				Name:    message.ToolResult.Name,
				Content: resultContent,
				IsError: message.ToolResult.IsError,
			}}
		}
		if len(message.ProviderData) != 0 {
			providerData, err := jsonv2.Marshal(message.ProviderData, jsonv1.DefaultOptionsV1())
			if err != nil {
				return nil, fmt.Errorf("messages[%d] provider data: %w", index, err)
			}
			canonical.ProviderData = providerData
		}
		converted = append(converted, canonical)
	}
	return converted, nil
}

func canonicalRole(role string) (llm.Role, error) {
	switch strings.TrimSpace(role) {
	case "system", "developer":
		return llm.RoleSystem, nil
	case "user":
		return llm.RoleUser, nil
	case "assistant":
		return llm.RoleAssistant, nil
	case "tool":
		return llm.RoleTool, nil
	default:
		return "", dspyerrors.New(dspyerrors.InvalidInput, "unsupported message role: "+role)
	}
}

func canonicalParts(blocks []core.ContentBlock) ([]llm.Part, error) {
	parts := make([]llm.Part, 0, len(blocks))
	for index, block := range blocks {
		switch block.Type {
		case core.FieldTypeText:
			parts = append(parts, llm.Part{Kind: llm.PartText, Text: block.Text})
		case core.FieldTypeImage:
			parts = append(parts, llm.Part{Kind: llm.PartImage, Data: append([]byte(nil), block.Data...), MediaType: block.MimeType})
		case core.FieldTypeAudio:
			parts = append(parts, llm.Part{Kind: llm.PartAudio, Data: append([]byte(nil), block.Data...), MediaType: block.MimeType})
		default:
			return nil, dspyerrors.WithFields(
				dspyerrors.New(dspyerrors.InvalidInput, "unsupported content block type"),
				dspyerrors.Fields{"index": index, "type": block.Type},
			)
		}
	}
	return parts, nil
}

func legacyParts(parts []llm.Part) []core.ContentBlock {
	blocks := make([]core.ContentBlock, 0, len(parts))
	for _, part := range parts {
		switch part.Kind {
		case llm.PartText:
			blocks = append(blocks, core.NewTextBlock(part.Text))
		case llm.PartImage:
			blocks = append(blocks, core.NewImageBlock(append([]byte(nil), part.Data...), part.MediaType))
		case llm.PartAudio:
			blocks = append(blocks, core.NewAudioBlock(append([]byte(nil), part.Data...), part.MediaType))
		}
	}
	return blocks
}

func canonicalTools(schemas []map[string]any) ([]llm.Tool, error) {
	tools := make([]llm.Tool, 0, len(schemas))
	for index, schema := range schemas {
		definition := schema
		if function, ok := schema["function"].(map[string]any); ok {
			definition = function
		}
		name, _ := definition["name"].(string)
		if strings.TrimSpace(name) == "" {
			return nil, dspyerrors.WithFields(
				dspyerrors.New(dspyerrors.InvalidInput, "tool schema requires a non-empty name"),
				dspyerrors.Fields{"index": index},
			)
		}
		parameters := definition["parameters"]
		if parameters == nil {
			parameters = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		inputSchema, err := jsonv2.Marshal(parameters, jsonv1.DefaultOptionsV1())
		if err != nil {
			return nil, fmt.Errorf("tool %q input schema: %w", name, err)
		}
		description, _ := definition["description"].(string)
		strict, _ := definition["strict"].(bool)
		tools = append(tools, llm.Tool{
			Name: name, Description: description, InputSchema: inputSchema, Strict: strict,
		})
	}
	return tools, nil
}

func legacyResponse(response *llm.Response) (*core.LLMResponse, error) {
	providerData, err := legacyProviderData(response.Message.ProviderData)
	if err != nil {
		return nil, err
	}
	metadata := responseMetadata(response)
	if providerData != nil {
		metadata["provider_data"] = providerData
	}
	if len(metadata) == 0 {
		metadata = nil
	}
	return &core.LLMResponse{
		Content:  response.Text(),
		Usage:    tokenInfo(response.Usage),
		Metadata: metadata,
	}, nil
}

func legacyToolResponse(response *llm.Response) (map[string]any, error) {
	result := responseMetadata(response)
	result["_usage"] = tokenInfo(response.Usage)
	if content := response.Text(); content != "" {
		result["content"] = content
	}
	if hasNonText(response.Message.Content) {
		result["content_blocks"] = legacyParts(response.Message.Content)
	}
	if len(response.Message.ToolCalls) != 0 {
		calls := make([]core.ToolCall, 0, len(response.Message.ToolCalls))
		for _, call := range response.Message.ToolCalls {
			arguments := map[string]any{}
			if len(call.Arguments) != 0 {
				if err := jsonv2.Unmarshal(call.Arguments, &arguments); err != nil {
					return nil, dspyerrors.WithFields(
						dspyerrors.Wrap(err, dspyerrors.InvalidResponse, "tool arguments must be a JSON object"),
						dspyerrors.Fields{"tool_name": call.Name},
					)
				}
			}
			calls = append(calls, core.ToolCall{ID: call.ID, Name: call.Name, Arguments: arguments})
		}
		result["tool_calls"] = calls
		result["function_call"] = map[string]any{
			"id": calls[0].ID, "name": calls[0].Name, "arguments": calls[0].Arguments,
		}
	}
	providerData, err := legacyProviderData(response.Message.ProviderData)
	if err != nil {
		return nil, err
	}
	if providerData != nil {
		result["provider_data"] = providerData
	}
	if _, hasContent := result["content"]; !hasContent && len(response.Message.ToolCalls) == 0 {
		result["content"] = "No content or function call received from model"
		result["provider_diagnostic"] = map[string]any{
			"provider": response.Model,
			"reason":   "empty_content_and_function_call",
		}
	}
	return result, nil
}

func responseMetadata(response *llm.Response) map[string]any {
	metadata := make(map[string]any)
	if response.ID != "" {
		metadata["response_id"] = response.ID
	}
	if response.Model != "" {
		metadata["model"] = response.Model
	}
	if response.FinishReason != "" {
		metadata["finish_reason"] = string(response.FinishReason)
	}
	if response.ReasoningSummary != "" {
		metadata["reasoning_summary"] = response.ReasoningSummary
	}
	if response.Usage != nil && response.Usage.Cost != nil {
		metadata["usage_cost"] = *response.Usage.Cost
	}
	return metadata
}

func legacyProviderData(data []byte) (map[string]any, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var providerData map[string]any
	if err := jsonv2.Unmarshal(data, &providerData); err != nil {
		return nil, dspyerrors.Wrap(err, dspyerrors.InvalidResponse, "provider data must be a JSON object")
	}
	return providerData, nil
}

func tokenInfo(usage *llm.Usage) *core.TokenInfo {
	if usage == nil {
		return nil
	}
	promptTokens := usage.InputTokens + usage.CacheReadTokens + usage.CacheWriteTokens
	totalTokens := usage.TotalTokens
	if totalTokens == 0 {
		totalTokens = promptTokens + usage.OutputTokens
	}
	return &core.TokenInfo{
		PromptTokens: promptTokens, CompletionTokens: usage.OutputTokens, TotalTokens: totalTokens,
	}
}

func canonicalText(parts []llm.Part) string {
	var text strings.Builder
	for _, part := range parts {
		if part.Kind == llm.PartText {
			text.WriteString(part.Text)
		}
	}
	return text.String()
}

func hasNonText(parts []llm.Part) bool {
	for _, part := range parts {
		if part.Kind != llm.PartText {
			return true
		}
	}
	return false
}

func hasCapability(capabilities []llm.Capability, expected llm.Capability) bool {
	for _, capability := range capabilities {
		if capability == expected {
			return true
		}
	}
	return false
}

func coreCapabilities(capabilities []llm.Capability) []core.Capability {
	converted := []core.Capability{core.CapabilityCompletion, core.CapabilityChat}
	for _, capability := range capabilities {
		switch capability {
		case llm.CapabilityStreaming:
			converted = appendCapability(converted, core.CapabilityStreaming)
		case llm.CapabilityTools:
			converted = appendCapability(converted, core.CapabilityToolCalling)
		case llm.CapabilityJSON:
			converted = appendCapability(converted, core.CapabilityJSON)
		case llm.CapabilityVision:
			converted = appendCapability(converted, core.CapabilityMultimodal)
			converted = appendCapability(converted, core.CapabilityVision)
		case llm.CapabilityAudio:
			converted = appendCapability(converted, core.CapabilityMultimodal)
			converted = appendCapability(converted, core.CapabilityAudio)
		}
	}
	return converted
}

func appendCapability(capabilities []core.Capability, capability core.Capability) []core.Capability {
	for _, existing := range capabilities {
		if existing == capability {
			return capabilities
		}
	}
	return append(capabilities, capability)
}

var (
	_ core.LLM                = (*GeneratorLLM)(nil)
	_ core.ToolCallingChatLLM = (*GeneratorLLM)(nil)
)
