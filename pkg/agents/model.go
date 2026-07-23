package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// JSONSchema is a protocol-neutral JSON Schema object. Tool input schemas use
// this open representation so the model boundary can preserve keywords beyond
// the subset supported by any one tool or provider package.
type JSONSchema map[string]any

// ModelTool describes a provider-visible tool without exposing its executor to
// the model boundary.
type ModelTool struct {
	Name        string
	Description string
	InputSchema JSONSchema
}

// ModelToolFromTool snapshots the provider-visible definition of an executable
// tool without exposing its executor to the model boundary.
func ModelToolFromTool(tool core.Tool) (ModelTool, error) {
	if tool == nil {
		return ModelTool{}, fmt.Errorf("tool is required")
	}
	return ModelTool{
		Name:        tool.Name(),
		Description: tool.Description(),
		InputSchema: jsonSchemaFromTool(tool),
	}, nil
}

// ModelRequest is the provider-neutral input for one assistant turn.
type ModelRequest struct {
	Messages []Message
	Tools    []ModelTool
	Options  []core.GenerateOption
}

// ModelResponse is the provider-neutral result of one assistant turn.
type ModelResponse struct {
	Message     Message
	Usage       *core.TokenInfo
	Diagnostics map[string]any
}

// Model produces one provider-neutral assistant response for an agent loop.
type Model interface {
	Complete(ctx context.Context, request ModelRequest) (ModelResponse, error)
	ModelID() string
	ProviderName() string
}

// PromptRenderer converts a typed model request to the text prompt required by
// legacy function-calling providers that do not accept chat history directly.
type PromptRenderer func(ModelRequest) (string, error)

// LLMAdapterOption configures an LLMAdapter.
type LLMAdapterOption func(*LLMAdapter)

// WithPromptRenderer replaces the default deterministic legacy prompt
// renderer. Native compatibility wrappers can use this to retain an existing
// flattened prompt while migrating to the typed model boundary.
func WithPromptRenderer(renderer PromptRenderer) LLMAdapterOption {
	return func(adapter *LLMAdapter) {
		if renderer != nil {
			adapter.renderPrompt = renderer
		}
	}
}

// LLMAdapter adapts the existing core.LLM compatibility interface to the
// narrow, typed agent Model contract.
type LLMAdapter struct {
	llm          core.LLM
	renderPrompt PromptRenderer
}

// NewLLMAdapter returns a typed agent-model adapter over an existing LLM.
func NewLLMAdapter(llm core.LLM, options ...LLMAdapterOption) (*LLMAdapter, error) {
	if llm == nil {
		return nil, fmt.Errorf("llm is required")
	}
	adapter := &LLMAdapter{
		llm:          llm,
		renderPrompt: renderModelPrompt,
	}
	for _, option := range options {
		if option != nil {
			option(adapter)
		}
	}
	return adapter, nil
}

// Complete calls the chat tool API when available and otherwise uses the
// configured prompt renderer with the legacy function-calling API.
func (a *LLMAdapter) Complete(ctx context.Context, request ModelRequest) (ModelResponse, error) {
	if a == nil || a.llm == nil {
		return ModelResponse{}, fmt.Errorf("llm adapter is not initialized")
	}

	toolSchemas, err := modelToolSchemas(request.Tools)
	if err != nil {
		return ModelResponse{}, err
	}
	baseLLM, err := unwrapModelLLM(a.llm)
	if err != nil {
		return ModelResponse{}, err
	}
	core.RecordLLMCall(ctx, baseLLM)

	var raw map[string]any
	if len(toolSchemas) == 0 {
		prompt, renderErr := a.renderPrompt(request)
		if renderErr != nil {
			return ModelResponse{}, fmt.Errorf("render model prompt: %w", renderErr)
		}
		response, generateErr := a.llm.Generate(ctx, prompt, request.Options...)
		if generateErr != nil {
			return ModelResponse{}, generateErr
		}
		if response == nil {
			return ModelResponse{}, fmt.Errorf("text generation returned a nil response")
		}
		raw = cloneAnyMap(response.Metadata)
		if raw == nil {
			raw = map[string]any{}
		}
		raw["content"] = response.Content
		raw["_usage"] = response.Usage
	} else if baseChat, ok := baseLLM.(core.ToolCallingChatLLM); ok {
		chat, outerSupportsChat := a.llm.(core.ToolCallingChatLLM)
		if !outerSupportsChat {
			chat = baseChat
		}
		raw, err = chat.GenerateWithTools(
			ctx,
			MessagesToChatMessages(request.Messages),
			toolSchemas,
			request.Options...,
		)
	} else {
		prompt, renderErr := a.renderPrompt(request)
		if renderErr != nil {
			return ModelResponse{}, fmt.Errorf("render model prompt: %w", renderErr)
		}
		raw, err = a.llm.GenerateWithFunctions(ctx, prompt, toolSchemas, request.Options...)
	}
	if err != nil {
		return ModelResponse{}, err
	}

	response, err := normalizeModelResponse(raw)
	if err != nil {
		return ModelResponse{}, fmt.Errorf("normalize model response: %w", err)
	}
	return response, nil
}

// ModelID returns the wrapped model identifier.
func (a *LLMAdapter) ModelID() string {
	if a == nil || a.llm == nil {
		return ""
	}
	llm, err := unwrapModelLLM(a.llm)
	if err != nil {
		return ""
	}
	return llm.ModelID()
}

// ProviderName returns the wrapped provider name.
func (a *LLMAdapter) ProviderName() string {
	if a == nil || a.llm == nil {
		return ""
	}
	llm, err := unwrapModelLLM(a.llm)
	if err != nil {
		return ""
	}
	return llm.ProviderName()
}

func jsonSchemaFromTool(tool core.Tool) JSONSchema {
	schema := tool.InputSchema()
	properties := make(map[string]any, len(schema.Properties))
	required := make([]string, 0, len(schema.Properties))
	for name, property := range schema.Properties {
		definition := map[string]any{
			"type":        property.Type,
			"description": property.Description,
		}
		if property.Minimum != nil {
			definition["minimum"] = *property.Minimum
		}
		if property.Maximum != nil {
			definition["maximum"] = *property.Maximum
		}
		properties[name] = definition
		if property.Required {
			required = append(required, name)
		}
	}
	sort.Strings(required)

	schemaType := strings.TrimSpace(schema.Type)
	if schemaType == "" {
		schemaType = "object"
	}
	return JSONSchema{
		"type":       schemaType,
		"properties": properties,
		"required":   required,
	}
}

func cloneJSONSchema(schema JSONSchema) JSONSchema {
	if schema == nil {
		return nil
	}
	return JSONSchema(cloneAnyMap(map[string]any(schema)))
}

func modelToolSchemas(tools []ModelTool) ([]map[string]any, error) {
	if len(tools) == 0 {
		return nil, nil
	}

	sorted := append([]ModelTool(nil), tools...)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name < sorted[j].Name
	})

	schemas := make([]map[string]any, 0, len(sorted))
	seen := make(map[string]struct{}, len(sorted))
	for _, tool := range sorted {
		name := strings.TrimSpace(tool.Name)
		if name == "" {
			return nil, fmt.Errorf("model tool name is required")
		}
		if _, exists := seen[name]; exists {
			return nil, fmt.Errorf("duplicate model tool %q", name)
		}
		seen[name] = struct{}{}

		parameters := cloneJSONSchema(tool.InputSchema)
		if parameters == nil {
			parameters = JSONSchema{}
		}
		schemaType, exists := parameters["type"]
		if !exists || schemaType == nil {
			parameters["type"] = "object"
		} else if text, ok := schemaType.(string); ok && strings.TrimSpace(text) == "" {
			parameters["type"] = "object"
		}
		if _, exists := parameters["properties"]; !exists {
			parameters["properties"] = map[string]any{}
		}
		if _, err := json.Marshal(parameters); err != nil {
			return nil, fmt.Errorf("model tool %q input schema is not valid JSON: %w", name, err)
		}
		schemas = append(schemas, map[string]any{
			"name":        name,
			"description": tool.Description,
			"parameters":  map[string]any(parameters),
		})
	}
	return schemas, nil
}

func unwrapModelLLM(llm core.LLM) (core.LLM, error) {
	type unwrapper interface {
		Unwrap() core.LLM
	}

	const maxUnwrapDepth = 100

	current := llm
	seen := make(map[core.LLM]struct{})
	for range maxUnwrapDepth {
		if current == nil {
			return nil, fmt.Errorf("llm unwrap returned nil")
		}
		value := reflect.ValueOf(current)
		if value.Comparable() {
			if _, exists := seen[current]; exists {
				return nil, fmt.Errorf("llm unwrap cycle detected")
			}
			seen[current] = struct{}{}
		}
		wrapped, ok := current.(unwrapper)
		if !ok {
			return current, nil
		}
		current = wrapped.Unwrap()
	}
	return nil, fmt.Errorf("llm unwrap exceeded %d layers", maxUnwrapDepth)
}

func normalizeModelResponse(raw map[string]any) (ModelResponse, error) {
	response := ModelResponse{
		Message: Message{Role: RoleAssistant},
	}
	if raw == nil {
		return response, nil
	}

	if value, exists := raw["content_blocks"]; exists {
		blocks, err := normalizeContentBlocks(value, "content_blocks")
		if err != nil {
			return ModelResponse{}, err
		}
		response.Message.Content = blocks
	} else if value, exists := raw["content"]; exists {
		content, ok := value.(string)
		if !ok {
			return ModelResponse{}, fmt.Errorf("content has type %T, want string", value)
		}
		if content != "" {
			response.Message.Content = []core.ContentBlock{core.NewTextBlock(content)}
		}
	}

	calls, err := modelResponseToolCalls(raw)
	if err != nil {
		return ModelResponse{}, err
	}
	response.Message.ToolCalls = calls

	if value, exists := raw["_usage"]; exists {
		usage, err := normalizeTokenInfo(value)
		if err != nil {
			return ModelResponse{}, err
		}
		response.Usage = usage
	}

	diagnostics := make(map[string]any)
	for key, value := range raw {
		switch key {
		case "content", "content_blocks", "tool_calls", "function_call", "_usage":
			continue
		default:
			if key == "thought_blocks" {
				blocks, err := normalizeContentBlocks(value, key)
				if err != nil {
					return ModelResponse{}, err
				}
				diagnostics[key] = blocks
				continue
			}
			diagnostics[key] = cloneAnyValue(value)
		}
	}
	if len(diagnostics) > 0 {
		response.Diagnostics = diagnostics
	}
	return response, nil
}

func normalizeContentBlocks(value any, field string) ([]core.ContentBlock, error) {
	if value == nil {
		return nil, nil
	}
	if blocks, ok := value.([]core.ContentBlock); ok {
		return cloneContentBlocks(blocks), nil
	}
	var blocks []core.ContentBlock
	if err := normalizeCachedValue(value, &blocks); err != nil {
		return nil, fmt.Errorf("%s has type %T, want []core.ContentBlock: %w", field, value, err)
	}
	return cloneContentBlocks(blocks), nil
}

func normalizeToolCalls(value any) ([]core.ToolCall, error) {
	if value == nil {
		return nil, nil
	}
	if calls, ok := value.([]core.ToolCall); ok {
		return cloneToolCalls(calls), nil
	}
	var calls []core.ToolCall
	if err := normalizeCachedValue(value, &calls); err != nil {
		return nil, fmt.Errorf("tool_calls has type %T, want []core.ToolCall: %w", value, err)
	}
	return cloneToolCalls(calls), nil
}

func normalizeTokenInfo(value any) (*core.TokenInfo, error) {
	if value == nil {
		return nil, nil
	}
	switch usage := value.(type) {
	case *core.TokenInfo:
		if usage == nil {
			return nil, nil
		}
		cloned := *usage
		return &cloned, nil
	case core.TokenInfo:
		cloned := usage
		return &cloned, nil
	}
	var usage core.TokenInfo
	if err := normalizeCachedValue(value, &usage); err != nil {
		return nil, fmt.Errorf("_usage has type %T, want *core.TokenInfo: %w", value, err)
	}
	return &usage, nil
}

func normalizeCachedValue(value any, target any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return json.Unmarshal(encoded, target)
}

func modelResponseToolCalls(raw map[string]any) ([]core.ToolCall, error) {
	if value, exists := raw["tool_calls"]; exists {
		calls, err := normalizeToolCalls(value)
		if err != nil {
			return nil, err
		}
		if len(calls) > 0 {
			return calls, nil
		}
	}

	value, exists := raw["function_call"]
	if !exists {
		return nil, nil
	}
	call, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("function_call has type %T, want map[string]any", value)
	}
	name, _ := call["name"].(string)
	if strings.TrimSpace(name) == "" {
		return nil, fmt.Errorf("function_call name is required")
	}
	arguments, ok := call["arguments"].(map[string]any)
	if call["arguments"] != nil && !ok {
		return nil, fmt.Errorf("function_call arguments has type %T, want map[string]any", call["arguments"])
	}
	metadata, ok := call["metadata"].(map[string]any)
	if call["metadata"] != nil && !ok {
		return nil, fmt.Errorf("function_call metadata has type %T, want map[string]any", call["metadata"])
	}
	id, _ := call["id"].(string)
	return []core.ToolCall{{
		ID:        id,
		Name:      name,
		Arguments: cloneAnyMap(arguments),
		Metadata:  cloneAnyMap(metadata),
	}}, nil
}

func renderModelPrompt(request ModelRequest) (string, error) {
	messages := MessagesToChatMessages(request.Messages)
	for _, message := range messages {
		for _, block := range message.Content {
			if block.Type != core.FieldTypeText {
				return "", fmt.Errorf("legacy prompt does not support %s content", block.Type)
			}
		}
		if message.ToolResult != nil {
			for _, block := range message.ToolResult.Content {
				if block.Type != core.FieldTypeText {
					return "", fmt.Errorf("legacy prompt does not support %s tool-result content", block.Type)
				}
			}
		}
	}
	encoded, err := json.Marshal(messages)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}
