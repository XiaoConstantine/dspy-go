package agents

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/cache"
	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelToolFromToolSnapshotsSchema(t *testing.T) {
	minimum := 1.0
	tool := modelAdapterTestTool{schema: models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"count": {Type: "number", Minimum: &minimum},
		},
	}}

	definition, err := ModelToolFromTool(tool)
	require.NoError(t, err)
	tool.schema.Properties["count"] = models.ParameterSchema{Type: "changed"}
	minimum = 9

	assert.Equal(t, "test", definition.Name)
	assert.Equal(t, "test tool", definition.Description)
	properties := definition.InputSchema["properties"].(map[string]any)
	count := properties["count"].(map[string]any)
	assert.Equal(t, "number", count["type"])
	assert.Equal(t, 1.0, count["minimum"])

	_, err = ModelToolFromTool(nil)
	require.EqualError(t, err, "tool is required")
}

func TestLLMAdapter_CompleteUsesTypedChatBoundary(t *testing.T) {
	pathEnum := []any{"main.go", "model.go"}
	inputSchema := JSONSchema{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{
				"type":        "string",
				"description": "path to read",
				"minLength":   1,
				"maxLength":   10,
				"enum":        pathEnum,
			},
			"ranges": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"start": map[string]any{"type": "integer"},
					},
				},
			},
		},
		"required":             []string{"path"},
		"additionalProperties": false,
	}
	rawArguments := map[string]any{"path": "main.go", "range": map[string]any{"start": 1}}
	rawMetadata := map[string]any{"signature": []byte{1, 2}}
	rawDiagnostic := map[string]any{"reason": "test", "nested": map[string]any{"value": "original"}}
	rawProviderData := map[string]any{"openai-codex": []any{map[string]any{"encrypted_content": "opaque"}}}
	rawThoughtBlocks := []core.ContentBlock{{
		Type:     core.FieldTypeImage,
		Data:     []byte{3, 4},
		MimeType: "image/png",
		Metadata: map[string]any{"nested": map[string]any{"value": "original"}},
	}}
	usage := &core.TokenInfo{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}

	llm := &chatAdapterTestLLM{adapterTestLLM: adapterTestLLM{
		result: map[string]any{
			"content":        "I will read the file.",
			"content_blocks": []core.ContentBlock{core.NewTextBlock("I will read the file.")},
			"tool_calls": []core.ToolCall{{
				ID:        "call-1",
				Name:      "read",
				Arguments: rawArguments,
				Metadata:  rawMetadata,
			}},
			"_usage":               usage,
			"provider_data":        rawProviderData,
			"provider_diagnostic":  rawDiagnostic,
			"thought_blocks":       rawThoughtBlocks,
			"thoughts_token_count": 4,
		},
	}}
	adapter, err := NewLLMAdapter(llm)
	require.NoError(t, err)

	response, err := adapter.Complete(context.Background(), ModelRequest{
		Messages: []Message{
			NewTextMessage(RoleSystem, "system"),
			NewTextMessage(RoleInternal, "secret"),
			NewTextMessage(RoleUser, "inspect main.go"),
		},
		Tools: []ModelTool{
			{Name: "write", Description: "write file"},
			{
				Name:        "read",
				Description: "read file",
				InputSchema: inputSchema,
			},
		},
		Options: []core.GenerateOption{core.WithMaxTokens(77)},
	})
	require.NoError(t, err)

	require.Len(t, llm.chatMessages, 2)
	assert.Equal(t, "system", llm.chatMessages[0].Content[0].Text)
	assert.Equal(t, "inspect main.go", llm.chatMessages[1].Content[0].Text)
	require.Len(t, llm.toolSchemas, 2)
	assert.Equal(t, "read", llm.toolSchemas[0]["name"])
	assert.Equal(t, "write", llm.toolSchemas[1]["name"])
	parameters := llm.toolSchemas[0]["parameters"].(map[string]any)
	properties := parameters["properties"].(map[string]any)
	path := properties["path"].(map[string]any)
	assert.Equal(t, 1, path["minLength"])
	assert.Equal(t, 10, path["maxLength"])
	assert.Equal(t, []string{"path"}, parameters["required"])
	assert.Equal(t, pathEnum, path["enum"])
	assert.Equal(t, false, parameters["additionalProperties"])
	ranges := properties["ranges"].(map[string]any)
	items := ranges["items"].(map[string]any)
	assert.Equal(t, "integer", items["properties"].(map[string]any)["start"].(map[string]any)["type"])
	assert.Equal(t, 77, llm.maxTokens)

	pathEnum[0] = "changed.go"
	inputSchema["properties"].(map[string]any)["path"].(map[string]any)["type"] = "changed"
	assert.Equal(t, "string", path["type"])
	assert.Equal(t, "main.go", path["enum"].([]any)[0])

	assert.Equal(t, RoleAssistant, response.Message.Role)
	assert.Equal(t, "I will read the file.", response.Message.Content[0].Text)
	require.Len(t, response.Message.ToolCalls, 1)
	assert.Equal(t, "call-1", response.Message.ToolCalls[0].ID)
	assert.Equal(t, "read", response.Message.ToolCalls[0].Name)
	assert.Equal(t, 5, response.Usage.TotalTokens)
	assert.Equal(t, 4, response.Diagnostics["thoughts_token_count"])
	assert.Equal(t, rawDiagnostic, response.Diagnostics["provider_diagnostic"])
	assert.Equal(t, "opaque", response.Message.ProviderData["openai-codex"].([]any)[0].(map[string]any)["encrypted_content"])
	assert.NotContains(t, response.Diagnostics, "provider_data")
	thoughtBlocks := response.Diagnostics["thought_blocks"].([]core.ContentBlock)
	require.Len(t, thoughtBlocks, 1)
	assert.Equal(t, []byte{3, 4}, thoughtBlocks[0].Data)

	rawArguments["range"].(map[string]any)["start"] = 9
	rawMetadata["signature"].([]byte)[0] = 9
	rawDiagnostic["nested"].(map[string]any)["value"] = "changed"
	rawProviderData["openai-codex"].([]any)[0].(map[string]any)["encrypted_content"] = "changed"
	rawThoughtBlocks[0].Data[0] = 9
	rawThoughtBlocks[0].Metadata["nested"].(map[string]any)["value"] = "changed"
	usage.TotalTokens = 99
	assert.Equal(t, 1, response.Message.ToolCalls[0].Arguments["range"].(map[string]any)["start"])
	assert.Equal(t, byte(1), response.Message.ToolCalls[0].Metadata["signature"].([]byte)[0])
	assert.Equal(t, "original", response.Diagnostics["provider_diagnostic"].(map[string]any)["nested"].(map[string]any)["value"])
	assert.Equal(t, "opaque", response.Message.ProviderData["openai-codex"].([]any)[0].(map[string]any)["encrypted_content"])
	assert.Equal(t, byte(3), thoughtBlocks[0].Data[0])
	assert.Equal(t, "original", thoughtBlocks[0].Metadata["nested"].(map[string]any)["value"])
	assert.Equal(t, 5, response.Usage.TotalTokens)
	assert.Equal(t, "adapter-model", adapter.ModelID())
	assert.Equal(t, "adapter-provider", adapter.ProviderName())
}

func TestLLMAdapter_CompleteUsesConfiguredFunctionFallback(t *testing.T) {
	llm := &adapterTestLLM{result: map[string]any{
		"content": "working",
		"function_call": map[string]any{
			"id":        "call-1",
			"name":      "read",
			"arguments": map[string]any{"path": "main.go"},
			"metadata":  map[string]any{"signature": "sig"},
		},
	}}
	rendered := false
	adapter, err := NewLLMAdapter(llm, WithPromptRenderer(func(request ModelRequest) (string, error) {
		rendered = true
		assert.Equal(t, "hello", request.Messages[0].TextContent())
		return "legacy prompt", nil
	}))
	require.NoError(t, err)

	response, err := adapter.Complete(context.Background(), ModelRequest{
		Messages: []Message{NewTextMessage(RoleUser, "hello")},
		Tools:    []ModelTool{{Name: "read"}},
	})
	require.NoError(t, err)
	assert.True(t, rendered)
	assert.Equal(t, "legacy prompt", llm.prompt)
	require.Len(t, response.Message.ToolCalls, 1)
	assert.Equal(t, "call-1", response.Message.ToolCalls[0].ID)
	assert.Equal(t, "read", response.Message.ToolCalls[0].Name)
	assert.Equal(t, map[string]any{"path": "main.go"}, response.Message.ToolCalls[0].Arguments)
	assert.Equal(t, map[string]any{"signature": "sig"}, response.Message.ToolCalls[0].Metadata)
}

func TestLLMAdapter_CompleteDispatchesWrappedModelsByUnderlyingCapability(t *testing.T) {
	functionOnly := &adapterTestLLM{result: map[string]any{
		"function_call": map[string]any{"name": "read", "arguments": map[string]any{}},
	}}
	wrappedFunctionOnly := core.NewModelContextDecorator(functionOnly)
	adapter, err := NewLLMAdapter(wrappedFunctionOnly, WithPromptRenderer(func(ModelRequest) (string, error) {
		return "wrapped function prompt", nil
	}))
	require.NoError(t, err)
	_, err = adapter.Complete(context.Background(), ModelRequest{Tools: []ModelTool{{Name: "read"}}})
	require.NoError(t, err)
	assert.Equal(t, "wrapped function prompt", functionOnly.prompt)

	chat := &chatAdapterTestLLM{adapterTestLLM: adapterTestLLM{result: map[string]any{
		"tool_calls": []core.ToolCall{{Name: "read"}},
	}}}
	wrappedChat := core.NewModelContextDecorator(chat)
	adapter, err = NewLLMAdapter(wrappedChat)
	require.NoError(t, err)
	_, err = adapter.Complete(context.Background(), ModelRequest{
		Messages: []Message{NewTextMessage(RoleUser, "wrapped chat")},
		Tools:    []ModelTool{{Name: "read"}},
	})
	require.NoError(t, err)
	require.Len(t, chat.chatMessages, 1)
	assert.Equal(t, "wrapped chat", chat.chatMessages[0].Content[0].Text)
}

func TestLLMAdapter_CompleteHandlesDynamicallyUncomparableWrapper(t *testing.T) {
	underlying := &adapterTestLLM{result: map[string]any{
		"function_call": map[string]any{"name": "read", "arguments": map[string]any{}},
	}}
	wrapped := uncomparableAdapterWrapper{
		LLM:   underlying,
		state: map[string]any{"mutable": true},
	}
	adapter, err := NewLLMAdapter(wrapped, WithPromptRenderer(func(ModelRequest) (string, error) {
		return "uncomparable wrapper", nil
	}))
	require.NoError(t, err)

	require.NotPanics(t, func() {
		_, err = adapter.Complete(context.Background(), ModelRequest{Tools: []ModelTool{{Name: "read"}}})
	})
	require.NoError(t, err)
	assert.Equal(t, "uncomparable wrapper", underlying.prompt)
}

func TestLLMAdapter_CompleteRejectsUnwrapCyclesAndExcessiveDepth(t *testing.T) {
	t.Run("self cycle", func(t *testing.T) {
		wrapper := &core.BaseDecorator{}
		wrapper.LLM = wrapper
		adapter, err := NewLLMAdapter(wrapper)
		require.NoError(t, err)

		ctx := core.WithExecutionState(context.Background())
		_, err = adapter.Complete(ctx, ModelRequest{Tools: []ModelTool{{Name: "read"}}})
		require.EqualError(t, err, "llm unwrap cycle detected")
		assert.Empty(t, adapter.ModelID())
		assert.Empty(t, adapter.ProviderName())
	})

	t.Run("two wrapper cycle", func(t *testing.T) {
		first := &core.BaseDecorator{}
		second := &core.BaseDecorator{}
		first.LLM = second
		second.LLM = first
		adapter, err := NewLLMAdapter(first)
		require.NoError(t, err)

		_, err = adapter.Complete(context.Background(), ModelRequest{Tools: []ModelTool{{Name: "read"}}})
		require.EqualError(t, err, "llm unwrap cycle detected")
	})

	t.Run("depth limit", func(t *testing.T) {
		var llm core.LLM = &adapterTestLLM{}
		for range 101 {
			llm = &core.BaseDecorator{LLM: llm}
		}
		adapter, err := NewLLMAdapter(llm)
		require.NoError(t, err)

		_, err = adapter.Complete(context.Background(), ModelRequest{Tools: []ModelTool{{Name: "read"}}})
		require.EqualError(t, err, "llm unwrap exceeded 100 layers")
	})
}

func TestLLMAdapter_CompleteNormalizesFunctionCacheHits(t *testing.T) {
	t.Setenv("DSPY_CACHE_ENABLED", "true")
	t.Setenv("DSPY_CACHE_TYPE", "memory")
	underlying := &adapterTestLLM{result: map[string]any{
		"content_blocks": []core.ContentBlock{core.NewTextBlock("cached response")},
		"tool_calls": []core.ToolCall{{
			ID:        "cached-call",
			Name:      "read",
			Arguments: map[string]any{"path": "cached.go"},
			Metadata:  map[string]any{"signature": "cached-signature"},
		}},
		"_usage": &core.TokenInfo{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
	}}
	wrapped := cache.WrapWithCache(underlying, &config.CachingConfig{
		Enabled: true,
		Type:    "memory",
		TTL:     time.Minute,
		MaxSize: 1 << 20,
	})
	cached, ok := wrapped.(*cache.CachedLLM)
	require.True(t, ok)
	t.Cleanup(func() { require.NoError(t, cached.ClearCache(context.Background())) })

	adapter, err := NewLLMAdapter(wrapped, WithPromptRenderer(func(ModelRequest) (string, error) {
		return "phase-2-cache-hit-contract", nil
	}))
	require.NoError(t, err)
	request := ModelRequest{Tools: []ModelTool{{Name: "read"}}}

	first, err := adapter.Complete(context.Background(), request)
	require.NoError(t, err)
	second, err := adapter.Complete(context.Background(), request)
	require.NoError(t, err)
	assert.Equal(t, 1, underlying.functionCalls)
	for _, response := range []ModelResponse{first, second} {
		assert.Equal(t, "cached response", response.Message.Content[0].Text)
		require.Len(t, response.Message.ToolCalls, 1)
		assert.Equal(t, "cached-call", response.Message.ToolCalls[0].ID)
		assert.Equal(t, "cached.go", response.Message.ToolCalls[0].Arguments["path"])
		assert.Equal(t, "cached-signature", response.Message.ToolCalls[0].Metadata["signature"])
		require.NotNil(t, response.Usage)
		assert.Equal(t, 8, response.Usage.TotalTokens)
	}
}

func TestLLMAdapter_CompletePreservesProviderAndRendererErrors(t *testing.T) {
	providerErr := errors.New("provider failed")
	adapter, err := NewLLMAdapter(&adapterTestLLM{err: providerErr})
	require.NoError(t, err)
	_, err = adapter.Complete(context.Background(), ModelRequest{Tools: []ModelTool{{Name: "read"}}})
	assert.ErrorIs(t, err, providerErr)

	renderErr := errors.New("render failed")
	adapter, err = NewLLMAdapter(&adapterTestLLM{}, WithPromptRenderer(func(ModelRequest) (string, error) {
		return "", renderErr
	}))
	require.NoError(t, err)
	_, err = adapter.Complete(context.Background(), ModelRequest{Tools: []ModelTool{{Name: "read"}}})
	assert.ErrorIs(t, err, renderErr)
	assert.ErrorContains(t, err, "render model prompt")
}

func TestLLMAdapter_CompleteSupportsTextWithoutTools(t *testing.T) {
	llm := &adapterTestLLM{textResponse: &core.LLMResponse{
		Content: "plain response", Usage: &core.TokenInfo{TotalTokens: 4},
		Metadata: map[string]any{"finish_reason": "stop"},
	}}
	adapter, err := NewLLMAdapter(llm)
	require.NoError(t, err)
	response, err := adapter.Complete(context.Background(), ModelRequest{
		Messages: []Message{NewTextMessage(RoleUser, "hello")},
	})
	require.NoError(t, err)
	assert.Equal(t, "plain response", response.Message.TextContent())
	assert.Equal(t, 4, response.Usage.TotalTokens)
	assert.Equal(t, "stop", response.Diagnostics["finish_reason"])
}

func TestLLMAdapter_CompleteRejectsInvalidInputsAndResults(t *testing.T) {
	_, err := NewLLMAdapter(nil)
	require.EqualError(t, err, "llm is required")

	tests := []struct {
		name   string
		result map[string]any
		match  string
	}{
		{name: "content", result: map[string]any{"content": 42}, match: "content has type int"},
		{name: "content blocks", result: map[string]any{"content_blocks": "bad"}, match: "content_blocks has type string"},
		{name: "tool calls", result: map[string]any{"tool_calls": "bad"}, match: "tool_calls has type string"},
		{name: "function call", result: map[string]any{"function_call": "bad"}, match: "function_call has type string"},
		{name: "function name", result: map[string]any{"function_call": map[string]any{}}, match: "function_call name is required"},
		{name: "function arguments", result: map[string]any{"function_call": map[string]any{"name": "read", "arguments": "bad"}}, match: "function_call arguments has type string"},
		{name: "function metadata", result: map[string]any{"function_call": map[string]any{"name": "read", "metadata": "bad"}}, match: "function_call metadata has type string"},
		{name: "usage", result: map[string]any{"_usage": "bad"}, match: "_usage has type string"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter, err := NewLLMAdapter(&adapterTestLLM{result: tt.result})
			require.NoError(t, err)
			_, err = adapter.Complete(context.Background(), ModelRequest{Tools: []ModelTool{{Name: "read"}}})
			require.ErrorContains(t, err, tt.match)
		})
	}
}

func TestLLMAdapter_DefaultRendererRejectsMultimodalFallback(t *testing.T) {
	adapter, err := NewLLMAdapter(&adapterTestLLM{})
	require.NoError(t, err)
	_, err = adapter.Complete(context.Background(), ModelRequest{
		Messages: []Message{{
			Role:    RoleUser,
			Content: []core.ContentBlock{core.NewImageBlock([]byte{1}, "image/png")},
		}},
		Tools: []ModelTool{{Name: "inspect"}},
	})
	require.ErrorContains(t, err, "legacy prompt does not support image content")
}

func TestModelToolSchemasRejectInvalidDefinitions(t *testing.T) {
	_, err := modelToolSchemas([]ModelTool{{Name: " "}})
	require.EqualError(t, err, "model tool name is required")

	_, err = modelToolSchemas([]ModelTool{{Name: "read"}, {Name: "read"}})
	require.EqualError(t, err, `duplicate model tool "read"`)

	_, err = modelToolSchemas([]ModelTool{{
		Name:        "read",
		InputSchema: JSONSchema{"invalid": func() {}},
	}})
	require.ErrorContains(t, err, `model tool "read" input schema is not valid JSON`)
}

type modelAdapterTestTool struct {
	schema models.InputSchema
}

func (modelAdapterTestTool) Name() string        { return "test" }
func (modelAdapterTestTool) Description() string { return "test tool" }
func (t modelAdapterTestTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{Name: t.Name(), Description: t.Description(), InputSchema: t.schema}
}
func (modelAdapterTestTool) CanHandle(context.Context, string) bool { return false }
func (modelAdapterTestTool) Execute(context.Context, map[string]any) (core.ToolResult, error) {
	return core.ToolResult{}, errors.New("unexpected Execute call")
}
func (modelAdapterTestTool) Validate(map[string]any) error { return nil }
func (t modelAdapterTestTool) InputSchema() models.InputSchema {
	return t.schema
}

type uncomparableAdapterWrapper struct {
	core.LLM
	state any
}

func (w uncomparableAdapterWrapper) Unwrap() core.LLM {
	return w.LLM
}

type adapterTestLLM struct {
	result        map[string]any
	textResponse  *core.LLMResponse
	err           error
	prompt        string
	toolSchemas   []map[string]any
	maxTokens     int
	functionCalls int
}

func (m *adapterTestLLM) Generate(context.Context, string, ...core.GenerateOption) (*core.LLMResponse, error) {
	if m.textResponse == nil && m.err == nil {
		return nil, errors.New("unexpected Generate call")
	}
	return m.textResponse, m.err
}
func (m *adapterTestLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]any, error) {
	return nil, errors.New("unexpected GenerateWithJSON call")
}
func (m *adapterTestLLM) GenerateWithFunctions(_ context.Context, prompt string, tools []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	m.functionCalls++
	m.prompt = prompt
	m.toolSchemas = tools
	configured := core.NewGenerateOptions()
	for _, option := range options {
		option(configured)
	}
	m.maxTokens = configured.MaxTokens
	return m.result, m.err
}
func (m *adapterTestLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, errors.New("unexpected CreateEmbedding call")
}
func (m *adapterTestLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, errors.New("unexpected CreateEmbeddings call")
}
func (m *adapterTestLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, errors.New("unexpected StreamGenerate call")
}
func (m *adapterTestLLM) GenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, errors.New("unexpected GenerateWithContent call")
}
func (m *adapterTestLLM) StreamGenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, errors.New("unexpected StreamGenerateWithContent call")
}
func (m *adapterTestLLM) ProviderName() string            { return "adapter-provider" }
func (m *adapterTestLLM) ModelID() string                 { return "adapter-model" }
func (m *adapterTestLLM) Capabilities() []core.Capability { return nil }

type chatAdapterTestLLM struct {
	adapterTestLLM
	chatMessages []core.ChatMessage
}

func (m *chatAdapterTestLLM) GenerateWithTools(_ context.Context, messages []core.ChatMessage, tools []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	m.chatMessages = messages
	m.toolSchemas = tools
	configured := core.NewGenerateOptions()
	for _, option := range options {
		option(configured)
	}
	m.maxTokens = configured.MaxTokens
	return m.result, m.err
}
