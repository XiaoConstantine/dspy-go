package native

import (
	"context"
	"fmt"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewAgent_RequiresToolCalling(t *testing.T) {
	_, err := NewAgent(&stubLLM{}, Config{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not support tool calling")
}

func TestAgent_Execute_CompletesWithToolAndFinish(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "write_note",
					"arguments": map[string]any{"content": "done"},
				},
				"_usage": &core.TokenInfo{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
			},
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "done"},
				},
				"_usage": &core.TokenInfo{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6},
			},
		},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 4})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{
		name: "write_note",
		run: func(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
			return core.ToolResult{Data: params["content"]}, nil
		},
	}))

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Write a note and finish.",
		"task_id": "task-1",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	assert.Equal(t, "done", result["final_answer"])

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	assert.True(t, trace.Completed)
	assert.Equal(t, "done", trace.FinalAnswer)
	assert.Len(t, trace.Steps, 2)
	assert.Equal(t, "write_note", trace.Steps[0].ToolName)
	assert.Equal(t, int64(9), trace.TokenUsage.PromptTokens)
	assert.Equal(t, int64(5), trace.TokenUsage.CompletionTokens)
}

func TestAgent_Execute_FailsFastAfterRepeatedNoCallResponses(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			noCallResult(),
			noCallResult(),
			noCallResult(),
		},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 20})
	require.NoError(t, err)

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Use a tool before finishing.",
		"task_id": "task-no-call",
	})
	require.NoError(t, err)
	require.False(t, result["completed"].(bool))
	assert.Contains(t, result["error"], "repeated model responses without tool calls")

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	assert.Len(t, trace.Steps, 3)
	assert.Equal(t, "empty_content_and_function_call", trace.Steps[0].Metadata["reason"])
}

func TestAgent_Execute_UsesNativeToolCallingWhenAvailable(t *testing.T) {
	llm := &nativeStubLLM{
		stubLLM: stubLLM{
			capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
			results: []map[string]any{
				{
					"function_call": map[string]any{
						"name":      "Finish",
						"arguments": map[string]any{"answer": "done"},
					},
				},
			},
		},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 1})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]interface{}{"task": "Finish immediately."})
	require.NoError(t, err)
	assert.Len(t, llm.messages, 1)
	assert.Empty(t, llm.prompts)
}

func TestAgent_Execute_FallsBackToFunctionsForWrappedFunctionOnlyLLM(t *testing.T) {
	base := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "done"},
				},
			},
		},
	}

	agent, err := NewAgent(core.NewModelContextDecorator(base), Config{MaxTurns: 1})
	require.NoError(t, err)

	result, err := agent.Execute(context.Background(), map[string]interface{}{"task": "Finish immediately."})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	assert.Len(t, base.prompts, 1)
	assert.Contains(t, base.prompts[0], "TURN BUDGET: turn 1 of 1")
	assert.Contains(t, base.prompts[0], "call Finish immediately")
}

func TestAgent_Clone_CopiesCloneableTools(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 1})
	require.NoError(t, err)
	tool := &simpleTool{name: "write_note"}
	require.NoError(t, agent.RegisterTool(tool))

	clonedAny, err := agent.Clone()
	require.NoError(t, err)

	cloned, ok := clonedAny.(*Agent)
	require.True(t, ok)

	originalTools := agent.toolRegistry.List()
	clonedTools := cloned.toolRegistry.List()
	require.Len(t, originalTools, 1)
	require.Len(t, clonedTools, 1)
	assert.NotSame(t, originalTools[0], clonedTools[0])
}

func TestAgent_Clone_FailsForNonCloneableTools(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 1})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(nonCloneableTool{name: "plain"}))

	_, err = agent.Clone()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not implement core.CloneableTool")
}

func TestTurnBudgetReminder(t *testing.T) {
	assert.Equal(t, "", turnBudgetReminder(1, 5))
	assert.Contains(t, turnBudgetReminder(3, 5), "3 turns remaining")
	assert.Contains(t, turnBudgetReminder(5, 5), "Final turn")
}

func noCallResult() map[string]any {
	return map[string]any{
		"content": "No content or function call received from model",
		"provider_diagnostic": map[string]any{
			"provider":      "google",
			"provider_mode": "tools",
			"reason":        "empty_content_and_function_call",
			"finish_reason": "STOP",
		},
	}
}

type simpleTool struct {
	name string
	run  func(context.Context, map[string]interface{}) (core.ToolResult, error)
}

func (t simpleTool) Name() string        { return t.name }
func (t simpleTool) Description() string { return t.name }
func (t simpleTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{Name: t.name, Description: t.name}
}
func (t simpleTool) CanHandle(context.Context, string) bool { return false }
func (t simpleTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	if t.run == nil {
		return core.ToolResult{}, nil
	}
	return t.run(ctx, params)
}
func (t simpleTool) Validate(map[string]interface{}) error { return nil }
func (t simpleTool) InputSchema() models.InputSchema {
	return models.InputSchema{Type: "object", Properties: map[string]models.ParameterSchema{}}
}
func (t *simpleTool) CloneTool() core.Tool {
	if t == nil {
		return nil
	}
	cloned := *t
	return &cloned
}

type nonCloneableTool struct {
	name string
}

func (t nonCloneableTool) Name() string        { return t.name }
func (t nonCloneableTool) Description() string { return t.name }
func (t nonCloneableTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{Name: t.name, Description: t.name}
}
func (t nonCloneableTool) CanHandle(context.Context, string) bool { return false }
func (t nonCloneableTool) Execute(context.Context, map[string]interface{}) (core.ToolResult, error) {
	return core.ToolResult{}, nil
}
func (t nonCloneableTool) Validate(map[string]interface{}) error { return nil }
func (t nonCloneableTool) InputSchema() models.InputSchema {
	return models.InputSchema{Type: "object", Properties: map[string]models.ParameterSchema{}}
}

type stubLLM struct {
	results      []map[string]any
	index        int
	capabilities []core.Capability
	prompts      []string
}

func (m *stubLLM) Generate(context.Context, string, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, fmt.Errorf("unexpected Generate call")
}

func (m *stubLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, fmt.Errorf("unexpected GenerateWithJSON call")
}

func (m *stubLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	m.prompts = append(m.prompts, prompt)
	if m.index >= len(m.results) {
		return nil, fmt.Errorf("no more stubbed results")
	}
	result := m.results[m.index]
	m.index++
	return result, nil
}

func (m *stubLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, fmt.Errorf("unexpected CreateEmbedding call")
}

func (m *stubLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, fmt.Errorf("unexpected CreateEmbeddings call")
}

func (m *stubLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("unexpected StreamGenerate call")
}

func (m *stubLLM) GenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, fmt.Errorf("unexpected GenerateWithContent call")
}

func (m *stubLLM) StreamGenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("unexpected StreamGenerateWithContent call")
}

func (m *stubLLM) ProviderName() string            { return "stub" }
func (m *stubLLM) ModelID() string                 { return "stub-model" }
func (m *stubLLM) Capabilities() []core.Capability { return m.capabilities }

type nativeStubLLM struct {
	stubLLM
	messages [][]core.ChatMessage
}

func (m *nativeStubLLM) GenerateWithTools(ctx context.Context, messages []core.ChatMessage, tools []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	m.messages = append(m.messages, append([]core.ChatMessage(nil), messages...))
	if m.index >= len(m.results) {
		return nil, fmt.Errorf("no more stubbed results")
	}
	result := m.results[m.index]
	m.index++
	return result, nil
}
