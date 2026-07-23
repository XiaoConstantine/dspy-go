package native

import (
	"context"
	"errors"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests intentionally characterize the native loop before it moves into
// pkg/agents. Some assertions record legacy event/error behavior that the typed
// loop will replace deliberately rather than changing accidentally.

func TestAgent_Execute_CharacterizesUnknownToolRecovery(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"id":        "call-missing",
					"name":      "missing_tool",
					"arguments": map[string]any{"value": "x"},
				},
			},
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "recovered"},
				},
			},
		},
	}

	var events []agents.AgentEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 2,
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)

	result, err := agent.Execute(context.Background(), map[string]any{"task": "Recover from an unknown tool."})
	require.NoError(t, err)
	assert.Equal(t, true, result["completed"])
	assert.Equal(t, "recovered", result["final_answer"])
	require.Len(t, llm.prompts, 2)
	assert.Contains(t, llm.prompts[1], `unknown tool "missing_tool"`)

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	require.Len(t, trace.Steps, 2)
	assert.True(t, trace.Steps[0].IsError)
	assert.Equal(t, "missing_tool", trace.Steps[0].ToolName)
	assert.Contains(t, trace.Steps[0].Observation, "tool not found")

	// Legacy behavior emits only a proposed event for lookup failures. The
	// typed loop will close this lifecycle with an error tool-result event.
	assert.NotNil(t, findEvent(events, agents.EventToolCallProposed, "missing_tool"))
	assert.Nil(t, findEvent(events, agents.EventToolCallStarted, "missing_tool"))
	assert.Nil(t, findEvent(events, agents.EventToolCallFinished, "missing_tool"))
}

func TestAgent_Execute_CharacterizesInvalidArgumentsRecovery(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"id":        "call-invalid",
					"name":      "strict_tool",
					"arguments": map[string]any{"value": 42},
				},
			},
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "recovered"},
				},
			},
		},
	}

	var events []agents.AgentEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 2,
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(validationFailureTool{name: "strict_tool"}))

	result, err := agent.Execute(context.Background(), map[string]any{"task": "Recover from invalid arguments."})
	require.NoError(t, err)
	assert.Equal(t, true, result["completed"])
	require.Len(t, llm.prompts, 2)
	assert.Contains(t, llm.prompts[1], "invalid tool arguments: value must be a string")

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	require.Len(t, trace.Steps, 2)
	assert.True(t, trace.Steps[0].IsError)
	assert.Contains(t, trace.Steps[0].Observation, "value must be a string")
	assert.NotNil(t, findEvent(events, agents.EventToolCallProposed, "strict_tool"))
	assert.Nil(t, findEvent(events, agents.EventToolCallStarted, "strict_tool"))
	assert.Nil(t, findEvent(events, agents.EventToolCallFinished, "strict_tool"))
}

func TestAgent_Execute_CharacterizesMaxTurnExhaustion(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "write_note",
					"arguments": map[string]any{"content": "not finished"},
				},
			},
		},
	}

	var events []agents.AgentEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 1,
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{name: "write_note"}))

	result, err := agent.Execute(context.Background(), map[string]any{"task": "Do not finish in time."})
	// Legacy Execute reports exhaustion in the result map, not the Go error.
	require.NoError(t, err)
	assert.Equal(t, false, result["completed"])
	assert.Contains(t, result["error"], "max turns reached without Finish after 1 turns")

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	assert.False(t, trace.Completed)
	assert.Contains(t, trace.Error, "max turns reached")
	assert.NotContains(t, eventTypes(events), agents.EventRunFailed)

	finished := firstEvent(events, agents.EventRunFinished)
	require.NotNil(t, finished)
	assert.Equal(t, false, finished.Data["completed"])
	assert.Contains(t, finished.Data["error"], "max turns reached")
}

func TestAgent_Execute_CharacterizesCanceledModelCall(t *testing.T) {
	llm := &cancelAwareLLM{stubLLM: stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
	}}

	var events []agents.AgentEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 1,
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := agent.Execute(ctx, map[string]any{"task": "Observe cancellation."})
	assert.Nil(t, result)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
	assert.Equal(t, []string{
		agents.EventRunStarted,
		agents.EventLLMTurnStarted,
		agents.EventLLMTurnFinished,
		agents.EventRunFailed,
		agents.EventRunFinished,
	}, eventTypes(events))
}

func eventTypes(events []agents.AgentEvent) []string {
	types := make([]string, 0, len(events))
	for _, event := range events {
		types = append(types, event.Type)
	}
	return types
}

func firstEvent(events []agents.AgentEvent, eventType string) *agents.AgentEvent {
	for i := range events {
		if events[i].Type == eventType {
			return &events[i]
		}
	}
	return nil
}

type validationFailureTool struct {
	name string
}

func (t validationFailureTool) Name() string        { return t.name }
func (t validationFailureTool) Description() string { return t.name }
func (t validationFailureTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{Name: t.name, Description: t.name}
}
func (t validationFailureTool) CanHandle(context.Context, string) bool { return false }
func (t validationFailureTool) Execute(context.Context, map[string]any) (core.ToolResult, error) {
	return core.ToolResult{}, errors.New("unexpected execution")
}
func (t validationFailureTool) Validate(map[string]any) error {
	return errors.New("value must be a string")
}
func (t validationFailureTool) InputSchema() models.InputSchema {
	return models.InputSchema{Type: "object", Properties: map[string]models.ParameterSchema{}}
}

type cancelAwareLLM struct {
	stubLLM
}

func (m *cancelAwareLLM) GenerateWithFunctions(ctx context.Context, _ string, _ []map[string]any, _ ...core.GenerateOption) (map[string]any, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return nil, errors.New("expected canceled context")
}
