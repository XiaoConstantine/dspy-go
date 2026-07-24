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

// These tests preserve native wrapper behavior after migration to the shared
// typed execution lifecycle.

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

	var events []agents.ExecutionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 2,
		EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
			events = append(events, event)
		}),
	})
	require.NoError(t, err)

	result, err := agent.Execute(context.Background(), map[string]any{"task": "Recover from an unknown tool."})
	require.NoError(t, err)
	assert.Equal(t, true, result["completed"])
	assert.Equal(t, "recovered", result["final_answer"])
	assert.Equal(t, 2, result["turns"])
	assert.Equal(t, 1, result["tool_calls"])
	require.Len(t, llm.prompts, 2)
	assert.Contains(t, llm.prompts[1], `unknown tool "missing_tool"`)

	trace := agent.LastExecutionTrace()
	require.NotNil(t, trace)
	require.Len(t, trace.Steps, 2)
	assert.False(t, trace.Steps[0].Success)
	assert.Equal(t, "missing_tool", trace.Steps[0].Tool)
	assert.Contains(t, trace.Steps[0].Observation, "tool not found")

	require.NoError(t, agents.ValidateEventLifecycle(events))
	require.NotNil(t, findToolProposedEvent(events, "missing_tool"))
	assert.Nil(t, findToolStartedEvent(events, "missing_tool"))
	finished := findToolFinishedEvent(events, "missing_tool")
	require.NotNil(t, finished)
	assert.Equal(t, agents.ToolCallOutcomeRejected, finished.Outcome)
	require.NotNil(t, finished.Result)
	require.NotNil(t, finished.Result.ToolResult)
	assert.True(t, finished.Result.ToolResult.IsError)
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

	var events []agents.ExecutionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 2,
		EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
			events = append(events, event)
		}),
	})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(validationFailureTool{name: "strict_tool"}))

	result, err := agent.Execute(context.Background(), map[string]any{"task": "Recover from invalid arguments."})
	require.NoError(t, err)
	assert.Equal(t, true, result["completed"])
	require.Len(t, llm.prompts, 2)
	assert.Contains(t, llm.prompts[1], "invalid tool arguments: value must be a string")

	trace := agent.LastExecutionTrace()
	require.NotNil(t, trace)
	require.Len(t, trace.Steps, 2)
	assert.False(t, trace.Steps[0].Success)
	assert.Contains(t, trace.Steps[0].Observation, "value must be a string")
	require.NoError(t, agents.ValidateEventLifecycle(events))
	require.NotNil(t, findToolProposedEvent(events, "strict_tool"))
	assert.Nil(t, findToolStartedEvent(events, "strict_tool"))
	finished := findToolFinishedEvent(events, "strict_tool")
	require.NotNil(t, finished)
	assert.Equal(t, agents.ToolCallOutcomeRejected, finished.Outcome)
	require.NotNil(t, finished.Result)
	require.NotNil(t, finished.Result.ToolResult)
	assert.True(t, finished.Result.ToolResult.IsError)
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

	var events []agents.ExecutionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 1,
		EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
			events = append(events, event)
		}),
	})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{name: "write_note"}))

	result, err := agent.Execute(context.Background(), map[string]any{"task": "Do not finish in time."})
	// Legacy Execute reports exhaustion in the result map, not the Go error.
	require.NoError(t, err)
	assert.Equal(t, false, result["completed"])
	assert.Contains(t, result["error"], "max turns reached without Finish after 1 turns")

	trace := agent.LastExecutionTrace()
	require.NotNil(t, trace)
	assert.Equal(t, agents.TraceStatusPartial, trace.Status)
	assert.Contains(t, trace.Error, "max turns reached")
	require.NoError(t, agents.ValidateEventLifecycle(events))
	finished := findRunFinishedEvent(events)
	require.NotNil(t, finished)
	assert.Equal(t, agents.RunStatusStopped, finished.Status)
	assert.Equal(t, agents.StopReasonMaxTurns, finished.StopReason)
	assert.Contains(t, finished.Diagnostic, "max turns reached without completion")
}

func TestAgent_Execute_CharacterizesCanceledModelCall(t *testing.T) {
	llm := &cancelAwareLLM{stubLLM: stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
	}}

	var events []agents.ExecutionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 1,
		EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
			events = append(events, event)
		}),
	})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := agent.Execute(ctx, map[string]any{"task": "Observe cancellation."})
	assert.Nil(t, result)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
	require.NoError(t, agents.ValidateEventLifecycle(events))
	runFinished := findRunFinishedEvent(events)
	require.NotNil(t, runFinished)
	assert.Equal(t, agents.RunStatusCanceled, runFinished.Status)
	assert.Equal(t, agents.StopReasonCanceled, runFinished.StopReason)
	assert.ErrorIs(t, runFinished.Err, context.Canceled)
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
