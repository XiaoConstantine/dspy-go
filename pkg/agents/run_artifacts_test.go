package agents

import (
	"errors"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExecutionTraceFromEvents_BuildsCanonicalTrace(t *testing.T) {
	events := testRunEvents(time.Date(2026, time.July, 23, 20, 0, 0, 0, time.UTC))
	trace, err := ExecutionTraceFromEvents(events, ExecutionTraceConfig{
		AgentID:   "native-test",
		AgentType: "native",
		Input:     map[string]any{"task_id": "task-1"},
		Task:      "Solve task-1",
		ContextMetadata: map[string]any{
			"session_id": "session-1",
		},
	})
	require.NoError(t, err)
	require.NotNil(t, trace)
	assert.Equal(t, TraceStatusSuccess, trace.Status)
	assert.Equal(t, "finish", trace.TerminationCause)
	assert.Equal(t, int64(5), trace.TokenUsage["prompt_tokens"])
	assert.Equal(t, int64(3), trace.TokenUsage["completion_tokens"])
	assert.Equal(t, int64(8), trace.TokenUsage["total_tokens"])
	require.Len(t, trace.Steps, 2)
	assert.Equal(t, "need lookup", trace.Steps[0].Thought)
	assert.Equal(t, "lookup", trace.Steps[0].Tool)
	assert.Equal(t, "model summary", trace.Steps[0].Observation)
	assert.Equal(t, "operator display", trace.Steps[0].ObservationDisplay)
	assert.Equal(t, map[string]any{"id": 1}, trace.Steps[0].ObservationDetails)
	assert.Equal(t, "Finish", trace.Steps[1].Tool)
	assert.Equal(t, map[string]int{"lookup": 1}, trace.ToolUsageCount)
	assert.Equal(t, "session-1", trace.ContextMetadata["session_id"])
	assert.Equal(t, 2, trace.ContextMetadata["turns"])
}

func TestExecutionTraceFromEvents_UsesTerminalErrorAndStopReason(t *testing.T) {
	base := time.Date(2026, time.July, 23, 22, 0, 0, 0, time.UTC)
	events := []ExecutionEvent{
		{Timestamp: base, Payload: RunStartedEvent{RunID: "run-err", Task: "broken task"}},
		{Timestamp: base.Add(time.Second), Payload: RunFinishedEvent{RunID: "run-err", Status: RunStatusFailed, StopReason: StopReasonError, Err: errors.New("boom")}},
	}
	trace, err := ExecutionTraceFromEvents(events, ExecutionTraceConfig{AgentID: "a", AgentType: "native"})
	require.NoError(t, err)
	require.NotNil(t, trace)
	assert.Equal(t, TraceStatusFailure, trace.Status)
	assert.Equal(t, "boom", trace.Error)
	assert.Equal(t, "error", trace.TerminationCause)
}

func TestExecutionTraceFromEvents_TextCompletionCountsAsCompleted(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 30, 0, 0, time.UTC)
	trace, err := ExecutionTraceFromEvents([]ExecutionEvent{
		{Timestamp: base, Payload: RunStartedEvent{RunID: "run-text", Task: "answer directly"}},
		{Timestamp: base.Add(time.Second), Payload: MessageAddedEvent{RunID: "run-text", Turn: 1, Message: NewTextMessage(RoleAssistant, "done")}},
		{Timestamp: base.Add(2 * time.Second), Payload: TurnFinishedEvent{RunID: "run-text", Turn: 1, Status: OperationStatusCompleted, Usage: &core.TokenInfo{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}}},
		{Timestamp: base.Add(3 * time.Second), Payload: RunFinishedEvent{RunID: "run-text", Status: RunStatusCompleted, StopReason: StopReasonText, Turns: 1, FinalAnswer: "done"}},
	}, ExecutionTraceConfig{RunID: "run-text", AgentID: "a", AgentType: "native"})
	require.NoError(t, err)
	require.NotNil(t, trace)
	assert.Equal(t, true, trace.Output["completed"])
	assert.Equal(t, "done", trace.Output["final_answer"])
	assert.Equal(t, TraceStatusSuccess, trace.Status)
	assert.Equal(t, "text", trace.TerminationCause)
	require.Len(t, trace.Steps, 1)
	assert.True(t, trace.Steps[0].Success)
	assert.Empty(t, trace.Steps[0].Error)
}

func TestExecutionTraceFromEvents_RejectsMixedRunsWithoutSelection(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 45, 0, 0, time.UTC)
	_, err := ExecutionTraceFromEvents([]ExecutionEvent{
		{Timestamp: base, Payload: RunStartedEvent{RunID: "run-a", Task: "a"}},
		{Timestamp: base.Add(time.Second), Payload: RunFinishedEvent{RunID: "run-a", Status: RunStatusCompleted, StopReason: StopReasonText, FinalAnswer: "a"}},
		{Timestamp: base.Add(2 * time.Second), Payload: RunStartedEvent{RunID: "run-b", Task: "b"}},
		{Timestamp: base.Add(3 * time.Second), Payload: RunFinishedEvent{RunID: "run-b", Status: RunStatusCompleted, StopReason: StopReasonText, FinalAnswer: "b"}},
	}, ExecutionTraceConfig{AgentID: "a", AgentType: "native"})
	require.ErrorContains(t, err, "matching run lifecycles")
}

func TestExecutionTraceFromEvents_RejectsRepeatedOrEmptyRunLifecycles(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 47, 0, 0, time.UTC)
	for _, test := range []struct {
		name     string
		runID    string
		selected string
	}{
		{name: "empty IDs"},
		{name: "reused ID", runID: "same", selected: "same"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := ExecutionTraceFromEvents([]ExecutionEvent{
				{Timestamp: base, Payload: RunStartedEvent{RunID: test.runID, Task: "prompt"}},
				{Timestamp: base.Add(time.Second), Payload: RunFinishedEvent{RunID: test.runID, Status: RunStatusCompleted, StopReason: StopReasonText, FinalAnswer: "first"}},
				{Timestamp: base.Add(2 * time.Second), Payload: RunStartedEvent{RunID: test.runID, Task: "continue"}},
				{Timestamp: base.Add(3 * time.Second), Payload: RunFinishedEvent{RunID: test.runID, Status: RunStatusCompleted, StopReason: StopReasonText, FinalAnswer: "second"}},
			}, ExecutionTraceConfig{RunID: test.selected, AgentID: "a", AgentType: "native"})
			require.ErrorContains(t, err, "matching run lifecycles")
		})
	}
}

func TestExecutionTraceFromEvents_SelectsExplicitRunID(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 50, 0, 0, time.UTC)
	trace, err := ExecutionTraceFromEvents([]ExecutionEvent{
		{Timestamp: base, Payload: RunStartedEvent{RunID: "run-a", Task: "a"}},
		{Timestamp: base.Add(time.Second), Payload: RunFinishedEvent{RunID: "run-a", Status: RunStatusCompleted, StopReason: StopReasonText, FinalAnswer: "a"}},
		{Timestamp: base.Add(2 * time.Second), Payload: RunStartedEvent{RunID: "run-b", Task: "b"}},
		{Timestamp: base.Add(3 * time.Second), Payload: RunFinishedEvent{RunID: "run-b", Status: RunStatusCompleted, StopReason: StopReasonText, FinalAnswer: "b"}},
	}, ExecutionTraceConfig{RunID: "run-b", AgentID: "a", AgentType: "native"})
	require.NoError(t, err)
	require.NotNil(t, trace)
	assert.Equal(t, "b", trace.Task)
	assert.Equal(t, "b", trace.Output["final_answer"])
}

func TestExecutionTraceFromEvents_CustomFinishOutcomeIsNotToolUsage(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 52, 0, 0, time.UTC)
	call := core.ToolCall{ID: "done-1", Name: "Done", Arguments: map[string]any{"answer": "complete"}}
	trace, err := ExecutionTraceFromEvents([]ExecutionEvent{
		{Timestamp: base, Payload: RunStartedEvent{RunID: "run-1", Task: "finish with Done"}},
		{Timestamp: base.Add(time.Second), Payload: MessageAddedEvent{RunID: "run-1", Turn: 1, Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{call}}}},
		{Timestamp: base.Add(2 * time.Second), Payload: ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: call, Outcome: ToolCallOutcomeFinish, Status: OperationStatusCompleted}},
		{Timestamp: base.Add(3 * time.Second), Payload: TurnFinishedEvent{RunID: "run-1", Turn: 1, Status: OperationStatusCompleted}},
		{Timestamp: base.Add(4 * time.Second), Payload: RunFinishedEvent{RunID: "run-1", Status: RunStatusCompleted, StopReason: StopReasonFinish, Turns: 1, FinalAnswer: "complete"}},
	}, ExecutionTraceConfig{RunID: "run-1", AgentID: "a", AgentType: "native"})
	require.NoError(t, err)
	require.NotNil(t, trace)
	assert.Empty(t, trace.ToolUsageCount)
	require.Len(t, trace.Steps, 1)
	assert.Equal(t, "Done", trace.Steps[0].Tool)
	assert.True(t, trace.Steps[0].Success)
}

func TestExecutionTraceFromEvents_CorrelatesRepeatedEmptyIDCallsByToolIndex(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 55, 0, 0, time.UTC)
	calls := []core.ToolCall{
		{Name: "lookup", Arguments: map[string]any{"query": "first"}},
		{Name: "lookup", Arguments: map[string]any{"query": "second"}},
	}
	first := NewToolResultMessage("", "lookup", core.ToolResult{Data: "first result"})
	second := NewToolResultMessage("", "lookup", core.ToolResult{Data: "second result"})
	trace, err := ExecutionTraceFromEvents([]ExecutionEvent{
		{Timestamp: base, Payload: RunStartedEvent{RunID: "run-1", Task: "lookup twice"}},
		{Timestamp: base.Add(time.Second), Payload: MessageAddedEvent{RunID: "run-1", Turn: 1, Message: Message{Role: RoleAssistant, ToolCalls: calls}}},
		{Timestamp: base.Add(2 * time.Second), Payload: ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: calls[0], Outcome: ToolCallOutcomeExecuted, Result: &first, Status: OperationStatusCompleted}},
		{Timestamp: base.Add(3 * time.Second), Payload: MessageAddedEvent{RunID: "run-1", Turn: 1, Message: first}},
		{Timestamp: base.Add(4 * time.Second), Payload: ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 1, Call: calls[1], Outcome: ToolCallOutcomeExecuted, Result: &second, Status: OperationStatusCompleted}},
		{Timestamp: base.Add(5 * time.Second), Payload: MessageAddedEvent{RunID: "run-1", Turn: 1, Message: second}},
		{Timestamp: base.Add(6 * time.Second), Payload: TurnFinishedEvent{RunID: "run-1", Turn: 1, Status: OperationStatusCompleted}},
		{Timestamp: base.Add(7 * time.Second), Payload: RunFinishedEvent{RunID: "run-1", Status: RunStatusStopped, StopReason: StopReasonMaxTurns, Turns: 1}},
	}, ExecutionTraceConfig{RunID: "run-1", AgentID: "a", AgentType: "native"})
	require.NoError(t, err)
	require.Len(t, trace.Steps, 2)
	assert.Equal(t, "first result", trace.Steps[0].Observation)
	assert.Equal(t, "second result", trace.Steps[1].Observation)
}

func TestExecutionTraceFromEvents_RejectsMismatchedOutcome(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 56, 0, 0, time.UTC)
	call := core.ToolCall{ID: "expected", Name: "lookup"}
	wrong := NewToolResultMessage("other", "lookup", core.ToolResult{Data: "wrong result"})
	_, err := ExecutionTraceFromEvents([]ExecutionEvent{
		{Timestamp: base, Payload: RunStartedEvent{RunID: "run-1", Task: "lookup"}},
		{Timestamp: base.Add(time.Second), Payload: MessageAddedEvent{RunID: "run-1", Turn: 1, Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{call}}}},
		{Timestamp: base.Add(2 * time.Second), Payload: ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: call, Outcome: ToolCallOutcomeExecuted, Result: &wrong, Status: OperationStatusCompleted}},
		{Timestamp: base.Add(3 * time.Second), Payload: MessageAddedEvent{RunID: "run-1", Turn: 1, Message: wrong}},
		{Timestamp: base.Add(4 * time.Second), Payload: RunFinishedEvent{RunID: "run-1", Status: RunStatusStopped, StopReason: StopReasonMaxTurns, Turns: 1}},
	}, ExecutionTraceConfig{RunID: "run-1", AgentID: "a", AgentType: "native"})
	require.ErrorContains(t, err, "does not match its call")
}

func testRunEvents(base time.Time) []ExecutionEvent {
	assistant := Message{
		Role:    RoleAssistant,
		Content: []core.ContentBlock{core.NewTextBlock("need lookup")},
		ToolCalls: []core.ToolCall{
			{ID: "call-1", Name: "lookup", Arguments: map[string]any{"query": "abc"}},
		},
	}
	toolResult := Message{
		Role: RoleTool,
		ToolResult: &MessageToolResult{
			ToolCallID:     "call-1",
			Name:           "lookup",
			Content:        []core.ContentBlock{core.NewTextBlock("model summary")},
			DisplayContent: []core.ContentBlock{core.NewTextBlock("operator display")},
			Details:        map[string]any{"id": 1},
		},
	}
	finish := Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{{ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "done"}}}}
	return []ExecutionEvent{
		{Timestamp: base, Payload: RunStartedEvent{RunID: "run-1", Task: "Solve task-1", Model: "model-1", Provider: "provider-1"}},
		{Timestamp: base.Add(time.Second), Payload: MessageAddedEvent{RunID: "run-1", Turn: 1, Message: assistant}},
		{Timestamp: base.Add(2 * time.Second), Payload: ToolCallProposedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: assistant.ToolCalls[0]}},
		{Timestamp: base.Add(2500 * time.Millisecond), Payload: ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: assistant.ToolCalls[0], Outcome: ToolCallOutcomeExecuted, Result: &toolResult, Status: OperationStatusCompleted}},
		{Timestamp: base.Add(3 * time.Second), Payload: MessageAddedEvent{RunID: "run-1", Turn: 1, Message: toolResult}},
		{Timestamp: base.Add(4 * time.Second), Payload: TurnFinishedEvent{RunID: "run-1", Turn: 1, Status: OperationStatusCompleted, Usage: &core.TokenInfo{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3}}},
		{Timestamp: base.Add(5 * time.Second), Payload: MessageAddedEvent{RunID: "run-1", Turn: 2, Message: finish}},
		{Timestamp: base.Add(5500 * time.Millisecond), Payload: ToolCallFinishedEvent{RunID: "run-1", Turn: 2, ToolIndex: 0, Call: finish.ToolCalls[0], Outcome: ToolCallOutcomeFinish, Status: OperationStatusCompleted}},
		{Timestamp: base.Add(6 * time.Second), Payload: TurnFinishedEvent{RunID: "run-1", Turn: 2, Status: OperationStatusCompleted, Usage: &core.TokenInfo{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}}},
		{Timestamp: base.Add(7 * time.Second), Payload: RunFinishedEvent{RunID: "run-1", Status: RunStatusCompleted, StopReason: StopReasonFinish, FinalAnswer: "done"}},
	}
}
