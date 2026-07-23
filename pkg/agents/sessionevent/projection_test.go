package sessionevent

import (
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEntriesFromEvents_PreservesOrderingAndCorrelation(t *testing.T) {
	events := testProjectionEvents(time.Date(2026, time.July, 23, 20, 0, 0, 0, time.UTC))
	entries, err := EntriesFromEvents(events, "session-1", "branch-1", EventProjectionConfig{
		RunID:    "run-1",
		TaskID:   "task-1",
		Source:   "native",
		UserText: "Solve task-1",
	})
	require.NoError(t, err)
	require.Len(t, entries, 6)
	assert.Equal(t, EntryKindUserMessage, entries[0].Kind)
	assert.Equal(t, EntryKindAssistantMessage, entries[1].Kind)
	assert.Equal(t, "need lookup", entries[1].Payload["text"])
	assert.Equal(t, EntryKindToolCall, entries[2].Kind)
	assert.Equal(t, "call-1", entries[2].Payload["tool_call_id"])
	assert.Equal(t, 0, entries[2].Payload["tool_index"])
	assert.Equal(t, EntryKindToolResult, entries[3].Kind)
	assert.Equal(t, "call-1", entries[3].Payload["tool_call_id"])
	assert.Equal(t, 0, entries[3].Payload["tool_index"])
	assert.Equal(t, "operator display", entries[3].Payload["observation_display"])
	assert.Equal(t, EntryKindAssistantMessage, entries[4].Kind)
	assert.Equal(t, "done", entries[4].Payload["text"])
	assert.Equal(t, EntryKindSystemEvent, entries[5].Kind)
	assert.Equal(t, "run_finished", entries[5].Payload["event"])
	assert.Equal(t, int64(8), entries[5].TotalTokens)
	assert.Equal(t, true, entries[5].Payload["completed"])
}

func TestEntriesFromEvents_AppendsFinalAnswerWithoutDuplicatingAssistantText(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 0, 0, 0, time.UTC)
	entries, err := EntriesFromEvents([]agents.ExecutionEvent{
		{Timestamp: base, Payload: agents.RunStartedEvent{RunID: "run-1", Task: "Finish task", Model: "m", Provider: "p"}},
		{Timestamp: base.Add(time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: agents.Message{Role: agents.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "done"}}}}}},
		{Timestamp: base.Add(1500 * time.Millisecond), Payload: agents.ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: core.ToolCall{ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "done"}}, Outcome: agents.ToolCallOutcomeFinish, Status: agents.OperationStatusCompleted}},
		{Timestamp: base.Add(2 * time.Second), Payload: agents.TurnFinishedEvent{RunID: "run-1", Turn: 1, Status: agents.OperationStatusCompleted, Usage: &core.TokenInfo{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}}},
		{Timestamp: base.Add(3 * time.Second), Payload: agents.RunFinishedEvent{RunID: "run-1", Status: agents.RunStatusCompleted, StopReason: agents.StopReasonFinish, FinalAnswer: "done"}},
	}, "session-1", "branch-1", EventProjectionConfig{RunID: "run-1", TaskID: "task-1"})
	require.NoError(t, err)
	require.Len(t, entries, 3)
	assert.Equal(t, EntryKindAssistantMessage, entries[1].Kind)
	assert.Equal(t, "done", entries[1].Payload["text"])
}

func TestEntriesFromEvents_TextCompletionMarksCompleted(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 10, 0, 0, time.UTC)
	entries, err := EntriesFromEvents([]agents.ExecutionEvent{
		{Timestamp: base, Payload: agents.RunStartedEvent{RunID: "run-text", Task: "Answer directly", Model: "m", Provider: "p"}},
		{Timestamp: base.Add(time.Second), Payload: agents.MessageAddedEvent{RunID: "run-text", Turn: 1, Message: agents.NewTextMessage(agents.RoleAssistant, "done")}},
		{Timestamp: base.Add(2 * time.Second), Payload: agents.TurnFinishedEvent{RunID: "run-text", Turn: 1, Status: agents.OperationStatusCompleted, Usage: &core.TokenInfo{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}}},
		{Timestamp: base.Add(3 * time.Second), Payload: agents.RunFinishedEvent{RunID: "run-text", Status: agents.RunStatusCompleted, StopReason: agents.StopReasonText, FinalAnswer: "done"}},
	}, "session-1", "branch-1", EventProjectionConfig{RunID: "run-text", TaskID: "task-text"})
	require.NoError(t, err)
	require.Len(t, entries, 3)
	assert.Equal(t, true, entries[2].Payload["completed"])
	assert.Equal(t, "text", entries[2].Payload["stop_reason"])
}

func TestEntriesFromEvents_RejectsMixedRunsWithoutSelection(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 20, 0, 0, time.UTC)
	_, err := EntriesFromEvents([]agents.ExecutionEvent{
		{Timestamp: base, Payload: agents.RunStartedEvent{RunID: "run-a", Task: "a"}},
		{Timestamp: base.Add(time.Second), Payload: agents.RunFinishedEvent{RunID: "run-a", Status: agents.RunStatusCompleted, StopReason: agents.StopReasonText, FinalAnswer: "a"}},
		{Timestamp: base.Add(2 * time.Second), Payload: agents.RunStartedEvent{RunID: "run-b", Task: "b"}},
		{Timestamp: base.Add(3 * time.Second), Payload: agents.RunFinishedEvent{RunID: "run-b", Status: agents.RunStatusCompleted, StopReason: agents.StopReasonText, FinalAnswer: "b"}},
	}, "session-1", "branch-1", EventProjectionConfig{})
	require.ErrorContains(t, err, "matching run lifecycles")
}

func TestEntriesFromEvents_RejectsRepeatedOrEmptyRunLifecycles(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 25, 0, 0, time.UTC)
	for _, test := range []struct {
		name     string
		runID    string
		selected string
	}{
		{name: "empty IDs"},
		{name: "reused ID", runID: "same", selected: "same"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := EntriesFromEvents([]agents.ExecutionEvent{
				{Timestamp: base, Payload: agents.RunStartedEvent{RunID: test.runID, Task: "prompt"}},
				{Timestamp: base.Add(time.Second), Payload: agents.RunFinishedEvent{RunID: test.runID, Status: agents.RunStatusCompleted, StopReason: agents.StopReasonText}},
				{Timestamp: base.Add(2 * time.Second), Payload: agents.RunStartedEvent{RunID: test.runID, Task: "continue"}},
				{Timestamp: base.Add(3 * time.Second), Payload: agents.RunFinishedEvent{RunID: test.runID, Status: agents.RunStatusCompleted, StopReason: agents.StopReasonText}},
			}, "session-1", "branch-1", EventProjectionConfig{RunID: test.selected})
			require.ErrorContains(t, err, "matching run lifecycles")
		})
	}
}

func TestEntriesFromEvents_CorrelatesRepeatedEmptyIDCallsByToolIndex(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 27, 0, 0, time.UTC)
	calls := []core.ToolCall{{Name: "lookup"}, {Name: "lookup"}}
	first := agents.NewToolResultMessage("", "lookup", core.ToolResult{Data: "first"})
	second := agents.NewToolResultMessage("", "lookup", core.ToolResult{Data: "second"})
	entries, err := EntriesFromEvents([]agents.ExecutionEvent{
		{Timestamp: base, Payload: agents.RunStartedEvent{RunID: "run-1", Task: "lookup twice"}},
		{Timestamp: base.Add(time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: agents.Message{Role: agents.RoleAssistant, ToolCalls: calls}}},
		{Timestamp: base.Add(2 * time.Second), Payload: agents.ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: calls[0], Outcome: agents.ToolCallOutcomeExecuted, Result: &first, Status: agents.OperationStatusCompleted}},
		{Timestamp: base.Add(3 * time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: first}},
		{Timestamp: base.Add(4 * time.Second), Payload: agents.ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 1, Call: calls[1], Outcome: agents.ToolCallOutcomeExecuted, Result: &second, Status: agents.OperationStatusCompleted}},
		{Timestamp: base.Add(5 * time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: second}},
		{Timestamp: base.Add(6 * time.Second), Payload: agents.RunFinishedEvent{RunID: "run-1", Status: agents.RunStatusStopped, StopReason: agents.StopReasonMaxTurns}},
	}, "session-1", "branch-1", EventProjectionConfig{RunID: "run-1"})
	require.NoError(t, err)
	require.Len(t, entries, 6)
	assert.Equal(t, 0, entries[1].Payload["tool_index"])
	assert.Equal(t, 1, entries[2].Payload["tool_index"])
	assert.Equal(t, 0, entries[3].Payload["tool_index"])
	assert.Equal(t, 1, entries[4].Payload["tool_index"])
}

func TestEntriesFromEvents_RejectsMismatchedToolResult(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 28, 0, 0, time.UTC)
	call := core.ToolCall{ID: "expected", Name: "lookup"}
	wrong := agents.NewToolResultMessage("other", "lookup", core.ToolResult{Data: "wrong"})
	_, err := EntriesFromEvents([]agents.ExecutionEvent{
		{Timestamp: base, Payload: agents.RunStartedEvent{RunID: "run-1", Task: "lookup"}},
		{Timestamp: base.Add(time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: agents.Message{Role: agents.RoleAssistant, ToolCalls: []core.ToolCall{call}}}},
		{Timestamp: base.Add(2 * time.Second), Payload: agents.ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: call, Outcome: agents.ToolCallOutcomeExecuted, Result: &wrong, Status: agents.OperationStatusCompleted}},
		{Timestamp: base.Add(3 * time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: wrong}},
		{Timestamp: base.Add(4 * time.Second), Payload: agents.RunFinishedEvent{RunID: "run-1", Status: agents.RunStatusStopped, StopReason: agents.StopReasonMaxTurns}},
	}, "session-1", "branch-1", EventProjectionConfig{RunID: "run-1"})
	require.ErrorContains(t, err, "no matching finished lifecycle")
}

func TestEntriesFromEvents_CustomFinishOutcomeDoesNotPersistToolCall(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 28, 30, 0, time.UTC)
	call := core.ToolCall{ID: "done-1", Name: "Done", Arguments: map[string]any{"answer": "complete"}}
	entries, err := EntriesFromEvents([]agents.ExecutionEvent{
		{Timestamp: base, Payload: agents.RunStartedEvent{RunID: "run-1", Task: "finish with Done"}},
		{Timestamp: base.Add(time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: agents.Message{Role: agents.RoleAssistant, ToolCalls: []core.ToolCall{call}}}},
		{Timestamp: base.Add(2 * time.Second), Payload: agents.ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: call, Outcome: agents.ToolCallOutcomeFinish, Status: agents.OperationStatusCompleted}},
		{Timestamp: base.Add(3 * time.Second), Payload: agents.TurnFinishedEvent{RunID: "run-1", Turn: 1, Status: agents.OperationStatusCompleted}},
		{Timestamp: base.Add(4 * time.Second), Payload: agents.RunFinishedEvent{RunID: "run-1", Status: agents.RunStatusCompleted, StopReason: agents.StopReasonFinish, Turns: 1, FinalAnswer: "complete"}},
	}, "session-1", "branch-1", EventProjectionConfig{RunID: "run-1"})
	require.NoError(t, err)
	require.Len(t, entries, 3)
	assert.Equal(t, EntryKindUserMessage, entries[0].Kind)
	assert.Equal(t, EntryKindAssistantMessage, entries[1].Kind)
	assert.Equal(t, "complete", entries[1].Payload["text"])
	assert.Equal(t, EntryKindSystemEvent, entries[2].Kind)
}

func TestEntriesFromEvents_PersistsRejectedFinishCallAndResult(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 29, 0, 0, time.UTC)
	call := core.ToolCall{ID: "finish-bad", Name: "Finish", Arguments: map[string]any{}}
	result := agents.NewToolResultMessage(call.ID, call.Name, core.ToolResult{Data: "missing answer", Metadata: map[string]any{core.ToolResultIsErrorMeta: true, core.ToolResultSyntheticMeta: true}})
	entries, err := EntriesFromEvents([]agents.ExecutionEvent{
		{Timestamp: base, Payload: agents.RunStartedEvent{RunID: "run-1", Task: "finish"}},
		{Timestamp: base.Add(time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: agents.Message{Role: agents.RoleAssistant, ToolCalls: []core.ToolCall{call}}}},
		{Timestamp: base.Add(2 * time.Second), Payload: agents.ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: call, Outcome: agents.ToolCallOutcomeRejected, Result: &result, Status: agents.OperationStatusFailed}},
		{Timestamp: base.Add(3 * time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: result}},
		{Timestamp: base.Add(4 * time.Second), Payload: agents.RunFinishedEvent{RunID: "run-1", Status: agents.RunStatusStopped, StopReason: agents.StopReasonMaxTurns}},
	}, "session-1", "branch-1", EventProjectionConfig{RunID: "run-1"})
	require.NoError(t, err)
	require.Len(t, entries, 4)
	assert.Equal(t, EntryKindToolCall, entries[1].Kind)
	assert.Equal(t, "Finish", entries[1].ToolName)
	assert.Equal(t, EntryKindToolResult, entries[2].Kind)
	assert.Equal(t, "finish-bad", entries[2].Payload["tool_call_id"])
	assert.True(t, entries[2].IsError)
}

func TestEntriesFromEvents_PersistsMixedRejectedFinishCorrelation(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 29, 30, 0, time.UTC)
	calls := []core.ToolCall{
		{ID: "lookup-1", Name: "lookup"},
		{ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "done"}},
	}
	lookupResult := agents.NewToolResultMessage("lookup-1", "lookup", core.ToolResult{Data: "found"})
	finishResult := agents.NewToolResultMessage("finish-1", "Finish", core.ToolResult{Data: "Finish must be the only tool call", Metadata: map[string]any{core.ToolResultIsErrorMeta: true, core.ToolResultSyntheticMeta: true}})
	entries, err := EntriesFromEvents([]agents.ExecutionEvent{
		{Timestamp: base, Payload: agents.RunStartedEvent{RunID: "run-1", Task: "mixed"}},
		{Timestamp: base.Add(time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: agents.Message{Role: agents.RoleAssistant, ToolCalls: calls}}},
		{Timestamp: base.Add(2 * time.Second), Payload: agents.ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: calls[0], Outcome: agents.ToolCallOutcomeExecuted, Result: &lookupResult, Status: agents.OperationStatusCompleted}},
		{Timestamp: base.Add(3 * time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: lookupResult}},
		{Timestamp: base.Add(4 * time.Second), Payload: agents.ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 1, Call: calls[1], Outcome: agents.ToolCallOutcomeRejected, Result: &finishResult, Status: agents.OperationStatusFailed}},
		{Timestamp: base.Add(5 * time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: finishResult}},
		{Timestamp: base.Add(6 * time.Second), Payload: agents.RunFinishedEvent{RunID: "run-1", Status: agents.RunStatusStopped, StopReason: agents.StopReasonMaxTurns}},
	}, "session-1", "branch-1", EventProjectionConfig{RunID: "run-1"})
	require.NoError(t, err)
	require.Len(t, entries, 6)
	assert.Equal(t, "lookup-1", entries[1].Payload["tool_call_id"])
	assert.Equal(t, "finish-1", entries[2].Payload["tool_call_id"])
	assert.Equal(t, "lookup-1", entries[3].Payload["tool_call_id"])
	assert.Equal(t, "finish-1", entries[4].Payload["tool_call_id"])
}

func TestEntriesFromEvents_PreservesStoppedRunDiagnosticForRecall(t *testing.T) {
	base := time.Date(2026, time.July, 23, 21, 30, 0, 0, time.UTC)
	entries, err := EntriesFromEvents([]agents.ExecutionEvent{
		{Timestamp: base, Payload: agents.RunStartedEvent{RunID: "run-stop", Task: "Use tool"}},
		{Timestamp: base.Add(time.Second), Payload: agents.RunFinishedEvent{RunID: "run-stop", Status: agents.RunStatusStopped, StopReason: agents.StopReasonNoToolCalls, Diagnostic: "repeated model responses without tool calls after 3 turns"}},
	}, "session-1", "branch-1", EventProjectionConfig{RunID: "run-stop", TaskID: "task-stop"})
	require.NoError(t, err)
	require.Len(t, entries, 2)
	assert.Equal(t, "repeated model responses without tool calls after 3 turns", entries[1].SearchText)
	assert.Equal(t, "no_tool_calls", entries[1].Payload["stop_reason"])
	assert.Equal(t, "repeated model responses without tool calls after 3 turns", entries[1].Payload["diagnostic"])
}

func testProjectionEvents(base time.Time) []agents.ExecutionEvent {
	assistant := agents.Message{
		Role:    agents.RoleAssistant,
		Content: []core.ContentBlock{core.NewTextBlock("need lookup")},
		ToolCalls: []core.ToolCall{{
			ID: "call-1", Name: "lookup", Arguments: map[string]any{"query": "abc"},
		}},
	}
	toolResult := agents.Message{
		Role: agents.RoleTool,
		ToolResult: &agents.MessageToolResult{
			ToolCallID:     "call-1",
			Name:           "lookup",
			Content:        []core.ContentBlock{core.NewTextBlock("model summary")},
			DisplayContent: []core.ContentBlock{core.NewTextBlock("operator display")},
			Details:        map[string]any{"id": 1},
		},
	}
	finish := agents.Message{Role: agents.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "done"}}}}
	return []agents.ExecutionEvent{
		{Timestamp: base, Payload: agents.RunStartedEvent{RunID: "run-1", Task: "Solve task-1", Model: "model-1", Provider: "provider-1"}},
		{Timestamp: base.Add(time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: assistant}},
		{Timestamp: base.Add(2500 * time.Millisecond), Payload: agents.ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: assistant.ToolCalls[0], Outcome: agents.ToolCallOutcomeExecuted, Result: &toolResult, Status: agents.OperationStatusCompleted}},
		{Timestamp: base.Add(3 * time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: toolResult}},
		{Timestamp: base.Add(4 * time.Second), Payload: agents.TurnFinishedEvent{RunID: "run-1", Turn: 1, Status: agents.OperationStatusCompleted, Usage: &core.TokenInfo{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3}}},
		{Timestamp: base.Add(5 * time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 2, Message: finish}},
		{Timestamp: base.Add(5500 * time.Millisecond), Payload: agents.ToolCallFinishedEvent{RunID: "run-1", Turn: 2, ToolIndex: 0, Call: finish.ToolCalls[0], Outcome: agents.ToolCallOutcomeFinish, Status: agents.OperationStatusCompleted}},
		{Timestamp: base.Add(6 * time.Second), Payload: agents.TurnFinishedEvent{RunID: "run-1", Turn: 2, Status: agents.OperationStatusCompleted, Usage: &core.TokenInfo{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}}},
		{Timestamp: base.Add(7 * time.Second), Payload: agents.RunFinishedEvent{RunID: "run-1", Status: agents.RunStatusCompleted, StopReason: agents.StopReasonFinish, FinalAnswer: "done"}},
	}
}
