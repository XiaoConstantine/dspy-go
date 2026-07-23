package agents

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEventEmitter_EmitsExactBalancedLifecycle(t *testing.T) {
	var events []ExecutionEvent
	emitter := NewEventEmitter(EventSinkFunc(func(_ context.Context, event ExecutionEvent) {
		events = append(events, event)
	}))
	emitter.now = func() time.Time { return time.Unix(10, 0) }

	call := core.ToolCall{ID: "call-1", Name: "read", Arguments: map[string]any{"path": "main.go"}}
	assistant := Message{Role: RoleAssistant, Content: []core.ContentBlock{core.NewTextBlock("reading")}, ToolCalls: []core.ToolCall{call}}
	result := NewToolResultMessage(call.ID, call.Name, core.ToolResult{Data: "contents"})
	usage := &core.TokenInfo{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}

	emitter.Emit(context.Background(), &RunStartedEvent{Messages: []Message{NewTextMessage(RoleUser, "inspect")}})
	emitter.Emit(context.Background(), TurnStartedEvent{Turn: 1})
	emitter.Emit(context.Background(), MessageAddedEvent{Turn: 1, Message: assistant})
	emitter.Emit(context.Background(), ToolCallProposedEvent{Turn: 1, ToolIndex: 0, Call: call})
	emitter.Emit(context.Background(), ToolExecutionStartedEvent{Turn: 1, ToolIndex: 0, Call: call})
	emitter.Emit(context.Background(), ToolCallFinishedEvent{
		Turn: 1, ToolIndex: 0, Call: call, Outcome: ToolCallOutcomeExecuted,
		Result: &result, Status: OperationStatusCompleted,
	})
	emitter.Emit(context.Background(), MessageAddedEvent{Turn: 1, Message: result})
	emitter.Emit(context.Background(), TurnFinishedEvent{Turn: 1, Status: OperationStatusCompleted, Usage: usage})
	emitter.Emit(context.Background(), RunFinishedEvent{Status: RunStatusCompleted, StopReason: StopReasonFinish})

	require.NoError(t, ValidateEventLifecycle(events))
	require.Len(t, events, 9)
	assert.Equal(t, []any{
		RunStartedEvent{}, TurnStartedEvent{}, MessageAddedEvent{}, ToolCallProposedEvent{},
		ToolExecutionStartedEvent{}, ToolCallFinishedEvent{}, MessageAddedEvent{},
		TurnFinishedEvent{}, RunFinishedEvent{},
	}, []any{
		typeOnly(events[0].Payload), typeOnly(events[1].Payload), typeOnly(events[2].Payload),
		typeOnly(events[3].Payload), typeOnly(events[4].Payload), typeOnly(events[5].Payload),
		typeOnly(events[6].Payload), typeOnly(events[7].Payload), typeOnly(events[8].Payload),
	})
	for i, event := range events {
		assert.Equal(t, uint64(i+1), event.Sequence)
		assert.Equal(t, time.Unix(10, 0).UTC(), event.Timestamp)
	}

	call.Arguments["path"] = "changed.go"
	assistant.Content[0].Text = "changed"
	result.ToolResult.Content[0].Text = "changed"
	usage.TotalTokens = 99
	assert.Equal(t, "main.go", events[3].Payload.(ToolCallProposedEvent).Call.Arguments["path"])
	assert.Equal(t, "reading", events[2].Payload.(MessageAddedEvent).Message.Content[0].Text)
	assert.Equal(t, "contents", events[5].Payload.(ToolCallFinishedEvent).Result.ToolResult.Content[0].Text)
	assert.Equal(t, 5, events[7].Payload.(TurnFinishedEvent).Usage.TotalTokens)
}

func TestValidateEventLifecycle_AcceptsNonExecutedAndBlockedToolOutcomes(t *testing.T) {
	call := core.ToolCall{ID: "call-1", Name: "read"}
	rejection := errors.New("unknown tool")
	blocked := errors.New("tool blocked")
	errorResult := NewToolResultMessage(call.ID, call.Name, core.ToolResult{
		Data: "failed", Metadata: map[string]any{core.ToolResultIsErrorMeta: true},
	})
	blockedResult := NewToolResultMessage(call.ID, call.Name, core.ToolResult{
		Data: "blocked", Metadata: map[string]any{core.ToolResultIsErrorMeta: true},
	})

	tests := []struct {
		name   string
		middle []EventPayload
	}{
		{
			name: "Finish completes without execution",
			middle: []EventPayload{
				ToolCallProposedEvent{Turn: 1, ToolIndex: 0, Call: core.ToolCall{ID: "finish-1", Name: "Finish"}},
				ToolCallFinishedEvent{Turn: 1, ToolIndex: 0, Call: core.ToolCall{ID: "finish-1", Name: "Finish"}, Outcome: ToolCallOutcomeFinish, Status: OperationStatusCompleted},
			},
		},
		{
			name: "unknown tool is rejected before execution",
			middle: []EventPayload{
				ToolCallProposedEvent{Turn: 1, ToolIndex: 0, Call: call},
				ToolCallFinishedEvent{Turn: 1, ToolIndex: 0, Call: call, Outcome: ToolCallOutcomeRejected, Result: &errorResult, Status: OperationStatusFailed, Err: rejection},
			},
		},
		{
			name: "blocked tool starts but does not execute",
			middle: []EventPayload{
				ToolCallProposedEvent{Turn: 1, ToolIndex: 0, Call: call},
				ToolExecutionStartedEvent{Turn: 1, ToolIndex: 0, Call: call},
				ToolCallFinishedEvent{Turn: 1, ToolIndex: 0, Call: call, Outcome: ToolCallOutcomeBlocked, Result: &blockedResult, Status: OperationStatusBlocked, Err: blocked},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var events []ExecutionEvent
			emitter := NewEventEmitter(EventSinkFunc(func(_ context.Context, event ExecutionEvent) {
				events = append(events, event)
			}))
			emitter.Emit(context.Background(), RunStartedEvent{})
			emitter.Emit(context.Background(), TurnStartedEvent{Turn: 1})
			for _, payload := range tt.middle {
				emitter.Emit(context.Background(), payload)
			}
			emitter.Emit(context.Background(), TurnFinishedEvent{Turn: 1, Status: OperationStatusCompleted})
			emitter.Emit(context.Background(), RunFinishedEvent{Status: RunStatusCompleted, StopReason: StopReasonText})
			require.NoError(t, ValidateEventLifecycle(events))
		})
	}
}

func TestValidateEventLifecycle_RejectsUnbalancedAndInvalidTerminalEvents(t *testing.T) {
	now := time.Now()
	event := func(sequence uint64, payload EventPayload) ExecutionEvent {
		return ExecutionEvent{Sequence: sequence, Timestamp: now, Payload: payload}
	}
	failure := errors.New("model failed")
	call := core.ToolCall{ID: "call-1", Name: "read"}
	wrongNameResult := NewToolResultMessage(call.ID, "write", core.ToolResult{Data: "contents"})

	tests := []struct {
		name   string
		events []ExecutionEvent
		match  string
	}{
		{
			name: "missing terminal event",
			events: []ExecutionEvent{
				event(1, RunStartedEvent{}),
			},
			match: "no terminal event",
		},
		{
			name: "negative turn cannot impersonate inactive sentinel",
			events: []ExecutionEvent{
				event(1, RunStartedEvent{}),
				event(2, TurnFinishedEvent{Turn: -1, Status: OperationStatusCompleted}),
				event(3, RunFinishedEvent{Status: RunStatusCompleted, StopReason: StopReasonText}),
			},
			match: "must not be negative",
		},
		{
			name: "run identity mismatch",
			events: []ExecutionEvent{
				event(1, RunStartedEvent{RunID: "run-a"}),
				event(2, RunFinishedEvent{RunID: "run-b", Status: RunStatusCompleted, StopReason: StopReasonText}),
			},
			match: "has run ID",
		},
		{
			name: "turn left open",
			events: []ExecutionEvent{
				event(1, RunStartedEvent{}),
				event(2, TurnStartedEvent{Turn: 1}),
				event(3, RunFinishedEvent{Status: RunStatusFailed, StopReason: StopReasonError, Err: failure}),
			},
			match: "active child lifecycle",
		},
		{
			name: "tool finish without start",
			events: []ExecutionEvent{
				event(1, RunStartedEvent{}),
				event(2, TurnStartedEvent{Turn: 1}),
				event(3, ToolCallFinishedEvent{Turn: 1, ToolIndex: 0}),
			},
			match: "finished without a proposal",
		},
		{
			name: "tool result name mismatch",
			events: []ExecutionEvent{
				event(1, RunStartedEvent{}),
				event(2, TurnStartedEvent{Turn: 1}),
				event(3, ToolCallProposedEvent{Turn: 1, ToolIndex: 0, Call: call}),
				event(4, ToolExecutionStartedEvent{Turn: 1, ToolIndex: 0, Call: call}),
				event(5, ToolCallFinishedEvent{Turn: 1, ToolIndex: 0, Call: call, Outcome: ToolCallOutcomeExecuted, Result: &wrongNameResult, Status: OperationStatusCompleted}),
			},
			match: "tool result name",
		},
		{
			name: "executed outcome cannot be blocked",
			events: []ExecutionEvent{
				event(1, RunStartedEvent{}),
				event(2, TurnStartedEvent{Turn: 1}),
				event(3, ToolCallProposedEvent{Turn: 1, ToolIndex: 0, Call: call}),
				event(4, ToolExecutionStartedEvent{Turn: 1, ToolIndex: 0, Call: call}),
				event(5, ToolCallFinishedEvent{Turn: 1, ToolIndex: 0, Call: call, Outcome: ToolCallOutcomeExecuted, Result: &wrongNameResult, Status: OperationStatusBlocked, Err: failure}),
			},
			match: "cannot have blocked status",
		},
		{
			name: "failed terminal without error",
			events: []ExecutionEvent{
				event(1, RunStartedEvent{}),
				event(2, RunFinishedEvent{Status: RunStatusFailed, StopReason: StopReasonError}),
			},
			match: "failed run requires",
		},
		{
			name: "event after terminal",
			events: []ExecutionEvent{
				event(1, RunStartedEvent{}),
				event(2, RunFinishedEvent{Status: RunStatusCompleted, StopReason: StopReasonText}),
				event(3, TurnStartedEvent{Turn: 1}),
			},
			match: "after terminal",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.ErrorContains(t, ValidateEventLifecycle(tt.events), tt.match)
		})
	}
}

func TestEventSinks_CallbackChannelAndLegacyProjection(t *testing.T) {
	t.Run("channel observes cancellation instead of blocking", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		sink := ChannelEventSink{Channel: make(chan ExecutionEvent)}
		done := make(chan struct{})
		go func() {
			sink.EmitEvent(ctx, ExecutionEvent{})
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("channel sink remained blocked after cancellation")
		}
	})

	t.Run("legacy projection has one terminal event", func(t *testing.T) {
		var mu sync.Mutex
		var legacy []AgentEvent
		emitter := NewEventEmitter(LegacyEventSink(func(event AgentEvent) {
			mu.Lock()
			defer mu.Unlock()
			legacy = append(legacy, event)
		}))
		failure := errors.New("provider failed")
		emitter.Emit(context.Background(), RunStartedEvent{})
		emitter.Emit(context.Background(), MessageAddedEvent{Turn: 1, Message: NewTextMessage(RoleAssistant, "omitted")})
		emitter.Emit(context.Background(), RunFinishedEvent{
			Status: RunStatusFailed, StopReason: StopReasonError, Err: failure,
		})

		require.Len(t, legacy, 3)
		assert.Equal(t, EventRunStarted, legacy[0].Type)
		assert.Equal(t, EventRunFailed, legacy[1].Type)
		assert.Equal(t, EventRunFinished, legacy[2].Type)
		assert.Equal(t, false, legacy[2].Data["completed"])
		assert.Equal(t, "provider failed", legacy[1].Data["error"])
	})

	t.Run("max turns is stopped rather than completed or failed", func(t *testing.T) {
		var legacy []AgentEvent
		var typed []ExecutionEvent
		emitter := NewEventEmitter(EventSinkFunc(func(ctx context.Context, event ExecutionEvent) {
			typed = append(typed, event)
			LegacyEventSink(func(event AgentEvent) { legacy = append(legacy, event) }).EmitEvent(ctx, event)
		}))
		emitter.Emit(context.Background(), RunStartedEvent{RunID: "task-1"})
		emitter.Emit(context.Background(), RunFinishedEvent{
			RunID: "task-1", Status: RunStatusStopped, StopReason: StopReasonMaxTurns,
			Turns: 3, ToolCalls: 2, Diagnostic: "max turns reached without Finish after 3 turns",
		})

		require.NoError(t, ValidateEventLifecycle(typed))
		require.Len(t, legacy, 2)
		assert.Equal(t, EventRunFinished, legacy[1].Type)
		assert.Equal(t, false, legacy[1].Data["completed"])
		assert.Equal(t, 3, legacy[1].Data["turns"])
		assert.Equal(t, "max turns reached without Finish after 3 turns", legacy[1].Data["error"])
	})

	t.Run("legacy blocked tool preserves keys and terminal pair", func(t *testing.T) {
		var legacy []AgentEvent
		emitter := NewEventEmitter(LegacyEventSink(func(event AgentEvent) {
			legacy = append(legacy, event)
		}))
		call := core.ToolCall{ID: "call-1", Name: "read", Arguments: map[string]any{"path": "main.go"}}
		blockedErr := errors.New("policy denied")
		result := NewToolResultMessage(call.ID, call.Name, core.ToolResult{
			Data: "blocked by policy", Metadata: map[string]any{core.ToolResultIsErrorMeta: true},
		})
		result.ToolResult.Details = map[string]any{
			"completed": true, "tool_name": "forged", "observation": "forged",
			"is_error": false, "reason": "forged reason",
		}
		emitter.Emit(context.Background(), ToolCallProposedEvent{RunID: "task-1", Turn: 1, Call: call})
		emitter.Emit(context.Background(), ToolExecutionStartedEvent{RunID: "task-1", Turn: 1, Call: call})
		emitter.Emit(context.Background(), ToolCallFinishedEvent{
			RunID: "task-1", Turn: 1, Call: call, Outcome: ToolCallOutcomeBlocked,
			Result: &result, Status: OperationStatusBlocked, Err: blockedErr,
		})

		require.Len(t, legacy, 4)
		assert.Equal(t, []string{EventToolCallProposed, EventToolCallStarted, EventToolCallBlocked, EventToolCallFinished}, []string{
			legacy[0].Type, legacy[1].Type, legacy[2].Type, legacy[3].Type,
		})
		assert.Equal(t, "read", legacy[3].Data["tool_name"])
		assert.Equal(t, "blocked by policy", legacy[3].Data["observation"])
		assert.Equal(t, "blocked by tool policy", legacy[2].Data["reason"])
		assert.Equal(t, true, legacy[3].Data["is_error"])
		assert.NotContains(t, legacy[3].Data, "child_completed")
		assert.NotContains(t, legacy[3].Data, "subagent_name")
		assert.Equal(t, "forged", legacy[3].Data["details"].(map[string]any)["tool_name"])
		assert.NotContains(t, legacy[3].Data, "tool")
	})
}

func TestEventEmitter_AllowsReentrantEmissionInSequence(t *testing.T) {
	var events []ExecutionEvent
	var emitter *EventEmitter
	emitter = NewEventEmitter(EventSinkFunc(func(ctx context.Context, event ExecutionEvent) {
		events = append(events, event)
		if _, ok := event.Payload.(RunStartedEvent); ok {
			emitter.Emit(ctx, RunFinishedEvent{Status: RunStatusCompleted, StopReason: StopReasonText})
		}
	}))

	done := make(chan struct{})
	go func() {
		emitter.Emit(context.Background(), RunStartedEvent{})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("reentrant event emission deadlocked")
	}
	require.Len(t, events, 2)
	assert.Nil(t, events[0].Payload.(RunStartedEvent).Messages)
	assert.Equal(t, uint64(1), events[0].Sequence)
	assert.Equal(t, uint64(2), events[1].Sequence)
	require.NoError(t, ValidateEventLifecycle(events))
}

func TestEventEmitter_SerializesConcurrentSinks(t *testing.T) {
	const count = 100
	var events []ExecutionEvent
	emitter := NewEventEmitter(EventSinkFunc(func(_ context.Context, event ExecutionEvent) {
		events = append(events, event)
	}))

	var wait sync.WaitGroup
	wait.Add(count)
	for i := range count {
		go func(turn int) {
			defer wait.Done()
			emitter.Emit(context.Background(), TurnStartedEvent{Turn: turn})
		}(i)
	}
	wait.Wait()

	require.Len(t, events, count)
	for i, event := range events {
		assert.Equal(t, uint64(i+1), event.Sequence)
	}
}

func typeOnly(payload EventPayload) any {
	switch payload.(type) {
	case RunStartedEvent:
		return RunStartedEvent{}
	case RunFinishedEvent:
		return RunFinishedEvent{}
	case TurnStartedEvent:
		return TurnStartedEvent{}
	case TurnFinishedEvent:
		return TurnFinishedEvent{}
	case MessageAddedEvent:
		return MessageAddedEvent{}
	case ToolCallProposedEvent:
		return ToolCallProposedEvent{}
	case ToolExecutionStartedEvent:
		return ToolExecutionStartedEvent{}
	case ToolCallFinishedEvent:
		return ToolCallFinishedEvent{}
	default:
		return nil
	}
}
