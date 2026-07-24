package agents

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// RunStatus is the terminal status carried by RunFinishedEvent.
type RunStatus string

const (
	RunStatusCompleted RunStatus = "completed"
	RunStatusStopped   RunStatus = "stopped"
	RunStatusFailed    RunStatus = "failed"
	RunStatusCanceled  RunStatus = "canceled"
)

// OperationStatus is the terminal status of a model turn or tool execution.
type OperationStatus string

const (
	OperationStatusCompleted OperationStatus = "completed"
	OperationStatusFailed    OperationStatus = "failed"
	OperationStatusCanceled  OperationStatus = "canceled"
	OperationStatusBlocked   OperationStatus = "blocked"
)

// EventPayload is one typed event in the provider-neutral execution lifecycle.
type EventPayload interface {
	executionEventPayload()
}

// ExecutionEvent adds ordering and time metadata to one typed event payload.
type ExecutionEvent struct {
	Sequence  uint64
	Timestamp time.Time
	Payload   EventPayload
}

// RunStartedEvent opens one run lifecycle.
type RunStartedEvent struct {
	RunID    string
	Task     string
	MaxTurns int
	Model    string
	Provider string
	Messages []Message
}

func (RunStartedEvent) executionEventPayload() {}

// RunFinishedEvent is the only terminal run event. Err is populated for failed
// and canceled runs and StopReason explains the terminal decision.
type RunFinishedEvent struct {
	RunID       string
	Status      RunStatus
	StopReason  StopReason
	Turns       int
	ToolCalls   int
	FinalAnswer string
	Diagnostic  string
	Err         error
}

func (RunFinishedEvent) executionEventPayload() {}

// TurnStartedEvent opens one model-turn lifecycle.
type TurnStartedEvent struct {
	RunID    string
	Turn     int
	MaxTurns int
}

func (TurnStartedEvent) executionEventPayload() {}

// TurnFinishedEvent closes one model-turn lifecycle.
type TurnFinishedEvent struct {
	RunID         string
	Turn          int
	Status        OperationStatus
	Assistant     *Message
	ToolCallCount int
	Usage         *core.TokenInfo
	Err           error
}

func (TurnFinishedEvent) executionEventPayload() {}

// MessageAddedEvent records one durable canonical transcript message.
type MessageAddedEvent struct {
	RunID   string
	Turn    int
	Message Message
}

func (MessageAddedEvent) executionEventPayload() {}

// ToolCallProposedEvent records a provider-requested tool call before dispatch.
type ToolCallProposedEvent struct {
	RunID     string
	Turn      int
	ToolIndex int
	Call      core.ToolCall
}

func (ToolCallProposedEvent) executionEventPayload() {}

// ToolExecutionStartedEvent opens one tool-execution lifecycle.
type ToolExecutionStartedEvent struct {
	RunID     string
	Turn      int
	ToolIndex int
	Call      core.ToolCall
}

func (ToolExecutionStartedEvent) executionEventPayload() {}

// ToolCallOutcome describes whether a proposed call executed or terminated
// before execution.
type ToolCallOutcome string

const (
	ToolCallOutcomeExecuted ToolCallOutcome = "executed"
	ToolCallOutcomeBlocked  ToolCallOutcome = "blocked"
	ToolCallOutcomeRejected ToolCallOutcome = "rejected"
	ToolCallOutcomeFinish   ToolCallOutcome = "finish"
)

// ToolCallFinishedEvent closes one proposed tool-call lifecycle. Execution is
// optional: Finish and pre-dispatch rejection terminate without a started event.
type ToolCallFinishedEvent struct {
	RunID     string
	Turn      int
	ToolIndex int
	Call      core.ToolCall
	Outcome   ToolCallOutcome
	Result    *Message
	Status    OperationStatus
	Err       error
}

func (ToolCallFinishedEvent) executionEventPayload() {}

// EventSink consumes typed execution events. Implementations should return
// promptly unless deliberate backpressure is desired.
type EventSink interface {
	EmitEvent(context.Context, ExecutionEvent)
}

// EventSinkFunc adapts a function to EventSink.
type EventSinkFunc func(context.Context, ExecutionEvent)

// EmitEvent implements EventSink.
func (f EventSinkFunc) EmitEvent(ctx context.Context, event ExecutionEvent) {
	if f != nil {
		f(ctx, event)
	}
}

// ChannelEventSink sends events to a channel and abandons a blocked send when
// the run context is canceled.
type ChannelEventSink struct {
	Channel chan<- ExecutionEvent
}

// EmitEvent implements EventSink.
func (s ChannelEventSink) EmitEvent(ctx context.Context, event ExecutionEvent) {
	if s.Channel == nil {
		return
	}
	select {
	case s.Channel <- event:
	case <-ctx.Done():
	}
}

// EventEmitter assigns a stable sequence and timestamp before delivering
// ownership-safe event snapshots. Emissions are serialized in sink order.
type EventEmitter struct {
	mu       sync.Mutex
	sink     EventSink
	sequence uint64
	now      func() time.Time
	queue    []queuedExecutionEvent
	draining bool
}

type queuedExecutionEvent struct {
	ctx   context.Context
	event ExecutionEvent
}

// NewEventEmitter creates an emitter for an optional sink.
func NewEventEmitter(sink EventSink) *EventEmitter {
	return &EventEmitter{sink: sink, now: time.Now}
}

// Emit publishes one event payload. A nil sink or payload is a no-op.
func (e *EventEmitter) Emit(ctx context.Context, payload EventPayload) {
	if e == nil || e.sink == nil || payload == nil {
		return
	}

	snapshot := cloneEventPayload(payload)
	if snapshot == nil {
		return
	}

	e.mu.Lock()
	e.sequence++
	e.queue = append(e.queue, queuedExecutionEvent{
		ctx: ctx,
		event: ExecutionEvent{
			Sequence:  e.sequence,
			Timestamp: e.now().UTC(),
			Payload:   snapshot,
		},
	})
	if e.draining {
		e.mu.Unlock()
		return
	}
	e.draining = true
	e.mu.Unlock()
	e.drain()
}

func (e *EventEmitter) drain() {
	defer func() {
		if recovered := recover(); recovered != nil {
			e.mu.Lock()
			e.draining = false
			e.mu.Unlock()
			panic(recovered)
		}
	}()
	for {
		e.mu.Lock()
		if len(e.queue) == 0 {
			e.draining = false
			e.mu.Unlock()
			return
		}
		queued := e.queue[0]
		e.queue[0] = queuedExecutionEvent{}
		e.queue = e.queue[1:]
		e.mu.Unlock()
		e.sink.EmitEvent(queued.ctx, queued.event)
	}
}

func cloneEventPayload(payload EventPayload) EventPayload {
	value := reflect.ValueOf(payload)
	if value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return nil
		}
		payload = value.Elem().Interface().(EventPayload)
	}

	switch event := payload.(type) {
	case RunStartedEvent:
		event.Messages = CloneMessages(event.Messages)
		return event
	case RunFinishedEvent:
		return event
	case TurnStartedEvent:
		return event
	case TurnFinishedEvent:
		if event.Assistant != nil {
			assistant := event.Assistant.Clone()
			event.Assistant = &assistant
		}
		if event.Usage != nil {
			usage := *event.Usage
			event.Usage = &usage
		}
		return event
	case MessageAddedEvent:
		event.Message = event.Message.Clone()
		return event
	case ToolCallProposedEvent:
		event.Call = cloneToolCall(event.Call)
		return event
	case ToolExecutionStartedEvent:
		event.Call = cloneToolCall(event.Call)
		return event
	case ToolCallFinishedEvent:
		event.Call = cloneToolCall(event.Call)
		if event.Result != nil {
			result := event.Result.Clone()
			event.Result = &result
		}
		return event
	default:
		return payload
	}
}

// CloneExecutionEvent returns an ownership-safe copy of one execution event.
func CloneExecutionEvent(event ExecutionEvent) ExecutionEvent {
	cloned := event
	cloned.Payload = cloneEventPayload(event.Payload)
	return cloned
}

func cloneToolCall(call core.ToolCall) core.ToolCall {
	return cloneToolCalls([]core.ToolCall{call})[0]
}

type toolEventKey struct {
	turn  int
	index int
}

func eventRunID(payload EventPayload) string {
	switch event := payload.(type) {
	case RunStartedEvent:
		return event.RunID
	case RunFinishedEvent:
		return event.RunID
	case TurnStartedEvent:
		return event.RunID
	case TurnFinishedEvent:
		return event.RunID
	case MessageAddedEvent:
		return event.RunID
	case ToolCallProposedEvent:
		return event.RunID
	case ToolExecutionStartedEvent:
		return event.RunID
	case ToolCallFinishedEvent:
		return event.RunID
	default:
		return ""
	}
}

// ValidateEventLifecycle verifies exact sequence numbers and balanced run,
// turn, and tool lifecycles. It is useful in tests and event consumers that
// require a complete execution trace.
func ValidateEventLifecycle(events []ExecutionEvent) error {
	if len(events) == 0 {
		return fmt.Errorf("event lifecycle is empty")
	}

	runActive := false
	runFinished := false
	runID := ""
	activeTurn := 0
	turnActive := false
	proposed := make(map[toolEventKey]core.ToolCall)
	activeTools := make(map[toolEventKey]core.ToolCall)
	for index, event := range events {
		if event.Sequence != uint64(index+1) {
			return fmt.Errorf("event %d has sequence %d, want %d", index, event.Sequence, index+1)
		}
		if event.Timestamp.IsZero() {
			return fmt.Errorf("event %d has no timestamp", index)
		}
		if event.Payload == nil {
			return fmt.Errorf("event %d has no payload", index)
		}
		if runFinished {
			return fmt.Errorf("event %d occurs after terminal run event", index)
		}
		if index > 0 && eventRunID(event.Payload) != runID {
			return fmt.Errorf("event %d has run ID %q, want %q", index, eventRunID(event.Payload), runID)
		}

		switch payload := event.Payload.(type) {
		case RunStartedEvent:
			if index != 0 || runActive {
				return fmt.Errorf("run started event must be first and unique")
			}
			runActive = true
			runID = payload.RunID
		case RunFinishedEvent:
			if !runActive {
				return fmt.Errorf("run finished without an active run")
			}
			if turnActive || len(proposed) != 0 || len(activeTools) != 0 {
				return fmt.Errorf("run finished with active child lifecycle")
			}
			if err := validateRunFinished(payload); err != nil {
				return err
			}
			runFinished = true
		case TurnStartedEvent:
			if payload.Turn < 0 {
				return fmt.Errorf("turn index must not be negative: %d", payload.Turn)
			}
			if !runActive || turnActive {
				return fmt.Errorf("turn %d started without an available run", payload.Turn)
			}
			activeTurn = payload.Turn
			turnActive = true
		case TurnFinishedEvent:
			if payload.Turn < 0 {
				return fmt.Errorf("turn index must not be negative: %d", payload.Turn)
			}
			if !turnActive || payload.Turn != activeTurn {
				return fmt.Errorf("turn %d finished without the matching active turn", payload.Turn)
			}
			if len(proposed) != 0 || len(activeTools) != 0 {
				return fmt.Errorf("turn %d finished with active tool execution", payload.Turn)
			}
			if err := validateOperationFinished(payload.Status, payload.Err); err != nil {
				return fmt.Errorf("turn %d: %w", payload.Turn, err)
			}
			turnActive = false
		case MessageAddedEvent:
			if payload.Turn < 0 {
				return fmt.Errorf("turn index must not be negative: %d", payload.Turn)
			}
			if !turnActive || payload.Turn != activeTurn {
				return fmt.Errorf("message for turn %d emitted while turn %d is active", payload.Turn, activeTurn)
			}
		case ToolCallProposedEvent:
			if payload.Turn < 0 || payload.ToolIndex < 0 {
				return fmt.Errorf("tool proposal indices must not be negative: turn %d, tool %d", payload.Turn, payload.ToolIndex)
			}
			if !turnActive || payload.Turn != activeTurn {
				return fmt.Errorf("tool proposal for turn %d emitted while turn %d is active", payload.Turn, activeTurn)
			}
			key := toolEventKey{turn: payload.Turn, index: payload.ToolIndex}
			if _, exists := proposed[key]; exists {
				return fmt.Errorf("tool %d in turn %d proposed more than once", payload.ToolIndex, payload.Turn)
			}
			proposed[key] = payload.Call
		case ToolExecutionStartedEvent:
			if payload.Turn < 0 || payload.ToolIndex < 0 {
				return fmt.Errorf("tool execution indices must not be negative: turn %d, tool %d", payload.Turn, payload.ToolIndex)
			}
			key := toolEventKey{turn: payload.Turn, index: payload.ToolIndex}
			if !turnActive || payload.Turn != activeTurn {
				return fmt.Errorf("tool execution for turn %d started while turn %d is active", payload.Turn, activeTurn)
			}
			proposedCall, exists := proposed[key]
			if !exists {
				return fmt.Errorf("tool %d in turn %d started without a proposal", payload.ToolIndex, payload.Turn)
			}
			if payload.Call.ID != proposedCall.ID || payload.Call.Name != proposedCall.Name {
				return fmt.Errorf("tool %d in turn %d started with a different call", payload.ToolIndex, payload.Turn)
			}
			if _, exists := activeTools[key]; exists {
				return fmt.Errorf("tool %d in turn %d started more than once", payload.ToolIndex, payload.Turn)
			}
			delete(proposed, key)
			activeTools[key] = payload.Call
		case ToolCallFinishedEvent:
			if payload.Turn < 0 || payload.ToolIndex < 0 {
				return fmt.Errorf("tool outcome indices must not be negative: turn %d, tool %d", payload.Turn, payload.ToolIndex)
			}
			key := toolEventKey{turn: payload.Turn, index: payload.ToolIndex}
			proposedCall, wasProposed := proposed[key]
			startedCall, wasStarted := activeTools[key]
			if !wasProposed && !wasStarted {
				return fmt.Errorf("tool %d in turn %d finished without a proposal", payload.ToolIndex, payload.Turn)
			}
			expectedCall := proposedCall
			if wasStarted {
				expectedCall = startedCall
			}
			if payload.Call.ID != expectedCall.ID || payload.Call.Name != expectedCall.Name {
				return fmt.Errorf("tool %d in turn %d finished with a different call", payload.ToolIndex, payload.Turn)
			}
			if err := validateToolCallFinished(payload, wasStarted); err != nil {
				return fmt.Errorf("tool %d in turn %d: %w", payload.ToolIndex, payload.Turn, err)
			}
			delete(proposed, key)
			delete(activeTools, key)
		default:
			return fmt.Errorf("event %d has unsupported payload %T", index, payload)
		}
	}
	if !runFinished {
		return fmt.Errorf("run lifecycle has no terminal event")
	}
	return nil
}

func validateOperationFinished(status OperationStatus, err error) error {
	switch status {
	case OperationStatusCompleted:
		if err != nil {
			return fmt.Errorf("completed operation has an error")
		}
	case OperationStatusFailed, OperationStatusCanceled, OperationStatusBlocked:
		if err == nil {
			return fmt.Errorf("%s operation requires an error", status)
		}
	default:
		return fmt.Errorf("operation has invalid status %q", status)
	}
	return nil
}

func validateToolCallFinished(event ToolCallFinishedEvent, wasStarted bool) error {
	if err := validateOperationFinished(event.Status, event.Err); err != nil {
		return err
	}
	switch event.Outcome {
	case ToolCallOutcomeExecuted:
		if !wasStarted {
			return fmt.Errorf("executed tool call has no execution start")
		}
		if event.Status == OperationStatusBlocked {
			return fmt.Errorf("executed tool call cannot have blocked status")
		}
	case ToolCallOutcomeBlocked:
		if !wasStarted || event.Status != OperationStatusBlocked {
			return fmt.Errorf("blocked tool call requires a started, blocked execution")
		}
	case ToolCallOutcomeRejected:
		if wasStarted || (event.Status != OperationStatusFailed && event.Status != OperationStatusCanceled) {
			return fmt.Errorf("rejected tool call must fail or cancel before execution")
		}
	case ToolCallOutcomeFinish:
		if wasStarted || event.Status != OperationStatusCompleted || event.Result != nil {
			return fmt.Errorf("Finish must complete without execution or a tool result")
		}
		return nil
	default:
		return fmt.Errorf("tool call has invalid outcome %q", event.Outcome)
	}
	if event.Result == nil || event.Result.Role != RoleTool || event.Result.ToolResult == nil {
		return fmt.Errorf("tool call has no canonical result message")
	}
	if event.Result.ToolResult.ToolCallID != event.Call.ID {
		return fmt.Errorf("tool result call ID %q does not match %q", event.Result.ToolResult.ToolCallID, event.Call.ID)
	}
	if event.Result.ToolResult.Name != event.Call.Name {
		return fmt.Errorf("tool result name %q does not match %q", event.Result.ToolResult.Name, event.Call.Name)
	}
	if event.Status == OperationStatusCompleted && event.Result.ToolResult.IsError {
		return fmt.Errorf("completed tool call has an error result")
	}
	if event.Status != OperationStatusCompleted && !event.Result.ToolResult.IsError {
		return fmt.Errorf("unsuccessful tool call requires an error result")
	}
	return nil
}

func validateRunFinished(event RunFinishedEvent) error {
	switch event.Status {
	case RunStatusCompleted:
		if event.Err != nil {
			return fmt.Errorf("completed run has an error")
		}
		if event.StopReason != StopReasonFinish && event.StopReason != StopReasonText {
			return fmt.Errorf("completed run has invalid stop reason %q", event.StopReason)
		}
	case RunStatusStopped:
		if event.Err != nil || (event.StopReason != StopReasonMaxTurns && event.StopReason != StopReasonNoToolCalls) {
			return fmt.Errorf("stopped run requires a non-error exhaustion stop reason")
		}
	case RunStatusFailed:
		if event.Err == nil || event.StopReason != StopReasonError {
			return fmt.Errorf("failed run requires error status and error stop reason")
		}
	case RunStatusCanceled:
		if event.Err == nil || event.StopReason != StopReasonCanceled {
			return fmt.Errorf("canceled run requires an error and canceled stop reason")
		}
	default:
		return fmt.Errorf("run has invalid status %q", event.Status)
	}
	return nil
}

// Deprecated: prefer typed ExecutionEvent consumers directly.
// LegacyEventSink keeps temporary string/map callback compatibility.
func LegacyEventSink(onEvent func(AgentEvent)) EventSink {
	return EventSinkFunc(func(_ context.Context, event ExecutionEvent) {
		if onEvent == nil {
			return
		}
		for _, legacy := range legacyEvents(event.Payload) {
			legacy.Timestamp = event.Timestamp
			onEvent(legacy)
		}
	})
}

func legacyEvents(payload EventPayload) []AgentEvent {
	switch event := payload.(type) {
	case RunStartedEvent:
		data := map[string]any{}
		data["task_id"] = event.RunID
		data["task"] = event.Task
		data["max_turns"] = event.MaxTurns
		data["model"] = event.Model
		data["provider"] = event.Provider
		return []AgentEvent{{Type: EventRunStarted, Data: data}}
	case RunFinishedEvent:
		data := map[string]any{}
		data["task_id"] = event.RunID
		data["completed"] = event.Status == RunStatusCompleted
		data["turns"] = event.Turns
		data["tool_calls"] = event.ToolCalls
		if event.FinalAnswer != "" {
			data["final_answer"] = event.FinalAnswer
		}
		if event.Err != nil {
			data["error"] = event.Err.Error()
		} else if event.Diagnostic != "" {
			data["error"] = event.Diagnostic
		}
		finished := AgentEvent{Type: EventRunFinished, Data: data}
		if event.Status == RunStatusFailed || event.Status == RunStatusCanceled {
			errorText := ""
			if event.Err != nil {
				errorText = event.Err.Error()
			}
			failed := AgentEvent{Type: EventRunFailed, Data: map[string]any{"task_id": event.RunID, "error": errorText}}
			return []AgentEvent{failed, finished}
		}
		return []AgentEvent{finished}
	case TurnStartedEvent:
		return []AgentEvent{{Type: EventLLMTurnStarted, Data: map[string]any{
			"task_id": event.RunID, "turn": event.Turn, "max_turns": event.MaxTurns,
		}}}
	case TurnFinishedEvent:
		data := map[string]any{
			"task_id": event.RunID, "turn": event.Turn, "tool_calls": event.ToolCallCount,
		}
		if event.Assistant != nil {
			data["assistant_text"] = event.Assistant.TextContent()
		}
		if event.Usage != nil {
			data["usage_total"] = int64(event.Usage.TotalTokens)
		} else {
			data["usage_total"] = int64(0)
		}
		if event.Err != nil {
			data["error"] = event.Err.Error()
		}
		return []AgentEvent{{Type: EventLLMTurnFinished, Data: data}}
	case ToolCallProposedEvent:
		return []AgentEvent{{Type: EventToolCallProposed, Data: legacyToolEventData(event.RunID, event.Turn, event.ToolIndex, event.Call)}}
	case ToolExecutionStartedEvent:
		return []AgentEvent{{Type: EventToolCallStarted, Data: legacyToolEventData(event.RunID, event.Turn, event.ToolIndex, event.Call)}}
	case ToolCallFinishedEvent:
		if event.Outcome == ToolCallOutcomeFinish || event.Outcome == ToolCallOutcomeRejected {
			return nil
		}
		data := legacyToolEventData(event.RunID, event.Turn, event.ToolIndex, event.Call)
		data["is_error"] = event.Err != nil
		if event.Result != nil && event.Result.ToolResult != nil {
			result := event.Result.ToolResult
			data["is_error"] = result.IsError || event.Err != nil
			data["observation"] = contentBlocksText(result.Content)
			data["synthetic"] = result.Synthetic
			data["redacted"] = result.Redacted
			data["truncated"] = result.Truncated
			if details := cloneAnyMap(result.Details); details != nil {
				data["details"] = details
			}
		}
		if event.Err != nil {
			data["error"] = event.Err.Error()
		}
		finished := AgentEvent{Type: EventToolCallFinished, Data: data}
		if event.Outcome == ToolCallOutcomeBlocked {
			blockedData := cloneAnyMap(data)
			blockedData["reason"] = blockedReasonText(event.Err)
			return []AgentEvent{{Type: EventToolCallBlocked, Data: blockedData}, finished}
		}
		return []AgentEvent{finished}
	case MessageAddedEvent:
		return nil
	default:
		return nil
	}
}

func legacyToolEventData(runID string, turn, toolIndex int, call core.ToolCall) map[string]any {
	data := map[string]any{
		"task_id":      runID,
		"turn":         turn,
		"tool_index":   toolIndex,
		"tool_call_id": call.ID,
		"tool_name":    call.Name,
		"arguments":    cloneAnyMap(call.Arguments),
		"metadata":     cloneAnyMap(call.Metadata),
	}
	return data
}

func contentBlocksText(blocks []core.ContentBlock) string {
	message := Message{Content: blocks}
	return message.TextContent()
}
