package agents

import (
	"maps"
	"time"
)

const (
	EventRunStarted       = "run_started"
	EventRunFinished      = "run_finished"
	EventRunFailed        = "run_failed"
	EventLLMTurnStarted   = "llm_turn_started"
	EventLLMTurnFinished  = "llm_turn_finished"
	EventToolCallProposed = "tool_call_proposed"
	EventToolCallStarted  = "tool_call_started"
	EventToolCallBlocked  = "tool_call_blocked"
	EventToolCallFinished = "tool_call_finished"
)

// Deprecated: prefer typed ExecutionEvent consumers via EventSink.
// AgentEvent remains only as a compatibility format for legacy callbacks.
type AgentEvent struct {
	Type      string
	Data      map[string]any
	Timestamp time.Time
}

// Deprecated: prefer typed ExecutionEvent consumers via EventSink.
// EmitEvent remains only for compatibility callbacks.
func EmitEvent(onEvent func(AgentEvent), eventType string, data map[string]any) {
	if onEvent == nil {
		return
	}

	onEvent(AgentEvent{
		Type:      eventType,
		Data:      maps.Clone(data),
		Timestamp: time.Now().UTC(),
	})
}
