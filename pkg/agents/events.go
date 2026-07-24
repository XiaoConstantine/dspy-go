package agents

import (
	"maps"
	"time"
)

const (
	EventRunStarted       = "run_started"
	EventRunFinished      = "run_finished"
	EventRunFailed        = "run_failed"
	EventSessionLoaded    = "session_loaded"
	EventSessionPersisted = "session_persisted"
	EventLLMTurnStarted   = "llm_turn_started"
	EventLLMTurnFinished  = "llm_turn_finished"
	EventToolCallProposed = "tool_call_proposed"
	EventToolCallStarted  = "tool_call_started"
	EventToolCallBlocked  = "tool_call_blocked"
	EventToolCallFinished = "tool_call_finished"
)

// Deprecated: prefer typed ExecutionEvent consumers via EventSink for portable
// run/turn/tool lifecycles. AgentEvent remains the compatibility format for
// legacy callbacks and native-only session notifications.
type AgentEvent struct {
	Type      string
	Data      map[string]any
	Timestamp time.Time
}

// Deprecated: prefer typed ExecutionEvent consumers via EventSink for portable
// run/turn/tool lifecycles. EmitEvent remains for compatibility callbacks and
// native-only session notifications.
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
