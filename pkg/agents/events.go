package agents

import (
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
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

// AgentEvent represents an event emitted during agent execution.
// This is intentionally minimal and will be extended by streaming/event work.
type AgentEvent struct {
	Type      string
	Data      map[string]any
	Timestamp time.Time
}

// EmitEvent safely emits an agent event to an optional callback.
func EmitEvent(onEvent func(AgentEvent), eventType string, data map[string]any) {
	if onEvent == nil {
		return
	}

	onEvent(AgentEvent{
		Type:      eventType,
		Data:      core.ShallowCopyMap(data),
		Timestamp: time.Now().UTC(),
	})
}
