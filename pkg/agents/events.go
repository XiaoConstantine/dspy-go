package agents

import "time"

// AgentEvent represents an event emitted during agent execution.
// This is intentionally minimal and will be extended by streaming/event work.
type AgentEvent struct {
	Type      string
	Data      map[string]any
	Timestamp time.Time
}

