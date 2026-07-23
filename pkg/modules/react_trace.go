package modules

import "time"

// ReActTraceStep captures a single iteration in the ReAct loop.
type ReActTraceStep struct {
	Index              int
	Thought            string
	ActionRaw          string
	Tool               string
	Arguments          map[string]any
	Observation        string
	ObservationDisplay string
	ObservationDetails map[string]any
	Duration           time.Duration
	Success            bool
	Error              string
	Synthetic          bool
	Redacted           bool
	Truncated          bool
}

// ReActTrace captures the full execution trace of a ReAct run.
type ReActTrace struct {
	Input            map[string]any
	Output           map[string]any
	Steps            []ReActTraceStep
	ProcessingTime   time.Duration
	TerminationCause string
	Error            string
}
