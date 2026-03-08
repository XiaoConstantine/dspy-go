package modules

import "time"

// ReActTraceStep captures a single iteration in the ReAct loop.
type ReActTraceStep struct {
	Index       int
	Thought     string
	ActionRaw   string
	Tool        string
	Arguments   map[string]interface{}
	Observation string
	Duration    time.Duration
	Success     bool
	Error       string
}

// ReActTrace captures the full execution trace of a ReAct run.
type ReActTrace struct {
	Input            map[string]interface{}
	Output           map[string]interface{}
	Steps            []ReActTraceStep
	ProcessingTime   time.Duration
	TerminationCause string
	Error            string
}
