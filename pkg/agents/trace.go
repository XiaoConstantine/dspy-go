package agents

import "time"

// TraceStatus captures the high-level outcome of an agent trace.
type TraceStatus string

const (
	TraceStatusSuccess TraceStatus = "success"
	TraceStatusFailure TraceStatus = "failure"
	TraceStatusPartial TraceStatus = "partial"
)

// TraceStep represents a structured step in an agent execution.
type TraceStep struct {
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

// ExecutionTrace captures an agent execution with step-level detail.
type ExecutionTrace struct {
	AgentID          string
	AgentType        string
	Task             string
	Input            map[string]interface{}
	Output           map[string]interface{}
	Steps            []TraceStep
	Status           TraceStatus
	Error            string
	StartedAt        time.Time
	CompletedAt      time.Time
	ProcessingTime   time.Duration
	TokenUsage       map[string]int64
	ToolUsageCount   map[string]int
	ContextMetadata  map[string]interface{}
	TerminationCause string
}
