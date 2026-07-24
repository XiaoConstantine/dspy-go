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
	Index              int
	Thought            string
	ActionRaw          string
	Tool               string
	Arguments          map[string]any
	Observation        string
	ObservationDisplay string
	ObservationDetails map[string]any
	TokenUsage         map[string]int64
	Metadata           map[string]any
	Duration           time.Duration
	Success            bool
	Error              string
	Synthetic          bool
	Redacted           bool
	Truncated          bool
}

// ExecutionTrace captures an agent execution with step-level detail.
type ExecutionTrace struct {
	RunID            string
	AgentID          string
	AgentType        string
	Model            string
	Provider         string
	Task             string
	Input            map[string]any
	Output           map[string]any
	Steps            []TraceStep
	Status           TraceStatus
	Error            string
	StartedAt        time.Time
	CompletedAt      time.Time
	ProcessingTime   time.Duration
	TokenUsage       map[string]int64
	ToolUsageCount   map[string]int
	ContextMetadata  map[string]any
	TerminationCause string
}

// Clone returns a deep copy of the trace step so callers can safely retain it.
func (s TraceStep) Clone() TraceStep {
	return TraceStep{
		Index:              s.Index,
		Thought:            s.Thought,
		ActionRaw:          s.ActionRaw,
		Tool:               s.Tool,
		Arguments:          cloneAnyMap(s.Arguments),
		Observation:        s.Observation,
		ObservationDisplay: s.ObservationDisplay,
		ObservationDetails: cloneAnyMap(s.ObservationDetails),
		TokenUsage:         cloneTypedMap(s.TokenUsage),
		Metadata:           cloneAnyMap(s.Metadata),
		Duration:           s.Duration,
		Success:            s.Success,
		Error:              s.Error,
		Synthetic:          s.Synthetic,
		Redacted:           s.Redacted,
		Truncated:          s.Truncated,
	}
}

// Clone returns a deep copy of the execution trace.
func (t *ExecutionTrace) Clone() *ExecutionTrace {
	if t == nil {
		return nil
	}

	cloned := &ExecutionTrace{
		RunID:            t.RunID,
		AgentID:          t.AgentID,
		AgentType:        t.AgentType,
		Model:            t.Model,
		Provider:         t.Provider,
		Task:             t.Task,
		Input:            cloneAnyMap(t.Input),
		Output:           cloneAnyMap(t.Output),
		Status:           t.Status,
		Error:            t.Error,
		StartedAt:        t.StartedAt,
		CompletedAt:      t.CompletedAt,
		ProcessingTime:   t.ProcessingTime,
		TokenUsage:       cloneTypedMap(t.TokenUsage),
		ToolUsageCount:   cloneTypedMap(t.ToolUsageCount),
		ContextMetadata:  cloneAnyMap(t.ContextMetadata),
		TerminationCause: t.TerminationCause,
	}

	if t.Steps != nil {
		cloned.Steps = make([]TraceStep, len(t.Steps))
		for i, step := range t.Steps {
			cloned.Steps[i] = step.Clone()
		}
	}

	return cloned
}
