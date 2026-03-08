package agents

import (
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

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

// Clone returns a deep copy of the trace step so callers can safely retain it.
func (s TraceStep) Clone() TraceStep {
	return TraceStep{
		Index:       s.Index,
		Thought:     s.Thought,
		ActionRaw:   s.ActionRaw,
		Tool:        s.Tool,
		Arguments:   core.ShallowCopyMap(s.Arguments),
		Observation: s.Observation,
		Duration:    s.Duration,
		Success:     s.Success,
		Error:       s.Error,
	}
}

// Clone returns a deep copy of the execution trace.
func (t *ExecutionTrace) Clone() *ExecutionTrace {
	if t == nil {
		return nil
	}

	cloned := &ExecutionTrace{
		AgentID:          t.AgentID,
		AgentType:        t.AgentType,
		Task:             t.Task,
		Input:            core.ShallowCopyMap(t.Input),
		Output:           core.ShallowCopyMap(t.Output),
		Status:           t.Status,
		Error:            t.Error,
		StartedAt:        t.StartedAt,
		CompletedAt:      t.CompletedAt,
		ProcessingTime:   t.ProcessingTime,
		TokenUsage:       core.ShallowCopyMap(t.TokenUsage),
		ToolUsageCount:   core.ShallowCopyMap(t.ToolUsageCount),
		ContextMetadata:  core.ShallowCopyMap(t.ContextMetadata),
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
