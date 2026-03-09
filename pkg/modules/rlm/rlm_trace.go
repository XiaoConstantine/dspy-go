package rlm

import (
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// RLMTraceStep captures one iteration of the RLM loop.
type RLMTraceStep struct {
	Index       int
	Thought     string
	Action      string
	Code        string
	SubQuery    string
	Observation string
	Duration    time.Duration
	Success     bool
	Error       string
}

// RLMTrace captures a structured RLM completion, including iterative steps.
type RLMTrace struct {
	Input            map[string]any
	Output           map[string]any
	Steps            []RLMTraceStep
	StartedAt        time.Time
	CompletedAt      time.Time
	ProcessingTime   time.Duration
	Iterations       int
	Usage            core.TokenUsage
	TerminationCause string
	Error            string
}

// Clone returns a deep copy of the trace.
func (t *RLMTrace) Clone() *RLMTrace {
	if t == nil {
		return nil
	}

	cloned := &RLMTrace{
		Input:            core.ShallowCopyMap(t.Input),
		Output:           core.ShallowCopyMap(t.Output),
		StartedAt:        t.StartedAt,
		CompletedAt:      t.CompletedAt,
		ProcessingTime:   t.ProcessingTime,
		Iterations:       t.Iterations,
		Usage:            t.Usage,
		TerminationCause: t.TerminationCause,
		Error:            t.Error,
	}
	if t.Steps != nil {
		cloned.Steps = make([]RLMTraceStep, len(t.Steps))
		copy(cloned.Steps, t.Steps)
	}

	return cloned
}
