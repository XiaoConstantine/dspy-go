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
	Input             map[string]any
	Output            map[string]any
	Steps             []RLMTraceStep
	StartedAt         time.Time
	CompletedAt       time.Time
	ProcessingTime    time.Duration
	Iterations        int
	Usage             core.TokenUsage
	RootUsage         core.TokenUsage
	SubUsage          core.TokenUsage
	SubRLMUsage       core.TokenUsage
	RootSnapshots     []RootIterationSnapshot
	SubLLMCallCount   int
	SubRLMCallCount   int
	ConfidenceSignals int
	CompressionCount  int
	TerminationCause  string
	Error             string

	tokenTracker *TokenTracker
}

// Clone returns a deep copy of the trace.
func (t *RLMTrace) Clone() *RLMTrace {
	if t == nil {
		return nil
	}

	cloned := &RLMTrace{
		Input:             core.ShallowCopyMap(t.Input),
		Output:            core.ShallowCopyMap(t.Output),
		StartedAt:         t.StartedAt,
		CompletedAt:       t.CompletedAt,
		ProcessingTime:    t.ProcessingTime,
		Iterations:        t.Iterations,
		Usage:             t.Usage,
		RootUsage:         t.RootUsage,
		SubUsage:          t.SubUsage,
		SubRLMUsage:       t.SubRLMUsage,
		SubLLMCallCount:   t.SubLLMCallCount,
		SubRLMCallCount:   t.SubRLMCallCount,
		ConfidenceSignals: t.ConfidenceSignals,
		CompressionCount:  t.CompressionCount,
		TerminationCause:  t.TerminationCause,
		Error:             t.Error,
	}
	if t.Steps != nil {
		cloned.Steps = make([]RLMTraceStep, len(t.Steps))
		copy(cloned.Steps, t.Steps)
	}
	if t.RootSnapshots != nil {
		cloned.RootSnapshots = make([]RootIterationSnapshot, len(t.RootSnapshots))
		copy(cloned.RootSnapshots, t.RootSnapshots)
	}

	return cloned
}
