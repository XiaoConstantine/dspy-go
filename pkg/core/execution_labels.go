package core

import (
	"context"
	"runtime/pprof"
)

// ExecutionLabels identifies an asynchronous execution boundary in profiles
// and goroutine tracebacks. Values must be stable, non-sensitive identifiers;
// callers must never use prompts, credentials, or request content as labels.
type ExecutionLabels struct {
	TaskID string
	StepID string
	Tool   string
}

// DoWithExecutionLabels runs f with the supplied execution labels. A trace ID
// is added automatically when ctx carries an ExecutionState. Goroutines started
// by f inherit the labels from the calling goroutine.
func DoWithExecutionLabels(ctx context.Context, labels ExecutionLabels, f func(context.Context)) {
	pairs := make([]string, 0, 8)
	if state := GetExecutionState(ctx); state != nil {
		if traceID := state.GetTraceID(); traceID != "" {
			pairs = append(pairs, "trace_id", traceID)
		}
	}
	if labels.TaskID != "" {
		pairs = append(pairs, "task_id", labels.TaskID)
	}
	if labels.StepID != "" {
		pairs = append(pairs, "step_id", labels.StepID)
	}
	if labels.Tool != "" {
		pairs = append(pairs, "tool", labels.Tool)
	}

	if len(pairs) == 0 {
		f(ctx)
		return
	}
	pprof.Do(ctx, pprof.Labels(pairs...), f)
}
