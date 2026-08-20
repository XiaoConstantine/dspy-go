package core

import (
	"context"
	"runtime/pprof"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDoWithExecutionLabels(t *testing.T) {
	ctx := WithExecutionState(context.Background())
	traceID := GetExecutionState(ctx).GetTraceID()

	DoWithExecutionLabels(ctx, ExecutionLabels{
		TaskID: "task-1",
		StepID: "step-1",
		Tool:   "search",
	}, func(labeledCtx context.Context) {
		assertPprofLabel(t, labeledCtx, "trace_id", traceID)
		assertPprofLabel(t, labeledCtx, "task_id", "task-1")
		assertPprofLabel(t, labeledCtx, "step_id", "step-1")
		assertPprofLabel(t, labeledCtx, "tool", "search")
	})

	_, ok := pprof.Label(ctx, "task_id")
	assert.False(t, ok, "labels must not mutate the parent context")
}

func TestDoWithExecutionLabelsWithoutValues(t *testing.T) {
	ctx := context.Background()
	called := false

	DoWithExecutionLabels(ctx, ExecutionLabels{}, func(got context.Context) {
		called = true
		assert.Equal(t, ctx, got)
	})

	assert.True(t, called)
}

func assertPprofLabel(t testing.TB, ctx context.Context, key, want string) {
	t.Helper()
	got, ok := pprof.Label(ctx, key)
	if !ok {
		t.Fatalf("pprof label %q is missing", key)
	}
	assert.Equal(t, want, got)
}
