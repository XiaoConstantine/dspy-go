package workflows

import (
	"context"
	"errors"
	"runtime/pprof"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestParallelWorkflow(t *testing.T) {
	t.Run("Execute steps in parallel", func(t *testing.T) {
		memory := new(MockMemory)
		workflow := NewParallelWorkflow(memory, 2)

		var wg sync.WaitGroup
		procCount := 0
		var mu sync.Mutex

		// Create mock modules that track concurrent execution
		createModule := func(id string, delay time.Duration) *MockModule {
			module := new(MockModule)
			module.On("GetSignature").Return(core.Signature{
				Inputs:  []core.InputField{{Name: "input"}},
				Outputs: []core.OutputField{{Name: id}},
			})
			module.On("Process", mock.Anything, mock.Anything).Run(func(args mock.Arguments) {
				mu.Lock()
				procCount++
				current := procCount
				mu.Unlock()

				// Ensure we don't exceed max concurrent processes
				assert.LessOrEqual(t, current, 2)
				time.Sleep(delay)

				mu.Lock()
				procCount--
				mu.Unlock()
				wg.Done()
			}).Return(map[string]any{id: "done"}, nil)
			return module
		}

		// Add three steps with different delays
		wg.Add(3)
		err := workflow.AddStep(&Step{ID: "step1", Module: createModule("output1", 100*time.Millisecond)})
		require.NoError(t, err, "Failed to add step1")

		err = workflow.AddStep(&Step{ID: "step2", Module: createModule("output2", 50*time.Millisecond)})
		require.NoError(t, err, "Failed to add step2")

		err = workflow.AddStep(&Step{ID: "step3", Module: createModule("output3", 75*time.Millisecond)})
		require.NoError(t, err, "Failed to add step3")

		// Execute workflow
		ctx := context.Background()
		result, err := workflow.Execute(ctx, map[string]any{
			"input": "value",
		})

		// Wait for all goroutines to complete
		wg.Wait()

		assert.NoError(t, err)
		assert.Equal(t, "done", result["output1"])
		assert.Equal(t, "done", result["output2"])
		assert.Equal(t, "done", result["output3"])
	})

	t.Run("Non-positive maxConcurrent means unlimited", func(t *testing.T) {
		for _, limit := range []int{0, -1} {
			memory := new(MockMemory)
			workflow := NewParallelWorkflow(memory, limit)

			module := new(MockModule)
			module.On("GetSignature").Return(core.Signature{
				Inputs:  []core.InputField{{Name: "input"}},
				Outputs: []core.OutputField{{Name: "output"}},
			})
			module.On("Process", mock.Anything, mock.Anything).Return(map[string]any{"output": "done"}, nil)

			require.NoError(t, workflow.AddStep(&Step{ID: "step1", Module: module}))

			result, err := workflow.Execute(context.Background(), map[string]any{"input": "value"})
			require.NoError(t, err, "maxConcurrent=%d must not deadlock or time out", limit)
			assert.Equal(t, "done", result["output"])
		}
	})

	t.Run("Cancellation while waiting on semaphore returns promptly", func(t *testing.T) {
		const limit = 2
		const steps = 6

		memory := new(MockMemory)
		workflow := NewParallelWorkflow(memory, limit)

		var active, maxActive int32
		entered := make(chan struct{}, steps)
		release := make(chan struct{})

		for i := 0; i < steps; i++ {
			id := "step" + string(rune('a'+i))
			module := new(MockModule)
			module.On("GetSignature").Return(core.Signature{
				Inputs:  []core.InputField{{Name: "input"}},
				Outputs: []core.OutputField{{Name: id}},
			})
			module.On("Process", mock.Anything, mock.Anything).Run(func(mock.Arguments) {
				n := atomic.AddInt32(&active, 1)
				for {
					m := atomic.LoadInt32(&maxActive)
					if n <= m || atomic.CompareAndSwapInt32(&maxActive, m, n) {
						break
					}
				}
				entered <- struct{}{}
				<-release
				atomic.AddInt32(&active, -1)
			}).Return(map[string]any{id: "done"}, nil).Maybe()
			require.NoError(t, workflow.AddStep(&Step{ID: id, Module: module}))
		}

		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan error, 1)
		go func() {
			_, err := workflow.Execute(ctx, map[string]any{"input": "value"})
			done <- err
		}()

		// Wait until the semaphore is saturated and the remaining steps are
		// waiting for a permit, then verify cancellation unblocks Execute.
		for i := 0; i < limit; i++ {
			<-entered
		}
		cancel()
		close(release)

		select {
		case err := <-done:
			assert.Error(t, err, "cancelled steps must surface the cancellation")
		case <-time.After(10 * time.Second):
			t.Fatal("Execute did not return after cancellation")
		}
		assert.LessOrEqual(t, atomic.LoadInt32(&maxActive), int32(limit),
			"active steps must never exceed maxConcurrent")
	})

	t.Run("Panicking steps become errors without crashing", func(t *testing.T) {
		memory := new(MockMemory)
		workflow := NewParallelWorkflow(memory, 2)

		// Several panicking steps stress the recovery path: the recovered
		// error must be sent before wg.Done, or the collector could close
		// the errors channel first and re-panic the process.
		for _, id := range []string{"boom1", "boom2", "boom3", "boom4"} {
			module := new(MockModule)
			module.On("GetSignature").Return(core.Signature{
				Inputs:  []core.InputField{{Name: "input"}},
				Outputs: []core.OutputField{{Name: id}},
			})
			module.On("Process", mock.Anything, mock.Anything).Run(func(mock.Arguments) {
				panic("kaboom: " + id)
			}).Return(map[string]any{}, nil)
			require.NoError(t, workflow.AddStep(&Step{ID: id, Module: module}))
		}

		_, err := workflow.Execute(context.Background(), map[string]any{"input": "value"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "panicked")
	})

	t.Run("Handle step failure", func(t *testing.T) {
		memory := new(MockMemory)
		workflow := NewParallelWorkflow(memory, 2)

		failingModule := new(MockModule)
		failingModule.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Name: "input"}},
			Outputs: []core.OutputField{{Name: "output"}},
		})
		failingModule.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{}, errors.New("step failed"),
		)

		err := workflow.AddStep(&Step{ID: "step1", Module: failingModule})
		require.NoError(t, err, "Failed to add step1")

		ctx := context.Background()
		_, err = workflow.Execute(ctx, map[string]any{
			"input": "value",
		})

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "step failed")
		failingModule.AssertExpectations(t)
	})
}

func TestAcquireSemaphoreCanceledWaiterDoesNotReleaseHolder(t *testing.T) {
	const limit = 2
	sem := make(chan struct{}, limit)
	sem <- struct{}{}
	sem <- struct{}{}

	ctx, cancel := context.WithCancel(context.Background())
	started := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		close(started)
		release, err := acquireSemaphore(ctx, sem, time.Minute)
		if release != nil {
			release()
		}
		done <- err
	}()

	// The channel is full before the waiter starts, so it cannot acquire a
	// permit. Canceling it must leave both holders' permits untouched.
	<-started
	cancel()
	require.ErrorIs(t, <-done, context.Canceled)
	assert.Equal(t, limit, len(sem), "a canceled waiter must not release a holder's permit")
}

func TestParallelWorkflowIsolatesExecutionState(t *testing.T) {
	const stepCount = 2

	workflow := NewParallelWorkflow(new(MockMemory), stepCount)
	states := make(chan *core.ExecutionState, stepCount)
	spans := make(chan *core.Span, stepCount)
	labels := make(chan map[string]string, stepCount)
	started := make(chan struct{}, stepCount)
	release := make(chan struct{})

	for _, id := range []string{"step1", "step2"} {
		module := new(MockModule)
		module.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Name: "input"}},
			Outputs: []core.OutputField{{Name: id}},
		})
		module.On("Process", mock.Anything, mock.Anything).Run(func(args mock.Arguments) {
			stepCtx := args.Get(0).(context.Context)
			states <- core.GetExecutionState(stepCtx)
			traceID, _ := pprof.Label(stepCtx, "trace_id")
			stepID, _ := pprof.Label(stepCtx, "step_id")
			labels <- map[string]string{"trace_id": traceID, "step_id": stepID}
			spanCtx, span := core.StartSpan(stepCtx, id)
			spans <- span
			started <- struct{}{}
			<-release
			core.EndSpan(spanCtx)
		}).Return(map[string]any{id: "done"}, nil)
		require.NoError(t, workflow.AddStep(&Step{ID: id, Module: module}))
	}

	parentCtx := core.WithExecutionState(context.Background())
	parentState := core.GetExecutionState(parentCtx)
	done := make(chan error, 1)
	go func() {
		_, err := workflow.Execute(parentCtx, map[string]any{"input": "value"})
		done <- err
	}()

	for range stepCount {
		<-started
	}
	close(release)
	require.NoError(t, <-done)

	firstState := <-states
	secondState := <-states
	require.NotNil(t, firstState)
	require.NotNil(t, secondState)
	assert.NotSame(t, parentState, firstState)
	assert.NotSame(t, parentState, secondState)
	assert.NotSame(t, firstState, secondState)
	assert.Equal(t, parentState.GetTraceID(), firstState.GetTraceID())
	assert.Equal(t, parentState.GetTraceID(), secondState.GetTraceID())
	seenStepIDs := make(map[string]bool, stepCount)
	for range stepCount {
		got := <-labels
		assert.Equal(t, parentState.GetTraceID(), got["trace_id"])
		seenStepIDs[got["step_id"]] = true
	}
	assert.Equal(t, map[string]bool{"step1": true, "step2": true}, seenStepIDs)

	for range stepCount {
		assert.Empty(t, (<-spans).ParentID, "parallel steps must not become sibling span parents")
	}
	assert.Empty(t, core.CollectSpans(parentCtx), "parallel steps must not mutate the parent execution state")
}
