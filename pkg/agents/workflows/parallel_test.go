package workflows

import (
	"context"
	"errors"
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
				Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
				Outputs: []core.OutputField{{Field: core.Field{Name: id}}},
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
				Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
				Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
			})
			module.On("Process", mock.Anything, mock.Anything).Return(map[string]any{"output": "done"}, nil)

			require.NoError(t, workflow.AddStep(&Step{ID: "step1", Module: module}))

			result, err := workflow.Execute(context.Background(), map[string]any{"input": "value"})
			require.NoError(t, err, "maxConcurrent=%d must not deadlock or time out", limit)
			assert.Equal(t, "done", result["output"])
		}
	})

	t.Run("Cancellation while waiting on semaphore keeps the concurrency bound", func(t *testing.T) {
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
				Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
				Outputs: []core.OutputField{{Field: core.Field{Name: id}}},
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

		// Wait until the semaphore is saturated: `limit` steps are running
		// and the rest are blocked acquiring a permit. Then cancel, which
		// previously could steal a running step's permit via the deferred
		// non-blocking drain.
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
				Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
				Outputs: []core.OutputField{{Field: core.Field{Name: id}}},
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
			Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
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
