package workflows

import (
	"context"
	"errors"
	"sync"
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
		result, err := workflow.Execute(ctx, map[string]interface{}{
			"input": "value",
		})

		// Wait for all goroutines to complete
		wg.Wait()

		assert.NoError(t, err)
		assert.Equal(t, "done", result["output1"])
		assert.Equal(t, "done", result["output2"])
		assert.Equal(t, "done", result["output3"])
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
		_, err = workflow.Execute(ctx, map[string]interface{}{
			"input": "value",
		})

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "step failed")
		failingModule.AssertExpectations(t)
	})
}
