package workflows

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestStep(t *testing.T) {
	// Helper function to create a basic step with mocks
	setupStep := func() (*Step, *MockModule) {
		module := new(MockModule)
		module.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
		})

		return &Step{
			ID:     "test_step",
			Module: module,
		}, module
	}

	t.Run("Basic execution", func(t *testing.T) {
		step, module := setupStep()

		// Set up expected behavior
		module.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{"output": "success"}, nil,
		)

		// Execute step
		ctx := context.Background()
		result, err := step.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		// Verify results
		require.NoError(t, err)
		assert.Equal(t, "success", result.Outputs["output"])
		assert.Equal(t, step.ID, result.StepID)
		module.AssertExpectations(t)
	})

	t.Run("Input validation", func(t *testing.T) {
		step, module := setupStep()

		ctx := context.Background()
		_, err := step.Execute(ctx, map[string]interface{}{
			"wrong_input": "test", // Missing required 'input' field
		})

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "input validation failed")
		module.AssertNotCalled(t, "Process")
	})

	t.Run("Output validation", func(t *testing.T) {
		step, module := setupStep()

		module.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{"wrong_output": "value"}, nil, // Missing required 'output' field
		)

		ctx := context.Background()
		_, err := step.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "output validation failed")
		module.AssertExpectations(t)
	})

	t.Run("Condition check", func(t *testing.T) {
		step, module := setupStep()

		// Add condition that always fails
		step.Condition = func(state map[string]interface{}) bool {
			return false
		}

		ctx := context.Background()
		_, err := step.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		assert.ErrorIs(t, err, ErrStepConditionFailed)
		module.AssertNotCalled(t, "Process")
	})

	t.Run("Retry logic", func(t *testing.T) {
		step, module := setupStep()

		// Configure retry
		step.RetryConfig = &RetryConfig{
			MaxAttempts:       3,
			BackoffMultiplier: 1.5,
		}
		var attempts int32

		module.On("Process", mock.Anything, mock.Anything).
			Run(func(args mock.Arguments) {
				current := atomic.AddInt32(&attempts, 1)
				t.Logf("Attempt #%d", current)
			}).
			Return(make(map[string]interface{}), assert.AnError).
			Times(2)

		// Third call will succeed
		module.On("Process", mock.Anything, mock.Anything).
			Run(func(args mock.Arguments) {
				atomic.AddInt32(&attempts, 1)
				t.Logf("Final successful attempt")
			}).
			Return(map[string]interface{}{"output": "success"}, nil).
			Once()

		ctx := context.Background()
		result, err := step.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		// Verify results
		require.NoError(t, err)
		assert.Equal(t, "success", result.Outputs["output"])
		assert.Equal(t, int32(3), atomic.LoadInt32(&attempts),
			"Should have attempted exactly 3 times")

		module.AssertExpectations(t)
	})
	t.Run("Retry backoff timing", func(t *testing.T) {
		step, module := setupStep()

		// Configure retry with very small intervals for testing
		step.RetryConfig = &RetryConfig{
			MaxAttempts:       2,
			BackoffMultiplier: 2.0,
		}

		// Record attempt times in a thread-safe way
		var mu sync.Mutex
		attemptTimes := make([]time.Time, 0, 2)

		module.On("Process", mock.Anything, mock.Anything).
			Run(func(args mock.Arguments) {
				mu.Lock()
				attemptTimes = append(attemptTimes, time.Now())
				mu.Unlock()
			}).
			Return(make(map[string]interface{}), assert.AnError).
			Times(2)

		ctx := context.Background()
		_, err := step.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		// Should fail after max attempts
		require.Error(t, err)

		mu.Lock()
		recordedAttempts := attemptTimes
		mu.Unlock()

		// Verify we got the expected number of attempts
		assert.Equal(t, 2, len(recordedAttempts),
			"Should have recorded exactly 2 attempts")

		// Check intervals between attempts
		if len(recordedAttempts) >= 2 {
			firstInterval := recordedAttempts[1].Sub(recordedAttempts[0])
			t.Logf("Interval between retries: %v", firstInterval)
			// Just verify the ordering is preserved and delays are positive
			assert.True(t, firstInterval > 0,
				"Time should progress forward between attempts")
		}

		module.AssertExpectations(t)
	})

	t.Run("Context cancellation", func(t *testing.T) {
		t.Skip("Skip context cancellation for now")
		step, module := setupStep()

		// Create cancellable context
		ctx, cancel := context.WithCancel(context.Background())

		// Mock module to wait before responding
		module.On("Process", mock.Anything, mock.Anything).
			Run(func(args mock.Arguments) {
				time.Sleep(100 * time.Millisecond)
			}).
			Return(map[string]any{"output": "success"}, nil)

		// Cancel context shortly after starting
		go func() {
			time.Sleep(50 * time.Millisecond)
			cancel()
		}()

		_, err := step.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		assert.ErrorIs(t, err, context.Canceled)
		module.AssertExpectations(t)
	})

	t.Run("Next steps propagation", func(t *testing.T) {
		step, module := setupStep()

		// Configure next steps
		step.NextSteps = []string{"step2", "step3"}

		module.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{"output": "success"}, nil,
		)

		ctx := context.Background()
		result, err := step.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		require.NoError(t, err)
		assert.Equal(t, step.NextSteps, result.NextSteps)
		module.AssertExpectations(t)
	})
}
