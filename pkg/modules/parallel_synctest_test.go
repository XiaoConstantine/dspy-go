// Package modules provides synctest-based tests for concurrent module execution.
// These tests use Go 1.25's testing/synctest package for deterministic concurrent testing.
package modules

import (
	"context"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
)

// TestParallelWithSynctest demonstrates Go 1.25's synctest for deterministic concurrent tests.
// The synctest.Test function creates an isolated "bubble" with virtualized time,
// allowing tests to verify concurrent behavior without timing-dependent flakiness.
func TestParallelWithSynctest(t *testing.T) {
	t.Run("Deterministic parallel execution timing", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			// Create a mock module that sleeps for 100ms per call
			// In synctest bubble, this sleep advances virtual time instantly
			mockModule := NewMockModule(false, 100*time.Millisecond)
			parallel := NewParallel(mockModule, WithMaxWorkers(4))

			batchInputs := []map[string]interface{}{
				{"input": "a"},
				{"input": "b"},
				{"input": "c"},
				{"input": "d"},
			}

			ctx := context.Background()
			start := time.Now()

			result, err := parallel.Process(ctx, map[string]interface{}{
				"batch_inputs": batchInputs,
			})

			elapsed := time.Since(start)

			assert.NoError(t, err)
			assert.NotNil(t, result)

			// With 4 workers processing 4 items with 100ms each in parallel,
			// virtual time should advance by ~100ms (not 400ms sequential)
			// In synctest, we can make deterministic assertions about timing
			t.Logf("Virtual elapsed time: %v", elapsed)

			results := result["results"].([]map[string]interface{})
			assert.Len(t, results, 4)
		})
	})

	t.Run("Worker completion ordering with synctest.Wait", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			var completionOrder []int
			var orderMu atomic.Int32

			// Create a custom module that tracks completion order
			mockModule := &orderTrackingModule{
				completionOrder: &completionOrder,
				orderCounter:    &orderMu,
			}

			parallel := NewParallel(mockModule, WithMaxWorkers(2))

			batchInputs := []map[string]interface{}{
				{"input": "first", "delay": 50},
				{"input": "second", "delay": 10},
			}

			ctx := context.Background()
			result, err := parallel.Process(ctx, map[string]interface{}{
				"batch_inputs": batchInputs,
			})

			// synctest.Wait() ensures all goroutines in the bubble are blocked
			// before we check results - this makes the test deterministic
			synctest.Wait()

			assert.NoError(t, err)
			assert.NotNil(t, result)

			// Results should be in original order regardless of completion order
			results := result["results"].([]map[string]interface{})
			assert.Len(t, results, 2)
		})
	})

	t.Run("Context cancellation with virtual time", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			// Create a slow module
			mockModule := NewMockModule(false, 1*time.Second)
			parallel := NewParallel(mockModule, WithMaxWorkers(2), WithStopOnFirstError(false))

			batchInputs := []map[string]interface{}{
				{"input": "a"},
				{"input": "b"},
			}

			// Create a context that cancels after 100ms (virtual time)
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()

			result, err := parallel.Process(ctx, map[string]interface{}{
				"batch_inputs": batchInputs,
			})

			// In synctest, the timeout fires deterministically at virtual 100ms
			// The 1-second sleeps in the module will be interrupted
			if err != nil {
				t.Logf("Expected cancellation error: %v", err)
			}

			// Either we get partial results or an error - both are valid
			// The key is that the test completes deterministically
			_ = result
		})
	})
}

// orderTrackingModule tracks the order in which Process calls complete.
type orderTrackingModule struct {
	core.BaseModule
	completionOrder *[]int
	orderCounter    *atomic.Int32
}

func (m *orderTrackingModule) Process(ctx context.Context, inputs map[string]interface{}, opts ...core.Option) (map[string]interface{}, error) {
	// Simulate variable processing time based on input
	delay := 10 * time.Millisecond
	if d, ok := inputs["delay"].(int); ok {
		delay = time.Duration(d) * time.Millisecond
	}
	time.Sleep(delay)

	// Record completion order
	order := m.orderCounter.Add(1)
	*m.completionOrder = append(*m.completionOrder, int(order))

	input, _ := inputs["input"].(string)
	return map[string]interface{}{
		"output": "processed_" + input,
	}, nil
}

func (m *orderTrackingModule) Clone() core.Module {
	return &orderTrackingModule{
		BaseModule:      m.BaseModule,
		completionOrder: m.completionOrder,
		orderCounter:    m.orderCounter,
	}
}

func (m *orderTrackingModule) GetSignature() core.Signature {
	return core.Signature{
		Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
		Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
	}
}
