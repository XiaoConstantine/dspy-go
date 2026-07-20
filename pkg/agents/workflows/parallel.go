package workflows

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
)

// ParallelWorkflow executes multiple steps concurrently.
type ParallelWorkflow struct {
	*BaseWorkflow
	// Maximum number of concurrent steps
	maxConcurrent int
}

// acquireSemaphore waits for a semaphore permit and returns the corresponding
// release function. A failed acquisition never changes the semaphore, which is
// important when cancellation races with existing permit holders.
func acquireSemaphore(ctx context.Context, sem chan struct{}, timeout time.Duration) (func(), error) {
	semCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	select {
	case sem <- struct{}{}:
		return func() { <-sem }, nil
	case <-semCtx.Done():
		return nil, semCtx.Err()
	}
}

// NewParallelWorkflow creates a workflow that runs its steps concurrently.
// A non-positive maxConcurrent means no concurrency limit.
func NewParallelWorkflow(memory agents.Memory, maxConcurrent int) *ParallelWorkflow {
	return &ParallelWorkflow{
		BaseWorkflow:  NewBaseWorkflow(memory),
		maxConcurrent: maxConcurrent,
	}
}

func (w *ParallelWorkflow) Execute(ctx context.Context, inputs map[string]any) (map[string]any, error) {
	state := make(map[string]any)
	for k, v := range inputs {
		state[k] = v
	}

	// Create channel for collecting results
	results := make(chan *StepResult, len(w.steps))
	errors := make(chan error, len(w.steps))

	// Create semaphore to limit concurrency. A non-positive limit means
	// unlimited, which a zero- or negative-capacity channel cannot
	// express (zero never admits work; negative panics).
	limit := w.maxConcurrent
	if limit <= 0 {
		limit = len(w.steps)
	}
	sem := make(chan struct{}, limit)

	// Launch goroutine for each step
	var wg sync.WaitGroup
	for _, step := range w.steps {
		wg.Add(1)
		go func(s *Step) {
			// Registered first so it runs last: the panic-recovery send
			// below must complete before wg.Done lets the collector
			// close the channels.
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					errors <- fmt.Errorf("step %s panicked: %v", s.ID, r)
				}
			}()

			// Check if context is already cancelled before execution
			select {
			case <-ctx.Done():
				errors <- ctx.Err()
				return
			default:
			}

			// Acquire semaphore with timeout to prevent deadlock.
			release, err := acquireSemaphore(ctx, sem, 30*time.Second)
			if err != nil {
				if err == context.DeadlineExceeded {
					errors <- fmt.Errorf("semaphore acquire timeout for step %s", s.ID)
				} else {
					errors <- ctx.Err() // Original context was cancelled
				}
				return
			}
			// Release only after a successful acquire; releasing on every
			// exit path could steal a permit another goroutine holds.
			defer release()

			// Prepare inputs for this step
			stepInputs := make(map[string]any)
			signature := s.Module.GetSignature()

			for _, field := range signature.Inputs {
				if val, ok := inputs[field.Name]; ok {
					stepInputs[field.Name] = val
				}
			}

			// Execute step
			result, err := s.Execute(ctx, stepInputs)
			if err != nil {
				errors <- fmt.Errorf("step %s failed: %w", s.ID, err)
				return
			}

			// Send result without blocking if context is cancelled
			select {
			case results <- result:
			case <-ctx.Done():
				return
			}
		}(step)
	}

	// Wait for all steps to complete
	go func() {
		defer func() {
			// Ensure goroutine cleanup on panic
			if r := recover(); r != nil {
				// Log panic but continue
				_ = r
			}
		}()

		wg.Wait()
		close(results)
		close(errors)
	}()

	// Collect results and errors
	var errs []error
	for err := range errors {
		errs = append(errs, err)
	}
	if len(errs) > 0 {
		return nil, fmt.Errorf("parallel execution failed with %d errors: %v", len(errs), errs)
	}

	// Merge results into final state
	for result := range results {
		for k, v := range result.Outputs {
			state[k] = v
		}
	}

	return state, nil
}
