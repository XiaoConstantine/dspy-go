package workflows

import (
	"context"
	"fmt"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
)

// ParallelWorkflow executes multiple steps concurrently.
type ParallelWorkflow struct {
	*BaseWorkflow
	// Maximum number of concurrent steps
	maxConcurrent int
}

func NewParallelWorkflow(memory agents.Memory, maxConcurrent int) *ParallelWorkflow {
	return &ParallelWorkflow{
		BaseWorkflow:  NewBaseWorkflow(memory),
		maxConcurrent: maxConcurrent,
	}
}

func (w *ParallelWorkflow) Execute(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	state := make(map[string]interface{})
	for k, v := range inputs {
		state[k] = v
	}

	// Create channel for collecting results
	results := make(chan *StepResult, len(w.steps))
	errors := make(chan error, len(w.steps))

	// Create semaphore to limit concurrency
	sem := make(chan struct{}, w.maxConcurrent)

	// Launch goroutine for each step
	var wg sync.WaitGroup
	for _, step := range w.steps {
		wg.Add(1)
		go func(s *Step) {
			defer wg.Done()

			// Acquire semaphore
			sem <- struct{}{}
			defer func() { <-sem }()

			// Prepare inputs for this step
			stepInputs := make(map[string]interface{})
			for _, field := range s.InputFields {
				if val, ok := inputs[field]; ok {
					stepInputs[field] = val
				}
			}

			// Execute step
			result, err := s.Execute(ctx, stepInputs)
			if err != nil {
				errors <- fmt.Errorf("step %s failed: %w", s.ID, err)
				return
			}
			results <- result
		}(step)
	}

	// Wait for all steps to complete
	go func() {
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
