// Package modules provides the Parallel execution wrapper for DSPy-Go.
//
// The Parallel module enables concurrent execution of any DSPy module across
// multiple inputs, providing significant performance improvements for batch processing.
//
// Example usage:
//
//	predict := modules.NewPredict(signature)
//	parallel := modules.NewParallel(predict,
//		modules.WithMaxWorkers(4),
//		modules.WithReturnFailures(true))
//
//	batchInputs := []map[string]interface{}{
//		{"input": "first example"},
//		{"input": "second example"},
//		{"input": "third example"},
//	}
//
//	result, err := parallel.Process(ctx, map[string]interface{}{
//		"batch_inputs": batchInputs,
//	})
//
//	results := result["results"].([]map[string]interface{})
//
// The parallel module automatically manages worker pools, error handling,
// and result collection while maintaining the order of inputs in outputs.
package modules

import (
	"context"
	"runtime"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// ParallelOptions configures parallel execution behavior.
type ParallelOptions struct {
	// MaxWorkers sets the maximum number of concurrent workers
	MaxWorkers int
	// ReturnFailures determines if failed results should be included in output
	ReturnFailures bool
	// StopOnFirstError stops execution on first error encountered
	StopOnFirstError bool
}

// ParallelResult contains the result of a parallel execution.
type ParallelResult struct {
	Index   int                    // Original index in the input batch
	Success bool                   // Whether execution succeeded
	Output  map[string]interface{} // The actual output
	Error   error                  // Error if execution failed
}

// Parallel executes a module against multiple inputs concurrently.
type Parallel struct {
	core.BaseModule
	innerModule core.Module
	options     ParallelOptions
}

// Ensure Parallel implements core.Module.
var _ core.Module = (*Parallel)(nil)

// NewParallel creates a new parallel execution wrapper around a module.
func NewParallel(module core.Module, opts ...ParallelOption) *Parallel {
	// Default options
	options := ParallelOptions{
		MaxWorkers:       runtime.NumCPU(),
		ReturnFailures:   false,
		StopOnFirstError: false,
	}

	// Apply custom options
	for _, opt := range opts {
		opt(&options)
	}

	// Copy the signature from the inner module
	signature := module.GetSignature()

	baseModule := core.NewModule(signature)
	baseModule.ModuleType = "Parallel"
	baseModule.DisplayName = "" // Will be set by user or derived from context

	return &Parallel{
		BaseModule:  *baseModule,
		innerModule: module,
		options:     options,
	}
}

// WithName sets a semantic name for this Parallel instance.
func (p *Parallel) WithName(name string) *Parallel {
	p.DisplayName = name
	return p
}

// ParallelOption is a function that configures ParallelOptions.
type ParallelOption func(*ParallelOptions)

// WithMaxWorkers sets the maximum number of concurrent workers.
func WithMaxWorkers(count int) ParallelOption {
	return func(opts *ParallelOptions) {
		if count > 0 {
			opts.MaxWorkers = count
		}
	}
}

// WithReturnFailures configures whether to return failed results.
func WithReturnFailures(returnFailures bool) ParallelOption {
	return func(opts *ParallelOptions) {
		opts.ReturnFailures = returnFailures
	}
}

// WithStopOnFirstError configures whether to stop on first error.
func WithStopOnFirstError(stopOnError bool) ParallelOption {
	return func(opts *ParallelOptions) {
		opts.StopOnFirstError = stopOnError
	}
}

// Process executes the inner module against multiple inputs in parallel.
func (p *Parallel) Process(ctx context.Context, inputs map[string]interface{}, opts ...core.Option) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	ctx, span := core.StartSpan(ctx, "ParallelProcess")
	defer core.EndSpan(ctx)

	// Extract batch inputs - expect a special key "batch_inputs" containing slice of input maps
	batchInputs, ok := inputs["batch_inputs"].([]map[string]interface{})
	if !ok {
		// If not batch, treat as single input
		result, err := p.innerModule.Process(ctx, inputs, opts...)
		return result, err
	}

	if len(batchInputs) == 0 {
		return map[string]interface{}{"results": []map[string]interface{}{}}, nil
	}

	logger.Debug(ctx, "Processing %d inputs in parallel with %d workers", len(batchInputs), p.options.MaxWorkers)

	// Create worker pool
	workers := p.options.MaxWorkers
	if workers > len(batchInputs) {
		workers = len(batchInputs)
	}

	// Channels for work distribution and result collection
	jobs := make(chan jobInput, len(batchInputs))
	results := make(chan ParallelResult, len(batchInputs))

	// Context for cancellation
	workerCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go p.worker(workerCtx, &wg, jobs, results, opts...)
	}

	// Send jobs
	go func() {
		defer func() {
			close(jobs)
			// Ensure goroutine cleanup on panic
			if r := recover(); r != nil {
				// Log panic but continue
				_ = r
			}
		}()

		for i, input := range batchInputs {
			select {
			case jobs <- jobInput{index: i, inputs: input}:
			case <-workerCtx.Done():
				return
			}
		}
	}()

	// Collect results
	go func() {
		defer func() {
			close(results)
			// Ensure goroutine cleanup on panic
			if r := recover(); r != nil {
				// Log panic but continue
				_ = r
			}
		}()

		wg.Wait()
	}()

	// Process results
	parallelResults := make([]ParallelResult, 0, len(batchInputs))
	var firstError error

	for result := range results {
		parallelResults = append(parallelResults, result)

		// Handle stop on first error
		if p.options.StopOnFirstError && result.Error != nil && firstError == nil {
			firstError = result.Error
			cancel() // Cancel remaining work
		}
	}

	// If we stopped on first error, return it
	if firstError != nil {
		span.WithError(firstError)
		return nil, errors.WithFields(
			errors.Wrap(firstError, errors.LLMGenerationFailed, "parallel execution failed"),
			errors.Fields{
				"module":     "Parallel",
				"batch_size": len(batchInputs),
			})
	}

	// Sort results by original index to maintain order
	sortedResults := make([]ParallelResult, len(batchInputs))
	for _, result := range parallelResults {
		if result.Index < len(sortedResults) {
			sortedResults[result.Index] = result
		}
	}

	// Format output based on options
	outputs := p.formatResults(sortedResults)

	span.WithAnnotation("batch_size", len(batchInputs))
	span.WithAnnotation("successful_results", p.countSuccessful(sortedResults))

	return outputs, nil
}

// jobInput represents a single job for the worker pool.
type jobInput struct {
	index  int
	inputs map[string]interface{}
}

// worker processes jobs from the jobs channel.
func (p *Parallel) worker(ctx context.Context, wg *sync.WaitGroup, jobs <-chan jobInput, results chan<- ParallelResult, opts ...core.Option) {
	defer wg.Done()
	logger := logging.GetLogger()

	for {
		select {
		case job, ok := <-jobs:
			if !ok {
				return // Channel closed, worker done
			}

			// Process the job
			logger.Debug(ctx, "Worker processing job %d", job.index)
			output, err := p.innerModule.Process(ctx, job.inputs, opts...)

			result := ParallelResult{
				Index:   job.index,
				Success: err == nil,
				Output:  output,
				Error:   err,
			}

			select {
			case results <- result:
			case <-ctx.Done():
				return
			}

		case <-ctx.Done():
			return // Context cancelled
		}
	}
}

// formatResults converts ParallelResult slice to output format.
// The outputs slice maintains the same length and order as the input batch,
// with nil placeholders for failed items to preserve index correspondence.
func (p *Parallel) formatResults(results []ParallelResult) map[string]interface{} {
	outputs := make([]map[string]interface{}, len(results))
	failures := make([]map[string]interface{}, 0)

	for i, result := range results {
		if result.Success {
			outputs[i] = result.Output
		} else {
			outputs[i] = nil // Use nil to preserve order for failed items
			if p.options.ReturnFailures {
				errStr := "an unknown error occurred"
				if result.Error != nil {
					errStr = result.Error.Error()
				}
				failureInfo := map[string]interface{}{
					"index": result.Index,
					"error": errStr,
				}
				failures = append(failures, failureInfo)
			}
		}
	}

	output := map[string]interface{}{
		"results": outputs,
	}

	if p.options.ReturnFailures && len(failures) > 0 {
		output["failures"] = failures
	}

	return output
}

// countSuccessful returns the number of successful results.
func (p *Parallel) countSuccessful(results []ParallelResult) int {
	count := 0
	for _, result := range results {
		if result.Success {
			count++
		}
	}
	return count
}

// Clone creates a deep copy of the Parallel module.
func (p *Parallel) Clone() core.Module {
	return &Parallel{
		BaseModule:  *p.BaseModule.Clone().(*core.BaseModule),
		innerModule: p.innerModule.Clone(),
		options:     p.options, // Options are copied by value
	}
}

// GetInnerModule returns the wrapped module.
func (p *Parallel) GetInnerModule() core.Module {
	return p.innerModule
}

// SetLLM sets the LLM for both this module and the inner module.
func (p *Parallel) SetLLM(llm core.LLM) {
	p.BaseModule.SetLLM(llm)
	p.innerModule.SetLLM(llm)
}
