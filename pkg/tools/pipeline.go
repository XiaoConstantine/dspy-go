package tools

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// DataTransformer defines a function that transforms data between tools in a pipeline.
type DataTransformer func(input interface{}) (map[string]interface{}, error)

// PipelineStep represents a single step in a tool pipeline.
type PipelineStep struct {
	ToolName    string          // Name of the tool to execute
	Transformer DataTransformer // Optional data transformer
	Timeout     time.Duration   // Step-specific timeout
	Retries     int             // Number of retries on failure
	Conditions  []Condition     // Conditions for step execution
}

// Condition defines when a pipeline step should execute.
type Condition struct {
	Field    string      // Field to check in previous step result
	Operator string      // eq, ne, gt, lt, contains, exists
	Value    interface{} // Value to compare against
}

// PipelineOptions configures pipeline execution.
type PipelineOptions struct {
	Timeout         time.Duration   // Overall pipeline timeout
	FailureStrategy FailureStrategy // How to handle step failures
	Parallel        bool            // Execute steps in parallel when possible
	CacheResults    bool            // Cache intermediate results
}

// FailureStrategy defines how to handle step failures.
type FailureStrategy int

const (
	FailFast        FailureStrategy = iota // Stop on first failure
	ContinueOnError                        // Continue with remaining steps
	Retry                                  // Retry failed steps
)

// PipelineResult contains the results of pipeline execution.
type PipelineResult struct {
	Results      []core.ToolResult          // Results from each step
	StepMetadata map[string]StepMetadata    // Metadata for each step
	Duration     time.Duration              // Total execution time
	Success      bool                       // Whether pipeline succeeded
	FailedStep   string                     // Name of failed step (if any)
	Error        error                      // Error details
	Cache        map[string]core.ToolResult // Cached intermediate results
}

// StepMetadata contains execution metadata for a pipeline step.
type StepMetadata struct {
	ToolName    string        // Tool that was executed
	Duration    time.Duration // Step execution time
	Success     bool          // Whether step succeeded
	Retries     int           // Number of retries attempted
	Cached      bool          // Whether result was cached
	Transformed bool          // Whether data was transformed
}

// ToolPipeline manages the execution of a sequence of tools.
type ToolPipeline struct {
	name     string
	steps    []PipelineStep
	options  PipelineOptions
	registry core.ToolRegistry
	cache    map[string]core.ToolResult
	mu       sync.RWMutex
}

// NewToolPipeline creates a new tool pipeline.
func NewToolPipeline(name string, registry core.ToolRegistry, options PipelineOptions) *ToolPipeline {
	return &ToolPipeline{
		name:     name,
		steps:    make([]PipelineStep, 0),
		options:  options,
		registry: registry,
		cache:    make(map[string]core.ToolResult),
	}
}

// AddStep adds a step to the pipeline.
func (tp *ToolPipeline) AddStep(step PipelineStep) error {
	// Validate that the tool exists
	_, err := tp.registry.Get(step.ToolName)
	if err != nil {
		return errors.WithFields(
			errors.New(errors.InvalidInput, "tool not found in registry"),
			errors.Fields{"tool_name": step.ToolName},
		)
	}

	// Set default values
	if step.Timeout == 0 {
		step.Timeout = 30 * time.Second
	}
	if step.Retries < 0 {
		step.Retries = 0
	}

	tp.steps = append(tp.steps, step)
	return nil
}

// Execute runs the pipeline with the given initial input.
func (tp *ToolPipeline) Execute(ctx context.Context, initialInput map[string]interface{}) (*PipelineResult, error) {
	if len(tp.steps) == 0 {
		return nil, errors.New(errors.InvalidInput, "pipeline has no steps")
	}

	start := time.Now()
	result := &PipelineResult{
		Results:      make([]core.ToolResult, 0, len(tp.steps)),
		StepMetadata: make(map[string]StepMetadata),
		Cache:        make(map[string]core.ToolResult),
		Success:      true,
	}

	// Set up pipeline timeout
	var cancel context.CancelFunc
	if tp.options.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, tp.options.Timeout)
		defer cancel()
	}

	currentInput := initialInput

	// Execute steps sequentially or in parallel based on options
	if tp.options.Parallel {
		return tp.executeParallel(ctx, currentInput, result, start)
	}

	return tp.executeSequential(ctx, currentInput, result, start)
}

// executeSequential executes pipeline steps one after another.
func (tp *ToolPipeline) executeSequential(ctx context.Context, initialInput map[string]interface{}, result *PipelineResult, start time.Time) (*PipelineResult, error) {
	currentInput := initialInput

	for i, step := range tp.steps {
		stepStart := time.Now()
		stepID := fmt.Sprintf("step_%d_%s", i, step.ToolName)

		// Check if we should execute this step based on conditions
		if !tp.evaluateConditions(step.Conditions, result.Results) {
			result.StepMetadata[stepID] = StepMetadata{
				ToolName: step.ToolName,
				Duration: time.Since(stepStart),
				Success:  true,
				Cached:   false,
			}
			continue
		}

		// Check cache first
		var stepResult core.ToolResult
		var fromCache bool
		if tp.options.CacheResults {
			if cached, found := tp.getCachedResult(step.ToolName, currentInput); found {
				stepResult = cached
				fromCache = true
			}
		}

		if !fromCache {
			// Execute the step
			var err error
			stepResult, err = tp.executeStep(ctx, step, currentInput)
			if err != nil {
				result.Success = false
				result.FailedStep = step.ToolName
				result.Error = err
				result.Duration = time.Since(start)

				if tp.options.FailureStrategy == FailFast {
					return result, err
				}
				// For ContinueOnError, we still record the failure but continue
			}

			// Cache the result if caching is enabled
			if tp.options.CacheResults && err == nil {
				tp.setCachedResult(step.ToolName, currentInput, stepResult)
			}
		}

		// Record step metadata
		result.StepMetadata[stepID] = StepMetadata{
			ToolName:    step.ToolName,
			Duration:    time.Since(stepStart),
			Success:     result.Error == nil,
			Cached:      fromCache,
			Transformed: step.Transformer != nil,
		}

		result.Results = append(result.Results, stepResult)

		// Transform output for next step if transformer is provided
		if step.Transformer != nil {
			transformed, err := step.Transformer(stepResult.Data)
			if err != nil {
				return result, errors.WithFields(
					errors.New(errors.Unknown, "data transformation failed"),
					errors.Fields{"step": step.ToolName, "error": err.Error()},
				)
			}
			currentInput = transformed
		} else {
			// Default transformation: use the result data as next input
			if resultMap, ok := stepResult.Data.(map[string]interface{}); ok {
				currentInput = resultMap
			} else {
				// Wrap non-map results
				currentInput = map[string]interface{}{
					"result": stepResult.Data,
				}
			}
		}
	}

	result.Duration = time.Since(start)
	return result, nil
}

// executeParallel executes independent steps in parallel.
func (tp *ToolPipeline) executeParallel(ctx context.Context, initialInput map[string]interface{}, result *PipelineResult, start time.Time) (*PipelineResult, error) {
	// For now, implement a simple parallel execution for independent steps
	// In a more sophisticated implementation, we would analyze dependencies

	type stepResult struct {
		index    int
		result   core.ToolResult
		error    error
		stepID   string
		metadata StepMetadata
	}

	resultChan := make(chan stepResult, len(tp.steps))
	var wg sync.WaitGroup

	// Execute all steps in parallel (assuming they're independent for now)
	for i, step := range tp.steps {
		wg.Add(1)
		go func(idx int, s PipelineStep) {
			defer func() {
				wg.Done()
				// Ensure goroutine cleanup on panic
				if r := recover(); r != nil {
					// Send panic as error result
					stepID := fmt.Sprintf("step_%d_%s", idx, s.ToolName)
					resultChan <- stepResult{
						index:  idx,
						result: core.ToolResult{},
						error:  fmt.Errorf("pipeline step %s panicked: %v", stepID, r),
						stepID: stepID,
						metadata: StepMetadata{
							ToolName: s.ToolName,
							Success:  false,
						},
					}
				}
			}()

			// Check if context is already cancelled before execution
			select {
			case <-ctx.Done():
				stepID := fmt.Sprintf("step_%d_%s", idx, s.ToolName)
				resultChan <- stepResult{
					index:  idx,
					result: core.ToolResult{},
					error:  ctx.Err(),
					stepID: stepID,
					metadata: StepMetadata{
						ToolName: s.ToolName,
						Success:  false,
					},
				}
				return
			default:
			}

			stepStart := time.Now()
			stepID := fmt.Sprintf("step_%d_%s", idx, s.ToolName)

			stepRes, err := tp.executeStep(ctx, s, initialInput)

			// Send result without blocking if context is cancelled
			select {
			case resultChan <- stepResult{
				index:  idx,
				result: stepRes,
				error:  err,
				stepID: stepID,
				metadata: StepMetadata{
					ToolName: s.ToolName,
					Duration: time.Since(stepStart),
					Success:  err == nil,
				},
			}:
			case <-ctx.Done():
				return
			}
		}(i, step)
	}

	// Wait for all steps to complete
	go func() {
		defer func() {
			close(resultChan)
			// Ensure goroutine cleanup on panic
			if r := recover(); r != nil {
				// Log panic but continue
				_ = r
			}
		}()

		wg.Wait()
	}()

	// Collect results
	results := make([]core.ToolResult, len(tp.steps))
	hasError := false
	var firstError error

	for stepRes := range resultChan {
		results[stepRes.index] = stepRes.result
		result.StepMetadata[stepRes.stepID] = stepRes.metadata

		if stepRes.error != nil {
			hasError = true
			if firstError == nil {
				firstError = stepRes.error
				result.FailedStep = tp.steps[stepRes.index].ToolName
			}
		}
	}

	result.Results = results
	result.Success = !hasError
	result.Error = firstError
	result.Duration = time.Since(start)

	return result, firstError
}

// executeStep executes a single pipeline step with retries.
func (tp *ToolPipeline) executeStep(ctx context.Context, step PipelineStep, input map[string]interface{}) (core.ToolResult, error) {
	tool, err := tp.registry.Get(step.ToolName)
	if err != nil {
		return core.ToolResult{}, err
	}

	// Set up step timeout
	stepCtx := ctx
	if step.Timeout > 0 {
		var cancel context.CancelFunc
		stepCtx, cancel = context.WithTimeout(ctx, step.Timeout)
		defer cancel()
	}

	// Execute with retries
	var lastError error
	for attempt := 0; attempt <= step.Retries; attempt++ {
		result, err := tool.Execute(stepCtx, input)
		if err == nil {
			return result, nil
		}
		lastError = err

		// Don't retry on context cancellation or timeout
		if stepCtx.Err() != nil {
			break
		}

		// Wait before retry (exponential backoff)
		if attempt < step.Retries {
			backoff := time.Duration(attempt+1) * 100 * time.Millisecond
			select {
			case <-time.After(backoff):
			case <-stepCtx.Done():
				return core.ToolResult{}, stepCtx.Err()
			}
		}
	}

	return core.ToolResult{}, lastError
}

// evaluateConditions checks if all conditions are met for step execution.
func (tp *ToolPipeline) evaluateConditions(conditions []Condition, previousResults []core.ToolResult) bool {
	if len(conditions) == 0 {
		return true // No conditions means always execute
	}

	if len(previousResults) == 0 {
		return false // Can't evaluate conditions without previous results
	}

	// Use the last result for condition evaluation
	lastResult := previousResults[len(previousResults)-1]
	resultData, ok := lastResult.Data.(map[string]interface{})
	if !ok {
		return false // Can't evaluate conditions on non-map data
	}

	for _, condition := range conditions {
		if !tp.evaluateCondition(condition, resultData) {
			return false
		}
	}

	return true
}

// evaluateCondition evaluates a single condition.
func (tp *ToolPipeline) evaluateCondition(condition Condition, data map[string]interface{}) bool {
	value, exists := data[condition.Field]

	switch condition.Operator {
	case "exists":
		return exists
	case "not_exists":
		return !exists
	case "eq":
		return exists && value == condition.Value
	case "ne":
		return exists && value != condition.Value
	case "contains":
		if str, ok := value.(string); ok {
			if search, ok := condition.Value.(string); ok {
				return contains(str, search)
			}
		}
		return false
	default:
		return false
	}
}

// Cache management methods

// getCachedResult retrieves a cached result if available.
func (tp *ToolPipeline) getCachedResult(toolName string, input map[string]interface{}) (core.ToolResult, bool) {
	tp.mu.RLock()
	defer tp.mu.RUnlock()

	key := tp.generateCacheKey(toolName, input)
	result, found := tp.cache[key]
	return result, found
}

// setCachedResult stores a result in the cache.
func (tp *ToolPipeline) setCachedResult(toolName string, input map[string]interface{}, result core.ToolResult) {
	tp.mu.Lock()
	defer tp.mu.Unlock()

	key := tp.generateCacheKey(toolName, input)
	tp.cache[key] = result
}

// generateCacheKey creates a cache key for the given tool and input.
func (tp *ToolPipeline) generateCacheKey(toolName string, input map[string]interface{}) string {
	// Simple key generation - in production, you might want a more sophisticated approach
	return fmt.Sprintf("%s:%v", toolName, input)
}

// GetName returns the pipeline name.
func (tp *ToolPipeline) GetName() string {
	return tp.name
}

// GetSteps returns a copy of the pipeline steps.
func (tp *ToolPipeline) GetSteps() []PipelineStep {
	steps := make([]PipelineStep, len(tp.steps))
	copy(steps, tp.steps)
	return steps
}

// ClearCache clears the pipeline cache.
func (tp *ToolPipeline) ClearCache() {
	tp.mu.Lock()
	defer tp.mu.Unlock()
	tp.cache = make(map[string]core.ToolResult)
}

// Helper function for string contains check.
func contains(s, substr string) bool { return strings.Contains(s, substr) }
