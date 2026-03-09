package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// ParallelExecutor manages parallel execution of tools with advanced scheduling.
type ParallelExecutor struct {
	registry   core.ToolRegistry
	maxWorkers int
	workerPool chan struct{} // Semaphore for worker management
	metrics    *ExecutorMetrics
}

// ExecutorMetrics tracks parallel execution statistics.
type ExecutorMetrics struct {
	TotalExecutions   int64         `json:"total_executions"`
	ParallelTasks     int64         `json:"parallel_tasks"`
	AverageWaitTime   time.Duration `json:"average_wait_time"`
	AverageExecTime   time.Duration `json:"average_exec_time"`
	WorkerUtilization float64       `json:"worker_utilization"`
	mu                sync.RWMutex
}

// ParallelTask represents a task to be executed in parallel.
type ParallelTask struct {
	ID         string                 // Unique task identifier
	ToolName   string                 // Tool to execute
	Input      map[string]interface{} // Input parameters
	Priority   int                    // Task priority (higher = more urgent)
	Timeout    time.Duration          // Task-specific timeout
	Retries    int                    // Number of retries on failure
	Context    context.Context        // Task-specific context
	ResultChan chan ParallelResult    // Channel to receive results
	SubmitTime time.Time              // When the task was submitted
}

// ParallelResult contains the result of parallel task execution.
type ParallelResult struct {
	TaskID   string          // Task identifier
	Result   core.ToolResult // Tool execution result
	Error    error           // Execution error
	Duration time.Duration   // Execution duration
	WaitTime time.Duration   // Time spent waiting for execution
	WorkerID int             // ID of worker that executed the task
	Retries  int             // Number of retries performed
}

// TaskScheduler defines the interface for task scheduling algorithms.
type TaskScheduler interface {
	Schedule(tasks []*ParallelTask) []*ParallelTask
	Name() string
}

// PriorityScheduler schedules tasks based on priority and submit time.
type PriorityScheduler struct{}

func (ps *PriorityScheduler) Name() string {
	return "priority"
}

func (ps *PriorityScheduler) Schedule(tasks []*ParallelTask) []*ParallelTask {
	// Create a copy to avoid modifying the original slice
	scheduled := make([]*ParallelTask, len(tasks))
	copy(scheduled, tasks)

	// Sort by priority (descending) then by submit time (ascending)
	for i := 0; i < len(scheduled)-1; i++ {
		for j := i + 1; j < len(scheduled); j++ {
			if scheduled[i].Priority < scheduled[j].Priority ||
				(scheduled[i].Priority == scheduled[j].Priority &&
					scheduled[i].SubmitTime.After(scheduled[j].SubmitTime)) {
				scheduled[i], scheduled[j] = scheduled[j], scheduled[i]
			}
		}
	}

	return scheduled
}

// FairShareScheduler provides fair sharing among different tools.
type FairShareScheduler struct {
	toolCounts map[string]int
	mu         sync.Mutex
}

func NewFairShareScheduler() *FairShareScheduler {
	return &FairShareScheduler{
		toolCounts: make(map[string]int),
	}
}

func (fs *FairShareScheduler) Name() string {
	return "fair-share"
}

func (fs *FairShareScheduler) Schedule(tasks []*ParallelTask) []*ParallelTask {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	// Create a copy
	scheduled := make([]*ParallelTask, len(tasks))
	copy(scheduled, tasks)

	// Sort by tool execution count (ascending) then priority (descending)
	for i := 0; i < len(scheduled)-1; i++ {
		for j := i + 1; j < len(scheduled); j++ {
			count1 := fs.toolCounts[scheduled[i].ToolName]
			count2 := fs.toolCounts[scheduled[j].ToolName]

			if count1 > count2 ||
				(count1 == count2 && scheduled[i].Priority < scheduled[j].Priority) {
				scheduled[i], scheduled[j] = scheduled[j], scheduled[i]
			}
		}
	}

	return scheduled
}

func (fs *FairShareScheduler) recordExecution(toolName string) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	fs.toolCounts[toolName]++
}

// NewParallelExecutor creates a new parallel executor.
func NewParallelExecutor(registry core.ToolRegistry, maxWorkers int) *ParallelExecutor {
	if maxWorkers <= 0 {
		maxWorkers = runtime.NumCPU() * 2 // Default to 2x CPU cores
	}

	return &ParallelExecutor{
		registry:   registry,
		maxWorkers: maxWorkers,
		workerPool: make(chan struct{}, maxWorkers),
		metrics:    &ExecutorMetrics{},
	}
}

// ExecuteParallel executes multiple tasks in parallel with advanced scheduling.
func (pe *ParallelExecutor) ExecuteParallel(ctx context.Context, tasks []*ParallelTask, scheduler TaskScheduler) ([]ParallelResult, error) {
	if len(tasks) == 0 {
		return []ParallelResult{}, nil
	}

	// Schedule tasks
	if scheduler == nil {
		scheduler = &PriorityScheduler{}
	}
	scheduledTasks := scheduler.Schedule(tasks)

	// Create result collection
	results := make([]ParallelResult, len(tasks))
	var wg sync.WaitGroup
	var mu sync.Mutex

	resultMap := make(map[string]int) // taskID -> result index
	for i, task := range tasks {
		resultMap[task.ID] = i
	}

	// Execute tasks
	for _, task := range scheduledTasks {
		wg.Add(1)
		go func(t *ParallelTask) {
			defer func() {
				wg.Done()
				// Ensure goroutine cleanup on panic
				if r := recover(); r != nil {
					// Create an error result for the panic
					result := ParallelResult{
						TaskID: t.ID,
						Error:  fmt.Errorf("task %s panicked: %v", t.ID, r),
					}

					// Store panic result
					mu.Lock()
					if idx, exists := resultMap[result.TaskID]; exists {
						results[idx] = result
					}
					mu.Unlock()
				}
			}()

			// Check if context is already cancelled before execution
			select {
			case <-ctx.Done():
				result := ParallelResult{
					TaskID: t.ID,
					Error:  ctx.Err(),
				}
				mu.Lock()
				if idx, exists := resultMap[result.TaskID]; exists {
					results[idx] = result
				}
				mu.Unlock()
				return
			default:
			}

			// Execute with worker pool
			result := pe.executeWithWorkerPool(ctx, t)

			// Store result
			mu.Lock()
			if idx, exists := resultMap[result.TaskID]; exists {
				results[idx] = result
			}
			mu.Unlock()

			// Update metrics
			pe.updateMetrics(result)

			// Update scheduler metrics if applicable
			if fairShare, ok := scheduler.(*FairShareScheduler); ok {
				fairShare.recordExecution(t.ToolName)
			}
		}(task)
	}

	// Wait for completion or context cancellation
	done := make(chan struct{})
	go func() {
		defer func() {
			// Ensure goroutine cleanup on panic
			if r := recover(); r != nil {
				// Log panic but continue
				_ = r
			}
		}()

		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return results, nil
	case <-ctx.Done():
		return results, ctx.Err()
	}
}

// executeWithWorkerPool executes a task using the worker pool.
func (pe *ParallelExecutor) executeWithWorkerPool(ctx context.Context, task *ParallelTask) ParallelResult {
	waitStart := time.Now()

	// Acquire worker
	select {
	case pe.workerPool <- struct{}{}:
		// Got a worker
	case <-ctx.Done():
		return ParallelResult{
			TaskID:   task.ID,
			Error:    ctx.Err(),
			WaitTime: time.Since(waitStart),
		}
	case <-task.Context.Done():
		return ParallelResult{
			TaskID:   task.ID,
			Error:    task.Context.Err(),
			WaitTime: time.Since(waitStart),
		}
	}

	defer func() { <-pe.workerPool }() // Release worker

	waitTime := time.Since(waitStart)
	execStart := time.Now()

	// Get worker ID (approximate)
	workerID := len(pe.workerPool)

	// Execute the task with retries
	var result core.ToolResult
	var err error
	retries := 0

	for attempt := 0; attempt <= task.Retries; attempt++ {
		// Set up task timeout
		taskCtx := task.Context
		if task.Timeout > 0 {
			var cancel context.CancelFunc
			taskCtx, cancel = context.WithTimeout(task.Context, task.Timeout)
			defer cancel()
		}

		// Execute tool
		result, err = pe.executeTool(taskCtx, task.ToolName, task.Input)
		if err == nil {
			break // Success
		}

		retries = attempt

		// Don't retry on context cancellation
		if taskCtx.Err() != nil {
			break
		}

		// Wait before retry with exponential backoff
		if attempt < task.Retries {
			backoff := time.Duration(attempt+1) * 100 * time.Millisecond
			select {
			case <-time.After(backoff):
			case <-taskCtx.Done():
				err = taskCtx.Err()
				break
			}
		}
	}

	return ParallelResult{
		TaskID:   task.ID,
		Result:   result,
		Error:    err,
		Duration: time.Since(execStart),
		WaitTime: waitTime,
		WorkerID: workerID,
		Retries:  retries,
	}
}

// executeTool executes a single tool.
func (pe *ParallelExecutor) executeTool(ctx context.Context, toolName string, input map[string]interface{}) (core.ToolResult, error) {
	tool, err := pe.registry.Get(toolName)
	if err != nil {
		return core.ToolResult{}, err
	}

	return tool.Execute(ctx, input)
}

// updateMetrics updates executor metrics.
func (pe *ParallelExecutor) updateMetrics(result ParallelResult) {
	pe.metrics.mu.Lock()
	defer pe.metrics.mu.Unlock()

	pe.metrics.TotalExecutions++
	pe.metrics.ParallelTasks++

	// Update average wait time
	oldAvgWait := pe.metrics.AverageWaitTime
	pe.metrics.AverageWaitTime = time.Duration(
		(int64(oldAvgWait)*(pe.metrics.TotalExecutions-1) + int64(result.WaitTime)) / pe.metrics.TotalExecutions,
	)

	// Update average execution time
	oldAvgExec := pe.metrics.AverageExecTime
	pe.metrics.AverageExecTime = time.Duration(
		(int64(oldAvgExec)*(pe.metrics.TotalExecutions-1) + int64(result.Duration)) / pe.metrics.TotalExecutions,
	)

	// Calculate worker utilization (approximate)
	activeWorkers := len(pe.workerPool)
	pe.metrics.WorkerUtilization = float64(activeWorkers) / float64(pe.maxWorkers)
}

// GetMetrics returns a copy of the current metrics.
func (pe *ParallelExecutor) GetMetrics() ExecutorMetrics {
	pe.metrics.mu.RLock()
	defer pe.metrics.mu.RUnlock()

	return ExecutorMetrics{
		TotalExecutions:   pe.metrics.TotalExecutions,
		ParallelTasks:     pe.metrics.ParallelTasks,
		AverageWaitTime:   pe.metrics.AverageWaitTime,
		AverageExecTime:   pe.metrics.AverageExecTime,
		WorkerUtilization: pe.metrics.WorkerUtilization,
	}
}

// GetWorkerPoolStatus returns current worker pool status.
func (pe *ParallelExecutor) GetWorkerPoolStatus() (total, active, available int) {
	total = pe.maxWorkers
	active = len(pe.workerPool) // Number of workers currently in use
	available = total - active  // Number of available workers
	return
}

// BatchExecutor provides convenient batch execution with automatic task creation.
type BatchExecutor struct {
	executor  *ParallelExecutor
	scheduler TaskScheduler
}

// NewBatchExecutor creates a new batch executor.
func NewBatchExecutor(executor *ParallelExecutor, scheduler TaskScheduler) *BatchExecutor {
	if scheduler == nil {
		scheduler = &PriorityScheduler{}
	}

	return &BatchExecutor{
		executor:  executor,
		scheduler: scheduler,
	}
}

// ExecuteBatch executes a batch of tool calls in parallel.
func (be *BatchExecutor) ExecuteBatch(ctx context.Context, calls []ToolCall) ([]ParallelResult, error) {
	tasks := make([]*ParallelTask, len(calls))

	for i := range calls {
		call := &calls[i]
		toolName := call.effectiveName()
		toolArgs := call.effectiveArguments()

		tasks[i] = &ParallelTask{
			ID:         fmt.Sprintf("batch_%d_%s", i, toolName),
			ToolName:   toolName,
			Input:      toolArgs,
			Priority:   call.Priority,
			Timeout:    call.Timeout,
			Retries:    call.Retries,
			Context:    ctx,
			SubmitTime: time.Now(),
		}
	}

	return be.executor.ExecuteParallel(ctx, tasks, be.scheduler)
}

// ToolCall represents a tool call for batch execution.
// Embeds core.ToolCall for the common LLM-originated fields (ID, Name, Arguments)
// and adds scheduling metadata for parallel execution.
type ToolCall struct {
	core.ToolCall

	// Deprecated: kept for backward compatibility with existing callers.
	ToolName string `json:"tool_name,omitempty"`
	// Deprecated: kept for backward compatibility with existing callers.
	Input map[string]interface{} `json:"input,omitempty"`

	Priority int           `json:"priority"`
	Timeout  time.Duration `json:"timeout"`
	Retries  int           `json:"retries"`
}

func (c *ToolCall) effectiveName() string {
	if c.Name != "" {
		return c.Name
	}
	return c.ToolName
}

func (c *ToolCall) effectiveArguments() map[string]interface{} {
	if c.Arguments != nil {
		return c.Arguments
	}
	return c.Input
}

type toolCallJSON struct {
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
	ToolName  string                 `json:"tool_name,omitempty"`
	Input     map[string]interface{} `json:"input,omitempty"`
	Priority  int                    `json:"priority"`
	Timeout   time.Duration          `json:"timeout"`
	Retries   int                    `json:"retries"`
}

// MarshalJSON emits only the canonical core.ToolCall field names while still
// allowing deprecated fields to be used as inputs to the wrapper.
func (c ToolCall) MarshalJSON() ([]byte, error) {
	return json.Marshal(toolCallJSON{
		ID:        c.ID,
		Name:      c.effectiveName(),
		Arguments: c.effectiveArguments(),
		Priority:  c.Priority,
		Timeout:   c.Timeout,
		Retries:   c.Retries,
	})
}

// UnmarshalJSON accepts both canonical and deprecated field names. When both
// are present, the embedded core.ToolCall fields take precedence.
func (c *ToolCall) UnmarshalJSON(data []byte) error {
	var decoded toolCallJSON
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	c.ToolCall = core.ToolCall{
		ID:        decoded.ID,
		Name:      decoded.Name,
		Arguments: decoded.Arguments,
	}
	c.ToolName = decoded.ToolName
	c.Input = decoded.Input
	c.Priority = decoded.Priority
	c.Timeout = decoded.Timeout
	c.Retries = decoded.Retries

	if c.Name == "" {
		c.Name = c.ToolName
	}
	if c.Arguments == nil && c.Input != nil {
		c.Arguments = c.Input
	}

	return nil
}

// ParallelPipelineExecutor combines pipeline execution with parallel capabilities.
type ParallelPipelineExecutor struct {
	pipeline *ToolPipeline
	executor *ParallelExecutor
}

// NewParallelPipelineExecutor creates a new parallel pipeline executor.
func NewParallelPipelineExecutor(pipeline *ToolPipeline, executor *ParallelExecutor) *ParallelPipelineExecutor {
	return &ParallelPipelineExecutor{
		pipeline: pipeline,
		executor: executor,
	}
}

// ExecuteWithParallelSteps executes pipeline with parallel step execution where possible.
func (ppe *ParallelPipelineExecutor) ExecuteWithParallelSteps(ctx context.Context, input map[string]interface{}) (*PipelineResult, error) {
	// This is a simplified implementation that identifies independent steps
	// In a full implementation, you would analyze step dependencies

	steps := ppe.pipeline.GetSteps()
	if len(steps) == 0 {
		return nil, errors.New(errors.InvalidInput, "pipeline has no steps")
	}

	// For now, execute all steps in parallel (assuming independence)
	// In practice, you'd need dependency analysis

	tasks := make([]*ParallelTask, len(steps))
	for i, step := range steps {
		tasks[i] = &ParallelTask{
			ID:         fmt.Sprintf("pipeline_step_%d_%s", i, step.ToolName),
			ToolName:   step.ToolName,
			Input:      input, // Simplified - would need proper input chaining
			Priority:   1,
			Timeout:    step.Timeout,
			Retries:    step.Retries,
			Context:    ctx,
			SubmitTime: time.Now(),
		}
	}

	start := time.Now()
	results, err := ppe.executor.ExecuteParallel(ctx, tasks, &PriorityScheduler{})

	// Convert to pipeline result format
	pipelineResult := &PipelineResult{
		Results:      make([]core.ToolResult, len(results)),
		StepMetadata: make(map[string]StepMetadata),
		Duration:     time.Since(start),
		Success:      err == nil,
		Cache:        make(map[string]core.ToolResult),
	}

	for i, result := range results {
		pipelineResult.Results[i] = result.Result
		pipelineResult.StepMetadata[result.TaskID] = StepMetadata{
			ToolName: steps[i].ToolName,
			Duration: result.Duration,
			Success:  result.Error == nil,
			Retries:  result.Retries,
		}

		if result.Error != nil && pipelineResult.Error == nil {
			pipelineResult.Error = result.Error
			pipelineResult.FailedStep = steps[i].ToolName
			pipelineResult.Success = false
		}
	}

	return pipelineResult, err
}
