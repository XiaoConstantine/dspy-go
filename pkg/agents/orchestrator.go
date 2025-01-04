package agents

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// TaskParser defines how to parse tasks from analyzer output.
type TaskParser interface {
	// Parse converts analyzer output into a slice of tasks
	Parse(analyzerOutput map[string]interface{}) ([]Task, error)
}

// PlanCreator defines how to create an execution plan from tasks.
type PlanCreator interface {
	// CreatePlan organizes tasks into execution phases
	CreatePlan(tasks []Task) ([][]Task, error)
}

// DefaultTaskParser provides a simple implementation for testing.
type DefaultTaskParser struct{}

func (p *DefaultTaskParser) Parse(analyzerOutput map[string]interface{}) ([]Task, error) {
	return nil, fmt.Errorf("default parser is a placeholder - please provide a custom implementation")
}

// DefaultPlanCreator provides a simple implementation for testing.
type DefaultPlanCreator struct{}

func (p *DefaultPlanCreator) CreatePlan(tasks []Task) ([][]Task, error) {
	return nil, fmt.Errorf("default planner is a placeholder - please provide a custom implementation")
}

// TaskProcessor defines how to process individual tasks.
type TaskProcessor interface {
	// Process handles a single task execution
	Process(ctx context.Context, task Task, context map[string]interface{}) (interface{}, error)
}

// Task represents a unit of work identified by the orchestrator.
type Task struct {
	// ID uniquely identifies the task
	ID string

	// Type indicates the kind of task
	Type string

	// Metadata holds task-specific information
	Metadata map[string]interface{}

	// Dependencies lists task IDs that must complete before this task
	Dependencies []string

	// Priority indicates task importance (lower number = higher priority)
	Priority int

	// ProcessorType indicates which processor should handle this task
	ProcessorType string
}

// OrchestrationConfig allows customizing orchestrator behavior.
type OrchestrationConfig struct {
	// MaxConcurrent controls maximum parallel task execution
	MaxConcurrent int

	// DefaultTimeout for task execution
	DefaultTimeout time.Duration

	// RetryConfig specifies retry behavior for failed tasks
	RetryConfig *RetryConfig

	// CustomProcessors maps processor types to implementations
	CustomProcessors map[string]TaskProcessor
	TaskParser       TaskParser
	PlanCreator      PlanCreator
}

// RetryConfig specifies retry behavior.
type RetryConfig struct {
	MaxAttempts       int
	BackoffMultiplier float64
}

// OrchestratorResult contains orchestration outputs.
type OrchestratorResult struct {
	// CompletedTasks holds results from successful tasks
	CompletedTasks map[string]interface{}

	// FailedTasks contains tasks that could not be completed
	FailedTasks map[string]error

	// Analysis contains orchestrator's task breakdown reasoning
	Analysis string

	// Metadata holds additional orchestration information
	Metadata map[string]interface{}
}

// FlexibleOrchestrator coordinates intelligent task decomposition and execution.
type FlexibleOrchestrator struct {
	memory     Memory
	config     OrchestrationConfig
	analyzer   *modules.Predict
	processors map[string]TaskProcessor
	parser     TaskParser
	planner    PlanCreator
	mu         sync.RWMutex
}

// NewFlexibleOrchestrator creates a new orchestrator instance.
func NewFlexibleOrchestrator(memory Memory, config OrchestrationConfig) *FlexibleOrchestrator {

	if config.TaskParser == nil {
		config.TaskParser = &DefaultTaskParser{}
	}
	if config.PlanCreator == nil {
		config.PlanCreator = &DefaultPlanCreator{}
	}
	// Create analyzer with a flexible prompt that can adapt to different domains
	analyzerSig := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "task"}},
			{Field: core.Field{Name: "context"}},
		},
		[]core.OutputField{
			{Field: core.NewField("analysis")},
			{Field: core.NewField("tasks")},
		},
	).WithInstruction(`Analyze the given task and break it down into well-defined subtasks.
        Consider:
        - Task dependencies and optimal execution order
        - Opportunities for parallel execution
        - Required processor types for each task
        - Task priorities and resource requirements`)

	orchestrator := &FlexibleOrchestrator{
		memory:     memory,
		config:     config,
		analyzer:   modules.NewPredict(analyzerSig),
		processors: make(map[string]TaskProcessor),
	}

	// Register custom processors
	for procType, processor := range config.CustomProcessors {
		orchestrator.RegisterProcessor(procType, processor)
	}

	return orchestrator
}

// RegisterProcessor adds a new task processor.
func (f *FlexibleOrchestrator) RegisterProcessor(processorType string, processor TaskProcessor) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.processors[processorType] = processor
}

// getProcessor returns the registered processor for a task type.
func (f *FlexibleOrchestrator) getProcessor(processorType string) (TaskProcessor, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	processor, exists := f.processors[processorType]
	if !exists {
		return nil, fmt.Errorf("no processor registered for type: %s", processorType)
	}
	return processor, nil
}

// Process handles complete orchestration workflow.
func (f *FlexibleOrchestrator) Process(ctx context.Context, task string, context map[string]interface{}) (*OrchestratorResult, error) {
	ctx, span := core.StartSpan(ctx, "FlexibleOrchestrator.Process")
	defer core.EndSpan(ctx)

	// Analyze and break down the task
	tasks, analysis, err := f.analyzeTasks(ctx, task, context)
	if err != nil {
		span.WithError(err)
		return nil, fmt.Errorf("task analysis failed: %w", err)
	}

	// Create execution plan based on dependencies
	plan, err := f.createExecutionPlan(tasks)
	if err != nil {
		span.WithError(err)
		return nil, fmt.Errorf("Plan failed: %w", err)

	}

	// Execute tasks according to plan
	result := &OrchestratorResult{
		CompletedTasks: make(map[string]interface{}),
		FailedTasks:    make(map[string]error),
		Analysis:       analysis,
		Metadata:       make(map[string]interface{}),
	}

	// Execute tasks with controlled concurrency
	if err := f.executePlan(ctx, plan, context, result); err != nil {
		result.Metadata["error"] = err.Error()
	}

	return result, nil
}

// analyzeTasks breaks down the high-level task into subtasks.
func (f *FlexibleOrchestrator) analyzeTasks(ctx context.Context, task string, context map[string]interface{}) ([]Task, string, error) {
	ctx, span := core.StartSpan(ctx, "AnalyzeTasks")
	defer core.EndSpan(ctx)

	// Get task breakdown from analyzer
	result, err := f.analyzer.Process(ctx, map[string]interface{}{
		"task":    task,
		"context": context,
	})
	if err != nil {
		span.WithError(err)
		return nil, "", err
	}
	tasks, err := f.parser.Parse(result)
	if err != nil {
		span.WithError(err)
		return nil, "", fmt.Errorf("task parsing failed: %w", err)
	}

	analysis, ok := result["analysis"].(string)
	if !ok {
		return nil, "", fmt.Errorf("invalid analysis format in analyzer output")
	}
	return tasks, analysis, nil
}

// createExecutionPlan organizes tasks based on dependencies.
func (f *FlexibleOrchestrator) createExecutionPlan(tasks []Task) ([][]Task, error) {
	return f.planner.CreatePlan(tasks)
}

// executePlan runs tasks according to the execution plan.
func (f *FlexibleOrchestrator) executePlan(ctx context.Context, plan [][]Task, context map[string]interface{}, result *OrchestratorResult) error {
	for phaseIdx, phase := range plan {
		phaseCtx, _ := core.StartSpan(ctx, fmt.Sprintf("Phase_%d", phaseIdx))

		// Execute tasks in this phase concurrently
		var wg sync.WaitGroup
		errors := make(chan error, len(phase))

		// Create semaphore for concurrency control
		sem := make(chan struct{}, f.config.MaxConcurrent)

		for _, task := range phase {
			wg.Add(1)
			go func(t Task) {
				defer wg.Done()

				// Acquire semaphore
				sem <- struct{}{}
				defer func() { <-sem }()

				if err := f.executeTask(phaseCtx, t, context, result); err != nil {
					errors <- fmt.Errorf("task %s failed: %w", t.ID, err)
				}
			}(task)
		}

		wg.Wait()
		close(errors)

		defer core.EndSpan(phaseCtx)

		// Check for phase errors
		if len(errors) > 0 {
			var errs []error
			for err := range errors {
				errs = append(errs, err)
			}
			return fmt.Errorf("phase %d had %d errors: %v", phaseIdx, len(errs), errs)
		}
	}

	return nil
}

// executeTask handles single task execution with retries.
func (f *FlexibleOrchestrator) executeTask(ctx context.Context, task Task, context map[string]interface{}, result *OrchestratorResult) error {
	processor, err := f.getProcessor(task.ProcessorType)
	if err != nil {
		result.FailedTasks[task.ID] = err
		return err
	}

	// Apply retry logic if configured
	var lastErr error
	attempts := 1
	if f.config.RetryConfig != nil {
		attempts = f.config.RetryConfig.MaxAttempts
	}

	for i := 0; i < attempts; i++ {
		taskResult, err := processor.Process(ctx, task, context)
		if err == nil {
			result.CompletedTasks[task.ID] = taskResult
			return nil
		}
		lastErr = err

		if i < attempts-1 {
			// Apply backoff before retry
			backoff := time.Duration(float64(time.Second) *
				math.Pow(f.config.RetryConfig.BackoffMultiplier, float64(i)))
			time.Sleep(backoff)
		}
	}

	result.FailedTasks[task.ID] = lastErr
	return lastErr
}
