package agents

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/sourcegraph/conc/pool"
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
	Process(ctx context.Context, task Task, taskContext map[string]interface{}) (interface{}, error)
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

	AnalyzerConfig AnalyzerConfig
}

// New type to encapsulate analyzer-specific configuration.
type AnalyzerConfig struct {
	// The base instruction for task analysis
	BaseInstruction string
	// Additional formatting instructions specific to the implementation
	FormatInstructions string
	// Any extra considerations for task analysis
	Considerations []string
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
	mu       sync.RWMutex
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
	if config.AnalyzerConfig.BaseInstruction == "" {
		config.AnalyzerConfig.BaseInstruction = `Analyze the given task and break it down into well-defined subtasks.

	IMPORTANT FORMAT RULES:
	1. Start fields exactly with 'analysis:' or 'tasks:' (no markdown formatting)
	2. Provide raw XML directly after 'tasks:' without any wrapping
	3. Keep the exact field prefix format - no decorations or modifications
	4. Ensure proper indentation and structure in the XML

        Consider:
        - Task dependencies and optimal execution order
        - Opportunities for parallel execution
        - Required processor types for each task
        - Task priorities and resource requirementsAnalyze the given task and break it down into well-defined subtasks.`
	}
	instruction := config.AnalyzerConfig.BaseInstruction
	if config.AnalyzerConfig.FormatInstructions != "" {
		instruction += "\n" + config.AnalyzerConfig.FormatInstructions
	}
	if len(config.AnalyzerConfig.Considerations) > 0 {
		instruction += "\nConsider:\n"
		for _, consideration := range config.AnalyzerConfig.Considerations {
			instruction += fmt.Sprintf("- %s\n", consideration)
		}
	}

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
	).WithInstruction(instruction)

	orchestrator := &FlexibleOrchestrator{
		memory:     memory,
		config:     config,
		analyzer:   modules.NewPredict(analyzerSig),
		processors: make(map[string]TaskProcessor),
		parser:     config.TaskParser,  // Add this line
		planner:    config.PlanCreator, // And this line
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
		logging.GetLogger().Error(context.Background(), "No processor found for type: %s", processorType)
		return nil, fmt.Errorf("no processor registered for type: %s", processorType)
	}
	return processor, nil
}

// Process handles complete orchestration workflow.
func (f *FlexibleOrchestrator) Process(ctx context.Context, task string, context map[string]interface{}) (*OrchestratorResult, error) {
	ctx, span := core.StartSpan(ctx, "FlexibleOrchestrator.Process")
	defer core.EndSpan(ctx)

	logger := logging.GetLogger()

	// Add context check at the start
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Analyze and break down the task
	tasks, analysis, err := f.analyzeTasks(ctx, task, context)
	logger.Info(ctx, "tasks: %v, analysis: %s", tasks, analysis)

	if err != nil {
		logger.Error(ctx, "Task analysis failed: %v", err)
		span.WithError(err)
		return nil, fmt.Errorf("task analysis failed: %w", err)
	}
	// Check context again after analysis
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(tasks) == 0 {
		logger.Error(ctx, "No tasks were generated from analysis")
		return nil, fmt.Errorf("no tasks generated from analysis")
	}

	// Create execution plan based on dependencies
	plan, err := f.createExecutionPlan(tasks)
	if err != nil {
		span.WithError(err)
		return nil, fmt.Errorf("Plan failed: %w", err)

	}

	// Check context before execution
	if err := ctx.Err(); err != nil {
		return nil, err
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
		return result, err
	}

	return result, nil
}

// analyzeTasks breaks down the high-level task into subtasks.
func (f *FlexibleOrchestrator) analyzeTasks(ctx context.Context, task string, context map[string]interface{}) ([]Task, string, error) {
	ctx, span := core.StartSpan(ctx, "AnalyzeTasks")
	defer core.EndSpan(ctx)
	logger := logging.GetLogger()

	// Get task breakdown from analyzer
	result, err := f.analyzer.Process(ctx, map[string]interface{}{
		"task":    task,
		"context": context,
	})

	if err != nil {
		span.WithError(err)
		return nil, "", err
	}
	logger.Info(ctx, "raw task: %s, result: %s", task, result)

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

func (f *FlexibleOrchestrator) executePlan(ctx context.Context, plan [][]Task, taskContext map[string]interface{}, result *OrchestratorResult) error {
	// We'll use a top-level pool to manage overall concurrency
	masterPool := pool.New().WithContext(ctx).WithCancelOnError().WithMaxGoroutines(f.config.MaxConcurrent)

	// Process each phase using the master pool
	for phaseIdx, phase := range plan {
		// Capture phase variables for closure
		currentPhase := phase
		currentPhaseIdx := phaseIdx

		// Add phase processing to master pool
		masterPool.Go(func(ctx context.Context) error {
			phaseCtx, span := core.StartSpan(ctx, fmt.Sprintf("Phase_%d", currentPhaseIdx))
			defer core.EndSpan(phaseCtx)

			// Create a phase-specific pool
			phasePool := pool.New().WithContext(phaseCtx).WithCancelOnError()
			// Process tasks in this phase
			for _, task := range currentPhase {
				t := task // Capture task for closure

				phasePool.Go(func(tCtx context.Context) error {
					// Execute task with proper context handling
					if err := f.executeTask(phaseCtx, t, taskContext, result); err != nil {
						span.WithError(err)
						return fmt.Errorf("task %s failed: %w", t.ID, err)
					}
					return nil
				})
			}

			// Wait for all tasks in this phase and handle errors
			if err := phasePool.Wait(); err != nil {
				return fmt.Errorf("phase %d failed: %w", currentPhaseIdx, err)
			}

			return nil
		})
	}

	// Wait for all phases to complete
	return masterPool.Wait()
}

// executeTask handles single task execution with retries.
func (f *FlexibleOrchestrator) executeTask(ctx context.Context, task Task, taskContext map[string]interface{}, result *OrchestratorResult) error {
	logger := logging.GetLogger()
	ctx, span := core.StartSpan(ctx, fmt.Sprintf("Task_%s", task.ID))
	defer core.EndSpan(ctx)

	// Add structured task information to span
	span.WithAnnotation("task", map[string]interface{}{
		"id":        task.ID,
		"type":      task.Type,
		"processor": task.ProcessorType,
		"priority":  task.Priority,
	})
	processor, err := f.getProcessor(task.ProcessorType)
	if err != nil {
		logger.Error(ctx, "Failed to get processor for task %s: %v",
			task.ID, err)
		span.WithError(err)
		result.mu.Lock()
		result.FailedTasks[task.ID] = err
		result.mu.Unlock()
		return err
	}
	logger.Debug(ctx, "Starting execution of task [%s] using processor [%T]",
		task.ID, processor)
	// Apply retry logic if configured
	var lastErr error
	attempts := 1
	if f.config.RetryConfig.MaxAttempts > 1 {
		attempts = f.config.RetryConfig.MaxAttempts
	}
	logger.Debug(ctx, "Starting execution of task [%s] with max attempts [%d]", task.ID, attempts)

	for i := 0; i < attempts; i++ {
		attemptCtx, _ := core.StartSpan(ctx, fmt.Sprintf("Attempt_%d", i+1))
		logger.Debug(attemptCtx, "Processing task [%s] (attempt %d/%d) with context: %+v",
			task.ID, i+1, attempts, taskContext)
		taskResult, err := processor.Process(attemptCtx, task, taskContext)

		logger.Info(ctx, "task: %v with result: %v", task, taskResult)
		if err == nil {
			logger.Info(attemptCtx, "Task [%s] completed successfully with result: %+v",
				task.ID, taskResult)

			result.mu.Lock()
			result.CompletedTasks[task.ID] = taskResult
			result.mu.Unlock()

			core.EndSpan(attemptCtx)
			return nil
		}

		lastErr = err
		if i < attempts-1 {
			backoff := time.Duration(float64(time.Second) *
				math.Pow(f.config.RetryConfig.BackoffMultiplier, float64(i)))

			logger.Warn(attemptCtx, "Task [%s] failed attempt %d/%d: %v. Retrying in %v",
				task.ID, i+1, attempts, err, backoff)
			time.Sleep(backoff)
		}

		core.EndSpan(attemptCtx)
	}

	result.mu.Lock()
	result.FailedTasks[task.ID] = lastErr
	result.mu.Unlock()

	logger.Error(ctx, "Task [%s] failed all %d attempts", task.ID, attempts)
	return lastErr
}
