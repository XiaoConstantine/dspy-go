package agents

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
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
	return nil, errors.New(errors.InvalidInput,
		"default parser is a placeholder - please provide a custom implementation")
}

// DefaultPlanCreator provides a simple implementation for testing.
type DefaultPlanCreator struct{}

func (p *DefaultPlanCreator) CreatePlan(tasks []Task) ([][]Task, error) {
	return nil, errors.New(errors.InvalidInput,
		"default planner is a placeholder - please provide a custom implementation")
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
	Options        core.Option
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
	4. Ensure proper indentation and structure in the XML.`
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
		parser:     config.TaskParser,
		planner:    config.PlanCreator,
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
func (f *FlexibleOrchestrator) GetProcessor(processorType string) (TaskProcessor, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	processor, exists := f.processors[processorType]
	if !exists {
		logging.GetLogger().Error(context.Background(), "No processor found for type: %s", processorType)
		return nil, errors.WithFields(
			errors.New(errors.ResourceNotFound, "processor not found"),
			errors.Fields{
				"processor_type":       processorType,
				"available_processors": getProcessorTypes(f.processors),
			})
	}
	return processor, nil
}

func getProcessorTypes(processors map[string]TaskProcessor) []string {
	types := make([]string, 0, len(processors))
	for pType := range processors {
		types = append(types, pType)
	}
	return types
}

// Process handles complete orchestration workflow.
func (f *FlexibleOrchestrator) Process(ctx context.Context, task string, context map[string]interface{}) (*OrchestratorResult, error) {
	ctx, span := core.StartSpan(ctx, "FlexibleOrchestrator.Process")
	defer core.EndSpan(ctx)

	logger := logging.GetLogger()

	// Add context check at the start
	if err := ctx.Err(); err != nil {

		return nil, errors.Wrap(err, errors.Canceled, "context canceled before processing started")
	}

	// Analyze and break down the task
	tasks, analysis, err := f.analyzeTasks(ctx, task, context)
	logger.Debug(ctx, "tasks: %v, analysis: %s", tasks, analysis)

	if err != nil {
		logger.Error(ctx, "Task analysis failed: %v", err)
		span.WithError(err)
		return nil, errors.WithFields(
			errors.Wrap(err, errors.WorkflowExecutionFailed, "task analysis failed"),
			errors.Fields{
				"task_input": task,
			})
	}
	// Check context again after analysis
	if err := ctx.Err(); err != nil {

		return nil, errors.Wrap(err, errors.Canceled, "context canceled after analysis")
	}

	if len(tasks) == 0 {
		logger.Error(ctx, "No tasks were generated from analysis")
		return nil, errors.New(errors.InvalidResponse, "no tasks generated from analysis")
	}

	// Create execution plan based on dependencies
	plan, err := f.createExecutionPlan(tasks)
	if err != nil {
		span.WithError(err)
		return nil, errors.WithFields(
			errors.Wrap(err, errors.WorkflowExecutionFailed, "execution plan creation failed"),
			errors.Fields{
				"task_count": len(tasks),
			})

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
		return result, errors.WithFields(
			errors.Wrap(err, errors.WorkflowExecutionFailed, "plan execution failed"),
			errors.Fields{
				"completed_tasks": len(result.CompletedTasks),
				"failed_tasks":    len(result.FailedTasks),
			})
	}

	return result, nil
}

// analyzeTasks breaks down the high-level task into subtasks.
func (f *FlexibleOrchestrator) analyzeTasks(ctx context.Context, task string, context map[string]interface{}) ([]Task, string, error) {
	ctx, span := core.StartSpan(ctx, "AnalyzeTasks")
	defer core.EndSpan(ctx)
	logger := logging.GetLogger()
	var lastErr error
	maxAttempts := 1
	if f.config.RetryConfig != nil {
		maxAttempts = f.config.RetryConfig.MaxAttempts
	}

	for attempt := 0; attempt < maxAttempts; attempt++ {
		attemptCtx, attemptSpan := core.StartSpan(ctx, fmt.Sprintf("AnalyzeAttempt_%d", attempt))

		// Check context cancellation
		if err := checkCtxErr(attemptCtx); err != nil {
			core.EndSpan(attemptCtx)

			return nil, "", errors.Wrap(err, errors.Canceled, "context canceled during analysis")
		}
		// Get task breakdown from analyzer
		result, err := f.analyzer.Process(ctx, map[string]interface{}{
			"task":    task,
			"context": context,
		}, f.config.Options)
		if err != nil {
			lastErr = err
			attemptSpan.WithError(err)
			logger.Warn(attemptCtx, "Analysis attempt %d failed: %v", attempt+1, err)
			core.EndSpan(attemptCtx)

			// Calculate backoff duration if configured
			if f.config.RetryConfig != nil && attempt < maxAttempts-1 {
				backoff := time.Duration(float64(time.Second) *
					math.Pow(f.config.RetryConfig.BackoffMultiplier, float64(attempt)))

				logger.Debug(attemptCtx, "Retrying analysis in %v", backoff)

				select {
				case <-ctx.Done():
					return nil, "", ctx.Err()
				case <-time.After(backoff):
					continue
				}
			}
			continue
		}

		logger.Debug(ctx, "raw task: %s, result: %s", task, result)

		tasks, err := f.parser.Parse(result)
		if err != nil {
			lastErr = errors.WithFields(
				errors.Wrap(err, errors.InvalidResponse, "failed to parse analyzer output"),
				errors.Fields{
					"attempt": attempt + 1,
				})
			attemptSpan.WithError(lastErr)
			logger.Warn(attemptCtx, "Task parsing failed in attempt %d: %v", attempt+1, err)
			core.EndSpan(attemptCtx)
			continue
		}

		analysis, ok := result["analysis"].(string)
		if !ok {
			lastErr = errors.New(errors.InvalidResponse, "invalid analysis format in analyzer output")
			attemptSpan.WithError(lastErr)
			logger.Warn(attemptCtx, "Invalid analysis format in attempt %d", attempt+1)
			core.EndSpan(attemptCtx)
			continue
		}
		logger.Debug(attemptCtx, "Successfully analyzed tasks on attempt %d", attempt+1)
		attemptSpan.WithAnnotation("tasks", tasks)
		attemptSpan.WithAnnotation("analysis", analysis)
		core.EndSpan(attemptCtx)

		return tasks, analysis, nil
	}

	span.WithError(lastErr)
	return nil, "", errors.WithFields(
		errors.Wrap(lastErr, errors.WorkflowExecutionFailed, "analysis failed after all attempts"),
		errors.Fields{
			"max_attempts": maxAttempts,
		})
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
						return errors.WithFields(
							errors.Wrap(err, errors.StepExecutionFailed, "task execution failed"),
							errors.Fields{
								"task_id":   t.ID,
								"task_type": t.Type,
							})
					}
					return nil
				})
			}

			// Wait for all tasks in this phase and handle errors
			if err := phasePool.Wait(); err != nil {
				return errors.WithFields(
					errors.Wrap(err, errors.WorkflowExecutionFailed, "phase execution failed"),
					errors.Fields{
						"phase_index": currentPhaseIdx,
						"task_count":  len(currentPhase),
					})
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
	ctx, span := core.StartSpan(ctx, fmt.Sprintf("Task_%s_%s", task.ID, task.Type))
	defer core.EndSpan(ctx)

	// Add structured task information to span
	span.WithAnnotation("task", map[string]interface{}{
		"id":        task.ID,
		"type":      task.Type,
		"processor": task.ProcessorType,
		"priority":  task.Priority,
	})
	processor, err := f.GetProcessor(task.ProcessorType)
	if err != nil {
		logger.Error(ctx, "Failed to get processor for [task-%s-%s]: %v",
			task.ID, task.Type, err)
		span.WithError(err)
		result.mu.Lock()
		result.FailedTasks[task.ID] = err
		result.mu.Unlock()
		return err
	}
	logger.Debug(ctx, "Starting execution of task [%s-%s] using processor [%T]",
		task.ID, task.Type, processor)
	// Apply retry logic if configured
	var lastErr error
	attempts := 1
	if f.config.RetryConfig.MaxAttempts > 1 {
		attempts = f.config.RetryConfig.MaxAttempts
	}
	logger.Debug(ctx, "Starting execution of task [%s-%s] with max attempts [%d]", task.ID, task.Type, attempts)

	for i := 0; i < attempts; i++ {
		attemptCtx, _ := core.StartSpan(ctx, fmt.Sprintf("Attempt_%d", i+1))
		logger.Debug(attemptCtx, "Processing task [%s-%s] (attempt %d/%d) with context: %+v",
			task.ID, task.Type, i+1, attempts, taskContext)
		taskResult, err := processor.Process(attemptCtx, task, taskContext)

		logger.Debug(ctx, "task: %v with result: %v", task, taskResult)
		if err == nil {
			logger.Debug(attemptCtx, "Task [%s-%s] completed successfully with result: %+v",
				task.ID, task.Type, taskResult)

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

			logger.Warn(attemptCtx, "Task [%s-%s] failed attempt %d/%d: %v. Retrying in %v",
				task.ID, task.Type, i+1, attempts, err, backoff)
			time.Sleep(backoff)
		}

		core.EndSpan(attemptCtx)
	}

	result.mu.Lock()
	result.FailedTasks[task.ID] = lastErr
	result.mu.Unlock()

	logger.Error(ctx, "Task [%s-%s] failed all %d attempts", task.ID, task.Type, attempts)
	return lastErr
}

func checkCtxErr(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil
	}
}
