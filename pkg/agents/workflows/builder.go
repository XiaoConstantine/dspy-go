package workflows

import (
	"context"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// Use the existing RetryConfig from step.go

// WorkflowBuilder provides a fluent API for constructing workflows declaratively.
// It maintains backward compatibility with the existing workflow system while
// providing a more intuitive and powerful interface for workflow composition.
type WorkflowBuilder struct {
	memory    agents.Memory
	stages    []*BuilderStage
	stepIndex map[string]*BuilderStage
	errors    []error
	config    *BuilderConfig
}

// BuilderConfig holds configuration options for the workflow builder.
type BuilderConfig struct {
	EnableValidation     bool // Whether to perform validation during build
	EnableOptimization   bool // Whether to optimize the workflow graph
	EnableTracing        bool // Whether to add tracing to steps
	MaxConcurrency       int  // Maximum concurrent steps for parallel execution
	DefaultRetryAttempts int  // Default retry attempts for steps
}

// DefaultBuilderConfig returns sensible defaults for workflow builder configuration.
func DefaultBuilderConfig() *BuilderConfig {
	return &BuilderConfig{
		EnableValidation:     true,
		EnableOptimization:   true,
		EnableTracing:        true,
		MaxConcurrency:       10,
		DefaultRetryAttempts: 3,
	}
}

// BuilderStage represents a stage in the workflow builder that can contain
// multiple execution patterns (sequential, parallel, conditional).
type BuilderStage struct {
	ID          string
	Type        StageType
	Module      core.Module
	Steps       []*BuilderStep
	Condition   ConditionalFunc
	Branches    map[string]*BuilderStage // For conditional execution
	Next        []string                 // IDs of next stages
	RetryConfig *RetryConfig
	Metadata    map[string]interface{} // Additional metadata
}

// BuilderStep represents a single step within a stage.
type BuilderStep struct {
	ID          string
	Module      core.Module
	Description string
	Metadata    map[string]interface{}
}

// StageType defines the execution pattern for a stage.
type StageType int

const (
	StageTypeSequential StageType = iota
	StageTypeParallel
	StageTypeConditional
	StageTypeLoop
)

// ConditionalFunc defines the signature for conditional execution logic.
type ConditionalFunc func(ctx context.Context, state map[string]interface{}) (bool, error)

// ConditionResult represents the result of a conditional evaluation.
type ConditionResult struct {
	ShouldExecute bool
	BranchName    string
	Error         error
}

// NewBuilder creates a new WorkflowBuilder instance with the provided memory store.
// If memory is nil, an in-memory store will be used.
func NewBuilder(memory agents.Memory) *WorkflowBuilder {
	if memory == nil {
		memory = agents.NewInMemoryStore()
	}

	return &WorkflowBuilder{
		memory:    memory,
		stages:    make([]*BuilderStage, 0),
		stepIndex: make(map[string]*BuilderStage),
		errors:    make([]error, 0),
		config:    DefaultBuilderConfig(),
	}
}

// WithConfig sets custom configuration for the workflow builder.
func (wb *WorkflowBuilder) WithConfig(config *BuilderConfig) *WorkflowBuilder {
	wb.config = config
	return wb
}

// Stage adds a sequential stage to the workflow with the given ID and module.
// This is the most common workflow pattern where steps execute one after another.
func (wb *WorkflowBuilder) Stage(id string, module core.Module) *WorkflowBuilder {
	// Don't return early on errors - accumulate all errors for better UX
	if err := wb.validateStageID(id); err != nil {
		wb.addError(err)
		return wb
	}

	stage := &BuilderStage{
		ID:       id,
		Type:     StageTypeSequential,
		Module:   module,
		Steps:    make([]*BuilderStep, 0),
		Branches: make(map[string]*BuilderStage),
		Next:     make([]string, 0),
		Metadata: make(map[string]interface{}),
	}

	wb.stages = append(wb.stages, stage)
	wb.stepIndex[id] = stage

	return wb
}

// Parallel creates a parallel execution stage with multiple steps that run concurrently.
// Steps can be added using the NewStep helper function.
func (wb *WorkflowBuilder) Parallel(id string, steps ...*BuilderStep) *WorkflowBuilder {
	// Don't return early on errors - accumulate all errors for better UX
	if err := wb.validateStageID(id); err != nil {
		wb.addError(err)
		return wb
	}

	if len(steps) == 0 {
		wb.addError(fmt.Errorf("parallel stage '%s' must have at least one step", id))
		return wb
	}

	stage := &BuilderStage{
		ID:       id,
		Type:     StageTypeParallel,
		Steps:    steps,
		Branches: make(map[string]*BuilderStage),
		Next:     make([]string, 0),
		Metadata: make(map[string]interface{}),
	}

	wb.stages = append(wb.stages, stage)
	wb.stepIndex[id] = stage

	return wb
}

// Conditional creates a conditional execution stage that branches based on a condition function.
// The condition function receives the current workflow state and returns whether to execute
// the associated module or branch to alternative execution paths.
func (wb *WorkflowBuilder) Conditional(id string, condition ConditionalFunc) *ConditionalBuilder {
	// Don't return early on errors - accumulate all errors for better UX
	if err := wb.validateStageID(id); err != nil {
		wb.addError(err)
		return &ConditionalBuilder{parent: wb, hasError: true}
	}

	if condition == nil {
		wb.addError(fmt.Errorf("conditional stage '%s' must have a condition function", id))
		return &ConditionalBuilder{parent: wb, hasError: true}
	}

	stage := &BuilderStage{
		ID:        id,
		Type:      StageTypeConditional,
		Condition: condition,
		Branches:  make(map[string]*BuilderStage),
		Next:      make([]string, 0),
		Metadata:  make(map[string]interface{}),
	}

	wb.stages = append(wb.stages, stage)
	wb.stepIndex[id] = stage

	return &ConditionalBuilder{
		parent: wb,
		stage:  stage,
	}
}

// Then creates a connection between the current stage and the next stage.
// This is used to explicitly define the workflow execution order.
func (wb *WorkflowBuilder) Then(nextStageID string) *WorkflowBuilder {
	// Don't return early on errors - accumulate all errors for better UX
	if len(wb.stages) == 0 {
		wb.addError(fmt.Errorf("cannot use Then() without any stages"))
		return wb
	}

	currentStage := wb.stages[len(wb.stages)-1]
	currentStage.Next = append(currentStage.Next, nextStageID)

	return wb
}

// WithRetry adds retry configuration to the most recently added stage.
func (wb *WorkflowBuilder) WithRetry(retryConfig *RetryConfig) *WorkflowBuilder {
	// Don't return early on errors - accumulate all errors for better UX
	if len(wb.stages) == 0 {
		wb.addError(fmt.Errorf("cannot use WithRetry() without any stages"))
		return wb
	}

	currentStage := wb.stages[len(wb.stages)-1]
	currentStage.RetryConfig = retryConfig

	return wb
}

// WithMetadata adds metadata to the most recently added stage.
func (wb *WorkflowBuilder) WithMetadata(key string, value interface{}) *WorkflowBuilder {
	// Don't return early on errors - accumulate all errors for better UX
	if len(wb.stages) == 0 {
		wb.addError(fmt.Errorf("cannot use WithMetadata() without any stages"))
		return wb
	}

	currentStage := wb.stages[len(wb.stages)-1]
	currentStage.Metadata[key] = value

	return wb
}

// Build constructs the final Workflow from the builder configuration.
// It performs validation, optimization, and creates the appropriate workflow type
// based on the stages and their connections.
func (wb *WorkflowBuilder) Build() (Workflow, error) {
	if wb.hasError() {
		return nil, fmt.Errorf("builder has errors: %v", wb.errors)
	}

	if len(wb.stages) == 0 {
		return nil, fmt.Errorf("workflow must have at least one stage")
	}

	// Perform validation if enabled
	if wb.config.EnableValidation {
		if err := wb.validate(); err != nil {
			return nil, fmt.Errorf("workflow validation failed: %w", err)
		}
	}

	// Optimize workflow graph if enabled
	if wb.config.EnableOptimization {
		wb.optimize()
	}

	// Determine the appropriate workflow type based on the stage patterns
	workflowType := wb.determineWorkflowType()

	// Build the appropriate workflow
	switch workflowType {
	case "chain":
		return wb.buildChainWorkflow()
	case "parallel":
		return wb.buildParallelWorkflow()
	case "router":
		return wb.buildRouterWorkflow()
	case "composite":
		return wb.buildCompositeWorkflow()
	default:
		return nil, fmt.Errorf("unsupported workflow type: %s", workflowType)
	}
}

// ConditionalBuilder provides a fluent interface for building conditional workflow stages.
type ConditionalBuilder struct {
	parent   *WorkflowBuilder
	stage    *BuilderStage
	hasError bool
}

// If adds a conditional branch that executes the given module when the condition is true.
func (cb *ConditionalBuilder) If(module core.Module) *ConditionalBuilder {
	if cb.hasError {
		return cb
	}

	cb.stage.Branches["true"] = &BuilderStage{
		ID:       cb.stage.ID + "_true",
		Type:     StageTypeSequential,
		Module:   module,
		Steps:    make([]*BuilderStep, 0),
		Branches: make(map[string]*BuilderStage),
		Next:     make([]string, 0),
		Metadata: make(map[string]interface{}),
	}

	return cb
}

// Else adds an alternative branch that executes when the condition is false.
func (cb *ConditionalBuilder) Else(module core.Module) *ConditionalBuilder {
	if cb.hasError {
		return cb
	}

	cb.stage.Branches["false"] = &BuilderStage{
		ID:       cb.stage.ID + "_false",
		Type:     StageTypeSequential,
		Module:   module,
		Steps:    make([]*BuilderStep, 0),
		Branches: make(map[string]*BuilderStage),
		Next:     make([]string, 0),
		Metadata: make(map[string]interface{}),
	}

	return cb
}

// ElseIf adds an additional conditional branch with its own condition.
func (cb *ConditionalBuilder) ElseIf(condition ConditionalFunc, module core.Module) *ConditionalBuilder {
	if cb.hasError {
		return cb
	}

	branchID := fmt.Sprintf("elseif_%d", len(cb.stage.Branches))
	cb.stage.Branches[branchID] = &BuilderStage{
		ID:        cb.stage.ID + "_" + branchID,
		Type:      StageTypeConditional,
		Module:    module,
		Condition: condition,
		Steps:     make([]*BuilderStep, 0),
		Branches:  make(map[string]*BuilderStage),
		Next:      make([]string, 0),
		Metadata:  make(map[string]interface{}),
	}

	return cb
}

// End completes the conditional builder and returns to the main workflow builder.
func (cb *ConditionalBuilder) End() *WorkflowBuilder {
	return cb.parent
}

// NewStep creates a new BuilderStep for use in parallel stages.
func NewStep(id string, module core.Module) *BuilderStep {
	return &BuilderStep{
		ID:          id,
		Module:      module,
		Description: "",
		Metadata:    make(map[string]interface{}),
	}
}

// WithDescription adds a description to a BuilderStep.
func (bs *BuilderStep) WithDescription(description string) *BuilderStep {
	bs.Description = description
	return bs
}

// WithStepMetadata adds metadata to a BuilderStep.
func (bs *BuilderStep) WithStepMetadata(key string, value interface{}) *BuilderStep {
	bs.Metadata[key] = value
	return bs
}

// Helper methods for WorkflowBuilder

func (wb *WorkflowBuilder) validateStageID(id string) error {
	if strings.TrimSpace(id) == "" {
		return fmt.Errorf("stage ID cannot be empty")
	}

	if _, exists := wb.stepIndex[id]; exists {
		return fmt.Errorf("stage with ID '%s' already exists", id)
	}

	return nil
}

func (wb *WorkflowBuilder) addError(err error) {
	wb.errors = append(wb.errors, err)
}

func (wb *WorkflowBuilder) hasError() bool {
	return len(wb.errors) > 0
}

func (wb *WorkflowBuilder) validate() error {
	// Check for cycles in the workflow graph
	if err := wb.checkForCycles(); err != nil {
		return err
	}

	// Validate all stage configurations
	for _, stage := range wb.stages {
		if err := wb.validateStage(stage); err != nil {
			return fmt.Errorf("stage '%s': %w", stage.ID, err)
		}
	}

	// Check that all referenced next stages exist
	if err := wb.validateStageReferences(); err != nil {
		return err
	}

	return nil
}

func (wb *WorkflowBuilder) validateStage(stage *BuilderStage) error {
	switch stage.Type {
	case StageTypeSequential:
		if stage.Module == nil {
			return fmt.Errorf("sequential stage must have a module")
		}
	case StageTypeParallel:
		if len(stage.Steps) == 0 {
			return fmt.Errorf("parallel stage must have at least one step")
		}
		for _, step := range stage.Steps {
			if step.Module == nil {
				return fmt.Errorf("parallel step '%s' must have a module", step.ID)
			}
		}
	case StageTypeConditional:
		if stage.Condition == nil {
			return fmt.Errorf("conditional stage must have a condition function")
		}
		if len(stage.Branches) == 0 {
			return fmt.Errorf("conditional stage must have at least one branch")
		}
	}

	return nil
}

func (wb *WorkflowBuilder) validateStageReferences() error {
	for _, stage := range wb.stages {
		for _, nextID := range stage.Next {
			if _, exists := wb.stepIndex[nextID]; !exists {
				return fmt.Errorf("stage '%s' references non-existent next stage '%s'", stage.ID, nextID)
			}
		}
	}
	return nil
}

func (wb *WorkflowBuilder) checkForCycles() error {
	// Use DFS to detect cycles in the workflow graph
	visited := make(map[string]bool)
	recStack := make(map[string]bool)

	for _, stage := range wb.stages {
		if !visited[stage.ID] {
			if wb.hasCycleDFS(stage.ID, visited, recStack) {
				return fmt.Errorf("workflow contains a cycle involving stage '%s'", stage.ID)
			}
		}
	}

	return nil
}

func (wb *WorkflowBuilder) hasCycleDFS(stageID string, visited, recStack map[string]bool) bool {
	visited[stageID] = true
	recStack[stageID] = true

	stage := wb.stepIndex[stageID]
	if stage == nil {
		return false // Stage doesn't exist, no cycle from this path
	}
	for _, nextID := range stage.Next {
		if !visited[nextID] {
			if wb.hasCycleDFS(nextID, visited, recStack) {
				return true
			}
		} else if recStack[nextID] {
			return true
		}
	}

	recStack[stageID] = false
	return false
}

func (wb *WorkflowBuilder) optimize() {
	// TODO: Implement workflow optimization strategies
	// - Merge sequential stages where possible
	// - Optimize parallel execution order
	// - Remove redundant conditional checks
	// - Precompute static values
}

func (wb *WorkflowBuilder) determineWorkflowType() string {
	hasParallel := false
	hasConditional := false

	for _, stage := range wb.stages {
		switch stage.Type {
		case StageTypeParallel:
			hasParallel = true
		case StageTypeConditional:
			hasConditional = true
		}
	}

	// Determine the most appropriate workflow implementation
	if hasConditional {
		return "router"
	}
	if hasParallel {
		return "parallel"
	}
	if len(wb.stages) == 1 {
		return "chain"
	}
	if wb.isLinearChain() {
		return "chain"
	}

	return "composite"
}

func (wb *WorkflowBuilder) isLinearChain() bool {
	// Check if all stages form a simple linear chain
	if len(wb.stages) <= 1 {
		return true
	}

	// Each stage (except the last) should have exactly one next stage
	for i, stage := range wb.stages[:len(wb.stages)-1] {
		if len(stage.Next) != 1 {
			return false
		}
		expectedNext := wb.stages[i+1].ID
		if stage.Next[0] != expectedNext {
			return false
		}
	}

	// Last stage should have no next stages
	lastStage := wb.stages[len(wb.stages)-1]
	return len(lastStage.Next) == 0
}

func (wb *WorkflowBuilder) buildChainWorkflow() (Workflow, error) {
	workflow := NewChainWorkflow(wb.memory)

	for _, stage := range wb.stages {
		step := &Step{
			ID:          stage.ID,
			Module:      stage.Module,
			NextSteps:   stage.Next,
			RetryConfig: stage.RetryConfig,
		}

		if err := workflow.AddStep(step); err != nil {
			return nil, fmt.Errorf("failed to add step '%s': %w", stage.ID, err)
		}
	}

	return workflow, nil
}

func (wb *WorkflowBuilder) buildParallelWorkflow() (Workflow, error) {
	// For now, create a basic parallel workflow
	// TODO: Implement more sophisticated parallel workflow building
	workflow := NewParallelWorkflow(wb.memory, wb.config.MaxConcurrency)

	for _, stage := range wb.stages {
		if stage.Type == StageTypeParallel {
			for _, step := range stage.Steps {
				workflowStep := &Step{
					ID:          step.ID,
					Module:      step.Module,
					NextSteps:   []string{},
					RetryConfig: stage.RetryConfig,
				}
				if err := workflow.AddStep(workflowStep); err != nil {
					return nil, fmt.Errorf("failed to add parallel step '%s': %w", step.ID, err)
				}
			}
		} else {
			step := &Step{
				ID:          stage.ID,
				Module:      stage.Module,
				NextSteps:   stage.Next,
				RetryConfig: stage.RetryConfig,
			}
			if err := workflow.AddStep(step); err != nil {
				return nil, fmt.Errorf("failed to add step '%s': %w", stage.ID, err)
			}
		}
	}

	return workflow, nil
}

func (wb *WorkflowBuilder) buildRouterWorkflow() (Workflow, error) {
	// For now, build a basic chain workflow that can handle conditionals
	// Full router workflow implementation is planned for Task 1.2
	workflow := NewChainWorkflow(wb.memory)

	for _, stage := range wb.stages {
		// For conditional stages, just add the main module for now
		// Full conditional logic will be implemented in Task 1.2
		var module core.Module
		if stage.Type == StageTypeConditional {
			// Use the first branch's module as fallback
			for _, branch := range stage.Branches {
				if branch.Module != nil {
					module = branch.Module
					break
				}
			}
			if module == nil {
				return nil, fmt.Errorf("conditional stage '%s' has no valid modules", stage.ID)
			}
		} else {
			module = stage.Module
		}

		step := &Step{
			ID:          stage.ID,
			Module:      module,
			NextSteps:   stage.Next,
			RetryConfig: stage.RetryConfig,
		}

		if err := workflow.AddStep(step); err != nil {
			return nil, fmt.Errorf("failed to add step '%s': %w", stage.ID, err)
		}
	}

	return workflow, nil
}

func (wb *WorkflowBuilder) buildCompositeWorkflow() (Workflow, error) {
	// For complex workflows that don't fit other patterns
	// TODO: Implement composite workflow building
	return nil, fmt.Errorf("composite workflow building not yet implemented")
}