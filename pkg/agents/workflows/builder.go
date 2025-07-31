package workflows

import (
	"context"
	"fmt"
	"sort"
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
// multiple execution patterns (sequential, parallel, conditional, loops, templates).
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

	// Advanced pattern fields
	LoopCondition     LoopConditionFunc     // For while/until loops
	IteratorFunc      IteratorFunc          // For forEach loops
	TemplateParams    TemplateParameterFunc // For templates
	MaxIterations     int                   // Safety limit for loops
	LoopBody          *WorkflowBuilder      // Nested workflow for loop body
	TemplateWorkflow  *WorkflowBuilder      // Template workflow definition
	TimeoutMs         int64                 // Timeout in milliseconds
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
	StageTypeForEach
	StageTypeWhile
	StageTypeUntil
	StageTypeTemplate
)

// ConditionalFunc defines the signature for conditional execution logic.
type ConditionalFunc func(ctx context.Context, state map[string]interface{}) (bool, error)

// LoopConditionFunc defines the signature for loop conditions (while/until).
type LoopConditionFunc func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error)

// IteratorFunc defines the signature for forEach iteration logic.
type IteratorFunc func(ctx context.Context, state map[string]interface{}) ([]interface{}, error)

// TemplateParameterFunc defines the signature for template parameter resolution.
type TemplateParameterFunc func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error)

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

// WithTimeout adds a timeout to the most recently added stage.
func (wb *WorkflowBuilder) WithTimeout(timeoutMs int64) *WorkflowBuilder {
	if len(wb.stages) == 0 {
		wb.addError(fmt.Errorf("cannot use WithTimeout() without any stages"))
		return wb
	}

	currentStage := wb.stages[len(wb.stages)-1]
	currentStage.TimeoutMs = timeoutMs

	return wb
}

// ForEach creates a loop stage that iterates over a collection.
// The iteratorFunc returns the collection to iterate over.
// The body builder defines the workflow to execute for each item.
func (wb *WorkflowBuilder) ForEach(id string, iteratorFunc IteratorFunc, bodyBuilder func(*WorkflowBuilder) *WorkflowBuilder) *WorkflowBuilder {
	if err := wb.validateStageID(id); err != nil {
		wb.addError(err)
		return wb
	}

	if iteratorFunc == nil {
		wb.addError(fmt.Errorf("forEach stage '%s' must have an iterator function", id))
		return wb
	}

	// Create a new builder for the loop body
	loopBody := NewBuilder(wb.memory)
	if bodyBuilder != nil {
		loopBody = bodyBuilder(loopBody)
	}

	stage := &BuilderStage{
		ID:           id,
		Type:         StageTypeForEach,
		IteratorFunc: iteratorFunc,
		LoopBody:     loopBody,
		Branches:     make(map[string]*BuilderStage),
		Next:         make([]string, 0),
		Metadata:     make(map[string]interface{}),
		MaxIterations: 1000, // Default safety limit
	}

	wb.stages = append(wb.stages, stage)
	wb.stepIndex[id] = stage

	return wb
}

// While creates a loop stage that continues while a condition is true.
func (wb *WorkflowBuilder) While(id string, condition LoopConditionFunc, bodyBuilder func(*WorkflowBuilder) *WorkflowBuilder) *WorkflowBuilder {
	if err := wb.validateStageID(id); err != nil {
		wb.addError(err)
		return wb
	}

	if condition == nil {
		wb.addError(fmt.Errorf("while stage '%s' must have a condition function", id))
		return wb
	}

	// Create a new builder for the loop body
	loopBody := NewBuilder(wb.memory)
	if bodyBuilder != nil {
		loopBody = bodyBuilder(loopBody)
	}

	stage := &BuilderStage{
		ID:            id,
		Type:          StageTypeWhile,
		LoopCondition: condition,
		LoopBody:      loopBody,
		Branches:      make(map[string]*BuilderStage),
		Next:          make([]string, 0),
		Metadata:      make(map[string]interface{}),
		MaxIterations: 1000, // Default safety limit
	}

	wb.stages = append(wb.stages, stage)
	wb.stepIndex[id] = stage

	return wb
}

// Until creates a loop stage that continues until a condition becomes true.
func (wb *WorkflowBuilder) Until(id string, condition LoopConditionFunc, bodyBuilder func(*WorkflowBuilder) *WorkflowBuilder) *WorkflowBuilder {
	if err := wb.validateStageID(id); err != nil {
		wb.addError(err)
		return wb
	}

	if condition == nil {
		wb.addError(fmt.Errorf("until stage '%s' must have a condition function", id))
		return wb
	}

	// Create a new builder for the loop body
	loopBody := NewBuilder(wb.memory)
	if bodyBuilder != nil {
		loopBody = bodyBuilder(loopBody)
	}

	stage := &BuilderStage{
		ID:            id,
		Type:          StageTypeUntil,
		LoopCondition: condition,
		LoopBody:      loopBody,
		Branches:      make(map[string]*BuilderStage),
		Next:          make([]string, 0),
		Metadata:      make(map[string]interface{}),
		MaxIterations: 1000, // Default safety limit
	}

	wb.stages = append(wb.stages, stage)
	wb.stepIndex[id] = stage

	return wb
}

// WithMaxIterations sets the maximum number of iterations for the most recently added loop stage.
func (wb *WorkflowBuilder) WithMaxIterations(maxIterations int) *WorkflowBuilder {
	if len(wb.stages) == 0 {
		wb.addError(fmt.Errorf("cannot use WithMaxIterations() without any stages"))
		return wb
	}

	currentStage := wb.stages[len(wb.stages)-1]
	if currentStage.Type != StageTypeForEach && currentStage.Type != StageTypeWhile && currentStage.Type != StageTypeUntil {
		wb.addError(fmt.Errorf("WithMaxIterations() can only be used with loop stages"))
		return wb
	}

	currentStage.MaxIterations = maxIterations
	return wb
}

// Template creates a reusable workflow template that can be instantiated with parameters.
func (wb *WorkflowBuilder) Template(id string, parameterFunc TemplateParameterFunc, templateBuilder func(*WorkflowBuilder) *WorkflowBuilder) *WorkflowBuilder {
	if err := wb.validateStageID(id); err != nil {
		wb.addError(err)
		return wb
	}

	if parameterFunc == nil {
		wb.addError(fmt.Errorf("template stage '%s' must have a parameter function", id))
		return wb
	}

	// Create a new builder for the template workflow
	templateWorkflow := NewBuilder(wb.memory)
	if templateBuilder != nil {
		templateWorkflow = templateBuilder(templateWorkflow)
	}

	stage := &BuilderStage{
		ID:               id,
		Type:             StageTypeTemplate,
		TemplateParams:   parameterFunc,
		TemplateWorkflow: templateWorkflow,
		Branches:         make(map[string]*BuilderStage),
		Next:             make([]string, 0),
		Metadata:         make(map[string]interface{}),
	}

	wb.stages = append(wb.stages, stage)
	wb.stepIndex[id] = stage

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
	// Check that all referenced next stages exist to ensure graph integrity
	if err := wb.validateStageReferences(); err != nil {
		return err
	}

	// Now that references are valid, check for cycles
	if err := wb.checkForCycles(); err != nil {
		return err
	}

	// Validate all stage configurations
	for _, stage := range wb.stages {
		if err := wb.validateStage(stage); err != nil {
			return fmt.Errorf("stage '%s': %w", stage.ID, err)
		}
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
	case StageTypeForEach:
		if stage.IteratorFunc == nil {
			return fmt.Errorf("forEach stage must have an iterator function")
		}
		if stage.LoopBody == nil {
			return fmt.Errorf("forEach stage must have a loop body")
		}
		if stage.MaxIterations <= 0 {
			return fmt.Errorf("forEach stage must have a positive maximum iteration count")
		}
	case StageTypeWhile, StageTypeUntil:
		if stage.LoopCondition == nil {
			return fmt.Errorf("%s stage must have a loop condition function",
				map[StageType]string{StageTypeWhile: "while", StageTypeUntil: "until"}[stage.Type])
		}
		if stage.LoopBody == nil {
			return fmt.Errorf("%s stage must have a loop body",
				map[StageType]string{StageTypeWhile: "while", StageTypeUntil: "until"}[stage.Type])
		}
		if stage.MaxIterations <= 0 {
			return fmt.Errorf("%s stage must have a positive maximum iteration count",
				map[StageType]string{StageTypeWhile: "while", StageTypeUntil: "until"}[stage.Type])
		}
	case StageTypeTemplate:
		if stage.TemplateParams == nil {
			return fmt.Errorf("template stage must have a parameter function")
		}
		if stage.TemplateWorkflow == nil {
			return fmt.Errorf("template stage must have a template workflow")
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
		recStack[stageID] = false // Clean up recursion stack for this path
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
	hasLoop := false
	hasTemplate := false

	for _, stage := range wb.stages {
		switch stage.Type {
		case StageTypeParallel:
			hasParallel = true
		case StageTypeConditional:
			hasConditional = true
		case StageTypeForEach, StageTypeWhile, StageTypeUntil:
			hasLoop = true
		case StageTypeTemplate:
			hasTemplate = true
		}
	}

	// Advanced patterns require composite workflow
	if hasLoop || hasTemplate {
		return "composite"
	}

	// Conditional patterns require router workflow
	if hasConditional {
		return "router"
	}

	// Only use parallel workflow if it's purely parallel stages
	if hasParallel && len(wb.stages) == 1 {
		return "parallel"
	}

	// Default to chain workflow for all other cases
	return "chain"
}

// isLinearChain method removed as it had fragile logic that assumed
// stages were added in execution order. The simplified workflow type
// determination no longer needs this method.

func (wb *WorkflowBuilder) buildChainWorkflow() (Workflow, error) {
	workflow := NewChainWorkflow(wb.memory)

	for _, stage := range wb.stages {
		var module core.Module

		// Handle different stage types
		switch stage.Type {
		case StageTypeSequential:
			module = stage.Module
		case StageTypeConditional:
			// For conditional stages, use the first available module as fallback
			// Full conditional logic will be implemented in Task 1.2
			branchKeys := make([]string, 0, len(stage.Branches))
			for k := range stage.Branches {
				branchKeys = append(branchKeys, k)
			}
			sort.Strings(branchKeys)

			for _, key := range branchKeys {
				if branch := stage.Branches[key]; branch.Module != nil {
					module = branch.Module
					break
				}
			}
			if module == nil {
				return nil, fmt.Errorf("conditional stage '%s' has no valid modules", stage.ID)
			}
		case StageTypeParallel:
			// For parallel stages in a chain workflow, this is a temporary fallback.
			// It only supports converting a parallel stage if it contains exactly one step.
			// Full support for mixed parallel/sequential workflows is planned for a future update.
			if len(stage.Steps) == 1 {
				module = stage.Steps[0].Module
			} else if len(stage.Steps) == 0 {
				return nil, fmt.Errorf("parallel stage '%s' has no steps", stage.ID)
			} else {
				return nil, fmt.Errorf("cannot convert parallel stage '%s' with %d steps to a single step in a chain workflow; this feature is not yet supported", stage.ID, len(stage.Steps))
			}
		default:
			return nil, fmt.Errorf("unsupported stage type for chain workflow: %v", stage.Type)
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

func (wb *WorkflowBuilder) buildParallelWorkflow() (Workflow, error) {
	// This function should only be called for pure parallel workflows
	workflow := NewParallelWorkflow(wb.memory, wb.config.MaxConcurrency)

	// We expect exactly one parallel stage
	if len(wb.stages) != 1 || wb.stages[0].Type != StageTypeParallel {
		return nil, fmt.Errorf("buildParallelWorkflow called for non-parallel workflow")
	}

	stage := wb.stages[0]
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

	return workflow, nil
}

func (wb *WorkflowBuilder) buildRouterWorkflow() (Workflow, error) {
	// Find the first conditional stage to use as classifier
	var classifierStage *BuilderStage
	for _, stage := range wb.stages {
		if stage.Type == StageTypeConditional {
			classifierStage = stage
			break
		}
	}

	if classifierStage == nil {
		return nil, fmt.Errorf("router workflow requires at least one conditional stage")
	}

	// Create a simple module that evaluates the condition and returns classification
	classifierModule := &conditionalClassifierModule{condition: classifierStage.Condition}
	workflow := NewConditionalRouterWorkflow(wb.memory, classifierModule)

	// Add routes based on branches
	for branchKey, branchStage := range classifierStage.Branches {
		if branchStage.Module != nil {
			step := &Step{
				ID:          branchStage.ID,
				Module:      branchStage.Module,
				NextSteps:   []string{},
				RetryConfig: branchStage.RetryConfig,
			}
			if err := workflow.AddRoute(branchKey, step); err != nil {
				return nil, fmt.Errorf("failed to add route '%s': %w", branchKey, err)
			}
		}
	}

	return workflow, nil
}

func (wb *WorkflowBuilder) buildCompositeWorkflow() (Workflow, error) {
	workflow := NewCompositeWorkflow(wb.memory)

	// Add all stages to the composite workflow
	for _, stage := range wb.stages {
		workflow.AddBuilderStage(stage)
	}

	return workflow, nil
}
