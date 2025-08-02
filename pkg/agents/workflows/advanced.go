package workflows

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// CompositeWorkflow handles complex workflow patterns including loops and templates.
type CompositeWorkflow struct {
	*BaseWorkflow
	stages []*BuilderStage
}

// NewCompositeWorkflow creates a new composite workflow.
func NewCompositeWorkflow(memory agents.Memory) *CompositeWorkflow {
	return &CompositeWorkflow{
		BaseWorkflow: NewBaseWorkflow(memory),
		stages:       make([]*BuilderStage, 0),
	}
}

// AddBuilderStage adds a builder stage to the composite workflow.
func (cw *CompositeWorkflow) AddBuilderStage(stage *BuilderStage) {
	cw.stages = append(cw.stages, stage)
}

// Execute runs the composite workflow with support for advanced patterns.
func (cw *CompositeWorkflow) Execute(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	state := make(map[string]interface{})
	for k, v := range inputs {
		state[k] = v
	}

	for _, stage := range cw.stages {
		result, err := cw.executeStage(ctx, stage, state)
		if err != nil {
			return nil, fmt.Errorf("stage '%s' failed: %w", stage.ID, err)
		}

		// Merge results into state
		for k, v := range result {
			state[k] = v
		}
	}

	return state, nil
}

// executeStage executes a single stage based on its type.
func (cw *CompositeWorkflow) executeStage(ctx context.Context, stage *BuilderStage, state map[string]interface{}) (map[string]interface{}, error) {
	// Apply timeout if specified
	if stage.TimeoutMs > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(stage.TimeoutMs)*time.Millisecond)
		defer cancel()
	}

	switch stage.Type {
	case StageTypeSequential:
		return cw.executeSequentialStage(ctx, stage, state)
	case StageTypeParallel:
		return cw.executeParallelStage(ctx, stage, state)
	case StageTypeConditional:
		return cw.executeConditionalStage(ctx, stage, state)
	case StageTypeForEach:
		return cw.executeForEachStage(ctx, stage, state)
	case StageTypeWhile:
		return cw.executeWhileStage(ctx, stage, state)
	case StageTypeUntil:
		return cw.executeUntilStage(ctx, stage, state)
	case StageTypeTemplate:
		return cw.executeTemplateStage(ctx, stage, state)
	default:
		return nil, fmt.Errorf("unsupported stage type: %v", stage.Type)
	}
}

// executeSequentialStage executes a sequential stage.
func (cw *CompositeWorkflow) executeSequentialStage(ctx context.Context, stage *BuilderStage, state map[string]interface{}) (map[string]interface{}, error) {
	if stage.Module == nil {
		return nil, fmt.Errorf("sequential stage must have a module")
	}

	result, err := stage.Module.Process(ctx, state)
	if err != nil {
		return nil, err
	}

	// Convert map[string]any to map[string]interface{}
	converted := make(map[string]interface{})
	for k, v := range result {
		converted[k] = v
	}
	return converted, nil
}

// executeParallelStage executes a parallel stage.
func (cw *CompositeWorkflow) executeParallelStage(ctx context.Context, stage *BuilderStage, state map[string]interface{}) (map[string]interface{}, error) {
	var wg sync.WaitGroup
	results := make(chan map[string]interface{}, len(stage.Steps))
	var allErrors []error
	var mu sync.Mutex

	for _, step := range stage.Steps {
		wg.Add(1)
		go func(step *BuilderStep) {
			defer func() {
				wg.Done()
				// Ensure goroutine cleanup on panic
				if r := recover(); r != nil {
					mu.Lock()
					allErrors = append(allErrors, fmt.Errorf("builder step panicked: %v", r))
					mu.Unlock()
				}
			}()

			// Check if context is already cancelled before execution
			select {
			case <-ctx.Done():
				mu.Lock()
				allErrors = append(allErrors, ctx.Err())
				mu.Unlock()
				return
			default:
			}

			result, err := step.Module.Process(ctx, state)
			if err != nil {
				mu.Lock()
				allErrors = append(allErrors, err)
				mu.Unlock()
			} else {
				// Convert map[string]any to map[string]interface{}
				converted := make(map[string]interface{})
				for k, v := range result {
					converted[k] = v
				}

				// Send result without blocking if context is cancelled
				select {
				case results <- converted:
				case <-ctx.Done():
					return
				}
			}
		}(step)
	}

	wg.Wait()
	close(results)

	if len(allErrors) > 0 {
		return nil, errors.Join(allErrors...)
	}

	// Collect results
	finalResult := make(map[string]interface{})
	for result := range results {
		for k, v := range result {
			finalResult[k] = v
		}
	}

	return finalResult, nil
}

// executeConditionalStage executes a conditional stage.
func (cw *CompositeWorkflow) executeConditionalStage(ctx context.Context, stage *BuilderStage, state map[string]interface{}) (map[string]interface{}, error) {
	shouldExecute, err := stage.Condition(ctx, state)
	if err != nil {
		return nil, fmt.Errorf("condition evaluation failed: %w", err)
	}

	branchKey := "false"
	if shouldExecute {
		branchKey = "true"
	}

	if branch, exists := stage.Branches[branchKey]; exists {
		return cw.executeStage(ctx, branch, state)
	}

	// No branch found, return state unchanged
	return state, nil
}

// executeForEachStage executes a forEach loop.
func (cw *CompositeWorkflow) executeForEachStage(ctx context.Context, stage *BuilderStage, state map[string]interface{}) (map[string]interface{}, error) {
	items, err := stage.IteratorFunc(ctx, state)
	if err != nil {
		return nil, fmt.Errorf("iterator function failed: %w", err)
	}

	results := make([]map[string]interface{}, 0, len(items))
	currentState := make(map[string]interface{})
	for k, v := range state {
		currentState[k] = v
	}

	for i, item := range items {
		if i >= stage.MaxIterations {
			break
		}

		// Add current item to state
		currentState["current_item"] = item
		currentState["current_index"] = i

		// Execute loop body
		loopResult, err := cw.executeNestedWorkflow(ctx, stage.LoopBody, currentState)
		if err != nil {
			return nil, fmt.Errorf("forEach iteration %d failed: %w", i, err)
		}

		results = append(results, loopResult)

		// Update state with loop results
		for k, v := range loopResult {
			currentState[k] = v
		}
	}

	// Add loop results to final state
	currentState["forEach_results"] = results
	return currentState, nil
}

// executeWhileStage executes a while loop.
func (cw *CompositeWorkflow) executeWhileStage(ctx context.Context, stage *BuilderStage, state map[string]interface{}) (map[string]interface{}, error) {
	iteration := 0
	currentState := make(map[string]interface{})
	for k, v := range state {
		currentState[k] = v
	}

	for iteration < stage.MaxIterations {
		shouldContinue, err := stage.LoopCondition(ctx, currentState, iteration)
		if err != nil {
			return nil, fmt.Errorf("while condition evaluation failed: %w", err)
		}

		if !shouldContinue {
			break
		}

		// Execute loop body
		loopResult, err := cw.executeNestedWorkflow(ctx, stage.LoopBody, currentState)
		if err != nil {
			return nil, fmt.Errorf("while iteration %d failed: %w", iteration, err)
		}

		// Update state with loop results
		for k, v := range loopResult {
			currentState[k] = v
		}

		iteration++
	}

	currentState["while_iterations"] = iteration
	return currentState, nil
}

// executeUntilStage executes an until loop.
func (cw *CompositeWorkflow) executeUntilStage(ctx context.Context, stage *BuilderStage, state map[string]interface{}) (map[string]interface{}, error) {
	iteration := 0
	currentState := make(map[string]interface{})
	for k, v := range state {
		currentState[k] = v
	}

	for iteration < stage.MaxIterations {
		shouldStop, err := stage.LoopCondition(ctx, currentState, iteration)
		if err != nil {
			return nil, fmt.Errorf("until condition evaluation failed: %w", err)
		}

		if shouldStop {
			break
		}

		// Execute loop body
		loopResult, err := cw.executeNestedWorkflow(ctx, stage.LoopBody, currentState)
		if err != nil {
			return nil, fmt.Errorf("until iteration %d failed: %w", iteration, err)
		}

		// Update state with loop results
		for k, v := range loopResult {
			currentState[k] = v
		}

		iteration++
	}

	currentState["until_iterations"] = iteration
	return currentState, nil
}

// executeTemplateStage executes a template with parameters.
func (cw *CompositeWorkflow) executeTemplateStage(ctx context.Context, stage *BuilderStage, state map[string]interface{}) (map[string]interface{}, error) {
	// Resolve template parameters
	params, err := stage.TemplateParams(ctx, state)
	if err != nil {
		return nil, fmt.Errorf("template parameter resolution failed: %w", err)
	}

	// Merge parameters with current state
	templateState := make(map[string]interface{})
	for k, v := range state {
		templateState[k] = v
	}
	for k, v := range params {
		templateState[k] = v
	}

	// Execute template workflow
	return cw.executeNestedWorkflow(ctx, stage.TemplateWorkflow, templateState)
}

// executeNestedWorkflow executes a nested WorkflowBuilder.
func (cw *CompositeWorkflow) executeNestedWorkflow(ctx context.Context, nestedBuilder *WorkflowBuilder, state map[string]interface{}) (map[string]interface{}, error) {
	if nestedBuilder == nil {
		return state, nil
	}

	workflow, err := nestedBuilder.Build()
	if err != nil {
		return nil, fmt.Errorf("failed to build nested workflow: %w", err)
	}

	return workflow.Execute(ctx, state)
}

// ConditionalRouterWorkflow handles conditional routing patterns.
type ConditionalRouterWorkflow struct {
	*BaseWorkflow
	classifier   core.Module
	routes       map[string]*Step
	defaultRoute *Step
}

// NewConditionalRouterWorkflow creates a new conditional router workflow.
func NewConditionalRouterWorkflow(memory agents.Memory, classifier core.Module) *ConditionalRouterWorkflow {
	return &ConditionalRouterWorkflow{
		BaseWorkflow: NewBaseWorkflow(memory),
		classifier:   classifier,
		routes:       make(map[string]*Step),
	}
}

// AddRoute adds a route to the router.
func (crw *ConditionalRouterWorkflow) AddRoute(route string, step *Step) error {
	if _, exists := crw.routes[route]; exists {
		return fmt.Errorf("route '%s' already exists", route)
	}
	crw.routes[route] = step
	return nil
}

// SetDefaultRoute sets the default route for unmatched classifications.
func (crw *ConditionalRouterWorkflow) SetDefaultRoute(step *Step) {
	crw.defaultRoute = step
}

// Execute runs the router workflow.
func (crw *ConditionalRouterWorkflow) Execute(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	// First classify the input
	classification, err := crw.classifier.Process(ctx, inputs)
	if err != nil {
		return nil, fmt.Errorf("classification failed: %w", err)
	}

	// Extract route from classification
	route, ok := classification["classification"]
	if !ok {
		return nil, fmt.Errorf("classifier did not return 'classification' field")
	}

	routeStr, ok := route.(string)
	if !ok {
		return nil, fmt.Errorf("classification must be a string, got %T", route)
	}

	// Find and execute the appropriate route
	if step, exists := crw.routes[routeStr]; exists {
		result, err := step.Execute(ctx, inputs)
		if err != nil {
			return nil, fmt.Errorf("route '%s' execution failed: %w", routeStr, err)
		}
		return result.Outputs, nil
	}

	// Use default route if available
	if crw.defaultRoute != nil {
		result, err := crw.defaultRoute.Execute(ctx, inputs)
		if err != nil {
			return nil, fmt.Errorf("default route execution failed: %w", err)
		}
		return result.Outputs, nil
	}

	return nil, fmt.Errorf("no route found for classification '%s' and no default route configured", routeStr)
}

// conditionalClassifierModule wraps a ConditionalFunc to act as a classifier module.
type conditionalClassifierModule struct {
	condition ConditionalFunc
}

func (ccm *conditionalClassifierModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	state := make(map[string]interface{}, len(inputs))
	for k, v := range inputs {
		state[k] = v
	}
	result, err := ccm.condition(ctx, state)
	if err != nil {
		return nil, err
	}

	classification := "false"
	if result {
		classification = "true"
	}

	return map[string]any{
		"classification": classification,
	}, nil
}

func (ccm *conditionalClassifierModule) GetSignature() core.Signature {
	return core.Signature{
		Inputs: []core.InputField{
			{Field: core.Field{Name: "input", Description: "input for conditional evaluation"}},
		},
		Outputs: []core.OutputField{
			{Field: core.Field{Name: "classification", Description: "true or false classification result"}},
		},
	}
}

func (ccm *conditionalClassifierModule) SetSignature(signature core.Signature) {}
func (ccm *conditionalClassifierModule) SetLLM(llm core.LLM)                   {}
func (ccm *conditionalClassifierModule) Clone() core.Module {
	return &conditionalClassifierModule{condition: ccm.condition}
}

// GetDisplayName returns a display name for the conditional classifier.
func (ccm *conditionalClassifierModule) GetDisplayName() string {
	return "ConditionalClassifier"
}

// GetModuleType returns the module type.
func (ccm *conditionalClassifierModule) GetModuleType() string {
	return "ConditionalClassifier"
}
