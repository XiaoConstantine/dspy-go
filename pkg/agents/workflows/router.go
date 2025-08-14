package workflows

import (
	"context"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"

	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// RouterWorkflow directs inputs to different processing paths based on
// a classification step.
type RouterWorkflow struct {
	*BaseWorkflow
	// The step that determines which path to take
	classifierStep *Step
	// Maps classification outputs to step sequences
	routes map[string][]*Step
}

func NewRouterWorkflow(memory agents.Memory, classifierStep *Step) *RouterWorkflow {
	return &RouterWorkflow{
		BaseWorkflow:   NewBaseWorkflow(memory),
		classifierStep: classifierStep,
		routes:         make(map[string][]*Step),
	}
}

// AddRoute associates a classification value with a sequence of steps.
func (w *RouterWorkflow) AddRoute(classification string, steps []*Step) error {
	// Validate steps exist in workflow
	for _, step := range steps {
		if _, exists := w.stepIndex[step.ID]; !exists {
			return fmt.Errorf("step %s not found in workflow", step.ID)
		}
	}
	w.routes[classification] = steps
	return nil
}

func (w *RouterWorkflow) Execute(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	// Initialize state
	state := make(map[string]interface{})
	for k, v := range inputs {
		state[k] = v
	}

	// Execute classifier step
	result, err := w.classifierStep.Execute(ctx, inputs)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.WorkflowExecutionFailed, "classifier step failed"),
			errors.Fields{
				"step_id": w.classifierStep.ID,
				"inputs":  inputs,
			})
	}

	// Get classification from result
	classificationRaw, ok := result.Outputs["classification"].(string)
	if !ok {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "classifier did not return a string classification"),
			errors.Fields{
				"actual_type": fmt.Sprintf("%T", result.Outputs["classification"]),
			})
	}

	// Clean up classification - remove field prefix and whitespace
	classification := strings.TrimSpace(classificationRaw)
	if strings.HasPrefix(classification, "classification") {
		// Remove "classification" prefix and any following whitespace/newlines
		classification = strings.TrimSpace(strings.TrimPrefix(classification, "classification"))
		// Remove any remaining colons or newlines
		classification = strings.TrimSpace(strings.TrimLeft(classification, ":\n"))
	}

	// Get route for this classification
	route, exists := w.routes[classification]
	if !exists {
		return nil, errors.WithFields(
			errors.New(errors.ResourceNotFound, "no route defined for classification"),
			errors.Fields{
				"classification": classification,
			})

	}

	// Execute steps in the selected route
	for _, step := range route {
		signature := step.Module.GetSignature()

		stepInputs := make(map[string]interface{})
		for _, field := range signature.Inputs {
			if val, ok := state[field.Name]; ok {
				stepInputs[field.Name] = val
			}
		}

		result, err := step.Execute(ctx, stepInputs)
		if err != nil {
			return nil, errors.WithFields(
				errors.Wrap(err, errors.StepExecutionFailed, "step execution failed"),
				errors.Fields{
					"step_id": step.ID,
					"inputs":  stepInputs,
				})
		}

		for k, v := range result.Outputs {
			state[k] = v
		}
	}

	return state, nil
}
