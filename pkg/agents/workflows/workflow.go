package workflows

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
)

// Workflow represents a sequence of steps that accomplish a task.
type Workflow interface {
	// Execute runs the workflow with the provided inputs
	Execute(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error)

	// GetSteps returns all steps in this workflow
	GetSteps() []*Step

	// AddStep adds a new step to the workflow
	AddStep(step *Step) error
}

// BaseWorkflow provides common workflow functionality.
type BaseWorkflow struct {
	// steps stores all steps in the workflow
	steps []*Step

	// stepIndex provides quick lookup of steps by ID
	stepIndex map[string]*Step

	// memory provides persistence between workflow runs
	memory agents.Memory
}

func NewBaseWorkflow(memory agents.Memory) *BaseWorkflow {
	return &BaseWorkflow{
		steps:     make([]*Step, 0),
		stepIndex: make(map[string]*Step),
		memory:    memory,
	}
}

func (w *BaseWorkflow) AddStep(step *Step) error {
	// Validate step ID is unique
	if _, exists := w.stepIndex[step.ID]; exists {
		return fmt.Errorf("step with ID %s already exists", step.ID)
	}

	// Add step to workflow
	w.steps = append(w.steps, step)
	w.stepIndex[step.ID] = step

	return nil
}

func (w *BaseWorkflow) GetSteps() []*Step {
	return w.steps
}

// ValidateWorkflow checks if the workflow structure is valid.
func (w *BaseWorkflow) ValidateWorkflow() error {
	// Check for cycles in step dependencies
	visited := make(map[string]bool)
	path := make(map[string]bool)

	var checkCycle func(stepID string) error
	checkCycle = func(stepID string) error {
		visited[stepID] = true
		path[stepID] = true

		step := w.stepIndex[stepID]
		for _, nextID := range step.NextSteps {
			if !visited[nextID] {
				if err := checkCycle(nextID); err != nil {
					return err
				}
			} else if path[nextID] {
				return fmt.Errorf("cycle detected in workflow")
			}
		}

		path[stepID] = false
		return nil
	}

	for _, step := range w.steps {
		if !visited[step.ID] {
			if err := checkCycle(step.ID); err != nil {
				return err
			}
		}
	}

	return nil
}
