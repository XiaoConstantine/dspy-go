package workflows

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
)

// ChainWorkflow executes steps in a linear sequence, where each step's output
// can be used as input for subsequent steps.
type ChainWorkflow struct {
	*BaseWorkflow
}

func NewChainWorkflow(memory agents.Memory) *ChainWorkflow {
	return &ChainWorkflow{
		BaseWorkflow: NewBaseWorkflow(memory),
	}
}

// Execute runs steps sequentially, passing state from one step to the next.
func (w *ChainWorkflow) Execute(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	// Initialize workflow state with input values
	state := make(map[string]interface{})
	for k, v := range inputs {
		state[k] = v
	}

	// Execute steps in sequence
	for _, step := range w.steps {
		signature := step.Module.GetSignature()

		// Create subset of state containing only the fields this step needs
		stepInputs := make(map[string]interface{})
		for _, field := range signature.Inputs {
			if val, ok := state[field.Name]; ok {
				stepInputs[field.Name] = val
			}
		}

		// Execute the step
		result, err := step.Execute(ctx, stepInputs)

		if err != nil {
			return nil, fmt.Errorf("step %s failed: %w", step.ID, err)
		}

		// Update state with step outputs
		for k, v := range result.Outputs {
			state[k] = v
		}
	}

	return state, nil
}
