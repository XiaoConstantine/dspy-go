package workflows

import (
	"context"
	"errors"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestChainWorkflow(t *testing.T) {
	t.Run("Execute steps in sequence", func(t *testing.T) {
		memory := new(MockMemory)
		workflow := NewChainWorkflow(memory)

		// Create mock modules with expected behavior
		module1 := new(MockModule)
		module2 := new(MockModule)

		// Setup mock responses
		module1.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Field: core.Field{Name: "input1"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "output1"}}},
		})
		module1.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{"output1": "intermediate"}, nil,
		)

		module2.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Field: core.Field{Name: "output1"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "final"}}},
		})
		module2.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{"final": "result"}, nil,
		)

		// Add steps to workflow
		err := workflow.AddStep(&Step{ID: "step1", Module: module1})
		require.NoError(t, err, "Failed to add step1")

		err = workflow.AddStep(&Step{ID: "step2", Module: module2})
		require.NoError(t, err, "Failed to add step2")
		// Execute workflow
		ctx := context.Background()
		result, err := workflow.Execute(ctx, map[string]interface{}{
			"input1": "initial",
		})

		assert.NoError(t, err)
		assert.Equal(t, "result", result["final"])

		// Verify mocks
		module1.AssertExpectations(t)
		module2.AssertExpectations(t)
	})

	t.Run("Step failure", func(t *testing.T) {
		memory := new(MockMemory)
		workflow := NewChainWorkflow(memory)

		module := new(MockModule)
		module.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
		})
		module.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{}, errors.New("step failed"),
		)

		err := workflow.AddStep(&Step{ID: "step1", Module: module})
		require.NoError(t, err, "Failed to add step1")
		ctx := context.Background()
		_, err = workflow.Execute(ctx, map[string]interface{}{
			"input": "value",
		})

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "step failed")
		module.AssertExpectations(t)
	})
}
