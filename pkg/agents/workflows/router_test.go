package workflows

import (
	"context"
	"testing"

	stderrors "errors" // Import standard errors package with an alias
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestRouterWorkflow(t *testing.T) {
	t.Run("Successful routing", func(t *testing.T) {
		memory := new(MockMemory)

		// Create classifier module
		classifier := new(MockModule)
		classifier.On("GetSignature").Return(core.Signature{
			Inputs: []core.InputField{{Field: core.Field{Name: "input"}}},
			Outputs: []core.OutputField{
				{Field: core.Field{Name: "classification"}},
			},
		})
		classifier.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{"classification": "route1"}, nil,
		)

		// Create handler module
		handler := new(MockModule)
		handler.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
		})
		handler.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{"output": "handled"}, nil,
		)

		// Create and setup workflow
		workflow := NewRouterWorkflow(memory, &Step{ID: "classifier", Module: classifier})
		err := workflow.AddStep(&Step{ID: "handler", Module: handler})
		require.NoError(t, err, "Failed to add handler")

		err = workflow.AddRoute("route1", []*Step{{ID: "handler", Module: handler}})
		require.NoError(t, err)

		// Execute workflow
		ctx := context.Background()
		result, err := workflow.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		assert.NoError(t, err)
		assert.Equal(t, "handled", result["output"])

		classifier.AssertExpectations(t)
		handler.AssertExpectations(t)
	})

	t.Run("Unknown route", func(t *testing.T) {
		memory := new(MockMemory)

		classifier := new(MockModule)
		classifier.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "classification"}}},
		})
		classifier.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{"classification": "unknown"}, nil,
		)

		workflow := NewRouterWorkflow(memory, &Step{ID: "classifier", Module: classifier})

		ctx := context.Background()
		_, err := workflow.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no route defined")
		classifier.AssertExpectations(t)
	})

	t.Run("Invalid classification type", func(t *testing.T) {
		memory := new(MockMemory)

		classifier := new(MockModule)
		classifier.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "classification"}}},
		})
		classifier.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{"classification": 123}, nil,
		)

		workflow := NewRouterWorkflow(memory, &Step{ID: "classifier", Module: classifier})

		ctx := context.Background()
		_, err := workflow.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "did not return a string classification")
		classifier.AssertExpectations(t)
	})

	t.Run("Missing classification output", func(t *testing.T) {
		memory := new(MockMemory)

		classifier := new(MockModule)
		classifier.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "classification"}}},
		})
		// Return empty map to simulate missing classification
		classifier.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{}, nil,
		)

		workflow := NewRouterWorkflow(memory, &Step{ID: "classifier", Module: classifier})

		ctx := context.Background()
		_, err := workflow.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		assert.Error(t, err)

		assert.Contains(t, err.Error(), "output validation failed: missing required output field")
		classifier.AssertExpectations(t)
	})

	t.Run("Classifier failure", func(t *testing.T) {
		memory := new(MockMemory)

		classifier := new(MockModule)
		classifier.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "classification"}}},
		})
		// Return empty map with error for failure case
		classifier.On("Process", mock.Anything, mock.Anything).Return(
			make(map[string]any), assert.AnError,
		)

		workflow := NewRouterWorkflow(memory, &Step{ID: "classifier", Module: classifier})

		ctx := context.Background()
		_, err := workflow.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "classifier step failed")
		classifier.AssertExpectations(t)
	})

	t.Run("Handler failure", func(t *testing.T) {
		memory := new(MockMemory)

		// Classifier returns valid route
		classifier := new(MockModule)
		classifier.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "classification"}}},
		})
		classifier.On("Process", mock.Anything, mock.Anything).Return(
			map[string]any{"classification": "route1"}, nil,
		)

		// Handler fails
		handler := new(MockModule)
		handler.On("GetSignature").Return(core.Signature{
			Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
			Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
		})
		// Return empty map with error for failure case
		handler.On("Process", mock.Anything, mock.Anything).Return(
			make(map[string]any), assert.AnError,
		)

		workflow := NewRouterWorkflow(memory, &Step{ID: "classifier", Module: classifier})
		err := workflow.AddStep(&Step{ID: "handler", Module: handler})
		require.NoError(t, err, "Failed to add handler")

		err = workflow.AddRoute("route1", []*Step{{ID: "handler", Module: handler}})
		require.NoError(t, err)

		ctx := context.Background()
		_, err = workflow.Execute(ctx, map[string]interface{}{
			"input": "test",
		})

		assert.Error(t, err)

		var stepErr *errors.Error
		assert.True(t, stderrors.As(err, &stepErr),
			"Expected error to be *errors.Error")

		// Check error code
		assert.Equal(t, errors.StepExecutionFailed, stepErr.Code())

		// Check fields
		fields := stepErr.Fields()
		assert.Equal(t, "handler", fields["step_id"])

		classifier.AssertExpectations(t)
		handler.AssertExpectations(t)
	})

	t.Run("Add route with non-existent step", func(t *testing.T) {
		memory := new(MockMemory)
		classifier := new(MockModule)

		workflow := NewRouterWorkflow(memory, &Step{ID: "classifier", Module: classifier})

		handler := new(MockModule)
		err := workflow.AddRoute("route1", []*Step{{ID: "non_existent", Module: handler}})

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "step non_existent not found in workflow")
	})
}
