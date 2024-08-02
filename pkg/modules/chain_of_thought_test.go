package modules

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestChainOfThought(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(MockLLM)

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return("rationale: Step 1, Step 2, Step 3\nanswer: 42", nil)

	// Create a ChainOfThought module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.Field{Name: "answer"}}},
	)
	cot := NewChainOfThought(signature)
	cot.SetLLM(mockLLM)

	// Create a context with TraceManager
	ctx := core.WithTraceManager(context.Background())

	inputs := map[string]any{"question": "What is the meaning of life?"}
	outputs, err := cot.Process(ctx, inputs)

	// Assert the results
	assert.NoError(t, err)
	assert.Equal(t, "42", outputs["answer"])
	assert.Equal(t, "Step 1, Step 2, Step 3", outputs["rationale"])
	// Verify traces
	tm := core.GetTraceManager(ctx)
	rootTrace := tm.RootTrace
	assert.Equal(t, "ChainOfThought", rootTrace.ModuleName)
	assert.Equal(t, "ChainOfThought", rootTrace.ModuleType)
	assert.Equal(t, inputs, rootTrace.Inputs)
	assert.Equal(t, outputs, rootTrace.Outputs)
	assert.Nil(t, rootTrace.Error)
	assert.True(t, rootTrace.Duration > 0)
	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}
