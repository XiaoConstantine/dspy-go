package modules

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestChainOfThought(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(MockLLM)

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(`
rationale:
Step 1, Step 2, Step 3

answer:
42
`, nil)
	// Create a ChainOfThought module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	cot := NewChainOfThought(signature)
	cot.SetLLM(mockLLM)

	// Create a context with TraceManager
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "What is the meaning of life?"}
	outputs, err := cot.Process(ctx, inputs)

	// Assert the results
	assert.NoError(t, err)
	assert.Equal(t, "42", outputs["answer"])
	assert.Equal(t, "Step 1, Step 2, Step 3", outputs["rationale"])
	// Verify traces
	spans := core.CollectSpans(ctx)
	require.Len(t, spans, 2, "Should have two spans")

	assert.Equal(t, "ChainOfThought", spans[0].Operation)
	assert.Equal(t, "Predict", spans[1].Operation)
	assert.Equal(t, spans[0].ID, spans[1].ParentID)
	rootSpan := spans[0]

	// Verify the span structure
	assert.Contains(t, rootSpan.Annotations, "inputs")
	assert.Contains(t, rootSpan.Annotations, "outputs")
	assert.NotZero(t, rootSpan.StartTime)
	assert.Nil(t, rootSpan.Error)
	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}
