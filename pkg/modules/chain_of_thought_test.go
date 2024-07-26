// modules/chainofthought_test.go

package modules

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
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

	// Test the Process method
	traces := &[]core.Trace{}
	ctx := context.WithValue(context.Background(), utils.TracesContextKey, traces)

	inputs := map[string]any{"question": "What is the meaning of life?"}
	outputs, err := cot.Process(ctx, inputs)

	// Assert the results
	assert.NoError(t, err)
	assert.Equal(t, "42", outputs["answer"])
	assert.Equal(t, "Step 1, Step 2, Step 3", outputs["rationale"])

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}
