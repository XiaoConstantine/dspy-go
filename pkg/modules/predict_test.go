package modules

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockLLM is a mock implementation of core.LLM.
type MockLLM struct {
	mock.Mock
}

func (m *MockLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (string, error) {
	args := m.Called(ctx, prompt, options)
	return args.String(0), args.Error(1)
}

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(map[string]interface{}), args.Error(1)
}

func TestPredict(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(MockLLM)

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return("answer: 42", nil)

	// Create a Predict module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.Field{Name: "answer"}}},
	)
	predict := NewPredict(signature)
	predict.SetLLM(mockLLM)

	// Test the Process method
	ctx := core.WithTraceManager(context.Background())

	inputs := map[string]any{"question": "What is the meaning of life?"}
	outputs, err := predict.Process(ctx, inputs)

	// Assert the results
	assert.NoError(t, err)
	assert.Equal(t, "42", outputs["answer"])

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
	// Verify traces
	tm := core.GetTraceManager(ctx)
	rootTrace := tm.RootTrace
	assert.Equal(t, "Predict", rootTrace.ModuleName)
	assert.Equal(t, "Predict", rootTrace.ModuleType)
	assert.Equal(t, inputs, rootTrace.Inputs)
	assert.Equal(t, outputs, rootTrace.Outputs)
	assert.Nil(t, rootTrace.Error)
	assert.True(t, rootTrace.Duration > 0)
}
