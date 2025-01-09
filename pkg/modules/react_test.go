package modules

import (
	"context"
	"testing"

	testutil "github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// MockTool is a mock implementation of the Tool interface.
type MockTool struct {
	mock.Mock
}

func (m *MockTool) CanHandle(action string) bool {
	args := m.Called(action)
	return args.Bool(0)
}

func (m *MockTool) Execute(ctx context.Context, action string) (string, error) {
	args := m.Called(ctx, action)
	return args.String(0), args.Error(1)
}

func TestReAct(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Create a mock Tool
	mockTool := new(MockTool)

	resp1Content := `
	thought:
	I should use the tool

	action:
	use_tool
	`
	resp1 := &core.LLMResponse{
		Content: resp1Content,
		Usage: &core.TokenInfo{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}

	resp2Content := `
	thought:
	I have the answer

	action:
	Finish

	answer:
	42
	`
	resp2 := &core.LLMResponse{
		Content: resp2Content,
		Usage: &core.TokenInfo{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(resp1, nil).Once()

	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(resp2, nil).Once()
	mockTool.On("CanHandle", "use_tool").Return(true)
	mockTool.On("Execute", mock.Anything, "use_tool").Return("Tool output", nil)

	// Create a ReAct module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, []Tool{mockTool}, 5)
	react.SetLLM(mockLLM)

	// Test the Process method
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	inputs := map[string]any{"question": "What is the meaning of life?"}
	outputs, err := react.Process(ctx, inputs)

	// Assert the results
	assert.NoError(t, err)
	assert.Equal(t, "42", outputs["answer"])

	// Verify that the mocks were called as expected
	mockLLM.AssertExpectations(t)
	mockTool.AssertExpectations(t)
	// Verify traces
	spans := core.CollectSpans(ctx)
	require.Len(t, spans, 3)
	assert.Equal(t, "ReAct", spans[0].Operation)
}
