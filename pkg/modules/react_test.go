package modules

import (
	"context"
	"testing"

	testutil "github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestReAct(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)
	schema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"question": {
				Type:        "string",
				Description: "Default parameter",
				Required:    true,
			},
			"action": {
				Type: "string",
			},
		},
	}

	// Create a mock Tool
	mockTool := testutil.NewMockTool("mock")
	toolMetadata := &core.ToolMetadata{
		Name:         "test_tool",
		Description:  "A test tool",
		InputSchema:  schema,
		Capabilities: []string{"test", "use_tool"},
	}
	mockTool.On("Metadata").Return(toolMetadata)
	mockTool.On("CanHandle", mock.Anything, "use_tool").Return(true)

	mockTool.On("Validate", mock.Anything).Return(nil)
	mockTool.On("Execute", mock.Anything, mock.Anything).Return(
		core.ToolResult{
			Data: "Tool execution result",
			Metadata: map[string]interface{}{
				"status": "success",
			},
		},
		nil,
	)

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

	// Create a ReAct module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, []core.Tool{mockTool}, 5)
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
