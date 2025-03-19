package modules

import (
	"context"
	"testing"

	testutil "github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
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

func TestReAct_WithErroredTool(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Create responses for the sequence
	resp1 := &core.LLMResponse{
		Content: `
		thought:
		I should check the weather
		
		action:
		weather_check
		`,
	}

	resp2 := &core.LLMResponse{
		Content: `
		thought:
		I encountered an error, I'll try another approach
		
		action:
		Finish
		
		answer:
		I apologize, but I wasn't able to get the weather information due to a technical issue.
		`,
	}

	// Set up the mock LLM behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(resp1, nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(resp2, nil).Once()

	// Create a tool that returns an error
	mockWeatherTool := testutil.NewMockTool("weather_check")
	weatherSchema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"location": {
				Type:        "string",
				Description: "Location to check weather for",
				Required:    true,
			},
		},
	}

	mockWeatherTool.On("Metadata").Return(&core.ToolMetadata{
		Name:         "weather_check",
		Description:  "Check the weather at a location",
		InputSchema:  weatherSchema,
		Capabilities: []string{"weather"},
	})

	mockWeatherTool.On("CanHandle", mock.Anything, "weather_check").Return(true)
	mockWeatherTool.On("Validate", mock.Anything).Return(nil)
	mockWeatherTool.On("Execute", mock.Anything, mock.Anything).Return(core.ToolResult{}, errors.New(errors.LLMGenerationFailed, "weather service down"))

	// Create ReAct module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, []core.Tool{mockWeatherTool}, 5)
	react.SetLLM(mockLLM)

	// Execute with a question
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "What's the weather like today?"}
	outputs, err := react.Process(ctx, inputs)

	// Verify results
	assert.NoError(t, err)
	answerStr, ok := outputs["answer"].(string)
	assert.True(t, ok, "answer should be a string")
	assert.Contains(t, answerStr, "technical issue")

	// Verify that the mocks were called as expected
	mockLLM.AssertExpectations(t)
	mockWeatherTool.AssertExpectations(t)
}

func TestReAct_MaxIterations(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Create responses that continue to call tools
	resp1 := &core.LLMResponse{
		Content: `
		thought:
		I'll check the database first
		
		action:
		database_query
		`,
	}

	resp2 := &core.LLMResponse{
		Content: `
		thought:
		Now I need to check the user profile
		
		action:
		user_profile
		`,
	}

	// Set up the mock behavior to keep calling tools
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(resp1, nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(resp2, nil).Once()

	// Create mock tools
	dbTool := testutil.NewMockTool("database_query")
	dbSchema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"query": {
				Type:        "string",
				Description: "SQL query",
				Required:    true,
			},
		},
	}

	dbTool.On("Metadata").Return(&core.ToolMetadata{
		Name:         "database_query",
		Description:  "Query the database",
		InputSchema:  dbSchema,
		Capabilities: []string{"database"},
	})

	dbTool.On("CanHandle", mock.Anything, "database_query").Return(true)
	dbTool.On("Validate", mock.Anything).Return(nil)
	dbTool.On("Execute", mock.Anything, mock.Anything).Return(core.ToolResult{
		Data: "Database results",
	}, nil)

	profileTool := testutil.NewMockTool("user_profile")
	profileSchema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"user_id": {
				Type:        "string",
				Description: "User ID",
				Required:    true,
			},
		},
	}

	profileTool.On("Metadata").Return(&core.ToolMetadata{
		Name:         "user_profile",
		Description:  "Get user profile",
		InputSchema:  profileSchema,
		Capabilities: []string{"user", "profile"},
	})

	profileTool.On("CanHandle", mock.Anything, "user_profile").Return(true)
	profileTool.On("Validate", mock.Anything).Return(nil)
	profileTool.On("Execute", mock.Anything, mock.Anything).Return(core.ToolResult{
		Data: "User profile data",
	}, nil)

	// Create ReAct module with max iterations of 2
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	tools := []core.Tool{dbTool, profileTool}
	react := NewReAct(signature, tools, 2) // Only 2 iterations max
	react.SetLLM(mockLLM)

	// Execute the module
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "Analyze this user data"}
	outputs, err := react.Process(ctx, inputs)

	// Verify results - we should get an error about max iterations
	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "max iterations reached")

	// Verify that the mocks were called as expected
	mockLLM.AssertExpectations(t)
	dbTool.AssertExpectations(t)
	profileTool.AssertExpectations(t)
}

func TestReAct_InvalidAction(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Create response with malformed action
	resp := &core.LLMResponse{
		Content: `
		thought:
		I need to check something
		
		action:
		`, // Empty action - added comma here
	}

	// Set up the mock behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(resp, nil).Once()

	// Create ReAct module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, []core.Tool{}, 5)
	react.SetLLM(mockLLM)

	// Execute with a question
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "What's the weather like today?"}
	outputs, err := react.Process(ctx, inputs)

	// Verify results
	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "invalid action")

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}

func TestReAct_NoMatchingTool(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Create response requesting a non-existent tool
	resp := &core.LLMResponse{
		Content: `
        thought:
        I'll check the calendar
        
        action:
        calendar_check
        `,
	}

	// Set up the mock behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(resp, nil).Once()

	// Create a tool but not the calendar one
	weatherTool := testutil.NewMockTool("weather_check")
	weatherSchema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"location": {
				Type:        "string",
				Description: "Location",
				Required:    true,
			},
		},
	}

	weatherTool.On("Metadata").Return(&core.ToolMetadata{
		Name:         "weather_check",
		Description:  "Check the weather at a location",
		InputSchema:  weatherSchema,
		Capabilities: []string{"weather"},
	})

	// Just set CanHandle to false - no need to set up Execute expectation since it won't be called
	weatherTool.On("CanHandle", mock.Anything, "calendar_check").Return(false)
	weatherTool.On("Execute", mock.Anything, mock.Anything).Return(
		core.ToolResult{}, errors.New(errors.InvalidInput, "no tool found for action"))

	// Create ReAct module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, []core.Tool{weatherTool}, 5)
	react.SetLLM(mockLLM)

	// Execute with a question
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "What's on my calendar today?"}
	outputs, err := react.Process(ctx, inputs)

	// Verify results
	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "no tool found for action")

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
	weatherTool.AssertExpectations(t)
}

func TestReAct_ToolValidationError(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Create a mock tool
	mockTool := testutil.NewMockTool("weather_check")

	// Define the signature for the ReAct module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.Field{Name: "answer"}}},
	)

	// Create the ReAct module with the mock tool
	react := NewReAct(signature, []core.Tool{mockTool}, 5)

	// Set the mock LLM
	react.SetLLM(mockLLM)

	// Setup mock LLM response to trigger tool execution
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: "Action: weather_check\nAction Input: invalid_params",
	}, nil)

	// Setup mock tool to expect Execute call and return a validation error
	mockTool.On("Execute", mock.Anything, mock.Anything).Return("", errors.New(errors.ValidationFailed, "validation error"))

	// Test input
	ctx := context.Background()
	inputs := map[string]interface{}{
		"question": "What is the weather like?",
	}

	// Execute the ReAct module
	result, err := react.Process(ctx, inputs)

	// Assertions
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "validation error")
	assert.Nil(t, result)

	// Verify mock expectations
	mockLLM.AssertExpectations(t)
	mockTool.AssertExpectations(t)
}

func TestReAct_Clone(t *testing.T) {
	// Create a ReAct module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	mockTool := testutil.NewMockTool("test_tool")
	originalReact := NewReAct(signature, []core.Tool{mockTool}, 5)

	// Clone the module
	clonedModule := originalReact.Clone()

	// Check that it's the right type
	clonedReact, ok := clonedModule.(*ReAct)
	assert.True(t, ok, "Cloned module should be a ReAct")

	// Check that values were correctly cloned
	assert.Equal(t, originalReact.MaxIters, clonedReact.MaxIters)
	assert.Equal(t, len(originalReact.Tools), len(clonedReact.Tools))

	// Verify that the Predict module was cloned (not just referenced)
	assert.NotSame(t, originalReact.Predict, clonedReact.Predict,
		"The Predict module should be cloned, not just referenced")
}

func TestReAct_WithDefaultOptions(t *testing.T) {
	// Create a ReAct module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	reactModule := NewReAct(signature, []core.Tool{}, 5)

	// Use WithDefaultOptions to set options
	result := reactModule.WithDefaultOptions(
		core.WithGenerateOptions(
			core.WithTemperature(0.7),
			core.WithMaxTokens(2000),
		),
	)

	// Verify the result is the same module (fluent interface)
	assert.Same(t, reactModule, result, "WithDefaultOptions should return the same instance")

	// We can't directly verify the options were set since they're encapsulated,
	// but we can verify the method can be called without errors
}

func TestReAct_FormatToolResult(t *testing.T) {
	// Create a tool result
	result := core.ToolResult{
		Data: "This is the result data",
		Metadata: map[string]interface{}{
			"source": "test_tool",
			"status": "success",
		},
		Annotations: map[string]interface{}{
			"confidence": "high",
		},
	}

	// Format the result
	observation := formatToolResult(result)

	// Verify the formatting
	assert.Contains(t, observation, "Result: This is the result data")
	assert.Contains(t, observation, "Metadata: map[source:test_tool status:success]")
	assert.Contains(t, observation, "Annotations: map[confidence:high]")
}

func TestReAct_CalculateToolMatchScore(t *testing.T) {
	// Create tool metadata
	metadata := &core.ToolMetadata{
		Name:         "weather_tool",
		Description:  "Check the weather at a location",
		Capabilities: []string{"weather", "temperature", "forecast"},
	}

	// Test exact match with tool name
	score1 := calculateToolMatchScore(metadata, "weather_tool")
	assert.Greater(t, score1, float64(0.5), "Exact name match should score high")

	// Test partial match with tool name
	score2 := calculateToolMatchScore(metadata, "Use weather_tool to check NYC")
	assert.Greater(t, score2, float64(0.5), "Partial name match should score high")

	// Test capability match
	score3 := calculateToolMatchScore(metadata, "check temperature in NYC")
	assert.Greater(t, score3, float64(0.3), "Capability match should have a moderate score")

	// Test no match
	score4 := calculateToolMatchScore(metadata, "search for restaurants")
	assert.Less(t, score4, float64(0.3), "No match should have a low score")
}
