package modules

import (
	"context"
	"testing"

	testutil "github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestReAct(t *testing.T) {
	mockLLM := new(testutil.MockLLM)
	// Define schema inline or ensure it's available if defined globally
	schema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"question": {
				Type:        "string",
				Description: "Default parameter",
				Required:    true,
			},
		},
	}
	mockTool := testutil.NewMockTool("test_tool") // Use the actual tool name

	toolMetadata := &core.ToolMetadata{
		Name:         "test_tool",
		Description:  "A test tool",
		InputSchema:  schema, // Use the defined schema
		Capabilities: []string{"test"},
	}
	mockTool.On("Metadata").Return(toolMetadata) // Called during NewReAct
	mockTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil)
	expectedArgs := map[string]interface{}{"question": "What is the meaning of\nlife?"}
	mockTool.On("Execute", mock.Anything, expectedArgs).Return(
		core.ToolResult{Data: "Tool execution result"}, nil,
	)
	resp1 := &core.LLMResponse{Content: `
  thought:
  I should use the tool

  action:
  <action><tool_name>test_tool</tool_name><arguments><arg key="question">What is the meaning of
  life?</arg></arguments></action>
  `}
	resp2 := &core.LLMResponse{Content: `
  thought:
  I have the answer

  action:
  Finish

  answer:
  42
  `}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1,
		nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2,
		nil).Once()

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, []core.Tool{mockTool}, 5)
	react.SetLLM(mockLLM)

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "What is the meaning of life?"}
	outputs, err := react.Process(ctx, inputs)

	assert.NoError(t, err)
	require.NotNil(t, outputs)
	assert.Equal(t, "42", outputs["answer"])
	mockLLM.AssertExpectations(t)
	mockTool.AssertExpectations(t) // This should now only check Metadata, Validate, Execute

	spans := core.CollectSpans(ctx)
	require.Len(t, spans, 4)
	assert.Equal(t, "ReAct", spans[0].Operation)
	assert.Equal(t, "Predict", spans[1].Operation)
	assert.Equal(t, "executeToolByName", spans[2].Operation)
	assert.Equal(t, "Predict", spans[3].Operation)
}

func TestReAct_WithErroredTool(t *testing.T) {
	mockLLM := new(testutil.MockLLM)

	resp1 := &core.LLMResponse{Content: `
  thought:
  I should check the weather

  action:
  <action><tool_name>weather_check</tool_name><arguments><arg key="location">nearby</arg></arguments></action>
  `}
	resp2 := &core.LLMResponse{Content: `
  thought:
  I encountered an error, I'll try another approach

  action:
  Finish

  answer:
  I apologize, but I wasn't able to get the weather information due to a technical issue.
  `}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1,
		nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2,
		nil).Once()

	mockWeatherTool := testutil.NewMockTool("weather_check")
	mockWeatherTool.On("Metadata").Return(&core.ToolMetadata{
		Name:         "weather_check",
		Description:  "Check the weather at a location",
		InputSchema:  models.InputSchema{ /* ... schema ... */ }, // Provide schema directly
		Capabilities: []string{"weather"},
	})
	mockWeatherTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil)
	expectedArgs := map[string]interface{}{"location": "nearby"}
	mockWeatherTool.On("Execute", mock.Anything, expectedArgs).Return(core.ToolResult{}, errors.New(errors.LLMGenerationFailed,
		"weather service down"))

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, []core.Tool{mockWeatherTool}, 5)
	react.SetLLM(mockLLM)

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "What's the weather like today?"}
	outputs, err := react.Process(ctx, inputs)

	assert.NoError(t, err)
	require.NotNil(t, outputs)
	answerStr, ok := outputs["answer"].(string)
	assert.True(t, ok)
	assert.Contains(t, answerStr, "technical issue")
	mockLLM.AssertExpectations(t)
	mockWeatherTool.AssertExpectations(t)
}

func TestReAct_MaxIterations(t *testing.T) {
	mockLLM := new(testutil.MockLLM)

	resp1 := &core.LLMResponse{Content: `
  thought:
  I'll check the database first

  action:
  <action><tool_name>database_query</tool_name><arguments><arg key="query">SELECT * FROM
  data</arg></arguments></action>
  `}
	resp2 := &core.LLMResponse{Content: `
  thought:
  Now I need to check the user profile

  action:
  <action><tool_name>user_profile</tool_name><arguments><arg key="user_id">123</arg></arguments></action>
  `}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1,
		nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2,
		nil).Once()

	dbTool := testutil.NewMockTool("database_query")
	// Remove unused variable declaration for dbSchema
	dbTool.On("Metadata").Return(&core.ToolMetadata{
		Name: "database_query", Description: "Query the database", InputSchema: models.InputSchema{ /*...*/ }, Capabilities: []string{"database"},
	})
	dbTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil)
	dbExpectedArgs := map[string]interface{}{"query": "SELECT * FROM\ndata"}
	dbTool.On("Execute", mock.Anything, dbExpectedArgs).Return(core.ToolResult{Data: "Database results"}, nil)

	profileTool := testutil.NewMockTool("user_profile")
	// Remove unused variable declaration for profileSchema
	profileTool.On("Metadata").Return(&core.ToolMetadata{
		Name: "user_profile", Description: "Get user profile", InputSchema: models.InputSchema{ /*...*/ }, Capabilities: []string{"user",
			"profile"},
	})
	profileTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil)
	profileExpectedArgs := map[string]interface{}{"user_id": "123"}
	profileTool.On("Execute", mock.Anything, profileExpectedArgs).Return(core.ToolResult{Data: "User profile data"}, nil)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	tools := []core.Tool{dbTool, profileTool}
	react := NewReAct(signature, tools, 2)
	react.SetLLM(mockLLM)

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "Analyze this user data"}
	outputs, err := react.Process(ctx, inputs)

	assert.Error(t, err)
	assert.NotNil(t, outputs)
	assert.Contains(t, err.Error(), "max iterations reached")
	mockLLM.AssertExpectations(t)
	dbTool.AssertExpectations(t)
	profileTool.AssertExpectations(t)
}

func TestReAct_InvalidAction(t *testing.T) {
	mockLLM := new(testutil.MockLLM)

	resp1 := &core.LLMResponse{Content: `
  thought:
  I need to check something

  action:
  <action><tool_name>some_tool</tool_name><arguments><arg key="p1">val1</arg</arguments></action>
  `} // Malformed XML
	resp2 := &core.LLMResponse{Content: `
  thought:
  The previous action was invalid. I will stop.

  action:
  Finish

  answer:
  Failed due to invalid action format.
  `}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1,
		nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2,
		nil).Once()

	// Fix: Provide arguments to core.NewSignature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, []core.Tool{}, 5)
	react.SetLLM(mockLLM)

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "What's the weather like today?"}
	outputs, err := react.Process(ctx, inputs)

	assert.NoError(t, err)
	require.NotNil(t, outputs)
	assert.Contains(t, outputs["answer"], "Failed due to invalid action format")
	mockLLM.AssertExpectations(t)
}

func TestReAct_NoMatchingTool(t *testing.T) {
	mockLLM := new(testutil.MockLLM)

	resp := &core.LLMResponse{Content: `
  thought:
  I'll check the calendar

  action:
  <action><tool_name>calendar_check</tool_name><arguments></arguments></action>
  `}
	resp2 := &core.LLMResponse{Content: `
  thought:
  Okay, that tool wasn't available. I'll finish.

  action:
  Finish

  answer:
  Could not check calendar.
  `}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp,
		nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2,
		nil).Once()

	weatherTool := testutil.NewMockTool("weather_check")
	// Remove unused variable declaration for weatherSchema
	weatherTool.On("Metadata").Return(&core.ToolMetadata{
		Name: "weather_check", Description: "Check weather", InputSchema: models.InputSchema{ /*...*/ }, Capabilities: []string{"weather"},
	})

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, []core.Tool{weatherTool}, 5)
	react.SetLLM(mockLLM)

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "What's on my calendar today?"}
	outputs, err := react.Process(ctx, inputs)

	assert.NoError(t, err)
	require.NotNil(t, outputs)
	assert.Contains(t, outputs["answer"], "Could not check calendar")
	mockLLM.AssertExpectations(t)
}

func TestReAct_ToolValidationError(t *testing.T) {
	mockLLM := new(testutil.MockLLM)

	mockTool := testutil.NewMockTool("weather_check")
	mockTool.ExpectedCalls = nil // Clear all default expectations
	mockTool.On("Metadata").Return(&core.ToolMetadata{Name: "weather_check"})
	mockTool.On("Validate", mock.Anything).Return(errors.New(errors.ValidationFailed, "validation error: missing location"))
	// Fix: Provide arguments to core.NewSignature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, []core.Tool{mockTool}, 5)
	react.SetLLM(mockLLM)

	resp1 := &core.LLMResponse{Content: `thought: I need the weather

  action:
  <action><tool_name>weather_check</tool_name><arguments><arg key="location">invalid-location</arg></arguments></action>
  `}
	resp2 := &core.LLMResponse{Content: `thought: Validation failed, cannot proceed.

  action:
  Finish

  answer:
  Failed due to validation error.
  `}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1,
		nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2,
		nil).Once()

	mockTool.On("Metadata").Return(&core.ToolMetadata{Name: "weather_check"})
	mockTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(errors.New(errors.ValidationFailed, "validation error: missing location"))

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]interface{}{"question": "What is the weather like?"}
	outputs, err := react.Process(ctx, inputs)

	assert.NoError(t, err)
	require.NotNil(t, outputs)
	assert.Contains(t, outputs["answer"], "Failed due to validation error")
	mockLLM.AssertExpectations(t)
	mockTool.AssertExpectations(t)
}

func TestReAct_Clone(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	mockTool := testutil.NewMockTool("test_tool")
	mockTool.On("Metadata").Return(&core.ToolMetadata{Name: "test_tool"})
	originalReact := NewReAct(signature, []core.Tool{mockTool}, 5)
	clonedModule := originalReact.Clone()
	clonedReact, ok := clonedModule.(*ReAct)

	assert.True(t, ok)
	assert.Equal(t, originalReact.MaxIters, clonedReact.MaxIters)
	assert.Equal(t, len(originalReact.Tools), len(clonedReact.Tools))
	assert.Equal(t, len(originalReact.toolMap), len(clonedReact.toolMap))
	assert.Contains(t, clonedReact.toolMap, "test_tool")
	assert.NotSame(t, originalReact.Predict, clonedReact.Predict)
}

func TestReAct_WithDefaultOptions(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	reactModule := NewReAct(signature, []core.Tool{}, 5)
	result := reactModule.WithDefaultOptions(
		core.WithGenerateOptions(core.WithTemperature(0.7), core.WithMaxTokens(2000)),
	)
	assert.Same(t, reactModule, result)
}

func TestReAct_FormatToolResult(t *testing.T) {
	testCases := []struct {
		name     string
		input    core.ToolResult
		expected string
	}{
		{name: "String data", input: core.ToolResult{Data: "This is the result data"}, expected: "Observation:\nThis is the result data"},
		{name: "JSON byte slice data", input: core.ToolResult{Data: []byte(`{"key": "value", "number": 123}`)}, expected: "Observation:\n{\n  \"key\": \"value\",\n  \"number\": 123\n}"},
		{name: "Non-JSON byte slice data", input: core.ToolResult{Data: []byte("Just plain text")}, expected: "Observation:\nJust plain text"},

		{name: "Struct data", input: core.ToolResult{Data: struct {
			Name string `json:"name"`
			Age  int    `json:"age"`
		}{Name: "Test", Age: 30}}, expected: "Observation:\n{\n  \"name\": \"Test\",\n  \"age\": 30\n}"},
		{name: "Nil data", input: core.ToolResult{Data: nil}, expected: "Observation:\n<empty result>"},
		{name: "Long data truncated", input: core.ToolResult{Data: string(make([]byte, 2100))}, expected: "Observation:\n" + string(make([]byte, 1971)) + "\n... (truncated)"},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			observation := formatToolResult(tc.input)
			assert.Equal(t, tc.expected, observation)
		})
	}
}
