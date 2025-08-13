package modules

import (
	"context"
	"testing"

	testutil "github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"

	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestReAct(t *testing.T) {
	mockLLM := new(testutil.MockLLM)
	// Removed unused schema definition
	mockTool := testutil.NewMockTool("test_tool") // Use the actual tool name
	mockTool.On("Name").Return("test_tool")
	mockTool.On("Description").Return("A test tool for ReAct testing")
	mockTool.On("InputSchema").Return(models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"question": {
				Type: "string",
				Description: "The question to ask the tool",
				Required: true,
			},
		},
	})
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
  <action>Finish</action>

  answer:
  42
  `}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1,
		nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2,
		nil).Once()

	// Create registry and add mock tool
	registry := tools.NewInMemoryToolRegistry()
	err := registry.Register(mockTool)
	require.NoError(t, err)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	// Removed old tool slice initialization
	react := NewReAct(signature, registry, 2)
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
	assert.Equal(t, "Predict (Predict)", spans[1].Operation)
	assert.Equal(t, "executeToolByName", spans[2].Operation)
	assert.Equal(t, "Predict (Predict)", spans[3].Operation)
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
  <action>Finish</action>

  answer:
  I apologize, but I wasn't able to get the weather information due to a technical issue.
  `}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1,
		nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2,
		nil).Once()

	mockWeatherTool := testutil.NewMockTool("weather_check")
	mockWeatherTool.On("Name").Return("weather_check")
	mockWeatherTool.On("Description").Return("A tool to check weather conditions")
	mockWeatherTool.On("InputSchema").Return(models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"location": {
				Type: "string",
				Description: "The location to check weather for",
				Required: true,
			},
		},
	})
	mockWeatherTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil)
	expectedArgs := map[string]interface{}{"location": "nearby"}
	mockWeatherTool.On("Execute", mock.Anything, expectedArgs).Return(core.ToolResult{}, errors.New(errors.LLMGenerationFailed,
		"weather service down"))

	// Create registry and add mock tool
	registry := tools.NewInMemoryToolRegistry()
	err := registry.Register(mockWeatherTool)
	require.NoError(t, err)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, registry, 5)
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
	dbTool.On("Name").Return("database_query")
	dbTool.On("Description").Return("A tool to query the database")
	dbTool.On("InputSchema").Return(models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"query": {
				Type: "string",
				Description: "SQL query to execute",
				Required: true,
			},
		},
	})
	dbTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil)
	dbExpectedArgs := map[string]interface{}{"query": "SELECT * FROM\ndata"}
	dbTool.On("Execute", mock.Anything, dbExpectedArgs).Return(core.ToolResult{Data: "Database results"}, nil)

	profileTool := testutil.NewMockTool("user_profile")
	profileTool.On("Name").Return("user_profile")
	profileTool.On("Description").Return("A tool to get user profile information")
	profileTool.On("InputSchema").Return(models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"user_id": {
				Type: "string",
				Description: "The user ID to look up",
				Required: true,
			},
		},
	})
	profileTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil)
	profileExpectedArgs := map[string]interface{}{"user_id": "123"}
	profileTool.On("Execute", mock.Anything, profileExpectedArgs).Return(core.ToolResult{Data: "User profile data"}, nil)

	// Create registry and add mock tools
	registry := tools.NewInMemoryToolRegistry()
	err := registry.Register(dbTool)
	require.NoError(t, err)
	err = registry.Register(profileTool)
	require.NoError(t, err)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	// tools := []core.Tool{dbTool, profileTool} // Old way
	react := NewReAct(signature, registry, 2)
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
  <action>Finish</action>

  answer:
  Failed due to invalid action format.
  `}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1,
		nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2,
		nil).Once()

	// Create registry, but don't add the 'unknown_tool'
	registry := tools.NewInMemoryToolRegistry()

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, registry, 5)
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
  <action>Finish</action>

  answer:
  Could not check calendar.
  `}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp,
		nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2,
		nil).Once()

	weatherTool := testutil.NewMockTool("weather_check")
	// Remove unused variable declaration for weatherSchema
	weatherTool.On("Name").Return("weather_check")
	weatherTool.On("Metadata").Return(&core.ToolMetadata{
		Name: "weather_check", Description: "Check weather", InputSchema: models.InputSchema{ /*...*/ }, Capabilities: []string{"weather"},
	})

	// Create registry, but don't add the 'unknown_tool'
	registry := tools.NewInMemoryToolRegistry()

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	react := NewReAct(signature, registry, 5)
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

	mockValidationTool := testutil.NewMockTool("weather_check")
	mockValidationTool.On("Name").Return("weather_check")
	mockValidationTool.On("Description").Return("A weather checking tool")
	mockValidationTool.On("InputSchema").Return(models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"location": {
				Type: "string",
				Description: "Location to check weather for",
				Required: true,
			},
		},
	})
	// Removed commented-out Metadata expectation
	// Add Validate expectation before register
	mockValidationTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(errors.New(errors.ValidationFailed, "validation error"))

	// Create registry and add mock tool
	registry := tools.NewInMemoryToolRegistry()
	err := registry.Register(mockValidationTool)
	require.NoError(t, err)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "data"}}},
		[]core.OutputField{{Field: core.NewField("status")}},
	)
	react := NewReAct(signature, registry, 5)
	react.SetLLM(mockLLM)

	resp1 := &core.LLMResponse{Content: `thought: I need the weather

  action:
  <action><tool_name>weather_check</tool_name><arguments><arg key="location">invalid-location</arg></arguments></action>
  `}
	resp2 := &core.LLMResponse{Content: `thought: Validation failed, cannot proceed.

  action:
  <action>Finish</action>

  answer:
  Failed due to validation error.
  `}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1,
		nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2,
		nil).Once()

	// Ensure no mockValidationTool.On(...) calls happen after Register

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]interface{}{"data": "What is the weather like?"}
	outputs, err := react.Process(ctx, inputs)

	assert.NoError(t, err)
	require.NotNil(t, outputs)
	assert.Contains(t, outputs["answer"], "Failed due to validation error")
	mockLLM.AssertExpectations(t)
	mockValidationTool.AssertExpectations(t)
}

func TestReAct_Clone(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	mockTool := testutil.NewMockTool("test_tool")
	mockTool.On("Name").Return("test_tool")
	mockTool.On("Description").Return("A test tool for cloning")
	mockTool.On("InputSchema").Return(models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"input": {
				Type: "string",
				Description: "Test input parameter",
				Required: false,
			},
		},
	})
	mockTool.On("Metadata").Return(&core.ToolMetadata{Name: "test_tool"})
	// Create registry for original
	registry := tools.NewInMemoryToolRegistry()
	err := registry.Register(mockTool)
	require.NoError(t, err)
	originalReact := NewReAct(signature, registry, 5)
	clonedModule := originalReact.Clone()
	clonedReact, ok := clonedModule.(*ReAct)

	assert.True(t, ok)
	assert.Equal(t, originalReact.MaxIters, clonedReact.MaxIters)
	// Assert based on registry content
	originalTools := originalReact.Registry.List()
	clonedTools := clonedReact.Registry.List()
	assert.Equal(t, len(originalTools), len(clonedTools))
	// Check if tool name exists in the cloned list
	found := false
	for _, tool := range clonedTools {
		if tool.Metadata().Name == "test_tool" {
			found = true
			break
		}
	}
	assert.True(t, found, "Tool 'test_tool' not found in cloned registry")
	assert.NotSame(t, originalReact.Predict, clonedReact.Predict)
	// Removed commented-out assert.Same check
}

func TestReAct_WithDefaultOptions(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	// Create empty registry
	emptyRegistry := tools.NewInMemoryToolRegistry()
	reactModule := NewReAct(signature, emptyRegistry, 5)
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

func TestReAct_WithXMLInterceptors(t *testing.T) {
	mockLLM := new(testutil.MockLLM)
	mockTool := testutil.NewMockTool("weather_tool")
	mockTool.On("Name").Return("weather_tool")
	mockTool.On("Description").Return("A weather tool with XML parsing")
	mockTool.On("InputSchema").Return(models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"location": {
				Type:        "string",
				Description: "The location to check weather for",
				Required:    true,
			},
		},
	})
	mockTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil)
	expectedArgs := map[string]interface{}{"location": "New York"}
	mockTool.On("Execute", mock.Anything, expectedArgs).Return(
		core.ToolResult{Data: "Sunny, 75°F"}, nil,
	)

	resp1 := &core.LLMResponse{Content: `
thought:
I need to check the weather

action:
<action><tool_name>weather_tool</tool_name><arguments><arg key="location">New York</arg></arguments></action>
`}
	resp2 := &core.LLMResponse{Content: `
thought:
I have the weather information

action:
<action><tool_name>Finish</tool_name></action>

answer:
The weather in New York is sunny, 75°F
`}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1, nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2, nil).Once()

	// Create registry and add mock tool
	registry := tools.NewInMemoryToolRegistry()
	err := registry.Register(mockTool)
	require.NoError(t, err)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create ReAct with XML interceptor configuration
	react := NewReAct(signature, registry, 2)
	xmlConfig := interceptors.DefaultXMLConfig().
		WithMaxSize(1024).
		WithStrictParsing(true).
		WithFallback(true)
	react.WithXMLParsing(xmlConfig)
	react.SetLLM(mockLLM)

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "What's the weather in New York?"}
	outputs, err := react.Process(ctx, inputs)

	assert.NoError(t, err)
	require.NotNil(t, outputs)
	assert.Contains(t, outputs["answer"], "sunny")
	mockLLM.AssertExpectations(t)
	mockTool.AssertExpectations(t)
}

func TestReAct_XMLInterceptors_InvalidAction(t *testing.T) {
	mockLLM := new(testutil.MockLLM)

	// Response with malformed XML that should trigger interceptor error handling
	resp1 := &core.LLMResponse{Content: `
thought:
I need to do something

action:
<action><tool_name>some_tool</tool_name><arguments><arg key="param">value</arg></arguments>
`} // Missing closing action tag
	resp2 := &core.LLMResponse{Content: `
thought:
The previous action failed, I'll finish

action:
<action><tool_name>Finish</tool_name></action>

answer:
Failed due to malformed action
`}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1, nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2, nil).Once()

	// Create empty registry
	registry := tools.NewInMemoryToolRegistry()

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create ReAct with XML interceptor configuration that has strict validation
	react := NewReAct(signature, registry, 3)
	xmlConfig := interceptors.DefaultXMLConfig().
		WithStrictParsing(true).
		WithFallback(false). // No fallback to test error handling
		WithValidation(true)
	react.WithXMLParsing(xmlConfig)
	react.SetLLM(mockLLM)

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "Test malformed XML handling"}
	outputs, err := react.Process(ctx, inputs)

	assert.NoError(t, err)
	require.NotNil(t, outputs)
	assert.Contains(t, outputs["answer"], "Failed")
	mockLLM.AssertExpectations(t)
}

func TestReAct_XMLInterceptors_SizeLimit(t *testing.T) {
	mockLLM := new(testutil.MockLLM)

	// Create a very large action string that exceeds size limit
	largeAction := "<action><tool_name>test_tool</tool_name><arguments><arg key=\"data\">" +
		string(make([]byte, 2000)) + // 2KB of data
		"</arg></arguments></action>"

	resp1 := &core.LLMResponse{Content: `
thought:
I need to send large data

action:
` + largeAction}
	resp2 := &core.LLMResponse{Content: `
thought:
The action was too large, I'll finish

action:
<action><tool_name>Finish</tool_name></action>

answer:
Action size exceeded limit
`}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1, nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2, nil).Once()

	// Create empty registry
	registry := tools.NewInMemoryToolRegistry()

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create ReAct with small size limit
	react := NewReAct(signature, registry, 3)
	xmlConfig := interceptors.DefaultXMLConfig().
		WithMaxSize(1024). // 1KB limit
		WithStrictParsing(true)
	react.WithXMLParsing(xmlConfig)
	react.SetLLM(mockLLM)

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "Test size limit"}
	outputs, err := react.Process(ctx, inputs)

	assert.NoError(t, err)
	require.NotNil(t, outputs)
	assert.Contains(t, outputs["answer"], "exceeded limit")
	mockLLM.AssertExpectations(t)
}

func TestReAct_BackwardCompatibility(t *testing.T) {
	// This test ensures that ReAct without XML interceptors still works exactly as before
	mockLLM := new(testutil.MockLLM)
	mockTool := testutil.NewMockTool("test_tool")
	mockTool.On("Name").Return("test_tool")
	mockTool.On("Description").Return("A test tool for backward compatibility")
	mockTool.On("InputSchema").Return(models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"query": {
				Type:        "string",
				Description: "The query parameter",
				Required:    true,
			},
		},
	})
	mockTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil)
	expectedArgs := map[string]interface{}{"query": "test query"}
	mockTool.On("Execute", mock.Anything, expectedArgs).Return(
		core.ToolResult{Data: "Backward compatibility works"}, nil,
	)

	resp1 := &core.LLMResponse{Content: `
thought:
I should use the tool

action:
<action><tool_name>test_tool</tool_name><arguments><arg key="query">test query</arg></arguments></action>
`}
	resp2 := &core.LLMResponse{Content: `
thought:
I have the result

action:
<action><tool_name>Finish</tool_name></action>

answer:
Success
`}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1, nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2, nil).Once()

	// Create registry and add mock tool
	registry := tools.NewInMemoryToolRegistry()
	err := registry.Register(mockTool)
	require.NoError(t, err)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create ReAct WITHOUT XML interceptors (should use hardcoded parsing)
	react := NewReAct(signature, registry, 2)
	// Intentionally NOT calling WithXMLParsing
	react.SetLLM(mockLLM)

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "Test backward compatibility"}
	outputs, err := react.Process(ctx, inputs)

	assert.NoError(t, err)
	require.NotNil(t, outputs)
	assert.Equal(t, "Success", outputs["answer"])
	mockLLM.AssertExpectations(t)
	mockTool.AssertExpectations(t)
}
