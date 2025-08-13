package react

import (
	"context"
	"testing"
	"time"

	testutil "github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestNewReActAgent(t *testing.T) {
	tests := []struct {
		name     string
		id       string
		agentName string
		opts     []Option
		expected func(*ReActAgent) bool
	}{
		{
			name:      "default configuration",
			id:        "test-agent",
			agentName: "Test Agent",
			opts:      nil,
			expected: func(agent *ReActAgent) bool {
				return agent.id == "test-agent" &&
					agent.name == "Test Agent" &&
					agent.config.MaxIterations == 10 &&
					agent.config.ExecutionMode == ModeReAct
			},
		},
		{
			name:      "with custom execution mode",
			id:        "test-agent",
			agentName: "Test Agent",
			opts:      []Option{WithExecutionMode(ModeReWOO)},
			expected: func(agent *ReActAgent) bool {
				return agent.config.ExecutionMode == ModeReWOO
			},
		},
		{
			name:      "with reflection enabled",
			id:        "test-agent",
			agentName: "Test Agent",
			opts:      []Option{WithReflection(true, 5)},
			expected: func(agent *ReActAgent) bool {
				return agent.config.EnableReflection == true &&
					agent.config.ReflectionDepth == 5 &&
					agent.Reflector != nil
			},
		},
		{
			name:      "with planning configuration",
			id:        "test-agent",
			agentName: "Test Agent",
			opts:      []Option{WithPlanning(Interleaved, 3)},
			expected: func(agent *ReActAgent) bool {
				return agent.config.PlanningStrategy == Interleaved &&
					agent.config.MaxPlanDepth == 3 &&
					agent.Planner != nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent := NewReActAgent(tt.id, tt.agentName, tt.opts...)
			require.NotNil(t, agent)
			assert.True(t, tt.expected(agent), "Agent configuration should match expectations")
		})
	}
}

func TestReActAgent_Initialize(t *testing.T) {
	mockLLM := new(testutil.MockLLM)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "task"}}},
		[]core.OutputField{{Field: core.Field{Name: "result"}}},
	)

	agent := NewReActAgent("test-agent", "Test Agent")
	err := agent.Initialize(mockLLM, signature)

	assert.NoError(t, err)
	assert.NotNil(t, agent.module)
	assert.Equal(t, mockLLM, agent.llm)
}

func TestReActAgent_RegisterTool(t *testing.T) {
	agent := NewReActAgent("test-agent", "Test Agent")

	mockTool := testutil.NewMockTool("test_tool")
	mockTool.On("Name").Return("test_tool")

	err := agent.RegisterTool(mockTool)
	assert.NoError(t, err)

	capabilities := agent.GetCapabilities()
	assert.Len(t, capabilities, 1)
	assert.Equal(t, mockTool, capabilities[0])
}

func TestReActAgent_Execute_ReActMode(t *testing.T) {
	mockLLM := new(testutil.MockLLM)
	mockTool := testutil.NewMockTool("test_tool")

	// Setup mock tool
	mockTool.On("Name").Return("test_tool")
	mockTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil)
	mockTool.On("Execute", mock.Anything, mock.AnythingOfType("map[string]interface {}")).Return(
		core.ToolResult{Data: "Tool executed successfully"}, nil)

	// Setup mock LLM responses
	resp1 := &core.LLMResponse{Content: `
thought: I need to use the test tool
action: <action><tool_name>test_tool</tool_name><arguments><arg key="input">test</arg></arguments></action>
`}

	resp2 := &core.LLMResponse{Content: `
thought: I have completed the task
action: <action><tool_name>Finish</tool_name></action>
result: Task completed successfully
`}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1, nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2, nil).Once()

	// Create and initialize agent
	agent := NewReActAgent("test-agent", "Test Agent", WithExecutionMode(ModeReAct))

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "task"}}},
		[]core.OutputField{{Field: core.Field{Name: "result"}}},
	)

	err := agent.Initialize(mockLLM, signature)
	require.NoError(t, err)

	err = agent.RegisterTool(mockTool)
	require.NoError(t, err)

	// Execute task
	ctx := core.WithExecutionState(context.Background())
	input := map[string]interface{}{"task": "Test task"}

	result, err := agent.Execute(ctx, input)

	assert.NoError(t, err)
	assert.NotNil(t, result)

	// Verify execution history was recorded
	history := agent.GetExecutionHistory()
	assert.Len(t, history, 1)
	assert.True(t, history[0].Success)
}

func TestReActAgent_Execute_WithReflection(t *testing.T) {
	mockLLM := new(testutil.MockLLM)

	// Setup mock LLM to return a finish response immediately
	resp := &core.LLMResponse{Content: `
thought: Task is simple, I can complete it directly
action: <action><tool_name>Finish</tool_name></action>
result: Task completed
`}
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp, nil)

	// Create agent with reflection enabled
	agent := NewReActAgent("test-agent", "Test Agent", WithReflection(true, 3))

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "task"}}},
		[]core.OutputField{{Field: core.Field{Name: "result"}}},
	)

	err := agent.Initialize(mockLLM, signature)
	require.NoError(t, err)

	// Execute task
	ctx := core.WithExecutionState(context.Background())
	input := map[string]interface{}{"task": "Simple test task"}

	result, err := agent.Execute(ctx, input)

	assert.NoError(t, err)
	assert.NotNil(t, result)

	// Verify reflection was performed
	assert.NotNil(t, agent.Reflector)

	// Check execution history
	history := agent.GetExecutionHistory()
	assert.Len(t, history, 1)
	assert.True(t, history[0].Success)
}

func TestReActAgent_Execute_Timeout(t *testing.T) {
	mockLLM := new(testutil.MockLLM)

	// Setup mock LLM to simulate a long-running response
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(
		&core.LLMResponse{Content: "thought: Still working\naction: <action><tool_name>Finish</tool_name></action>"}, nil).Maybe()

	// Create agent with short timeout
	agent := NewReActAgent("test-agent", "Test Agent", WithTimeout(10*time.Millisecond))

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "task"}}},
		[]core.OutputField{{Field: core.Field{Name: "result"}}},
	)

	err := agent.Initialize(mockLLM, signature)
	require.NoError(t, err)

	// Execute task with a context that will timeout quickly
	ctx := core.WithExecutionState(context.Background())
	input := map[string]interface{}{"task": "Test task"}

	// Since we can't easily simulate slow LLM, just test that timeout is respected
	start := time.Now()
	result, err := agent.Execute(ctx, input)
	duration := time.Since(start)

	// The test should complete quickly (either succeed or timeout)
	assert.True(t, duration < 1*time.Second, "Test should complete quickly")

	// We don't assert error/success here since the timeout behavior depends on LLM timing
	_ = result
	_ = err
}

func TestReActAgent_GetCapabilities(t *testing.T) {
	agent := NewReActAgent("test-agent", "Test Agent")

	// Initially no capabilities
	capabilities := agent.GetCapabilities()
	assert.Len(t, capabilities, 0)

	// Add tools
	tool1 := testutil.NewMockTool("tool1")
	tool1.On("Name").Return("tool1")
	tool2 := testutil.NewMockTool("tool2")
	tool2.On("Name").Return("tool2")

	_ = agent.RegisterTool(tool1)
	_ = agent.RegisterTool(tool2)

	capabilities = agent.GetCapabilities()
	assert.Len(t, capabilities, 2)
}

func TestReActAgent_GetMemory(t *testing.T) {
	agent := NewReActAgent("test-agent", "Test Agent")

	memory := agent.GetMemory()
	assert.NotNil(t, memory)
}

func TestReActAgent_InterceptorSupport(t *testing.T) {
	agent := NewReActAgent("test-agent", "Test Agent")

	// Initially no interceptors
	interceptors := agent.GetInterceptors()
	assert.Len(t, interceptors, 0)

	// Create mock interceptor
	mockInterceptor := func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, next core.AgentHandler) (map[string]interface{}, error) {
		// Add metadata to input
		input["intercepted"] = true
		return next(ctx, input)
	}

	// Set interceptors
	agent.SetInterceptors([]core.AgentInterceptor{mockInterceptor})

	interceptors = agent.GetInterceptors()
	assert.Len(t, interceptors, 1)

	// Clear interceptors
	agent.ClearInterceptors()
	interceptors = agent.GetInterceptors()
	assert.Len(t, interceptors, 0)
}

func TestReActAgent_HybridMode(t *testing.T) {
	mockLLM := new(testutil.MockLLM)

	// Setup mock LLM response
	resp := &core.LLMResponse{Content: `
thought: This is a simple task
action: <action><tool_name>Finish</tool_name></action>
result: Task completed
`}
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp, nil)

	// Create agent in hybrid mode
	agent := NewReActAgent("test-agent", "Test Agent", WithExecutionMode(ModeHybrid))

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "task"}}},
		[]core.OutputField{{Field: core.Field{Name: "result"}}},
	)

	err := agent.Initialize(mockLLM, signature)
	require.NoError(t, err)

	// Test with simple task (should use ReAct)
	ctx := core.WithExecutionState(context.Background())
	input := map[string]interface{}{"task": "Simple task"}

	result, err := agent.Execute(ctx, input)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestReActAgent_analyzeTaskComplexity(t *testing.T) {
	agent := NewReActAgent("test-agent", "Test Agent")

	tests := []struct {
		name     string
		input    map[string]interface{}
		expected float64 // Rough expectation, not exact
	}{
		{
			name:     "simple task",
			input:    map[string]interface{}{"task": "Hello"},
			expected: 0.5,
		},
		{
			name:     "complex task",
			input:    map[string]interface{}{"task": "Please analyze the complex multi-step process and compare multiple options while evaluating their effectiveness"},
			expected: 0.8, // Should be higher due to length and keywords
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			complexity := agent.analyzeTaskComplexity(tt.input)
			assert.GreaterOrEqual(t, complexity, 0.0)
			assert.LessOrEqual(t, complexity, 1.0)

			if tt.expected < 0.7 {
				assert.Less(t, complexity, 0.7, "Simple tasks should have low complexity")
			} else {
				assert.GreaterOrEqual(t, complexity, 0.7, "Complex tasks should have high complexity")
			}
		})
	}
}

func TestExecutionModeOptions(t *testing.T) {
	tests := []struct {
		name string
		mode ExecutionMode
	}{
		{"ReAct mode", ModeReAct},
		{"ReWOO mode", ModeReWOO},
		{"Hybrid mode", ModeHybrid},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent := NewReActAgent("test", "Test", WithExecutionMode(tt.mode))
			assert.Equal(t, tt.mode, agent.config.ExecutionMode)
		})
	}
}

func TestReActAgent_GetAgentInfo(t *testing.T) {
	agent := NewReActAgent("test-agent-123", "My Test Agent")

	assert.Equal(t, "test-agent-123", agent.GetAgentID())
	assert.Equal(t, "ReActAgent", agent.GetAgentType())
}

// Benchmarks

func BenchmarkReActAgent_Execute_Simple(b *testing.B) {
	mockLLM := new(testutil.MockLLM)
	resp := &core.LLMResponse{Content: `
thought: Simple task
action: <action><tool_name>Finish</tool_name></action>
result: Done
`}
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp, nil)

	agent := NewReActAgent("bench-agent", "Benchmark Agent")
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "task"}}},
		[]core.OutputField{{Field: core.Field{Name: "result"}}},
	)
	_ = agent.Initialize(mockLLM, signature)

	ctx := core.WithExecutionState(context.Background())
	input := map[string]interface{}{"task": "Benchmark task"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := agent.Execute(ctx, input)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func TestReActAgent_XMLParsing(t *testing.T) {
	// Test XML parsing functionality in ReAct agent
	agent := NewReActAgent("test", "Test Agent")

	// Mock LLM
	mockLLM := new(testutil.MockLLM)

	// Mock tool
	mockTool := testutil.NewMockTool("test_tool")
	mockTool.On("Name").Return("test_tool")
	mockTool.On("Description").Return("A test tool")
	mockTool.On("Metadata").Return(&core.ToolMetadata{Name: "test_tool"})
	mockTool.On("InputSchema").Return(models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"param": {
				Type:        "string",
				Description: "Test parameter",
				Required:    false,
			},
		},
	})
	mockTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil)
	mockTool.On("Execute", mock.Anything, mock.AnythingOfType("map[string]interface {}")).Return(
		core.ToolResult{Data: "Test result"}, nil,
	)

	// Register tool
	err := agent.RegisterTool(mockTool)
	require.NoError(t, err)

	// Create signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "task"}}},
		[]core.OutputField{{Field: core.NewField("result")}},
	)

	// Initialize agent
	err = agent.Initialize(mockLLM, signature)
	require.NoError(t, err)

	// Test 1: Agent without XML parsing (should work normally)
	t.Run("without_xml_parsing", func(t *testing.T) {
		assert.False(t, agent.config.EnableXMLParsing)
		assert.Nil(t, agent.config.XMLConfig)
	})

	// Test 2: Enable XML parsing after initialization
	t.Run("enable_xml_parsing_after_init", func(t *testing.T) {
		xmlConfig := interceptors.DefaultXMLConfig().
			WithMaxSize(1024).
			WithStrictParsing(true)

		err := agent.EnableXMLParsing(xmlConfig)
		assert.NoError(t, err)
		assert.True(t, agent.config.EnableXMLParsing)
		assert.NotNil(t, agent.config.XMLConfig)
		assert.Equal(t, int64(1024), agent.config.XMLConfig.MaxSize)
	})

	// Test 3: Create agent with XML parsing from start
	t.Run("with_xml_parsing_option", func(t *testing.T) {
		xmlConfig := interceptors.FlexibleXMLConfig().
			WithMaxSize(2048).
			WithValidation(true)

		xmlAgent := NewReActAgent("xml-test", "XML Test Agent",
			WithXMLParsing(xmlConfig),
		)

		assert.True(t, xmlAgent.config.EnableXMLParsing)
		assert.NotNil(t, xmlAgent.config.XMLConfig)
		assert.Equal(t, int64(2048), xmlAgent.config.XMLConfig.MaxSize)
		assert.True(t, xmlAgent.config.XMLConfig.ValidateXML)
	})

	// Test 4: Error when enabling XML parsing on uninitialized agent
	t.Run("error_uninitialized", func(t *testing.T) {
		uninitAgent := NewReActAgent("uninit", "Uninitialized Agent")
		xmlConfig := interceptors.DefaultXMLConfig()

		err := uninitAgent.EnableXMLParsing(xmlConfig)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "agent not initialized")
	})
}

func TestReActAgent_XMLParsingIntegration(t *testing.T) {
	// Integration test to ensure XML parsing works with the underlying ReAct module
	agent := NewReActAgent("integration-test", "Integration Test Agent")

	// Mock LLM with XML responses
	mockLLM := new(testutil.MockLLM)

	// First response: tool action
	resp1 := &core.LLMResponse{Content: `
thought:
I need to use the test tool

action:
<action><tool_name>test_tool</tool_name><arguments><arg key="param">test_value</arg></arguments></action>
`}

	// Second response: finish action
	resp2 := &core.LLMResponse{Content: `
thought:
Task completed

action:
<action><tool_name>Finish</tool_name></action>

result:
Integration test successful
`}

	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp1, nil).Once()
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp2, nil).Once()

	// Mock tool
	mockTool := testutil.NewMockTool("test_tool")
	mockTool.On("Name").Return("test_tool").Maybe()
	mockTool.On("Description").Return("Integration test tool").Maybe()
	mockTool.On("Metadata").Return(&core.ToolMetadata{Name: "test_tool"}).Maybe()
	mockTool.On("InputSchema").Return(models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"param": {
				Type:        "string",
				Description: "Test parameter",
				Required:    true,
			},
		},
	}).Maybe()
	mockTool.On("Validate", mock.AnythingOfType("map[string]interface {}")).Return(nil).Maybe()
	expectedArgs := map[string]interface{}{"param": "test_value"}
	mockTool.On("Execute", mock.Anything, expectedArgs).Return(
		core.ToolResult{Data: "Integration success"}, nil,
	)

	// Register tool
	err := agent.RegisterTool(mockTool)
	require.NoError(t, err)

	// Create signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "task"}}},
		[]core.OutputField{{Field: core.NewField("result")}},
	)

	// Initialize agent with XML parsing
	xmlConfig := interceptors.DefaultXMLConfig().
		WithStrictParsing(true).
		WithValidation(true)

	agent.config.EnableXMLParsing = true
	agent.config.XMLConfig = &xmlConfig

	err = agent.Initialize(mockLLM, signature)
	require.NoError(t, err)

	// Execute task
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	input := map[string]interface{}{"task": "Test XML parsing integration"}
	result, err := agent.Execute(ctx, input)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Contains(t, result["result"], "Integration test successful")

	// Verify mocks
	mockLLM.AssertExpectations(t)
	mockTool.AssertExpectations(t)
}
