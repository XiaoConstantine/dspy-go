package react

import (
	"context"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestReActAgent_IntegrationTestsClean(t *testing.T) {
	t.Run("Complete Agent Workflow", func(t *testing.T) {
		// Create agent with all features enabled
		agent := NewReActAgent("integration-agent", "Integration Test Agent",
			WithExecutionMode(ModeHybrid),
			WithReflection(true, 3),
			WithPlanning(DecompositionFirst, 2),
			WithMemoryOptimization(24*time.Hour, 0.6),
			WithTimeout(30*time.Second),
		)

		// Register test tools using testutil mocks
		mathTool := testutil.NewMockCoreTool("math_calculator", "Performs math calculations", func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
			return core.ToolResult{Data: "42"}, nil
		})
		err := agent.RegisterTool(mathTool)
		require.NoError(t, err)

		// Initialize with mock LLM from testutil
		mockLLM := new(testutil.MockLLM)
		mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.Anything).Return(
			&core.LLMResponse{Content: `thought: I need to calculate this
action: <action><tool_name>Finish</tool_name></action>`}, nil)

		signature := core.NewSignature(
			[]core.InputField{{Field: core.Field{Name: "task"}}},
			[]core.OutputField{{Field: core.Field{Name: "result"}}},
		)

		err = agent.Initialize(mockLLM, signature)
		require.NoError(t, err)

		ctx := context.Background()

		// Test simple task execution
		simpleInput := map[string]interface{}{"task": "calculate 15 + 27"}
		result, err := agent.Execute(ctx, simpleInput)
		require.NoError(t, err)
		assert.NotNil(t, result)

		// Verify execution history
		history := agent.GetExecutionHistory()
		assert.Equal(t, 1, len(history))
		assert.True(t, history[0].Success)

		// Test agent properties
		assert.Equal(t, "integration-agent", agent.GetAgentID())
		assert.Equal(t, "ReActAgent", agent.GetAgentType())
		assert.NotNil(t, agent.GetCapabilities())
		assert.NotNil(t, agent.GetMemory())
	})

	t.Run("Memory System Integration", func(t *testing.T) {
		agent := NewReActAgent("memory-agent", "Memory Test Agent",
			WithMemoryOptimization(1*time.Hour, 0.5),
		)

		mockLLM := new(testutil.MockLLM)
		mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.Anything).Return(
			&core.LLMResponse{Content: `thought: Processing task
action: <action><tool_name>Finish</tool_name></action>`}, nil)

		signature := core.NewSignature(
			[]core.InputField{{Field: core.Field{Name: "task"}}},
			[]core.OutputField{{Field: core.Field{Name: "result"}}},
		)

		err := agent.Initialize(mockLLM, signature)
		require.NoError(t, err)

		ctx := context.Background()
		input := map[string]interface{}{"task": "test memory"}
		_, err = agent.Execute(ctx, input)
		require.NoError(t, err)

		// Check memory statistics
		if agent.Optimizer != nil {
			stats := agent.Optimizer.GetStatistics()
			assert.NotNil(t, stats)
			assert.Contains(t, stats, "total_items")
		}
	})

	t.Run("Reflection System", func(t *testing.T) {
		agent := NewReActAgent("reflection-agent", "Reflection Test Agent",
			WithReflection(true, 5),
		)

		mockLLM := new(testutil.MockLLM)
		mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.Anything).Return(
			&core.LLMResponse{Content: `thought: Running reflection test
action: <action><tool_name>Finish</tool_name></action>`}, nil)

		signature := core.NewSignature(
			[]core.InputField{{Field: core.Field{Name: "task"}}},
			[]core.OutputField{{Field: core.Field{Name: "result"}}},
		)

		err := agent.Initialize(mockLLM, signature)
		require.NoError(t, err)

		ctx := context.Background()
		input := map[string]interface{}{"task": "test reflection"}
		_, err = agent.Execute(ctx, input)
		require.NoError(t, err)

		// Check reflection capabilities
		if agent.Reflector != nil {
			metrics := agent.Reflector.GetMetrics()
			assert.NotNil(t, metrics)
			assert.Greater(t, metrics.TotalExecutions, 0)
		}
	})

	t.Run("Planning System", func(t *testing.T) {
		agent := NewReActAgent("planning-agent", "Planning Test Agent",
			WithPlanning(DecompositionFirst, 3),
		)

		mockLLM := new(testutil.MockLLM)

		signature := core.NewSignature(
			[]core.InputField{{Field: core.Field{Name: "task"}}},
			[]core.OutputField{{Field: core.Field{Name: "result"}}},
		)

		err := agent.Initialize(mockLLM, signature)
		require.NoError(t, err)

		// Test planning capabilities
		if agent.Planner != nil {
			ctx := context.Background()
			task := "analyze data and create report"
			input := map[string]interface{}{"task": task}
			tools := []core.Tool{}

			plan, err := agent.Planner.CreatePlan(ctx, input, tools)
			require.NoError(t, err)
			assert.NotNil(t, plan)
			assert.Equal(t, task, plan.Goal)

			// Validate the plan
			err = agent.Planner.ValidatePlan(plan)
			assert.NoError(t, err)

			// Get plan metrics
			metrics := agent.Planner.GetPlanMetrics(plan)
			assert.NotNil(t, metrics)
			assert.Contains(t, metrics, "total_steps")
		}
	})

	t.Run("Tool Integration", func(t *testing.T) {
		agent := NewReActAgent("tool-agent", "Tool Test Agent")

		// Test tool registration
		testTool := testutil.NewMockCoreTool("test_tool", "Test tool", func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
			return core.ToolResult{Data: "test result"}, nil
		})
		err := agent.RegisterTool(testTool)
		require.NoError(t, err)

		capabilities := agent.GetCapabilities()
		assert.Greater(t, len(capabilities), 0)

		// Test plan step execution
		mockLLM := new(testutil.MockLLM)
		signature := core.NewSignature(
			[]core.InputField{{Field: core.Field{Name: "input"}}},
			[]core.OutputField{{Field: core.Field{Name: "output"}}},
		)
		err = agent.Initialize(mockLLM, signature)
		require.NoError(t, err)

		ctx := context.Background()
		step := PlanStep{
			ID:        "step1",
			Tool:      "test_tool",
			Arguments: map[string]interface{}{"arg1": "value1"},
			Critical:  false,
			Timeout:   5 * time.Second,
		}

		result, err := agent.executePlanStep(ctx, step, map[string]interface{}{})
		require.NoError(t, err)
		assert.NotNil(t, result)
	})
}

func TestUtilityIntegration(t *testing.T) {
	t.Run("Tool Registry", func(t *testing.T) {
		registry := tools.NewInMemoryToolRegistry()

		mathTool := testutil.NewMockCoreTool("math_calculator", "Performs calculations", nil)
		err := registry.Register(mathTool)
		require.NoError(t, err)

		retrieved, err := registry.Get("math_calculator")
		require.NoError(t, err)
		assert.Equal(t, mathTool.Name(), retrieved.Name())
	})

	t.Run("Mock LLM", func(t *testing.T) {
		mockLLM := new(testutil.MockLLM)
		mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.Anything).Return(
			&core.LLMResponse{Content: "test response"}, nil)

		ctx := context.Background()
		response, err := mockLLM.Generate(ctx, "test prompt")
		require.NoError(t, err)
		assert.Equal(t, "test response", response.Content)

		mockLLM.AssertExpectations(t)
	})
}
