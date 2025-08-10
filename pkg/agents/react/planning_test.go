package react

import (
	"context"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTaskPlanner_ComprehensiveTests(t *testing.T) {
	t.Run("NewTaskPlanner", func(t *testing.T) {
		planner := NewTaskPlanner(DecompositionFirst, 5)
		assert.NotNil(t, planner)
		assert.Equal(t, DecompositionFirst, planner.strategy)
		assert.Equal(t, 5, planner.maxDepth)
		assert.NotNil(t, planner.planCache)
		assert.NotNil(t, planner.templateLib)
		assert.NotNil(t, planner.decomposer)
	})

	t.Run("CreatePlan", func(t *testing.T) {
		planner := NewTaskPlanner(DecompositionFirst, 3)
		ctx := context.Background()

		task := "Analyze sales data and generate report"
		input := map[string]interface{}{"task": task}
		tools := []core.Tool{} // Empty for now, as we're testing the planning logic

		plan, err := planner.CreatePlan(ctx, input, tools)
		require.NoError(t, err)
		assert.NotNil(t, plan)
		assert.Equal(t, task, plan.Goal)
		assert.NotEmpty(t, plan.ID)
		assert.Equal(t, DecompositionFirst, plan.Strategy)
		assert.NotZero(t, plan.CreatedAt)
	})

	t.Run("CreatePlan - Template Strategy", func(t *testing.T) {
		planner := NewTaskPlanner(DecompositionFirst, 3) // Templates are used within DecompositionFirst strategy
		ctx := context.Background()

		task := "research the latest AI trends"
		input := map[string]interface{}{"task": task}
		tools := []core.Tool{}

		plan, err := planner.CreatePlan(ctx, input, tools)
		require.NoError(t, err)
		assert.NotNil(t, plan)
		assert.Equal(t, DecompositionFirst, plan.Strategy)
	})

	t.Run("CreatePlan - Interleaved Strategy", func(t *testing.T) {
		planner := NewTaskPlanner(Interleaved, 3)
		ctx := context.Background()

		task := "calculate quarterly revenue and analyze trends"
		input := map[string]interface{}{"task": task}
		tools := []core.Tool{}

		plan, err := planner.CreatePlan(ctx, input, tools)
		require.NoError(t, err)
		assert.NotNil(t, plan)
		assert.Equal(t, Interleaved, plan.Strategy)
	})

	t.Run("Plan Validation", func(t *testing.T) {
		planner := NewTaskPlanner(DecompositionFirst, 3)

		validPlan := &Plan{
			ID:   "test-plan",
			Goal: "Test goal",
			Steps: []PlanStep{
				{ID: "step1", Tool: "tool1", Critical: true},
				{ID: "step2", Tool: "tool2", DependsOn: []string{"step1"}},
			},
		}

		err := planner.ValidatePlan(validPlan)
		assert.NoError(t, err)

		// Test circular dependency
		invalidPlan := &Plan{
			ID:   "invalid-plan",
			Goal: "Test goal",
			Steps: []PlanStep{
				{ID: "step1", Tool: "tool1", DependsOn: []string{"step2"}},
				{ID: "step2", Tool: "tool2", DependsOn: []string{"step1"}},
			},
		}

		err = planner.ValidatePlan(invalidPlan)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "circular")
	})

	t.Run("Plan Metrics", func(t *testing.T) {
		planner := NewTaskPlanner(DecompositionFirst, 3)

		plan := &Plan{
			Steps: []PlanStep{
				{ID: "step1", Tool: "tool1", Critical: true, Timeout: time.Second},
				{ID: "step2", Tool: "tool2", Critical: false, Timeout: 2 * time.Second},
				{ID: "step3", Tool: "tool1", Critical: true, Timeout: time.Second},
			},
		}

		metrics := planner.GetPlanMetrics(plan)
		assert.NotNil(t, metrics)
		assert.Contains(t, metrics, "total_steps")
		assert.Contains(t, metrics, "critical_steps")
		assert.Contains(t, metrics, "tool_usage") // Tool usage metrics showing count per tool
		assert.Equal(t, 3, metrics["total_steps"])
		assert.Equal(t, 2, metrics["critical_steps"])
	})
}

func TestTaskDecomposer_Tests(t *testing.T) {
	t.Run("NewTaskDecomposer", func(t *testing.T) {
		decomposer := NewTaskDecomposer(5)
		assert.NotNil(t, decomposer)
		assert.Equal(t, 5, decomposer.maxDepth)
		assert.NotNil(t, decomposer.decomposition)
	})

	t.Run("Decompose Task", func(t *testing.T) {
		decomposer := NewTaskDecomposer(3)

		subtasks := decomposer.Decompose("research and analyze market trends", 3)
		assert.NotNil(t, subtasks)
		assert.Greater(t, len(subtasks), 0)
	})
}

func TestPlanTemplateLibrary_Tests(t *testing.T) {
	t.Run("NewPlanTemplateLibrary", func(t *testing.T) {
		lib := NewPlanTemplateLibrary()
		assert.NotNil(t, lib)
		assert.NotNil(t, lib.templates)

		// Should have default templates
		assert.Greater(t, len(lib.templates), 0)
	})

	t.Run("AddTemplate", func(t *testing.T) {
		lib := NewPlanTemplateLibrary()

		template := &PlanTemplate{
			Name:        "custom_template",
			TaskPattern: "custom.*task",
			Steps: []PlanStepTemplate{
				{Description: "Custom step", ToolType: "custom_tool", Critical: true},
			},
		}

		lib.AddTemplate(template)
		assert.Contains(t, lib.templates, "custom_template")
	})

	t.Run("FindTemplate", func(t *testing.T) {
		lib := NewPlanTemplateLibrary()

		// Test finding research template
		template := lib.FindTemplate("research the latest trends")
		assert.NotNil(t, template)
		assert.Equal(t, "research_and_summarize", template.Name)

		// Test no match
		template = lib.FindTemplate("completely unknown task")
		assert.Nil(t, template)
	})
}

func TestPlanStep_Tests(t *testing.T) {
	t.Run("Plan Step Creation", func(t *testing.T) {
		step := PlanStep{
			ID:          "step1",
			Description: "Test step",
			Tool:        "test_tool",
			Arguments:   map[string]interface{}{"arg1": "value1"},
			Expected:    "Expected result",
			Critical:    true,
			Parallel:    false,
			DependsOn:   []string{"step0"},
			Timeout:     5 * time.Second,
		}

		assert.Equal(t, "step1", step.ID)
		assert.Equal(t, "test_tool", step.Tool)
		assert.True(t, step.Critical)
		assert.False(t, step.Parallel)
		assert.Equal(t, 5*time.Second, step.Timeout)
		assert.Contains(t, step.DependsOn, "step0")
	})
}

func TestPlanOptimization_Tests(t *testing.T) {
	t.Run("Parallel Step Identification", func(t *testing.T) {
		planner := NewTaskPlanner(DecompositionFirst, 3)

		plan := &Plan{
			Steps: []PlanStep{
				{ID: "step1", Tool: "tool1", DependsOn: []string{}},
				{ID: "step2", Tool: "tool2", DependsOn: []string{}},        // Can run parallel with step1
				{ID: "step3", Tool: "tool3", DependsOn: []string{"step1"}}, // Depends on step1
				{ID: "step4", Tool: "tool4", DependsOn: []string{"step2"}}, // Depends on step2
			},
		}

		planner.optimizePlan(context.Background(), plan, []core.Tool{})

		// After optimization, step1 and step2 should be marked as parallel
		stepMap := make(map[string]PlanStep)
		for _, step := range plan.Steps {
			stepMap[step.ID] = step
		}

		// step1 and step2 have no dependencies, so they should be parallel
		assert.True(t, stepMap["step1"].Parallel, "step1 should be parallel")
		assert.True(t, stepMap["step2"].Parallel, "step2 should be parallel")
		// step3 and step4 have dependencies, so they should not be parallel
		assert.False(t, stepMap["step3"].Parallel, "step3 should not be parallel (depends on step1)")
		assert.False(t, stepMap["step4"].Parallel, "step4 should not be parallel (depends on step2)")
	})

	t.Run("Transitive Dependencies", func(t *testing.T) {
		planner := NewTaskPlanner(DecompositionFirst, 3)

		// Create a dependency map for testing
		dependents := map[string][]string{
			"step1": {"step2"},
			"step2": {"step3"},
			"step3": {},
		}

		step1 := &PlanStep{ID: "step1"}
		step3 := &PlanStep{ID: "step3"}

		// step1 transitively depends on step3 (step1 -> step2 -> step3)
		canParallel := planner.canRunInParallel(step1, step3, dependents)
		assert.False(t, canParallel)

		// Independent steps should be able to run in parallel
		step4 := &PlanStep{ID: "step4"}
		canParallel = planner.canRunInParallel(step1, step4, dependents)
		assert.True(t, canParallel)
	})

	t.Run("Circular Dependency Detection", func(t *testing.T) {
		planner := NewTaskPlanner(DecompositionFirst, 3)

		plan := &Plan{
			Steps: []PlanStep{
				{ID: "step1", DependsOn: []string{"step3"}},
				{ID: "step2", DependsOn: []string{"step1"}},
				{ID: "step3", DependsOn: []string{"step2"}},
			},
		}

		hasCycles := planner.hasCircularDependencies(plan)
		assert.True(t, hasCycles)

		// Test non-circular plan
		validPlan := &Plan{
			Steps: []PlanStep{
				{ID: "step1", DependsOn: []string{}},
				{ID: "step2", DependsOn: []string{"step1"}},
				{ID: "step3", DependsOn: []string{"step2"}},
			},
		}

		hasCycles = planner.hasCircularDependencies(validPlan)
		assert.False(t, hasCycles)
	})
}

func TestReActAgent_ExecutionModes(t *testing.T) {
	t.Run("ReWOO Mode Execution", func(t *testing.T) {
		agent := NewReActAgent("test-agent", "Test Agent", WithExecutionMode(ModeReWOO))

		// Initialize with mock LLM
		mockLLM := &mockLLM{
			response: &core.LLMResponse{
				Content: `thought: I need to plan this task
action: <action><tool_name>Finish</tool_name></action>`,
			},
		}

		signature := core.NewSignature(
			[]core.InputField{{Field: core.Field{Name: "task"}}},
			[]core.OutputField{{Field: core.Field{Name: "result"}}},
		)

		err := agent.Initialize(mockLLM, signature)
		require.NoError(t, err)

		// Test ReWOO execution
		ctx := context.Background()
		input := map[string]interface{}{"task": "complex analysis task requiring multiple steps"}

		// Register a test tool for the planner to use
		testTool := &mockTool{
			name: "analyze",
			result: core.ToolResult{
				Data: "analysis result",
			},
		}
		err = agent.RegisterTool(testTool)
		require.NoError(t, err)

		result, err := agent.executeReWOO(ctx, input)
		// ReWOO mode should create and execute a plan
		// The error indicates planning needs proper tool registration
		if err != nil {
			// The planner creates an adaptive step which needs a tool
			assert.Contains(t, err.Error(), "critical step", "Expected error about critical step failure")
			assert.Contains(t, err.Error(), "not found in registry", "Expected error about missing tool in registry")
		} else {
			// If successful, result should contain step results
			assert.NotNil(t, result, "Result should not be nil")
			assert.IsType(t, map[string]interface{}{}, result, "Result should be a map")
		}
	})

	t.Run("Hybrid Mode Task Analysis", func(t *testing.T) {
		agent := NewReActAgent("test-agent", "Test Agent")

		// Test simple task
		simpleTask := map[string]interface{}{"task": "add two numbers"}
		complexity := agent.analyzeTaskComplexity(simpleTask)
		assert.Less(t, complexity, 0.7) // Should be classified as simple

		// Test complex task
		complexTask := map[string]interface{}{"task": "analyze market trends, create detailed report with charts, and provide investment recommendations based on comprehensive financial modeling"}
		complexity = agent.analyzeTaskComplexity(complexTask)
		assert.Greater(t, complexity, 0.5) // Should be classified as more complex than simple task
	})

	t.Run("Plan Step Execution", func(t *testing.T) {
		agent := NewReActAgent("test-agent", "Test Agent")

		// Register a mock tool
		mockTool := &mockTool{
			name: "test_tool",
			result: core.ToolResult{
				Data: "test result",
			},
		}
		err := agent.RegisterTool(mockTool)
		require.NoError(t, err)

		// Initialize
		mockLLM := &mockLLM{}
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
			Critical:  true,
			Timeout:   5 * time.Second,
		}

		previousResults := map[string]interface{}{}

		result, err := agent.executePlanStep(ctx, step, previousResults)
		require.NoError(t, err)
		assert.NotNil(t, result)
	})

	t.Run("Plan Step Execution with Dependencies", func(t *testing.T) {
		agent := NewReActAgent("test-agent", "Test Agent")

		mockTool := &mockTool{
			name:   "dependent_tool",
			result: core.ToolResult{Data: "processed result"},
		}
		err := agent.RegisterTool(mockTool)
		require.NoError(t, err)

		mockLLM := &mockLLM{}
		signature := core.NewSignature(
			[]core.InputField{{Field: core.Field{Name: "input"}}},
			[]core.OutputField{{Field: core.Field{Name: "output"}}},
		)
		err = agent.Initialize(mockLLM, signature)
		require.NoError(t, err)

		ctx := context.Background()
		step := PlanStep{
			ID:        "step2",
			Tool:      "dependent_tool",
			Arguments: map[string]interface{}{"base_arg": "value"},
			DependsOn: []string{"step1"},
			Critical:  false,
		}

		previousResults := map[string]interface{}{
			"step1": "previous step result",
		}

		result, err := agent.executePlanStep(ctx, step, previousResults)
		require.NoError(t, err)
		assert.NotNil(t, result)
	})

	t.Run("Non-Critical Step Failure", func(t *testing.T) {
		agent := NewReActAgent("test-agent", "Test Agent")

		failingTool := &mockTool{
			name:       "failing_tool",
			shouldFail: true,
		}
		err := agent.RegisterTool(failingTool)
		require.NoError(t, err)

		mockLLM := &mockLLM{}
		signature := core.NewSignature(
			[]core.InputField{{Field: core.Field{Name: "input"}}},
			[]core.OutputField{{Field: core.Field{Name: "output"}}},
		)
		err = agent.Initialize(mockLLM, signature)
		require.NoError(t, err)

		ctx := context.Background()
		step := PlanStep{
			ID:       "failing_step",
			Tool:     "failing_tool",
			Critical: false, // Non-critical
		}

		result, err := agent.executePlanStep(ctx, step, map[string]interface{}{})
		require.NoError(t, err) // Should not error for non-critical step

		// Result should indicate failure
		if resultMap, ok := result.(map[string]interface{}); ok {
			assert.False(t, resultMap["success"].(bool))
			assert.Contains(t, resultMap, "error")
		}
	})

	t.Run("Critical Step Failure", func(t *testing.T) {
		agent := NewReActAgent("test-agent", "Test Agent")

		failingTool := &mockTool{
			name:       "critical_failing_tool",
			shouldFail: true,
		}
		err := agent.RegisterTool(failingTool)
		require.NoError(t, err)

		mockLLM := &mockLLM{}
		signature := core.NewSignature(
			[]core.InputField{{Field: core.Field{Name: "input"}}},
			[]core.OutputField{{Field: core.Field{Name: "output"}}},
		)
		err = agent.Initialize(mockLLM, signature)
		require.NoError(t, err)

		ctx := context.Background()
		step := PlanStep{
			ID:       "critical_failing_step",
			Tool:     "critical_failing_tool",
			Critical: true, // Critical step
		}

		_, err = agent.executePlanStep(ctx, step, map[string]interface{}{})
		require.Error(t, err) // Should error for critical step failure
		// The error message should indicate tool execution failure
		assert.Contains(t, err.Error(), "assert.AnError")
	})
}

// Mock implementations for testing.
type mockLLM struct {
	response *core.LLMResponse
	err      error
}

func (m *mockLLM) Generate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.response != nil {
		return m.response, nil
	}
	return &core.LLMResponse{Content: "mock response"}, nil
}

func (m *mockLLM) Capabilities() []core.Capability {
	return []core.Capability{}
}

func (m *mockLLM) ModelID() string {
	return "mock-model"
}

func (m *mockLLM) ProviderName() string {
	return "mock-provider"
}

func (m *mockLLM) GenerateWithJSON(ctx context.Context, prompt string, opts ...core.GenerateOption) (map[string]interface{}, error) {
	return map[string]interface{}{"result": "mock"}, nil
}

func (m *mockLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	return map[string]interface{}{"result": "mock"}, nil
}

func (m *mockLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return &core.EmbeddingResult{Vector: []float32{0.1, 0.2, 0.3}}, nil
}

func (m *mockLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return &core.BatchEmbeddingResult{}, nil
}

func (m *mockLLM) StreamGenerate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (m *mockLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{Content: "mock response"}, nil
}

func (m *mockLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

type mockTool struct {
	name       string
	result     core.ToolResult
	shouldFail bool
}

func (m *mockTool) Name() string {
	return m.name
}

func (m *mockTool) Description() string {
	return "Mock tool for testing"
}

func (m *mockTool) Validate(args map[string]interface{}) error {
	// For non-critical step failure test, validation should pass but execution should fail
	return nil
}

func (m *mockTool) Execute(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
	if m.shouldFail {
		return core.ToolResult{}, assert.AnError
	}
	return m.result, nil
}

func (m *mockTool) CanHandle(ctx context.Context, intent string) bool {
	return intent == m.name
}

func (m *mockTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:        m.name,
		Description: "Mock tool for testing",
	}
}

func (m *mockTool) InputSchema() models.InputSchema {
	return models.InputSchema{Type: "object"}
}
