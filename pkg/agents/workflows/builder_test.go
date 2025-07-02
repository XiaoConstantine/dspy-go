package workflows

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Simple mock for testing - avoiding conflicts with existing MockModule.
func NewMockModule(id string) core.Module {
	return &testModule{id: id}
}

type testModule struct {
	id string
}

func (m *testModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	return map[string]any{"output": "processed_" + m.id}, nil
}

func (m *testModule) GetSignature() core.Signature {
	return core.Signature{
		Inputs:  []core.InputField{{Field: core.Field{Name: "input", Description: "test input"}}},
		Outputs: []core.OutputField{{Field: core.Field{Name: "output", Description: "test output"}}},
	}
}

func (m *testModule) SetSignature(signature core.Signature) {}
func (m *testModule) SetLLM(llm core.LLM)                   {}
func (m *testModule) Clone() core.Module                    { return &testModule{id: m.id + "_clone"} }

func (m *testModule) GetDisplayName() string {
	return "TestModule_" + m.id
}

func (m *testModule) GetModuleType() string {
	return "test"
}

func TestWorkflowBuilder_NewBuilder(t *testing.T) {
	tests := []struct {
		name   string
		memory agents.Memory
	}{
		{
			name:   "with provided memory",
			memory: agents.NewInMemoryStore(),
		},
		{
			name:   "with nil memory",
			memory: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := NewBuilder(tt.memory)
			assert.NotNil(t, builder)
			assert.NotNil(t, builder.memory)
			assert.Equal(t, 0, len(builder.stages))
			assert.Equal(t, 0, len(builder.errors))
			assert.NotNil(t, builder.config)
		})
	}
}

func TestWorkflowBuilder_WithConfig(t *testing.T) {
	builder := NewBuilder(nil)
	customConfig := &BuilderConfig{
		EnableValidation:     false,
		EnableOptimization:   false,
		EnableTracing:        false,
		MaxConcurrency:       5,
		DefaultRetryAttempts: 1,
	}

	result := builder.WithConfig(customConfig)
	assert.Equal(t, builder, result) // Should return the same builder
	assert.Equal(t, customConfig, builder.config)
}

func TestWorkflowBuilder_Stage(t *testing.T) {
	tests := []struct {
		name        string
		stageID     string
		module      core.Module
		expectError bool
	}{
		{
			name:        "valid stage",
			stageID:     "test_stage",
			module:      NewMockModule("test"),
			expectError: false,
		},
		{
			name:        "empty stage ID",
			stageID:     "",
			module:      NewMockModule("test"),
			expectError: true,
		},
		{
			name:        "whitespace only stage ID",
			stageID:     "   ",
			module:      NewMockModule("test"),
			expectError: true,
		},
		{
			name:        "duplicate stage ID",
			stageID:     "duplicate",
			module:      NewMockModule("test"),
			expectError: false, // First addition should succeed
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := NewBuilder(nil)

			if tt.name == "duplicate stage ID" {
				// Add the stage once first
				builder.Stage(tt.stageID, NewMockModule("first"))
				assert.False(t, builder.hasError())

				// Adding again should cause error
				result := builder.Stage(tt.stageID, tt.module)
				assert.True(t, result.hasError())
				return
			}

			result := builder.Stage(tt.stageID, tt.module)

			if tt.expectError {
				assert.True(t, result.hasError())
			} else {
				assert.False(t, result.hasError())
				assert.Equal(t, 1, len(builder.stages))
				assert.Equal(t, tt.stageID, builder.stages[0].ID)
				assert.Equal(t, StageTypeSequential, builder.stages[0].Type)
				assert.Equal(t, tt.module, builder.stages[0].Module)
			}
		})
	}
}

func TestWorkflowBuilder_Parallel(t *testing.T) {
	tests := []struct {
		name        string
		stageID     string
		steps       []*BuilderStep
		expectError bool
	}{
		{
			name:    "valid parallel stage",
			stageID: "parallel_stage",
			steps: []*BuilderStep{
				NewStep("step1", NewMockModule("step1")),
				NewStep("step2", NewMockModule("step2")),
			},
			expectError: false,
		},
		{
			name:        "empty steps",
			stageID:     "empty_parallel",
			steps:       []*BuilderStep{},
			expectError: true,
		},
		{
			name:        "nil steps",
			stageID:     "nil_parallel",
			steps:       nil,
			expectError: true,
		},
		{
			name:        "empty stage ID",
			stageID:     "",
			steps:       []*BuilderStep{NewStep("step1", NewMockModule("step1"))},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := NewBuilder(nil)
			result := builder.Parallel(tt.stageID, tt.steps...)

			if tt.expectError {
				assert.True(t, result.hasError())
			} else {
				assert.False(t, result.hasError())
				assert.Equal(t, 1, len(builder.stages))
				assert.Equal(t, tt.stageID, builder.stages[0].ID)
				assert.Equal(t, StageTypeParallel, builder.stages[0].Type)
				assert.Equal(t, len(tt.steps), len(builder.stages[0].Steps))
			}
		})
	}
}

func TestWorkflowBuilder_Conditional(t *testing.T) {
	conditionFunc := func(ctx context.Context, state map[string]interface{}) (bool, error) {
		return true, nil
	}

	tests := []struct {
		name        string
		stageID     string
		condition   ConditionalFunc
		expectError bool
	}{
		{
			name:        "valid conditional stage",
			stageID:     "conditional_stage",
			condition:   conditionFunc,
			expectError: false,
		},
		{
			name:        "nil condition",
			stageID:     "nil_condition",
			condition:   nil,
			expectError: true,
		},
		{
			name:        "empty stage ID",
			stageID:     "",
			condition:   conditionFunc,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := NewBuilder(nil)
			result := builder.Conditional(tt.stageID, tt.condition)

			if tt.expectError {
				assert.True(t, result.hasError)
			} else {
				assert.False(t, result.hasError)
				assert.Equal(t, 1, len(builder.stages))
				assert.Equal(t, tt.stageID, builder.stages[0].ID)
				assert.Equal(t, StageTypeConditional, builder.stages[0].Type)
				assert.NotNil(t, builder.stages[0].Condition)
			}
		})
	}
}

func TestConditionalBuilder_IfElse(t *testing.T) {
	builder := NewBuilder(nil)
	condition := func(ctx context.Context, state map[string]interface{}) (bool, error) {
		return true, nil
	}

	conditionalBuilder := builder.Conditional("test_conditional", condition)
	result := conditionalBuilder.If(NewMockModule("if_module")).Else(NewMockModule("else_module")).End()

	assert.False(t, result.hasError())
	assert.Equal(t, 1, len(builder.stages))

	stage := builder.stages[0]
	assert.Equal(t, 2, len(stage.Branches))
	assert.Contains(t, stage.Branches, "true")
	assert.Contains(t, stage.Branches, "false")
}

func TestWorkflowBuilder_Then(t *testing.T) {
	builder := NewBuilder(nil)

	// Test Then without any stages - should error
	result := builder.Then("next_stage")
	assert.True(t, result.hasError())

	// Add stages and test Then
	builder = NewBuilder(nil)
	builder.Stage("stage1", NewMockModule("stage1"))
	result = builder.Then("stage2")

	assert.False(t, result.hasError())
	assert.Equal(t, 1, len(builder.stages[0].Next))
	assert.Equal(t, "stage2", builder.stages[0].Next[0])
}

func TestWorkflowBuilder_WithRetry(t *testing.T) {
	builder := NewBuilder(nil)
	retryConfig := &RetryConfig{
		MaxAttempts:       3,
		BackoffMultiplier: 2.0,
	}

	// Test WithRetry without any stages - should error
	result := builder.WithRetry(retryConfig)
	assert.True(t, result.hasError())

	// Add stage and test WithRetry
	builder = NewBuilder(nil)
	builder.Stage("stage1", NewMockModule("stage1"))
	result = builder.WithRetry(retryConfig)

	assert.False(t, result.hasError())
	assert.Equal(t, retryConfig, builder.stages[0].RetryConfig)
}

func TestWorkflowBuilder_WithMetadata(t *testing.T) {
	builder := NewBuilder(nil)

	// Test WithMetadata without any stages - should error
	result := builder.WithMetadata("key", "value")
	assert.True(t, result.hasError())

	// Add stage and test WithMetadata
	builder = NewBuilder(nil)
	builder.Stage("stage1", NewMockModule("stage1"))
	result = builder.WithMetadata("description", "test stage").WithMetadata("priority", 1)

	assert.False(t, result.hasError())
	assert.Equal(t, "test stage", builder.stages[0].Metadata["description"])
	assert.Equal(t, 1, builder.stages[0].Metadata["priority"])
}

func TestWorkflowBuilder_ValidationErrors(t *testing.T) {
	builder := NewBuilder(nil)

	// Add stages with validation issues
	builder.Stage("", NewMockModule("test")) // Empty ID
	builder.Stage("stage1", NewMockModule("test"))
	builder.Stage("stage1", NewMockModule("test2")) // Duplicate ID

	// Try to build - should fail validation
	workflow, err := builder.Build()
	assert.Nil(t, workflow)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "builder has errors")
}

func TestWorkflowBuilder_CycleDetection(t *testing.T) {
	builder := NewBuilder(nil)

	// Create a cycle: stage1 -> stage2 -> stage1
	builder.Stage("stage1", NewMockModule("stage1")).Then("stage2")
	builder.Stage("stage2", NewMockModule("stage2")).Then("stage1")

	workflow, err := builder.Build()
	assert.Nil(t, workflow)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cycle")
}

func TestWorkflowBuilder_Build_SimpleChain(t *testing.T) {
	builder := NewBuilder(nil)

	// Create a simple linear chain
	workflow, err := builder.
		Stage("stage1", NewMockModule("stage1")).Then("stage2").
		Stage("stage2", NewMockModule("stage2")).Then("stage3").
		Stage("stage3", NewMockModule("stage3")).
		Build()

	require.NoError(t, err)
	require.NotNil(t, workflow)

	// Verify it's a chain workflow
	chainWorkflow, ok := workflow.(*ChainWorkflow)
	assert.True(t, ok)
	assert.Equal(t, 3, len(chainWorkflow.GetSteps()))
}

func TestWorkflowBuilder_Build_ParallelWorkflow(t *testing.T) {
	builder := NewBuilder(nil)

	// Create a workflow with parallel stages
	workflow, err := builder.
		Parallel("parallel_stage",
			NewStep("step1", NewMockModule("step1")),
			NewStep("step2", NewMockModule("step2")),
		).
		Build()

	require.NoError(t, err)
	require.NotNil(t, workflow)

	// Verify it's a parallel workflow
	parallelWorkflow, ok := workflow.(*ParallelWorkflow)
	assert.True(t, ok)
	assert.True(t, len(parallelWorkflow.GetSteps()) >= 2)
}

func TestWorkflowBuilder_Build_EmptyWorkflow(t *testing.T) {
	builder := NewBuilder(nil)

	workflow, err := builder.Build()
	assert.Nil(t, workflow)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must have at least one stage")
}

func TestStep_Creation(t *testing.T) {
	module := NewMockModule("test")
	step := NewStep("test_step", module)

	assert.Equal(t, "test_step", step.ID)
	assert.Equal(t, module, step.Module)
	assert.Equal(t, "", step.Description)
	assert.NotNil(t, step.Metadata)
}

func TestStep_WithDescription(t *testing.T) {
	step := NewStep("test_step", NewMockModule("test")).
		WithDescription("This is a test step")

	assert.Equal(t, "This is a test step", step.Description)
}

func TestStep_WithStepMetadata(t *testing.T) {
	step := NewStep("test_step", NewMockModule("test")).
		WithStepMetadata("priority", "high").
		WithStepMetadata("timeout", 30)

	assert.Equal(t, "high", step.Metadata["priority"])
	assert.Equal(t, 30, step.Metadata["timeout"])
}

func TestWorkflowBuilder_ComplexWorkflow(t *testing.T) {
	// Test a complex workflow with multiple patterns
	builder := NewBuilder(nil)

	condition := func(ctx context.Context, state map[string]interface{}) (bool, error) {
		// Simple condition based on previous results
		if result, ok := state["analysis_result"]; ok {
			if confidence, ok := result.(map[string]interface{})["confidence"]; ok {
				if confValue, ok := confidence.(float64); ok {
					return confValue > 0.8, nil
				}
			}
		}
		return false, nil
	}

	retryConfig := &RetryConfig{
		MaxAttempts:       3,
		BackoffMultiplier: 2.0,
	}

	workflow, err := builder.
		Stage("analysis", NewMockModule("analysis")).
		WithMetadata("description", "Initial analysis stage").
		WithRetry(retryConfig).
		Then("validation").
		Parallel("validation",
			NewStep("context_check", NewMockModule("context_validator")).
				WithDescription("Validate context requirements"),
		).
		Then("decision").
		Conditional("decision", condition).
		If(NewMockModule("high_confidence_processor")).
		Else(NewMockModule("refinement_processor")).
		End().
		Build()

	require.NoError(t, err)
	require.NotNil(t, workflow)
	
	// Verify it's a router workflow (because it contains conditional stages)
	_, ok := workflow.(*ConditionalRouterWorkflow)
	assert.True(t, ok, "Complex workflow with conditional stages should create a ConditionalRouterWorkflow")
	
	// The router workflow should handle conditional routing
	// We can test its basic functionality by executing it
	result, err := workflow.Execute(context.Background(), map[string]interface{}{
		"input": "test_input",
		"analysis_result": map[string]interface{}{
			"confidence": 0.9, // High confidence should route to "true" branch
		},
	})
	
	require.NoError(t, err)
	require.NotNil(t, result)
	// The result should come from the high confidence processor since confidence > 0.8
	assert.Equal(t, "processed_high_confidence_processor", result["output"])
}

func TestWorkflowBuilder_ValidationStageReferences(t *testing.T) {
	builder := NewBuilder(nil)

	// Create stages that reference non-existent stages
	builder.Stage("stage1", NewMockModule("stage1")).Then("nonexistent")

	workflow, err := builder.Build()
	assert.Nil(t, workflow)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "references non-existent")
}

func TestDefaultBuilderConfig(t *testing.T) {
	config := DefaultBuilderConfig()

	assert.NotNil(t, config)
	assert.True(t, config.EnableValidation)
	assert.True(t, config.EnableOptimization)
	assert.True(t, config.EnableTracing)
	assert.Equal(t, 10, config.MaxConcurrency)
	assert.Equal(t, 3, config.DefaultRetryAttempts)
}

// TestWorkflowBuilder_IsLinearChain removed as the isLinearChain method
// was removed due to fragile logic. Workflow type determination is now
// simplified and doesn't rely on this method.

func TestWorkflowBuilder_ErrorAccumulation(t *testing.T) {
	builder := NewBuilder(nil)

	// Add multiple errors
	builder.Stage("", NewMockModule("test"))        // Error 1: empty ID
	builder.Stage("stage1", NewMockModule("test"))  // OK
	builder.Stage("stage1", NewMockModule("test2")) // Error 2: duplicate ID
	builder.Then("nonexistent")                     // This will add to next steps but validation will catch it

	// Should accumulate multiple errors
	assert.True(t, builder.hasError())
	assert.True(t, len(builder.errors) >= 2)

	workflow, err := builder.Build()
	assert.Nil(t, workflow)
	assert.Error(t, err)
}

// Benchmark tests.
func BenchmarkWorkflowBuilder_SimpleChain(b *testing.B) {
	for i := 0; i < b.N; i++ {
		builder := NewBuilder(nil)
		_, err := builder.
			Stage("stage1", NewMockModule("stage1")).Then("stage2").
			Stage("stage2", NewMockModule("stage2")).Then("stage3").
			Stage("stage3", NewMockModule("stage3")).
			Build()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWorkflowBuilder_ComplexWorkflow(b *testing.B) {
	condition := func(ctx context.Context, state map[string]interface{}) (bool, error) {
		return true, nil
	}

	for i := 0; i < b.N; i++ {
		builder := NewBuilder(nil)
		_, err := builder.
			Stage("analysis", NewMockModule("analysis")).
			Then("validation").
			Parallel("validation",
				NewStep("step1", NewMockModule("step1")),
				NewStep("step2", NewMockModule("step2")),
				NewStep("step3", NewMockModule("step3")),
			).
			Then("decision").
			Conditional("decision", condition).
			If(NewMockModule("if_module")).
			Else(NewMockModule("else_module")).
			End().
			Build()
		if err != nil {
			b.Fatal(err)
		}
	}
}
