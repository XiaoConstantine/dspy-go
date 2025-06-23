package workflows

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mock module for testing advanced patterns.
type advancedMockModule struct {
	id      string
	outputs map[string]interface{}
}

func NewAdvancedMockModule(id string) *advancedMockModule {
	return &advancedMockModule{
		id:      id,
		outputs: map[string]interface{}{"output": "processed_" + id},
	}
}

func (m *advancedMockModule) WithOutputs(outputs map[string]interface{}) *advancedMockModule {
	m.outputs = outputs
	return m
}

func (m *advancedMockModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Convert outputs to map[string]any
	result := make(map[string]any)
	for k, v := range m.outputs {
		result[k] = v
	}
	return result, nil
}

func (m *advancedMockModule) GetSignature() core.Signature {
	return core.Signature{
		Inputs:  []core.InputField{{Field: core.Field{Name: "input", Description: "test input"}}},
		Outputs: []core.OutputField{{Field: core.Field{Name: "output", Description: "test output"}}},
	}
}

func (m *advancedMockModule) SetSignature(signature core.Signature) {}
func (m *advancedMockModule) SetLLM(llm core.LLM)                   {}
func (m *advancedMockModule) Clone() core.Module {
	return &advancedMockModule{id: m.id + "_clone", outputs: m.outputs}
}

func TestWorkflowBuilder_ForEach(t *testing.T) {
	tests := []struct {
		name         string
		iteratorFunc IteratorFunc
		expectError  bool
		expectedLen  int
	}{
		{
			name: "valid forEach with items",
			iteratorFunc: func(ctx context.Context, state map[string]interface{}) ([]interface{}, error) {
				return []interface{}{"item1", "item2", "item3"}, nil
			},
			expectError: false,
			expectedLen: 3,
		},
		{
			name:         "nil iterator function",
			iteratorFunc: nil,
			expectError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := NewBuilder(nil)
			result := builder.ForEach("forEach_test", tt.iteratorFunc, func(bodyBuilder *WorkflowBuilder) *WorkflowBuilder {
				return bodyBuilder.Stage("process_item", NewAdvancedMockModule("item_processor"))
			})

			if tt.expectError {
				assert.True(t, result.hasError())
			} else {
				assert.False(t, result.hasError())
				assert.Equal(t, 1, len(builder.stages))
				assert.Equal(t, StageTypeForEach, builder.stages[0].Type)
				assert.NotNil(t, builder.stages[0].LoopBody)
			}
		})
	}
}

func TestWorkflowBuilder_While(t *testing.T) {
	tests := []struct {
		name        string
		condition   LoopConditionFunc
		expectError bool
	}{
		{
			name: "valid while condition",
			condition: func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error) {
				return iteration < 3, nil
			},
			expectError: false,
		},
		{
			name:        "nil condition",
			condition:   nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := NewBuilder(nil)
			result := builder.While("while_test", tt.condition, func(bodyBuilder *WorkflowBuilder) *WorkflowBuilder {
				return bodyBuilder.Stage("loop_body", NewAdvancedMockModule("loop_processor"))
			})

			if tt.expectError {
				assert.True(t, result.hasError())
			} else {
				assert.False(t, result.hasError())
				assert.Equal(t, 1, len(builder.stages))
				assert.Equal(t, StageTypeWhile, builder.stages[0].Type)
				assert.NotNil(t, builder.stages[0].LoopBody)
			}
		})
	}
}

func TestWorkflowBuilder_Until(t *testing.T) {
	tests := []struct {
		name        string
		condition   LoopConditionFunc
		expectError bool
	}{
		{
			name: "valid until condition",
			condition: func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error) {
				return iteration >= 3, nil
			},
			expectError: false,
		},
		{
			name:        "nil condition",
			condition:   nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := NewBuilder(nil)
			result := builder.Until("until_test", tt.condition, func(bodyBuilder *WorkflowBuilder) *WorkflowBuilder {
				return bodyBuilder.Stage("loop_body", NewAdvancedMockModule("loop_processor"))
			})

			if tt.expectError {
				assert.True(t, result.hasError())
			} else {
				assert.False(t, result.hasError())
				assert.Equal(t, 1, len(builder.stages))
				assert.Equal(t, StageTypeUntil, builder.stages[0].Type)
				assert.NotNil(t, builder.stages[0].LoopBody)
			}
		})
	}
}

func TestWorkflowBuilder_Template(t *testing.T) {
	tests := []struct {
		name          string
		parameterFunc TemplateParameterFunc
		expectError   bool
	}{
		{
			name: "valid template",
			parameterFunc: func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error) {
				return map[string]interface{}{"template_param": "value"}, nil
			},
			expectError: false,
		},
		{
			name:          "nil parameter function",
			parameterFunc: nil,
			expectError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := NewBuilder(nil)
			result := builder.Template("template_test", tt.parameterFunc, func(templateBuilder *WorkflowBuilder) *WorkflowBuilder {
				return templateBuilder.Stage("template_stage", NewAdvancedMockModule("template_processor"))
			})

			if tt.expectError {
				assert.True(t, result.hasError())
			} else {
				assert.False(t, result.hasError())
				assert.Equal(t, 1, len(builder.stages))
				assert.Equal(t, StageTypeTemplate, builder.stages[0].Type)
				assert.NotNil(t, builder.stages[0].TemplateWorkflow)
			}
		})
	}
}

func TestWorkflowBuilder_WithMaxIterations(t *testing.T) {
	builder := NewBuilder(nil)

	// Test without any stages
	result := builder.WithMaxIterations(10)
	assert.True(t, result.hasError())

	// Test with non-loop stage
	builder = NewBuilder(nil)
	builder.Stage("regular_stage", NewAdvancedMockModule("regular"))
	result = builder.WithMaxIterations(10)
	assert.True(t, result.hasError())

	// Test with loop stage
	builder = NewBuilder(nil)
	builder.While("while_stage", func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error) {
		return iteration < 5, nil
	}, nil)
	result = builder.WithMaxIterations(10)
	assert.False(t, result.hasError())
	assert.Equal(t, 10, builder.stages[0].MaxIterations)
}

func TestWorkflowBuilder_WithTimeout(t *testing.T) {
	builder := NewBuilder(nil)

	// Test without any stages
	result := builder.WithTimeout(5000)
	assert.True(t, result.hasError())

	// Test with stage
	builder = NewBuilder(nil)
	builder.Stage("timed_stage", NewAdvancedMockModule("timed"))
	result = builder.WithTimeout(5000)
	assert.False(t, result.hasError())
	assert.Equal(t, int64(5000), builder.stages[0].TimeoutMs)
}

func TestCompositeWorkflow_ForEachExecution(t *testing.T) {
	builder := NewBuilder(agents.NewInMemoryStore())

	// Create a forEach workflow
	workflow, err := builder.
		ForEach("process_items",
			func(ctx context.Context, state map[string]interface{}) ([]interface{}, error) {
				return []interface{}{"apple", "banana", "cherry"}, nil
			},
			func(bodyBuilder *WorkflowBuilder) *WorkflowBuilder {
				return bodyBuilder.Stage("process_item",
					NewAdvancedMockModule("item_processor").WithOutputs(map[string]interface{}{
						"output":    "processed_item",
						"processed": true,
					}))
			}).
		Build()

	require.NoError(t, err)
	require.NotNil(t, workflow)

	// Execute the workflow
	result, err := workflow.Execute(context.Background(), map[string]interface{}{
		"input": "test_input",
	})

	require.NoError(t, err)
	require.NotNil(t, result)

	// Check that forEach results are present
	forEachResults, exists := result["forEach_results"]
	assert.True(t, exists)
	assert.IsType(t, []map[string]interface{}{}, forEachResults)

	results := forEachResults.([]map[string]interface{})
	assert.Equal(t, 3, len(results)) // Should have processed 3 items
}

func TestCompositeWorkflow_WhileExecution(t *testing.T) {
	builder := NewBuilder(agents.NewInMemoryStore())

	// Create a while workflow that runs 3 times
	workflow, err := builder.
		While("count_loop",
			func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error) {
				return iteration < 3, nil
			},
			func(bodyBuilder *WorkflowBuilder) *WorkflowBuilder {
				return bodyBuilder.Stage("increment",
					NewAdvancedMockModule("counter").WithOutputs(map[string]interface{}{
						"output": "incremented",
						"count":  1,
					}))
			}).
		Build()

	require.NoError(t, err)
	require.NotNil(t, workflow)

	// Execute the workflow
	result, err := workflow.Execute(context.Background(), map[string]interface{}{
		"input": "test_input",
	})

	require.NoError(t, err)
	require.NotNil(t, result)

	// Check iteration count
	iterations, exists := result["while_iterations"]
	assert.True(t, exists)
	assert.Equal(t, 3, iterations)
}

func TestCompositeWorkflow_TemplateExecution(t *testing.T) {
	builder := NewBuilder(agents.NewInMemoryStore())

	// Create a template workflow
	workflow, err := builder.
		Template("process_template",
			func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error) {
				return map[string]interface{}{
					"template_param": "resolved_value",
					"multiplier":     2,
				}, nil
			},
			func(templateBuilder *WorkflowBuilder) *WorkflowBuilder {
				return templateBuilder.Stage("apply_template",
					NewAdvancedMockModule("template_applier").WithOutputs(map[string]interface{}{
						"output":          "template_applied",
						"template_result": "applied",
					}))
			}).
		Build()

	require.NoError(t, err)
	require.NotNil(t, workflow)

	// Execute the workflow
	result, err := workflow.Execute(context.Background(), map[string]interface{}{
		"input": "test_input",
	})

	require.NoError(t, err)
	require.NotNil(t, result)

	// Check that template parameters were resolved
	assert.Equal(t, "resolved_value", result["template_param"])
	assert.Equal(t, 2, result["multiplier"])
	assert.Equal(t, "applied", result["template_result"])
}

func TestRouterWorkflow_ConditionalRouting(t *testing.T) {
	builder := NewBuilder(agents.NewInMemoryStore())

	// Create a conditional workflow
	workflow, err := builder.
		Conditional("route_decision",
			func(ctx context.Context, state map[string]interface{}) (bool, error) {
				confidence, ok := state["confidence"].(float64)
				if !ok {
					return false, nil
				}
				return confidence > 0.8, nil
			}).
		If(NewAdvancedMockModule("high_confidence").WithOutputs(map[string]interface{}{
			"output":      "processed_high",
			"route_taken": "high_confidence",
		})).
		Else(NewAdvancedMockModule("low_confidence").WithOutputs(map[string]interface{}{
			"output":      "processed_low",
			"route_taken": "low_confidence",
		})).
		End().
		Build()

	require.NoError(t, err)
	require.NotNil(t, workflow)

	// Test high confidence routing
	result, err := workflow.Execute(context.Background(), map[string]interface{}{
		"input":      "test_data",
		"confidence": 0.9,
	})

	require.NoError(t, err)
	assert.Equal(t, "high_confidence", result["route_taken"])

	// Test low confidence routing
	result, err = workflow.Execute(context.Background(), map[string]interface{}{
		"input":      "test_data",
		"confidence": 0.5,
	})

	require.NoError(t, err)
	assert.Equal(t, "low_confidence", result["route_taken"])
}

func TestWorkflowBuilder_AdvancedValidation(t *testing.T) {
	tests := []struct {
		name          string
		buildWorkflow func() *WorkflowBuilder
		expectError   bool
		errorContains string
	}{
		{
			name: "forEach without iterator function",
			buildWorkflow: func() *WorkflowBuilder {
				return NewBuilder(nil).ForEach("test", nil, nil)
			},
			expectError:   true,
			errorContains: "iterator function",
		},
		{
			name: "while without condition",
			buildWorkflow: func() *WorkflowBuilder {
				return NewBuilder(nil).While("test", nil, nil)
			},
			expectError:   true,
			errorContains: "condition function",
		},
		{
			name: "template without parameter function",
			buildWorkflow: func() *WorkflowBuilder {
				return NewBuilder(nil).Template("test", nil, nil)
			},
			expectError:   true,
			errorContains: "parameter function",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := tt.buildWorkflow()
			_, err := builder.Build()

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// Additional comprehensive tests for better coverage

func TestCompositeWorkflow_ExecuteStageFailures(t *testing.T) {
	cw := NewCompositeWorkflow(agents.NewInMemoryStore())

	tests := []struct {
		name          string
		stage         *BuilderStage
		state         map[string]interface{}
		expectError   bool
		errorContains string
	}{
		{
			name: "sequential stage with nil module",
			stage: &BuilderStage{
				ID:     "nil_module",
				Type:   StageTypeSequential,
				Module: nil,
			},
			state:         map[string]interface{}{"input": "test"},
			expectError:   true,
			errorContains: "sequential stage must have a module",
		},
		{
			name: "unsupported stage type",
			stage: &BuilderStage{
				ID:   "unsupported",
				Type: StageType(999), // Invalid stage type
			},
			state:         map[string]interface{}{"input": "test"},
			expectError:   true,
			errorContains: "unsupported stage type",
		},
		{
			name: "parallel stage with empty steps",
			stage: &BuilderStage{
				ID:    "empty_parallel",
				Type:  StageTypeParallel,
				Steps: []*BuilderStep{},
			},
			state:       map[string]interface{}{"input": "test"},
			expectError: false, // Should not error, just return empty result
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := cw.executeStage(context.Background(), tt.stage, tt.state)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
			}
		})
	}
}

// slowMockModule for timeout testing.
type slowMockModule struct {
	id      string
	outputs map[string]interface{}
	delay   time.Duration
}

func NewSlowMockModule(id string, delay time.Duration) *slowMockModule {
	return &slowMockModule{
		id:      id,
		outputs: map[string]interface{}{"output": "processed_" + id},
		delay:   delay,
	}
}

func (m *slowMockModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(m.delay):
		result := make(map[string]any)
		for k, v := range m.outputs {
			result[k] = v
		}
		return result, nil
	}
}

func (m *slowMockModule) GetSignature() core.Signature {
	return core.Signature{
		Inputs:  []core.InputField{{Field: core.Field{Name: "input", Description: "test input"}}},
		Outputs: []core.OutputField{{Field: core.Field{Name: "output", Description: "test output"}}},
	}
}

func (m *slowMockModule) SetSignature(signature core.Signature) {}
func (m *slowMockModule) SetLLM(llm core.LLM)                   {}
func (m *slowMockModule) Clone() core.Module {
	return &slowMockModule{id: m.id + "_clone", outputs: m.outputs, delay: m.delay}
}

// Error scenario mock module.
type errorMockModule struct {
	id string
}

func NewErrorMockModule(id string) *errorMockModule {
	return &errorMockModule{id: id}
}

func (m *errorMockModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	return nil, fmt.Errorf("mock error from %s", m.id)
}

func (m *errorMockModule) GetSignature() core.Signature {
	return core.Signature{
		Inputs:  []core.InputField{{Field: core.Field{Name: "input", Description: "test input"}}},
		Outputs: []core.OutputField{{Field: core.Field{Name: "output", Description: "test output"}}},
	}
}

func (m *errorMockModule) SetSignature(signature core.Signature) {}
func (m *errorMockModule) SetLLM(llm core.LLM)                   {}
func (m *errorMockModule) Clone() core.Module {
	return &errorMockModule{id: m.id + "_clone"}
}

// badFieldClassifierModule returns wrong field name.
type badFieldClassifierModule struct{}

func (bfcm *badFieldClassifierModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	return map[string]any{"wrong_field": "value"}, nil
}

func (bfcm *badFieldClassifierModule) GetSignature() core.Signature {
	return core.Signature{}
}

func (bfcm *badFieldClassifierModule) SetSignature(signature core.Signature) {}
func (bfcm *badFieldClassifierModule) SetLLM(llm core.LLM)                   {}
func (bfcm *badFieldClassifierModule) Clone() core.Module {
	return &badFieldClassifierModule{}
}

// intClassifierModule returns integer classification.
type intClassifierModule struct{}

func (icm *intClassifierModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	return map[string]any{"classification": 123}, nil // Integer instead of string
}

func (icm *intClassifierModule) GetSignature() core.Signature {
	return core.Signature{}
}

func (icm *intClassifierModule) SetSignature(signature core.Signature) {}
func (icm *intClassifierModule) SetLLM(llm core.LLM)                   {}
func (icm *intClassifierModule) Clone() core.Module {
	return &intClassifierModule{}
}

func TestCompositeWorkflow_TimeoutHandling(t *testing.T) {
	cw := NewCompositeWorkflow(agents.NewInMemoryStore())

	// Create a slow mock module
	slowModule := NewSlowMockModule("slow_module", 100*time.Millisecond)

	stage := &BuilderStage{
		ID:        "timeout_stage",
		Type:      StageTypeSequential,
		Module:    slowModule,
		TimeoutMs: 50, // Very short timeout
	}

	result, err := cw.executeStage(context.Background(), stage, map[string]interface{}{"input": "test"})

	// Should timeout
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestCompositeWorkflow_ParallelExecutionErrors(t *testing.T) {
	cw := NewCompositeWorkflow(agents.NewInMemoryStore())

	// Create modules that will fail
	failingModule := NewErrorMockModule("failing")

	successModule := NewAdvancedMockModule("success").WithOutputs(map[string]interface{}{
		"output": "success_result",
	})

	stage := &BuilderStage{
		ID:   "mixed_parallel",
		Type: StageTypeParallel,
		Steps: []*BuilderStep{
			{ID: "failing_step", Module: failingModule},
			{ID: "success_step", Module: successModule},
		},
	}

	result, err := cw.executeStage(context.Background(), stage, map[string]interface{}{"input": "test"})

	// Should fail because one module failed
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestCompositeWorkflow_ConditionalExecutionErrors(t *testing.T) {
	cw := NewCompositeWorkflow(agents.NewInMemoryStore())

	tests := []struct {
		name          string
		condition     ConditionalFunc
		branches      map[string]*BuilderStage
		expectError   bool
		errorContains string
	}{
		{
			name: "condition function fails",
			condition: func(ctx context.Context, state map[string]interface{}) (bool, error) {
				return false, fmt.Errorf("condition failed")
			},
			branches:      make(map[string]*BuilderStage),
			expectError:   true,
			errorContains: "condition evaluation failed",
		},
		{
			name: "no matching branch",
			condition: func(ctx context.Context, state map[string]interface{}) (bool, error) {
				return true, nil
			},
			branches:    make(map[string]*BuilderStage), // No branches
			expectError: false,                          // Should return state unchanged
		},
		{
			name: "branch execution fails",
			condition: func(ctx context.Context, state map[string]interface{}) (bool, error) {
				return true, nil
			},
			branches: map[string]*BuilderStage{
				"true": {
					ID:     "failing_branch",
					Type:   StageTypeSequential,
					Module: nil, // Will cause error
				},
			},
			expectError:   true,
			errorContains: "sequential stage must have a module",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stage := &BuilderStage{
				ID:        "conditional_test",
				Type:      StageTypeConditional,
				Condition: tt.condition,
				Branches:  tt.branches,
			}

			result, err := cw.executeStage(context.Background(), stage, map[string]interface{}{"input": "test"})

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
			}
		})
	}
}

func TestCompositeWorkflow_ForEachExecutionErrors(t *testing.T) {
	cw := NewCompositeWorkflow(agents.NewInMemoryStore())

	tests := []struct {
		name          string
		iteratorFunc  IteratorFunc
		loopBody      *WorkflowBuilder
		maxIterations int
		expectError   bool
		errorContains string
	}{
		{
			name: "iterator function fails",
			iteratorFunc: func(ctx context.Context, state map[string]interface{}) ([]interface{}, error) {
				return nil, fmt.Errorf("iterator failed")
			},
			loopBody:      NewBuilder(nil),
			maxIterations: 10,
			expectError:   true,
			errorContains: "iterator function failed",
		},
		{
			name: "empty items list",
			iteratorFunc: func(ctx context.Context, state map[string]interface{}) ([]interface{}, error) {
				return []interface{}{}, nil
			},
			loopBody:      NewBuilder(nil),
			maxIterations: 10,
			expectError:   false,
		},
		{
			name: "max iterations exceeded",
			iteratorFunc: func(ctx context.Context, state map[string]interface{}) ([]interface{}, error) {
				return []interface{}{"item1", "item2", "item3", "item4", "item5"}, nil
			},
			loopBody:      NewBuilder(nil).Stage("process", NewAdvancedMockModule("processor")),
			maxIterations: 3, // Will stop at 3 iterations
			expectError:   false,
		},
		{
			name: "nested workflow execution fails",
			iteratorFunc: func(ctx context.Context, state map[string]interface{}) ([]interface{}, error) {
				return []interface{}{"item1"}, nil
			},
			loopBody: func() *WorkflowBuilder {
				// Create a workflow that will fail
				failingModule := NewErrorMockModule("failing")
				return NewBuilder(nil).Stage("failing", failingModule)
			}(),
			maxIterations: 10,
			expectError:   true,
			errorContains: "forEach iteration 0 failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stage := &BuilderStage{
				ID:            "forEach_test",
				Type:          StageTypeForEach,
				IteratorFunc:  tt.iteratorFunc,
				LoopBody:      tt.loopBody,
				MaxIterations: tt.maxIterations,
			}

			result, err := cw.executeStage(context.Background(), stage, map[string]interface{}{"input": "test"})

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				if tt.name == "max iterations exceeded" {
					// Should have stopped at maxIterations
					results := result["forEach_results"].([]map[string]interface{})
					assert.Equal(t, tt.maxIterations, len(results))
				}
			}
		})
	}
}

func TestCompositeWorkflow_WhileExecutionErrors(t *testing.T) {
	cw := NewCompositeWorkflow(agents.NewInMemoryStore())

	tests := []struct {
		name          string
		condition     LoopConditionFunc
		loopBody      *WorkflowBuilder
		maxIterations int
		expectError   bool
		errorContains string
	}{
		{
			name: "condition function fails",
			condition: func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error) {
				return false, fmt.Errorf("condition failed")
			},
			loopBody:      NewBuilder(nil),
			maxIterations: 10,
			expectError:   true,
			errorContains: "while condition evaluation failed",
		},
		{
			name: "max iterations reached",
			condition: func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error) {
				return true, nil // Always true, will hit max iterations
			},
			loopBody:      NewBuilder(nil).Stage("process", NewAdvancedMockModule("processor")),
			maxIterations: 3,
			expectError:   false,
		},
		{
			name: "nested workflow fails",
			condition: func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error) {
				return iteration == 0, nil // Run once
			},
			loopBody: func() *WorkflowBuilder {
				failingModule := NewErrorMockModule("failing")
				return NewBuilder(nil).Stage("failing", failingModule)
			}(),
			maxIterations: 10,
			expectError:   true,
			errorContains: "while iteration 0 failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stage := &BuilderStage{
				ID:            "while_test",
				Type:          StageTypeWhile,
				LoopCondition: tt.condition,
				LoopBody:      tt.loopBody,
				MaxIterations: tt.maxIterations,
			}

			result, err := cw.executeStage(context.Background(), stage, map[string]interface{}{"input": "test"})

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				if tt.name == "max iterations reached" {
					assert.Equal(t, tt.maxIterations, result["while_iterations"])
				}
			}
		})
	}
}

func TestCompositeWorkflow_UntilExecutionErrors(t *testing.T) {
	cw := NewCompositeWorkflow(agents.NewInMemoryStore())

	tests := []struct {
		name          string
		condition     LoopConditionFunc
		loopBody      *WorkflowBuilder
		maxIterations int
		expectError   bool
		errorContains string
	}{
		{
			name: "condition function fails",
			condition: func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error) {
				return false, fmt.Errorf("condition failed")
			},
			loopBody:      NewBuilder(nil),
			maxIterations: 10,
			expectError:   true,
			errorContains: "until condition evaluation failed",
		},
		{
			name: "max iterations reached",
			condition: func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error) {
				return false, nil // Never true, will hit max iterations
			},
			loopBody:      NewBuilder(nil).Stage("process", NewAdvancedMockModule("processor")),
			maxIterations: 3,
			expectError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stage := &BuilderStage{
				ID:            "until_test",
				Type:          StageTypeUntil,
				LoopCondition: tt.condition,
				LoopBody:      tt.loopBody,
				MaxIterations: tt.maxIterations,
			}

			result, err := cw.executeStage(context.Background(), stage, map[string]interface{}{"input": "test"})

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				if tt.name == "max iterations reached" {
					assert.Equal(t, tt.maxIterations, result["until_iterations"])
				}
			}
		})
	}
}

func TestCompositeWorkflow_TemplateExecutionErrors(t *testing.T) {
	cw := NewCompositeWorkflow(agents.NewInMemoryStore())

	tests := []struct {
		name             string
		templateParams   TemplateParameterFunc
		templateWorkflow *WorkflowBuilder
		expectError      bool
		errorContains    string
	}{
		{
			name: "template parameter resolution fails",
			templateParams: func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error) {
				return nil, fmt.Errorf("parameter resolution failed")
			},
			templateWorkflow: NewBuilder(nil),
			expectError:      true,
			errorContains:    "template parameter resolution failed",
		},
		{
			name: "template workflow execution fails",
			templateParams: func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error) {
				return map[string]interface{}{"param": "value"}, nil
			},
			templateWorkflow: func() *WorkflowBuilder {
				failingModule := NewErrorMockModule("failing")
				return NewBuilder(nil).Stage("failing", failingModule)
			}(),
			expectError:   true,
			errorContains: "step execution failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stage := &BuilderStage{
				ID:               "template_test",
				Type:             StageTypeTemplate,
				TemplateParams:   tt.templateParams,
				TemplateWorkflow: tt.templateWorkflow,
			}

			result, err := cw.executeStage(context.Background(), stage, map[string]interface{}{"input": "test"})

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
			}
		})
	}
}

func TestCompositeWorkflow_ExecuteNestedWorkflow(t *testing.T) {
	cw := NewCompositeWorkflow(agents.NewInMemoryStore())

	tests := []struct {
		name          string
		nestedBuilder *WorkflowBuilder
		state         map[string]interface{}
		expectError   bool
		errorContains string
	}{
		{
			name:          "nil nested builder",
			nestedBuilder: nil,
			state:         map[string]interface{}{"input": "test"},
			expectError:   false, // Should return state unchanged
		},
		{
			name: "build failure",
			nestedBuilder: func() *WorkflowBuilder {
				// Create a builder that will fail to build
				builder := NewBuilder(nil)
				builder.addError(fmt.Errorf("forced build error"))
				return builder
			}(),
			state:         map[string]interface{}{"input": "test"},
			expectError:   true,
			errorContains: "failed to build nested workflow",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := cw.executeNestedWorkflow(context.Background(), tt.nestedBuilder, tt.state)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
			}
		})
	}
}

func TestConditionalRouterWorkflow_EdgeCases(t *testing.T) {
	classifier := &conditionalClassifierModule{
		condition: func(ctx context.Context, state map[string]interface{}) (bool, error) {
			return true, nil
		},
	}
	workflow := NewConditionalRouterWorkflow(agents.NewInMemoryStore(), classifier)

	t.Run("add duplicate route", func(t *testing.T) {
		step := &Step{ID: "test_step", Module: NewAdvancedMockModule("test")}

		// Add route first time - should succeed
		err := workflow.AddRoute("test_route", step)
		assert.NoError(t, err)

		// Add same route again - should fail
		err = workflow.AddRoute("test_route", step)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "already exists")
	})

	t.Run("classifier failure", func(t *testing.T) {
		failingClassifier := &conditionalClassifierModule{
			condition: func(ctx context.Context, state map[string]interface{}) (bool, error) {
				return false, fmt.Errorf("classifier failed")
			},
		}
		failingWorkflow := NewConditionalRouterWorkflow(agents.NewInMemoryStore(), failingClassifier)

		result, err := failingWorkflow.Execute(context.Background(), map[string]interface{}{"input": "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "classification failed")
		assert.Nil(t, result)
	})

	t.Run("no classification field", func(t *testing.T) {
		badClassifier := &badFieldClassifierModule{}

		badWorkflow := NewConditionalRouterWorkflow(agents.NewInMemoryStore(), badClassifier)

		result, err := badWorkflow.Execute(context.Background(), map[string]interface{}{"input": "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "did not return 'classification' field")
		assert.Nil(t, result)
	})

	t.Run("non-string classification", func(t *testing.T) {
		intClassifier := &intClassifierModule{}

		intWorkflow := NewConditionalRouterWorkflow(agents.NewInMemoryStore(), intClassifier)

		result, err := intWorkflow.Execute(context.Background(), map[string]interface{}{"input": "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "classification must be a string")
		assert.Nil(t, result)
	})

	t.Run("no matching route and no default", func(t *testing.T) {
		workflow := NewConditionalRouterWorkflow(agents.NewInMemoryStore(), classifier)

		result, err := workflow.Execute(context.Background(), map[string]interface{}{"input": "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no route found")
		assert.Nil(t, result)
	})

	t.Run("route execution failure", func(t *testing.T) {
		failingModule := NewErrorMockModule("failing")

		failingStep := &Step{ID: "failing_step", Module: failingModule}
		workflow := NewConditionalRouterWorkflow(agents.NewInMemoryStore(), classifier)
		err := workflow.AddRoute("true", failingStep)
		require.NoError(t, err)

		result, err := workflow.Execute(context.Background(), map[string]interface{}{"input": "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "route 'true' execution failed")
		assert.Nil(t, result)
	})

	t.Run("default route execution failure", func(t *testing.T) {
		failingModule := NewErrorMockModule("failing")

		failingStep := &Step{ID: "default_step", Module: failingModule}
		workflow := NewConditionalRouterWorkflow(agents.NewInMemoryStore(), classifier)
		workflow.SetDefaultRoute(failingStep)

		result, err := workflow.Execute(context.Background(), map[string]interface{}{"input": "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "default route execution failed")
		assert.Nil(t, result)
	})
}

func TestConditionalClassifierModule(t *testing.T) {
	t.Run("condition function failure", func(t *testing.T) {
		ccm := &conditionalClassifierModule{
			condition: func(ctx context.Context, state map[string]interface{}) (bool, error) {
				return false, fmt.Errorf("condition error")
			},
		}

		result, err := ccm.Process(context.Background(), map[string]any{"input": "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "condition error")
		assert.Nil(t, result)
	})

	t.Run("successful true condition", func(t *testing.T) {
		ccm := &conditionalClassifierModule{
			condition: func(ctx context.Context, state map[string]interface{}) (bool, error) {
				return true, nil
			},
		}

		result, err := ccm.Process(context.Background(), map[string]any{"input": "test"})
		assert.NoError(t, err)
		assert.Equal(t, "true", result["classification"])
	})

	t.Run("successful false condition", func(t *testing.T) {
		ccm := &conditionalClassifierModule{
			condition: func(ctx context.Context, state map[string]interface{}) (bool, error) {
				return false, nil
			},
		}

		result, err := ccm.Process(context.Background(), map[string]any{"input": "test"})
		assert.NoError(t, err)
		assert.Equal(t, "false", result["classification"])
	})

	t.Run("clone functionality", func(t *testing.T) {
		original := &conditionalClassifierModule{
			condition: func(ctx context.Context, state map[string]interface{}) (bool, error) {
				return true, nil
			},
		}

		cloned := original.Clone()
		assert.NotNil(t, cloned)
		assert.IsType(t, &conditionalClassifierModule{}, cloned)

		// Test that clone works
		result, err := cloned.Process(context.Background(), map[string]any{"input": "test"})
		assert.NoError(t, err)
		assert.Equal(t, "true", result["classification"])
	})

	t.Run("signature and interface methods", func(t *testing.T) {
		ccm := &conditionalClassifierModule{condition: func(ctx context.Context, state map[string]interface{}) (bool, error) { return true, nil }}

		// Test GetSignature
		sig := ccm.GetSignature()
		assert.NotNil(t, sig)
		assert.Equal(t, 1, len(sig.Inputs))
		assert.Equal(t, 1, len(sig.Outputs))
		assert.Equal(t, "input", sig.Inputs[0].Name)
		assert.Equal(t, "classification", sig.Outputs[0].Name)

		// Test SetSignature and SetLLM (should not panic)
		ccm.SetSignature(core.Signature{})
		ccm.SetLLM(nil)
	})
}

// Additional tests for better coverage of builder.go edge cases

func TestWorkflowBuilder_ValidateStageIDEdgeCases(t *testing.T) {
	builder := NewBuilder(nil)

	// Test various invalid stage IDs
	testCases := []struct {
		id       string
		expected bool
	}{
		{"", true},           // Empty string should error
		{"  ", true},         // Whitespace only should error
		{"\t\n", true},       // Other whitespace should error
		{"valid_id", false},  // Valid ID should not error
		{"stage-1", false},   // Valid with dash should not error
		{"Stage_123", false}, // Valid with underscore and numbers should not error
	}

	for _, tc := range testCases {
		t.Run("ID: '"+tc.id+"'", func(t *testing.T) {
			err := builder.validateStageID(tc.id)
			if tc.expected {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}

	// Test duplicate ID detection
	builder.Stage("existing_id", NewAdvancedMockModule("test"))
	err := builder.validateStageID("existing_id")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "already exists")
}

func TestWorkflowBuilder_BuildValidationStageTypes(t *testing.T) {
	tests := []struct {
		name          string
		setupWorkflow func() *WorkflowBuilder
		expectError   bool
		errorContains string
	}{
		{
			name: "sequential stage validation - missing module",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:       "test",
					Type:     StageTypeSequential,
					Module:   nil, // Missing module
					Branches: make(map[string]*BuilderStage),
					Next:     make([]string, 0),
					Metadata: make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "sequential stage must have a module",
		},
		{
			name: "parallel stage validation - no steps",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:       "test",
					Type:     StageTypeParallel,
					Steps:    []*BuilderStep{}, // Empty steps
					Branches: make(map[string]*BuilderStage),
					Next:     make([]string, 0),
					Metadata: make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "parallel stage must have at least one step",
		},
		{
			name: "parallel stage validation - step with nil module",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:   "test",
					Type: StageTypeParallel,
					Steps: []*BuilderStep{
						{ID: "step1", Module: nil}, // Nil module
					},
					Branches: make(map[string]*BuilderStage),
					Next:     make([]string, 0),
					Metadata: make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "parallel step 'step1' must have a module",
		},
		{
			name: "conditional stage validation - missing condition",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:        "test",
					Type:      StageTypeConditional,
					Condition: nil, // Missing condition
					Branches:  make(map[string]*BuilderStage),
					Next:      make([]string, 0),
					Metadata:  make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "conditional stage must have a condition function",
		},
		{
			name: "conditional stage validation - no branches",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:   "test",
					Type: StageTypeConditional,
					Condition: func(ctx context.Context, state map[string]interface{}) (bool, error) {
						return true, nil
					},
					Branches: make(map[string]*BuilderStage), // Empty branches
					Next:     make([]string, 0),
					Metadata: make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "conditional stage must have at least one branch",
		},
		{
			name: "forEach stage validation - missing iterator",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:            "test",
					Type:          StageTypeForEach,
					IteratorFunc:  nil, // Missing iterator
					LoopBody:      NewBuilder(nil),
					MaxIterations: 10,
					Branches:      make(map[string]*BuilderStage),
					Next:          make([]string, 0),
					Metadata:      make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "forEach stage must have an iterator function",
		},
		{
			name: "forEach stage validation - missing loop body",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:   "test",
					Type: StageTypeForEach,
					IteratorFunc: func(ctx context.Context, state map[string]interface{}) ([]interface{}, error) {
						return []interface{}{"item"}, nil
					},
					LoopBody:      nil, // Missing loop body
					MaxIterations: 10,
					Branches:      make(map[string]*BuilderStage),
					Next:          make([]string, 0),
					Metadata:      make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "forEach stage must have a loop body",
		},
		{
			name: "forEach stage validation - invalid max iterations",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:   "test",
					Type: StageTypeForEach,
					IteratorFunc: func(ctx context.Context, state map[string]interface{}) ([]interface{}, error) {
						return []interface{}{"item"}, nil
					},
					LoopBody:      NewBuilder(nil),
					MaxIterations: 0, // Invalid max iterations
					Branches:      make(map[string]*BuilderStage),
					Next:          make([]string, 0),
					Metadata:      make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "forEach stage must have a positive maximum iteration count",
		},
		{
			name: "while stage validation - missing condition",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:            "test",
					Type:          StageTypeWhile,
					LoopCondition: nil, // Missing condition
					LoopBody:      NewBuilder(nil),
					MaxIterations: 10,
					Branches:      make(map[string]*BuilderStage),
					Next:          make([]string, 0),
					Metadata:      make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "while stage must have a loop condition function",
		},
		{
			name: "until stage validation - missing condition",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:            "test",
					Type:          StageTypeUntil,
					LoopCondition: nil, // Missing condition
					LoopBody:      NewBuilder(nil),
					MaxIterations: 10,
					Branches:      make(map[string]*BuilderStage),
					Next:          make([]string, 0),
					Metadata:      make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "until stage must have a loop condition function",
		},
		{
			name: "template stage validation - missing parameter function",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:               "test",
					Type:             StageTypeTemplate,
					TemplateParams:   nil, // Missing parameter function
					TemplateWorkflow: NewBuilder(nil),
					Branches:         make(map[string]*BuilderStage),
					Next:             make([]string, 0),
					Metadata:         make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "template stage must have a parameter function",
		},
		{
			name: "template stage validation - missing template workflow",
			setupWorkflow: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				stage := &BuilderStage{
					ID:   "test",
					Type: StageTypeTemplate,
					TemplateParams: func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error) {
						return map[string]interface{}{}, nil
					},
					TemplateWorkflow: nil, // Missing template workflow
					Branches:         make(map[string]*BuilderStage),
					Next:             make([]string, 0),
					Metadata:         make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["test"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "template stage must have a template workflow",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := tt.setupWorkflow()
			_, err := builder.Build()

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestWorkflowBuilder_DetermineWorkflowType(t *testing.T) {
	tests := []struct {
		name         string
		setupBuilder func() *WorkflowBuilder
		expectedType string
	}{
		{
			name: "composite workflow with forEach",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.ForEach("test", func(ctx context.Context, state map[string]interface{}) ([]interface{}, error) {
					return []interface{}{}, nil
				}, nil)
				return builder
			},
			expectedType: "composite",
		},
		{
			name: "composite workflow with while",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.While("test", func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error) {
					return false, nil
				}, nil)
				return builder
			},
			expectedType: "composite",
		},
		{
			name: "composite workflow with until",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Until("test", func(ctx context.Context, state map[string]interface{}, iteration int) (bool, error) {
					return false, nil
				}, nil)
				return builder
			},
			expectedType: "composite",
		},
		{
			name: "composite workflow with template",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Template("test", func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error) {
					return map[string]interface{}{}, nil
				}, nil)
				return builder
			},
			expectedType: "composite",
		},
		{
			name: "router workflow with conditional",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Conditional("test", func(ctx context.Context, state map[string]interface{}) (bool, error) {
					return true, nil
				}).If(NewAdvancedMockModule("test")).End()
				return builder
			},
			expectedType: "router",
		},
		{
			name: "parallel workflow with single parallel stage",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Parallel("test", NewStep("step1", NewAdvancedMockModule("step1")))
				return builder
			},
			expectedType: "parallel",
		},
		{
			name: "chain workflow with multiple stages",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Stage("stage1", NewAdvancedMockModule("stage1"))
				builder.Stage("stage2", NewAdvancedMockModule("stage2"))
				return builder
			},
			expectedType: "chain",
		},
		{
			name: "chain workflow with mixed parallel and sequential (multiple stages)",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Parallel("parallel1", NewStep("step1", NewAdvancedMockModule("step1")))
				builder.Stage("sequential1", NewAdvancedMockModule("sequential1"))
				return builder
			},
			expectedType: "chain",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := tt.setupBuilder()
			actualType := builder.determineWorkflowType()
			assert.Equal(t, tt.expectedType, actualType)
		})
	}
}

func TestWorkflowBuilder_BuildChainWorkflowEdgeCases(t *testing.T) {
	tests := []struct {
		name          string
		setupBuilder  func() *WorkflowBuilder
		expectError   bool
		errorContains string
	}{
		{
			name: "unsupported stage type in build",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)

				// Manually create a stage with unsupported type for chain workflow
				stage := &BuilderStage{
					ID:       "unsupported",
					Type:     StageType(999), // Invalid stage type
					Branches: make(map[string]*BuilderStage),
					Next:     make([]string, 0),
					Metadata: make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["unsupported"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "unsupported stage type for chain workflow",
		},
		{
			name: "parallel stage with no steps",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				// Manually create a parallel stage with no steps
				stage := &BuilderStage{
					ID:       "empty_parallel",
					Type:     StageTypeParallel,
					Steps:    []*BuilderStep{},
					Branches: make(map[string]*BuilderStage),
					Next:     make([]string, 0),
					Metadata: make(map[string]interface{}),
				}
				builder.stages = append(builder.stages, stage)
				builder.stepIndex["empty_parallel"] = stage
				return builder
			},
			expectError:   true,
			errorContains: "parallel stage must have at least one step",
		},
		{
			name: "chain workflow should build successfully",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Stage("stage1", NewAdvancedMockModule("stage1"))
				builder.Stage("stage2", NewAdvancedMockModule("stage2"))
				return builder
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := tt.setupBuilder()
			_, err := builder.Build()

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" && err != nil {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestWorkflowBuilder_BuildParallelWorkflowValidation(t *testing.T) {
	builder := NewBuilder(nil)

	// Create a builder with multiple stages (not pure parallel)
	builder.Stage("stage1", NewAdvancedMockModule("stage1"))
	builder.Parallel("parallel1", NewStep("step1", NewAdvancedMockModule("step1")))

	// Force workflow type to parallel (this simulates incorrect workflow type determination)
	builder.config = DefaultBuilderConfig()
	originalStages := builder.stages

	// Test buildParallelWorkflow with non-parallel configuration
	parallelStage := &BuilderStage{
		ID:   "parallel_only",
		Type: StageTypeParallel,
		Steps: []*BuilderStep{
			NewStep("step1", NewAdvancedMockModule("step1")),
		},
		Branches: make(map[string]*BuilderStage),
		Next:     make([]string, 0),
		Metadata: make(map[string]interface{}),
	}

	// Test with multiple stages (should fail)
	builder.stages = originalStages
	result, err := builder.buildParallelWorkflow()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "buildParallelWorkflow called for non-parallel workflow")
	assert.Nil(t, result)

	// Test with single non-parallel stage (should fail)
	builder.stages = []*BuilderStage{
		{
			ID:     "sequential",
			Type:   StageTypeSequential,
			Module: NewAdvancedMockModule("test"),
		},
	}
	result, err = builder.buildParallelWorkflow()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "buildParallelWorkflow called for non-parallel workflow")
	assert.Nil(t, result)

	// Test with correct single parallel stage (should succeed)
	builder.stages = []*BuilderStage{parallelStage}
	result, err = builder.buildParallelWorkflow()
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestWorkflowBuilder_BuildRouterWorkflowValidation(t *testing.T) {
	tests := []struct {
		name          string
		setupBuilder  func() *WorkflowBuilder
		expectError   bool
		errorContains string
	}{
		{
			name: "no conditional stages",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Stage("stage1", NewAdvancedMockModule("stage1"))
				return builder
			},
			expectError:   true,
			errorContains: "router workflow requires at least one conditional stage",
		},
		{
			name: "conditional stage with branches containing nil modules",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				conditional := builder.Conditional("test", func(ctx context.Context, state map[string]interface{}) (bool, error) {
					return true, nil
				})
				// Add a branch with nil module
				conditional.stage.Branches["true"] = &BuilderStage{
					ID:     "nil_module_branch",
					Type:   StageTypeSequential,
					Module: nil, // Nil module - should be skipped
				}
				conditional.stage.Branches["false"] = &BuilderStage{
					ID:     "valid_branch",
					Type:   StageTypeSequential,
					Module: NewAdvancedMockModule("valid"),
				}
				conditional.End()
				return builder
			},
			expectError: false, // Should succeed, nil module branches are skipped
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := tt.setupBuilder()

			// Force router workflow type
			originalType := builder.determineWorkflowType()
			if originalType != "router" {
				// Manually build router workflow
				result, err := builder.buildRouterWorkflow()
				if tt.expectError {
					assert.Error(t, err)
					if tt.errorContains != "" {
						assert.Contains(t, err.Error(), tt.errorContains)
					}
				} else {
					assert.NoError(t, err)
					assert.NotNil(t, result)
				}
			} else {
				// Use normal Build() path
				_, err := builder.Build()
				if tt.expectError {
					assert.Error(t, err)
					if tt.errorContains != "" {
						assert.Contains(t, err.Error(), tt.errorContains)
					}
				} else {
					assert.NoError(t, err)
				}
			}
		})
	}
}

func TestWorkflowBuilder_CycleDetectionEdgeCases(t *testing.T) {
	tests := []struct {
		name         string
		setupBuilder func() *WorkflowBuilder
		expectCycle  bool
	}{
		{
			name: "self-referencing stage",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Stage("self", NewAdvancedMockModule("self")).Then("self")
				return builder
			},
			expectCycle: true,
		},
		{
			name: "complex cycle with three stages",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Stage("a", NewAdvancedMockModule("a")).Then("b")
				builder.Stage("b", NewAdvancedMockModule("b")).Then("c")
				builder.Stage("c", NewAdvancedMockModule("c")).Then("a") // Creates cycle
				return builder
			},
			expectCycle: true,
		},
		{
			name: "no cycle - linear chain",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Stage("a", NewAdvancedMockModule("a")).Then("b")
				builder.Stage("b", NewAdvancedMockModule("b")).Then("c")
				builder.Stage("c", NewAdvancedMockModule("c"))
				return builder
			},
			expectCycle: false,
		},
		{
			name: "no cycle - tree structure",
			setupBuilder: func() *WorkflowBuilder {
				builder := NewBuilder(nil)
				builder.Stage("root", NewAdvancedMockModule("root")).Then("left").Then("right")
				builder.Stage("left", NewAdvancedMockModule("left"))
				builder.Stage("right", NewAdvancedMockModule("right"))
				return builder
			},
			expectCycle: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := tt.setupBuilder()

			err := builder.checkForCycles()
			if tt.expectCycle {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "cycle")
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestWorkflowBuilder_ValidationDisabled(t *testing.T) {
	builder := NewBuilder(nil)

	// Disable validation
	config := DefaultBuilderConfig()
	config.EnableValidation = false
	builder.WithConfig(config)

	// Create a workflow with cycles (would normally fail)
	builder.Stage("a", NewAdvancedMockModule("a")).Then("b")
	builder.Stage("b", NewAdvancedMockModule("b")).Then("a") // Creates cycle

	// Should succeed because validation is disabled
	workflow, err := builder.Build()
	assert.NoError(t, err)
	assert.NotNil(t, workflow)
}

func TestWorkflowBuilder_OptimizationPlaceholder(t *testing.T) {
	builder := NewBuilder(nil)

	// Enable optimization
	config := DefaultBuilderConfig()
	config.EnableOptimization = true
	builder.WithConfig(config)

	builder.Stage("stage1", NewAdvancedMockModule("stage1"))

	// Test that optimize() doesn't break anything (it's currently a placeholder)
	builder.optimize()

	// Should still build successfully
	workflow, err := builder.Build()
	assert.NoError(t, err)
	assert.NotNil(t, workflow)
}

func TestWorkflowBuilder_ConditionalBuilderErrorHandling(t *testing.T) {
	builder := NewBuilder(nil)

	// Create a conditional builder with error
	conditionalBuilder := &ConditionalBuilder{
		parent:   builder,
		hasError: true,
	}

	// All methods should return the same builder when hasError is true
	result1 := conditionalBuilder.If(NewAdvancedMockModule("test"))
	assert.Equal(t, conditionalBuilder, result1)
	assert.True(t, result1.hasError)

	result2 := conditionalBuilder.Else(NewAdvancedMockModule("test"))
	assert.Equal(t, conditionalBuilder, result2)
	assert.True(t, result2.hasError)

	result3 := conditionalBuilder.ElseIf(func(ctx context.Context, state map[string]interface{}) (bool, error) {
		return true, nil
	}, NewAdvancedMockModule("test"))
	assert.Equal(t, conditionalBuilder, result3)
	assert.True(t, result3.hasError)

	// End should still return parent
	parent := conditionalBuilder.End()
	assert.Equal(t, builder, parent)
}

func TestWorkflowBuilder_ElseIfBranchNaming(t *testing.T) {
	builder := NewBuilder(nil)

	conditional := builder.Conditional("test", func(ctx context.Context, state map[string]interface{}) (bool, error) {
		return true, nil
	})

	// Add multiple ElseIf branches
	conditional.ElseIf(func(ctx context.Context, state map[string]interface{}) (bool, error) {
		return false, nil
	}, NewAdvancedMockModule("elseif1"))

	conditional.ElseIf(func(ctx context.Context, state map[string]interface{}) (bool, error) {
		return false, nil
	}, NewAdvancedMockModule("elseif2"))

	conditional.End()

	// Check that branches are named correctly
	stage := builder.stages[0]
	assert.Contains(t, stage.Branches, "elseif_0")
	assert.Contains(t, stage.Branches, "elseif_1")

	// Check branch IDs
	assert.Equal(t, "test_elseif_0", stage.Branches["elseif_0"].ID)
	assert.Equal(t, "test_elseif_1", stage.Branches["elseif_1"].ID)
}
