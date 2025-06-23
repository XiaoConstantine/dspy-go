package workflows

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mock module for testing advanced patterns
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
		name      string
		condition LoopConditionFunc
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
			name:      "nil condition",
			condition: nil,
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
		name      string
		condition LoopConditionFunc
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
			name:      "nil condition",
			condition: nil,
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
		name         string
		parameterFunc TemplateParameterFunc
		expectError  bool
	}{
		{
			name: "valid template",
			parameterFunc: func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error) {
				return map[string]interface{}{"template_param": "value"}, nil
			},
			expectError: false,
		},
		{
			name:         "nil parameter function",
			parameterFunc: nil,
			expectError:  true,
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
						"output": "processed_item",
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
						"count": 1,
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
					"multiplier": 2,
				}, nil
			},
			func(templateBuilder *WorkflowBuilder) *WorkflowBuilder {
				return templateBuilder.Stage("apply_template", 
					NewAdvancedMockModule("template_applier").WithOutputs(map[string]interface{}{
						"output": "template_applied",
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
			"output": "processed_high",
			"route_taken": "high_confidence",
		})).
		Else(NewAdvancedMockModule("low_confidence").WithOutputs(map[string]interface{}{
			"output": "processed_low",
			"route_taken": "low_confidence",
		})).
		End().
		Build()

	require.NoError(t, err)
	require.NotNil(t, workflow)

	// Test high confidence routing
	result, err := workflow.Execute(context.Background(), map[string]interface{}{
		"input": "test_data",
		"confidence": 0.9,
	})

	require.NoError(t, err)
	assert.Equal(t, "high_confidence", result["route_taken"])

	// Test low confidence routing
	result, err = workflow.Execute(context.Background(), map[string]interface{}{
		"input": "test_data",
		"confidence": 0.5,
	})

	require.NoError(t, err)
	assert.Equal(t, "low_confidence", result["route_taken"])
}

func TestWorkflowBuilder_AdvancedValidation(t *testing.T) {
	tests := []struct {
		name         string
		buildWorkflow func() *WorkflowBuilder
		expectError  bool
		errorContains string
	}{
		{
			name: "forEach without iterator function",
			buildWorkflow: func() *WorkflowBuilder {
				return NewBuilder(nil).ForEach("test", nil, nil)
			},
			expectError: true,
			errorContains: "iterator function",
		},
		{
			name: "while without condition",
			buildWorkflow: func() *WorkflowBuilder {
				return NewBuilder(nil).While("test", nil, nil)
			},
			expectError: true,
			errorContains: "condition function",
		},
		{
			name: "template without parameter function",
			buildWorkflow: func() *WorkflowBuilder {
				return NewBuilder(nil).Template("test", nil, nil)
			},
			expectError: true,
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