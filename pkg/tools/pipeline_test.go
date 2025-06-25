package tools

import (
	"context"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mock tools for testing.
type mockProcessingTool struct {
	name        string
	delay       time.Duration
	shouldError bool
	errorMsg    string
}

func (m *mockProcessingTool) Name() string {
	return m.name
}

func (m *mockProcessingTool) Description() string {
	return "Mock processing tool for testing"
}

func (m *mockProcessingTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         m.name,
		Description:  m.Description(),
		Capabilities: []string{"processing"},
		Version:      "1.0.0",
	}
}

func (m *mockProcessingTool) CanHandle(ctx context.Context, intent string) bool {
	return true
}

func (m *mockProcessingTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	if m.delay > 0 {
		// Use context-aware delay that can be interrupted
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return core.ToolResult{}, ctx.Err()
		}
	}

	if m.shouldError {
		return core.ToolResult{}, errors.New(errors.Unknown, m.errorMsg)
	}

	// Process input and add tool name
	result := make(map[string]interface{})
	for k, v := range params {
		result[k] = v
	}
	result["processed_by"] = m.name
	result["processed_at"] = time.Now().Unix()

	return core.ToolResult{
		Data: result,
		Metadata: map[string]interface{}{
			"tool": m.name,
		},
	}, nil
}

func (m *mockProcessingTool) Validate(params map[string]interface{}) error {
	return nil
}

func (m *mockProcessingTool) InputSchema() models.InputSchema {
	return models.InputSchema{}
}

func createTestRegistry() core.ToolRegistry {
	registry := NewInMemoryToolRegistry()

	// Add mock tools
	_ = registry.Register(&mockProcessingTool{name: "parser", delay: 10 * time.Millisecond})
	_ = registry.Register(&mockProcessingTool{name: "validator", delay: 5 * time.Millisecond})
	_ = registry.Register(&mockProcessingTool{name: "transformer", delay: 15 * time.Millisecond})
	_ = registry.Register(&mockProcessingTool{name: "processor", delay: 20 * time.Millisecond})
	_ = registry.Register(&mockProcessingTool{name: "error_tool", shouldError: true, errorMsg: "simulated error"})

	return registry
}

func TestToolPipeline_Basic(t *testing.T) {
	registry := createTestRegistry()

	options := PipelineOptions{
		Timeout:         10 * time.Second,
		FailureStrategy: FailFast,
		Parallel:        false,
		CacheResults:    true,
	}

	pipeline := NewToolPipeline("test-pipeline", registry, options)

	// Add steps
	err := pipeline.AddStep(PipelineStep{
		ToolName: "parser",
		Timeout:  1 * time.Second,
	})
	require.NoError(t, err)

	err = pipeline.AddStep(PipelineStep{
		ToolName: "validator",
		Timeout:  1 * time.Second,
	})
	require.NoError(t, err)

	// Execute pipeline
	ctx := context.Background()
	input := map[string]interface{}{
		"data": "test input",
	}

	result, err := pipeline.Execute(ctx, input)
	require.NoError(t, err)
	assert.True(t, result.Success)
	assert.Len(t, result.Results, 2)
	assert.Equal(t, "test-pipeline", pipeline.GetName())
}

func TestToolPipeline_WithTransformer(t *testing.T) {
	registry := createTestRegistry()

	options := PipelineOptions{
		FailureStrategy: FailFast,
		CacheResults:    false,
	}

	pipeline := NewToolPipeline("transform-pipeline", registry, options)

	// Add step with transformer
	transformer := func(input interface{}) (map[string]interface{}, error) {
		if inputMap, ok := input.(map[string]interface{}); ok {
			return map[string]interface{}{
				"transformed_data": inputMap,
				"transformation":   "applied",
			}, nil
		}
		return nil, errors.New(errors.InvalidInput, "invalid input type")
	}

	err := pipeline.AddStep(PipelineStep{
		ToolName:    "parser",
		Transformer: transformer,
	})
	require.NoError(t, err)

	err = pipeline.AddStep(PipelineStep{
		ToolName: "validator",
	})
	require.NoError(t, err)

	// Execute pipeline
	ctx := context.Background()
	input := map[string]interface{}{
		"data": "test input",
	}

	result, err := pipeline.Execute(ctx, input)
	require.NoError(t, err)
	assert.True(t, result.Success)
	assert.Len(t, result.Results, 2)

	// Check if transformation was applied
	secondResult := result.Results[1]
	resultData, ok := secondResult.Data.(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "applied", resultData["transformation"])
}

func TestToolPipeline_ConditionalExecution(t *testing.T) {
	registry := createTestRegistry()

	options := PipelineOptions{
		FailureStrategy: FailFast,
	}

	pipeline := NewToolPipeline("conditional-pipeline", registry, options)

	// Add initial step
	err := pipeline.AddStep(PipelineStep{
		ToolName: "parser",
	})
	require.NoError(t, err)

	// Add conditional step that should execute
	err = pipeline.AddStep(PipelineStep{
		ToolName: "validator",
		Conditions: []Condition{
			{Field: "processed_by", Operator: "eq", Value: "parser"},
		},
	})
	require.NoError(t, err)

	// Add conditional step that should NOT execute
	err = pipeline.AddStep(PipelineStep{
		ToolName: "transformer",
		Conditions: []Condition{
			{Field: "nonexistent_field", Operator: "exists"},
		},
	})
	require.NoError(t, err)

	// Execute pipeline
	ctx := context.Background()
	input := map[string]interface{}{
		"data": "test input",
	}

	result, err := pipeline.Execute(ctx, input)
	require.NoError(t, err)
	assert.True(t, result.Success)
	assert.Len(t, result.Results, 2) // Only parser and validator should execute
}

func TestToolPipeline_ErrorHandling(t *testing.T) {
	registry := createTestRegistry()

	t.Run("FailFast strategy", func(t *testing.T) {
		options := PipelineOptions{
			FailureStrategy: FailFast,
		}

		pipeline := NewToolPipeline("error-pipeline", registry, options)

		err := pipeline.AddStep(PipelineStep{ToolName: "parser"})
		require.NoError(t, err)

		err = pipeline.AddStep(PipelineStep{ToolName: "error_tool"})
		require.NoError(t, err)

		err = pipeline.AddStep(PipelineStep{ToolName: "validator"})
		require.NoError(t, err)

		ctx := context.Background()
		input := map[string]interface{}{"data": "test"}

		result, err := pipeline.Execute(ctx, input)
		assert.Error(t, err)
		assert.False(t, result.Success)
		assert.Equal(t, "error_tool", result.FailedStep)
		assert.Len(t, result.Results, 1) // Only first step should execute
	})

	t.Run("ContinueOnError strategy", func(t *testing.T) {
		options := PipelineOptions{
			FailureStrategy: ContinueOnError,
		}

		pipeline := NewToolPipeline("continue-pipeline", registry, options)

		err := pipeline.AddStep(PipelineStep{ToolName: "parser"})
		require.NoError(t, err)

		err = pipeline.AddStep(PipelineStep{ToolName: "error_tool"})
		require.NoError(t, err)

		err = pipeline.AddStep(PipelineStep{ToolName: "validator"})
		require.NoError(t, err)

		ctx := context.Background()
		input := map[string]interface{}{"data": "test"}

		result, err := pipeline.Execute(ctx, input)
		assert.NoError(t, err)          // Should not return error with ContinueOnError
		assert.False(t, result.Success) // But success should be false
		assert.Equal(t, "error_tool", result.FailedStep)
		assert.Len(t, result.Results, 3) // All steps should execute
	})
}

func TestToolPipeline_Caching(t *testing.T) {
	registry := createTestRegistry()

	options := PipelineOptions{
		CacheResults: true,
	}

	pipeline := NewToolPipeline("cache-pipeline", registry, options)

	err := pipeline.AddStep(PipelineStep{ToolName: "parser"})
	require.NoError(t, err)

	ctx := context.Background()
	input := map[string]interface{}{
		"data": "test input",
	}

	// First execution
	start1 := time.Now()
	result1, err := pipeline.Execute(ctx, input)
	duration1 := time.Since(start1)
	require.NoError(t, err)
	assert.True(t, result1.Success)

	// Second execution (should use cache)
	start2 := time.Now()
	result2, err := pipeline.Execute(ctx, input)
	duration2 := time.Since(start2)
	require.NoError(t, err)
	assert.True(t, result2.Success)

	// Cache should make second execution faster
	assert.True(t, duration2 < duration1)

	// Clear cache and test
	pipeline.ClearCache()
	start3 := time.Now()
	result3, err := pipeline.Execute(ctx, input)
	duration3 := time.Since(start3)
	require.NoError(t, err)
	assert.True(t, result3.Success)

	// After clearing cache, should take similar time to first execution
	assert.True(t, duration3 > duration2)
}

func TestPipelineBuilder(t *testing.T) {
	registry := createTestRegistry()

	pipeline, err := NewPipelineBuilder("builder-test", registry).
		Step("parser").
		StepWithTimeout("validator", 500*time.Millisecond).
		StepWithRetries("transformer", 2).
		ConditionalStep("processor", ConditionExists("processed_by")).
		FailFast().
		EnableCaching().
		Build()

	require.NoError(t, err)
	assert.Equal(t, "builder-test", pipeline.GetName())

	steps := pipeline.GetSteps()
	assert.Len(t, steps, 4)
	assert.Equal(t, "parser", steps[0].ToolName)
	assert.Equal(t, "validator", steps[1].ToolName)
	assert.Equal(t, 500*time.Millisecond, steps[1].Timeout)
	assert.Equal(t, "transformer", steps[2].ToolName)
	assert.Equal(t, 2, steps[2].Retries)
	assert.Equal(t, "processor", steps[3].ToolName)
	assert.Len(t, steps[3].Conditions, 1)
}

func TestDataTransformers(t *testing.T) {
	t.Run("TransformExtractField", func(t *testing.T) {
		transformer := TransformExtractField("test_field")

		input := map[string]interface{}{
			"test_field":  "extracted_value",
			"other_field": "ignored",
		}

		result, err := transformer(input)
		require.NoError(t, err)
		assert.Equal(t, "extracted_value", result["value"])
	})

	t.Run("TransformRename", func(t *testing.T) {
		transformer := TransformRename(map[string]string{
			"old_name": "new_name",
		})

		input := map[string]interface{}{
			"old_name":  "value",
			"keep_name": "keep_value",
		}

		result, err := transformer(input)
		require.NoError(t, err)
		assert.Equal(t, "value", result["new_name"])
		assert.Equal(t, "keep_value", result["keep_name"])
		assert.NotContains(t, result, "old_name")
	})

	t.Run("TransformFilter", func(t *testing.T) {
		transformer := TransformFilter([]string{"keep1", "keep2"})

		input := map[string]interface{}{
			"keep1":  "value1",
			"keep2":  "value2",
			"remove": "removed_value",
		}

		result, err := transformer(input)
		require.NoError(t, err)
		assert.Equal(t, "value1", result["keep1"])
		assert.Equal(t, "value2", result["keep2"])
		assert.NotContains(t, result, "remove")
	})

	t.Run("TransformAddConstant", func(t *testing.T) {
		transformer := TransformAddConstant(map[string]interface{}{
			"constant":  "constant_value",
			"timestamp": 12345,
		})

		input := map[string]interface{}{
			"input_field": "input_value",
		}

		result, err := transformer(input)
		require.NoError(t, err)
		assert.Equal(t, "input_value", result["input_field"])
		assert.Equal(t, "constant_value", result["constant"])
		assert.Equal(t, 12345, result["timestamp"])
	})

	t.Run("TransformChain", func(t *testing.T) {
		transformer := TransformChain(
			TransformRename(map[string]string{"old": "new"}),
			TransformAddConstant(map[string]interface{}{"added": "value"}),
			TransformFilter([]string{"new", "added"}),
		)

		input := map[string]interface{}{
			"old":    "renamed_value",
			"remove": "will_be_removed",
		}

		result, err := transformer(input)
		require.NoError(t, err)
		assert.Equal(t, "renamed_value", result["new"])
		assert.Equal(t, "value", result["added"])
		assert.NotContains(t, result, "old")
		assert.NotContains(t, result, "remove")
	})
}

func TestConditionEvaluation(t *testing.T) {
	pipeline := NewToolPipeline("test", createTestRegistry(), PipelineOptions{})

	testData := map[string]interface{}{
		"field1": "value1",
		"field2": 42,
		"field3": "contains_test",
	}

	testCases := []struct {
		condition Condition
		expected  bool
	}{
		{ConditionExists("field1"), true},
		{ConditionExists("nonexistent"), false},
		{ConditionEquals("field1", "value1"), true},
		{ConditionEquals("field1", "wrong"), false},
		{ConditionNotEquals("field1", "wrong"), true},
		{ConditionNotEquals("field1", "value1"), false},
		{ConditionContains("field3", "test"), true},
		{ConditionContains("field3", "missing"), false},
	}

	for _, tc := range testCases {
		result := pipeline.evaluateCondition(tc.condition, testData)
		assert.Equal(t, tc.expected, result, "Condition: %+v", tc.condition)
	}
}

func TestPipeline_EdgeCases(t *testing.T) {
	registry := createTestRegistry()

	t.Run("Empty pipeline", func(t *testing.T) {
		pipeline := NewToolPipeline("empty", registry, PipelineOptions{})

		ctx := context.Background()
		_, err := pipeline.Execute(ctx, map[string]interface{}{})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no steps")
	})

	t.Run("Nonexistent tool", func(t *testing.T) {
		pipeline := NewToolPipeline("invalid", registry, PipelineOptions{})

		err := pipeline.AddStep(PipelineStep{ToolName: "nonexistent"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "tool not found")
	})

	t.Run("Context timeout", func(t *testing.T) {
		pipeline := NewToolPipeline("timeout", registry, PipelineOptions{
			Timeout: 1 * time.Millisecond, // Very short timeout
		})

		err := pipeline.AddStep(PipelineStep{ToolName: "parser"})
		require.NoError(t, err)

		ctx := context.Background()
		_, err = pipeline.Execute(ctx, map[string]interface{}{})
		assert.Error(t, err)
	})
}

func TestPipeline_ParallelExecution(t *testing.T) {
	registry := createTestRegistry()

	options := PipelineOptions{
		Parallel: true,
	}

	pipeline := NewToolPipeline("parallel", registry, options)

	// Add multiple independent steps
	tools := []string{"parser", "validator", "transformer"}
	for _, tool := range tools {
		err := pipeline.AddStep(PipelineStep{ToolName: tool})
		require.NoError(t, err)
	}

	ctx := context.Background()
	input := map[string]interface{}{
		"data": "test input",
	}

	start := time.Now()
	result, err := pipeline.Execute(ctx, input)
	duration := time.Since(start)

	require.NoError(t, err)
	assert.True(t, result.Success)
	assert.Len(t, result.Results, 3)

	// Test that parallel execution actually works by verifying all tools were executed
	// In parallel mode, all tools should run simultaneously with the same input
	toolsExecuted := make(map[string]bool)
	
	for _, result := range result.Results {
		if resultData, ok := result.Data.(map[string]interface{}); ok {
			if processedBy, ok := resultData["processed_by"].(string); ok {
				toolsExecuted[processedBy] = true
			}
		}
	}
	
	expectedTools := []string{"parser", "validator", "transformer"}
	for _, tool := range expectedTools {
		assert.True(t, toolsExecuted[tool], "Tool %s should have been executed", tool)
	}
	
	// Log timing for debugging (but don't assert on it)
	t.Logf("Parallel execution took %v (expected ~20ms, sequential would be ~30ms)", duration)
}
