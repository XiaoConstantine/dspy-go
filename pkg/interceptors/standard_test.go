package interceptors

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

func TestLoggingModuleInterceptor(t *testing.T) {
	interceptor := LoggingModuleInterceptor()

	// Create test data
	ctx := core.WithExecutionState(context.Background())
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	// Create mock handler
	handlerCalled := false
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		handlerCalled = true
		return map[string]any{"result": "success"}, nil
	}

	// Test successful execution
	result, err := interceptor(ctx, inputs, info, handler)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !handlerCalled {
		t.Error("Handler should have been called")
	}
	if result["result"] != "success" {
		t.Errorf("Expected result 'success', got %v", result["result"])
	}
}

func TestLoggingModuleInterceptorWithError(t *testing.T) {
	interceptor := LoggingModuleInterceptor()

	ctx := core.WithExecutionState(context.Background())
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	// Create mock handler that returns error
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return nil, errors.New("test error")
	}

	// Test error handling
	result, err := interceptor(ctx, inputs, info, handler)

	if err == nil {
		t.Error("Expected error, got nil")
	}
	if result != nil {
		t.Errorf("Expected nil result, got %v", result)
	}
}

func TestLoggingAgentInterceptor(t *testing.T) {
	interceptor := LoggingAgentInterceptor()

	ctx := core.WithExecutionState(context.Background())
	input := map[string]interface{}{"test": "value"}
	info := core.NewAgentInfo("TestAgent", "TestType", []core.Tool{})

	handlerCalled := false
	handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
		handlerCalled = true
		return map[string]interface{}{"result": "success"}, nil
	}

	result, err := interceptor(ctx, input, info, handler)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !handlerCalled {
		t.Error("Handler should have been called")
	}
	if result["result"] != "success" {
		t.Errorf("Expected result 'success', got %v", result["result"])
	}
}

func TestLoggingToolInterceptor(t *testing.T) {
	interceptor := LoggingToolInterceptor()

	ctx := core.WithExecutionState(context.Background())
	args := map[string]interface{}{"test": "value"}
	info := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})

	handlerCalled := false
	handler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		handlerCalled = true
		return core.ToolResult{Data: "success"}, nil
	}

	result, err := interceptor(ctx, args, info, handler)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !handlerCalled {
		t.Error("Handler should have been called")
	}
	if result.Data != "success" {
		t.Errorf("Expected result data 'success', got %v", result.Data)
	}
}

func TestTracingModuleInterceptor(t *testing.T) {
	interceptor := TracingModuleInterceptor()

	ctx := core.WithExecutionState(context.Background())
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	handlerCalled := false
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		handlerCalled = true
		// Verify that span context is available
		state := core.GetExecutionState(ctx)
		if state == nil {
			t.Error("Expected execution state in context")
		}
		return map[string]any{"result": "success"}, nil
	}

	result, err := interceptor(ctx, inputs, info, handler)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !handlerCalled {
		t.Error("Handler should have been called")
	}
	if result["result"] != "success" {
		t.Errorf("Expected result 'success', got %v", result["result"])
	}

	// Verify span was created
	spans := core.CollectSpans(ctx)
	if len(spans) == 0 {
		t.Error("Expected at least one span to be created")
	}
}

func TestTracingModuleInterceptorWithError(t *testing.T) {
	interceptor := TracingModuleInterceptor()

	ctx := core.WithExecutionState(context.Background())
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return nil, errors.New("test error")
	}

	_, err := interceptor(ctx, inputs, info, handler)

	if err == nil {
		t.Error("Expected error, got nil")
	}

	// Verify span recorded the error
	spans := core.CollectSpans(ctx)
	if len(spans) == 0 {
		t.Error("Expected at least one span to be created")
	}

	span := spans[len(spans)-1]
	if span.Error == nil {
		t.Error("Expected span to record error")
	}
}

func TestTracingAgentInterceptor(t *testing.T) {
	interceptor := TracingAgentInterceptor()

	ctx := core.WithExecutionState(context.Background())
	input := map[string]interface{}{"test": "value"}
	info := core.NewAgentInfo("TestAgent", "TestType", []core.Tool{})

	handlerCalled := false
	handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
		handlerCalled = true
		return map[string]interface{}{"result": "success"}, nil
	}

	_, err := interceptor(ctx, input, info, handler)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !handlerCalled {
		t.Error("Handler should have been called")
	}

	// Verify span was created
	spans := core.CollectSpans(ctx)
	if len(spans) == 0 {
		t.Error("Expected at least one span to be created")
	}
}

func TestTracingToolInterceptor(t *testing.T) {
	interceptor := TracingToolInterceptor()

	ctx := core.WithExecutionState(context.Background())
	args := map[string]interface{}{"test": "value"}
	info := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})

	handlerCalled := false
	handler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		handlerCalled = true
		return core.ToolResult{Data: "success"}, nil
	}

	_, err := interceptor(ctx, args, info, handler)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !handlerCalled {
		t.Error("Handler should have been called")
	}

	// Verify span was created
	spans := core.CollectSpans(ctx)
	if len(spans) == 0 {
		t.Error("Expected at least one span to be created")
	}
}

func TestMetricsModuleInterceptor(t *testing.T) {
	interceptor := MetricsModuleInterceptor()

	ctx := core.WithExecutionState(context.Background())
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	handlerCalled := false
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		handlerCalled = true
		time.Sleep(10 * time.Millisecond) // Simulate some work
		return map[string]any{"result": "success"}, nil
	}

	_, err := interceptor(ctx, inputs, info, handler)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !handlerCalled {
		t.Error("Handler should have been called")
	}

	// Verify metrics were recorded - the metrics interceptor doesn't create spans
	// it just records metrics in the execution state
}

func TestMetricsAgentInterceptor(t *testing.T) {
	interceptor := MetricsAgentInterceptor()

	ctx := core.WithExecutionState(context.Background())
	input := map[string]interface{}{"test": "value"}
	info := core.NewAgentInfo("TestAgent", "TestType", []core.Tool{})

	handlerCalled := false
	handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
		handlerCalled = true
		time.Sleep(5 * time.Millisecond)
		return map[string]interface{}{"result": "success"}, nil
	}

	_, err := interceptor(ctx, input, info, handler)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !handlerCalled {
		t.Error("Handler should have been called")
	}
}

func TestMetricsToolInterceptor(t *testing.T) {
	interceptor := MetricsToolInterceptor()

	ctx := core.WithExecutionState(context.Background())
	args := map[string]interface{}{"test": "value"}
	info := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})

	handlerCalled := false
	handler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		handlerCalled = true
		time.Sleep(5 * time.Millisecond)
		return core.ToolResult{Data: "success"}, nil
	}

	_, err := interceptor(ctx, args, info, handler)

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !handlerCalled {
		t.Error("Handler should have been called")
	}
}

func TestStandardHelperFunctions(t *testing.T) {
	// Test getInputFieldNames
	inputs := map[string]any{"field1": "value1", "field2": "value2"}
	fieldNames := getInputFieldNames(inputs)

	if len(fieldNames) != 2 {
		t.Errorf("Expected 2 field names, got %d", len(fieldNames))
	}

	// Test getOutputFieldNames
	outputs := map[string]any{"output1": "result1", "output2": "result2"}
	outputNames := getOutputFieldNames(outputs)

	if len(outputNames) != 2 {
		t.Errorf("Expected 2 output names, got %d", len(outputNames))
	}

	// Test getMapKeys
	m := map[string]interface{}{"key1": "value1", "key2": "value2"}
	keys := getMapKeys(m)

	if len(keys) != 2 {
		t.Errorf("Expected 2 keys, got %d", len(keys))
	}
}

// Benchmark tests.
func BenchmarkLoggingModuleInterceptor(b *testing.B) {
	interceptor := LoggingModuleInterceptor()
	ctx := core.WithExecutionState(context.Background())
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{"result": "success"}, nil
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := interceptor(ctx, inputs, info, handler)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTracingModuleInterceptor(b *testing.B) {
	interceptor := TracingModuleInterceptor()
	ctx := core.WithExecutionState(context.Background())
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{"result": "success"}, nil
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := interceptor(ctx, inputs, info, handler)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMetricsModuleInterceptor(b *testing.B) {
	interceptor := MetricsModuleInterceptor()
	ctx := core.WithExecutionState(context.Background())
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{"result": "success"}, nil
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := interceptor(ctx, inputs, info, handler)
		if err != nil {
			b.Fatal(err)
		}
	}
}
func TestMetricsWithTokenUsage(t *testing.T) {
	interceptor := MetricsModuleInterceptor()

	ctx := core.WithExecutionState(context.Background())
	state := core.GetExecutionState(ctx)
	// Set token usage to test metrics collection
	state.WithTokenUsage(&core.TokenUsage{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
		Cost:             0.01,
	})

	inputs := map[string]any{"test": "value", "other": "data"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{"result": "success", "data": "output"}, nil
	}

	_, err := interceptor(ctx, inputs, info, handler)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
}
