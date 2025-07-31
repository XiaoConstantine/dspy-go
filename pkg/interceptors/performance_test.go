package interceptors

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

func TestMemoryCache(t *testing.T) {
	cache := NewMemoryCache()

	// Test Set and Get
	cache.Set("key1", "value1", time.Second)

	value, found := cache.Get("key1")
	if !found {
		t.Error("Expected to find key1")
	}
	if value != "value1" {
		t.Errorf("Expected 'value1', got %v", value)
	}

	// Test non-existent key
	_, found = cache.Get("nonexistent")
	if found {
		t.Error("Expected not to find nonexistent key")
	}

	// Test expiration
	cache.Set("expire", "value", 50*time.Millisecond)
	time.Sleep(100 * time.Millisecond)

	_, found = cache.Get("expire")
	if found {
		t.Error("Expected expired key to not be found")
	}

	// Test Delete
	cache.Set("delete_me", "value", time.Second)
	cache.Delete("delete_me")

	_, found = cache.Get("delete_me")
	if found {
		t.Error("Expected deleted key to not be found")
	}

	// Test Clear
	cache.Set("key1", "value1", time.Second)
	cache.Set("key2", "value2", time.Second)
	cache.Clear()

	_, found = cache.Get("key1")
	if found {
		t.Error("Expected cache to be cleared")
	}
}

func TestCircuitBreaker(t *testing.T) {
	cb := NewCircuitBreaker(2, 100*time.Millisecond, 1)

	// Initially closed
	if cb.GetState() != CircuitClosed {
		t.Error("Expected circuit to be initially closed")
	}

	// Should allow requests when closed
	if !cb.Allow() {
		t.Error("Expected request to be allowed when circuit is closed")
	}

	// Record failures to trip the circuit
	cb.RecordFailure()
	cb.RecordFailure()

	// Should be open after threshold failures
	if cb.GetState() != CircuitOpen {
		t.Error("Expected circuit to be open after failures")
	}

	// Should block requests when open
	if cb.Allow() {
		t.Error("Expected request to be blocked when circuit is open")
	}

	// Wait for reset timeout
	time.Sleep(150 * time.Millisecond)

	// Should allow limited requests when half-open
	if !cb.Allow() {
		t.Error("Expected request to be allowed when circuit transitions to half-open")
	}

	if cb.GetState() != CircuitHalfOpen {
		t.Error("Expected circuit to be half-open")
	}

	// Success should close the circuit
	cb.RecordSuccess()

	if cb.GetState() != CircuitClosed {
		t.Error("Expected circuit to be closed after success")
	}
}

func TestCachingModuleInterceptor(t *testing.T) {
	cache := NewMemoryCache()
	interceptor := CachingModuleInterceptor(cache, time.Second)

	ctx := context.Background()
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	callCount := 0
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		callCount++
		return map[string]any{"result": "success", "call_count": callCount}, nil
	}

	// First call should execute handler
	result1, err := interceptor(ctx, inputs, info, handler)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result1["call_count"] != 1 {
		t.Errorf("Expected call count 1, got %v", result1["call_count"])
	}

	// Second call with same inputs should use cache
	result2, err := interceptor(ctx, inputs, info, handler)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result2["call_count"] != 1 {
		t.Errorf("Expected cached result with call count 1, got %v", result2["call_count"])
	}

	// Different inputs should execute handler again
	differentInputs := map[string]any{"test": "different"}
	result3, err := interceptor(ctx, differentInputs, info, handler)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result3["call_count"] != 2 {
		t.Errorf("Expected call count 2 for different inputs, got %v", result3["call_count"])
	}
}

func TestCachingToolInterceptor(t *testing.T) {
	cache := NewMemoryCache()
	interceptor := CachingToolInterceptor(cache, time.Second)

	ctx := context.Background()
	args := map[string]interface{}{"test": "value"}
	info := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})

	callCount := 0
	handler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		callCount++
		return core.ToolResult{Data: map[string]interface{}{"result": "success", "call_count": callCount}}, nil
	}

	// First call should execute handler
	result1, err := interceptor(ctx, args, info, handler)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	// Second call should use cache
	result2, err := interceptor(ctx, args, info, handler)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	// Results should be identical (from cache)
	data1 := result1.Data.(map[string]interface{})
	data2 := result2.Data.(map[string]interface{})
	if data1["call_count"] != data2["call_count"] {
		t.Error("Expected cached result to be identical")
	}
}

func TestTimeoutModuleInterceptor(t *testing.T) {
	t.Run("Successful execution within timeout", func(t *testing.T) {
		interceptor := TimeoutModuleInterceptor(100 * time.Millisecond)

		ctx := context.Background()
		inputs := map[string]any{"test": "value"}
		info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

		handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
			time.Sleep(10 * time.Millisecond) // Within timeout
			return map[string]any{"result": "success"}, nil
		}

		result, err := interceptor(ctx, inputs, info, handler)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result["result"] != "success" {
			t.Errorf("Expected success result, got %v", result["result"])
		}
	})

	t.Run("Execution timeout", func(t *testing.T) {
		interceptor := TimeoutModuleInterceptor(50 * time.Millisecond)

		ctx := context.Background()
		inputs := map[string]any{"test": "value"}
		info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

		handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
			time.Sleep(100 * time.Millisecond) // Exceeds timeout
			return map[string]any{"result": "success"}, nil
		}

		_, err := interceptor(ctx, inputs, info, handler)
		if err == nil {
			t.Error("Expected timeout error")
		}
		if !strings.Contains(err.Error(), "timed out") {
			t.Errorf("Expected timeout error, got: %v", err)
		}
	})
}

func TestTimeoutAgentInterceptor(t *testing.T) {
	interceptor := TimeoutAgentInterceptor(50 * time.Millisecond)

	ctx := context.Background()
	input := map[string]interface{}{"test": "value"}
	info := core.NewAgentInfo("TestAgent", "TestType", []core.Tool{})

	handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
		time.Sleep(100 * time.Millisecond) // Exceeds timeout
		return map[string]interface{}{"result": "success"}, nil
	}

	_, err := interceptor(ctx, input, info, handler)
	if err == nil {
		t.Error("Expected timeout error")
	}
}

func TestTimeoutToolInterceptor(t *testing.T) {
	interceptor := TimeoutToolInterceptor(50 * time.Millisecond)

	ctx := context.Background()
	args := map[string]interface{}{"test": "value"}
	info := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})

	handler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		time.Sleep(100 * time.Millisecond) // Exceeds timeout
		return core.ToolResult{Data: "success"}, nil
	}

	_, err := interceptor(ctx, args, info, handler)
	if err == nil {
		t.Error("Expected timeout error")
	}
}

func TestCircuitBreakerModuleInterceptor(t *testing.T) {
	cb := NewCircuitBreaker(2, 100*time.Millisecond, 1)
	interceptor := CircuitBreakerModuleInterceptor(cb)

	ctx := context.Background()
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	t.Run("Success case", func(t *testing.T) {
		handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
			return map[string]any{"result": "success"}, nil
		}

		result, err := interceptor(ctx, inputs, info, handler)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result["result"] != "success" {
			t.Errorf("Expected success result, got %v", result["result"])
		}
	})

	t.Run("Failure case", func(t *testing.T) {
		handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
			return nil, errors.New("handler error")
		}

		// First two failures should be allowed through
		for i := 0; i < 2; i++ {
			_, err := interceptor(ctx, inputs, info, handler)
			if err == nil {
				t.Errorf("Expected handler error on attempt %d", i+1)
			}
		}

		// Third attempt should be blocked by circuit breaker
		_, err := interceptor(ctx, inputs, info, handler)
		if err == nil {
			t.Error("Expected circuit breaker to block request")
		}
		if !strings.Contains(err.Error(), "circuit breaker is open") {
			t.Errorf("Expected circuit breaker error, got: %v", err)
		}
	})
}

func TestCircuitBreakerAgentInterceptor(t *testing.T) {
	cb := NewCircuitBreaker(1, 100*time.Millisecond, 1)
	interceptor := CircuitBreakerAgentInterceptor(cb)

	ctx := context.Background()
	input := map[string]interface{}{"test": "value"}
	info := core.NewAgentInfo("TestAgent", "TestType", []core.Tool{})

	failingHandler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
		return nil, errors.New("handler error")
	}

	// First failure should trip the circuit
	_, err := interceptor(ctx, input, info, failingHandler)
	if err == nil {
		t.Error("Expected handler error")
	}

	// Second attempt should be blocked
	_, err = interceptor(ctx, input, info, failingHandler)
	if err == nil {
		t.Error("Expected circuit breaker to block request")
	}
	if !strings.Contains(err.Error(), "circuit breaker is open") {
		t.Errorf("Expected circuit breaker error, got: %v", err)
	}
}

func TestCircuitBreakerToolInterceptor(t *testing.T) {
	cb := NewCircuitBreaker(1, 100*time.Millisecond, 1)
	interceptor := CircuitBreakerToolInterceptor(cb)

	ctx := context.Background()
	args := map[string]interface{}{"test": "value"}
	info := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})

	successHandler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		return core.ToolResult{Data: "success"}, nil
	}

	// Success should work normally
	result, err := interceptor(ctx, args, info, successHandler)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result.Data != "success" {
		t.Errorf("Expected success result, got %v", result.Data)
	}
}

func TestRetryModuleInterceptor(t *testing.T) {
	config := RetryConfig{
		MaxAttempts: 3,
		Delay:       10 * time.Millisecond,
		Backoff:     2.0,
	}
	interceptor := RetryModuleInterceptor(config)

	ctx := context.Background()
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	t.Run("Success on first attempt", func(t *testing.T) {
		handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
			return map[string]any{"result": "success"}, nil
		}

		result, err := interceptor(ctx, inputs, info, handler)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result["result"] != "success" {
			t.Errorf("Expected success result, got %v", result["result"])
		}
	})

	t.Run("Success on retry", func(t *testing.T) {
		attemptCount := 0
		handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
			attemptCount++
			if attemptCount < 2 {
				return nil, errors.New("temporary error")
			}
			return map[string]any{"result": "success"}, nil
		}

		result, err := interceptor(ctx, inputs, info, handler)
		if err != nil {
			t.Errorf("Expected no error after retry, got %v", err)
		}
		if result["result"] != "success" {
			t.Errorf("Expected success result, got %v", result["result"])
		}
		if attemptCount != 2 {
			t.Errorf("Expected 2 attempts, got %d", attemptCount)
		}
	})

	t.Run("All attempts fail", func(t *testing.T) {
		attemptCount := 0
		handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
			attemptCount++
			return nil, errors.New("persistent error")
		}

		_, err := interceptor(ctx, inputs, info, handler)
		if err == nil {
			t.Error("Expected error after all retries failed")
		}
		if !strings.Contains(err.Error(), "failed after 3 attempts") {
			t.Errorf("Expected retry exhausted error, got: %v", err)
		}
		if attemptCount != 3 {
			t.Errorf("Expected 3 attempts, got %d", attemptCount)
		}
	})
}

func TestRetryAgentInterceptor(t *testing.T) {
	config := RetryConfig{
		MaxAttempts: 2,
		Delay:       10 * time.Millisecond,
		Backoff:     2.0,
	}
	interceptor := RetryAgentInterceptor(config)

	ctx := context.Background()
	input := map[string]interface{}{"test": "value"}
	info := core.NewAgentInfo("TestAgent", "TestType", []core.Tool{})

	attemptCount := 0
	handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
		attemptCount++
		if attemptCount < 2 {
			return nil, errors.New("temporary error")
		}
		return map[string]interface{}{"result": "success"}, nil
	}

	result, err := interceptor(ctx, input, info, handler)
	if err != nil {
		t.Errorf("Expected no error after retry, got %v", err)
	}
	if result["result"] != "success" {
		t.Errorf("Expected success result, got %v", result["result"])
	}
}

func TestRetryToolInterceptor(t *testing.T) {
	config := RetryConfig{
		MaxAttempts: 2,
		Delay:       10 * time.Millisecond,
		Backoff:     2.0,
	}
	interceptor := RetryToolInterceptor(config)

	ctx := context.Background()
	args := map[string]interface{}{"test": "value"}
	info := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})

	attemptCount := 0
	handler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		attemptCount++
		if attemptCount < 2 {
			return core.ToolResult{}, errors.New("temporary error")
		}
		return core.ToolResult{Data: "success"}, nil
	}

	result, err := interceptor(ctx, args, info, handler)
	if err != nil {
		t.Errorf("Expected no error after retry, got %v", err)
	}
	if result.Data != "success" {
		t.Errorf("Expected success result, got %v", result.Data)
	}
}

func TestBatchingModuleInterceptor(t *testing.T) {
	config := BatchingConfig{
		BatchSize:    2,
		BatchTimeout: 100 * time.Millisecond,
	}
	interceptor := BatchingModuleInterceptor(config)

	ctx := context.Background()
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	callCount := 0
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		callCount++
		return map[string]any{"result": "success", "call": callCount}, nil
	}

	// Test batching - this is a simplified test since the actual batching
	// logic would need more sophisticated module-specific implementation
	result1, err := interceptor(ctx, map[string]any{"test": "1"}, info, handler)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result1["result"] != "success" {
		t.Errorf("Expected success result, got %v", result1["result"])
	}
}

func TestPerformanceHelperFunctions(t *testing.T) {
	t.Run("generateModuleCacheKey", func(t *testing.T) {
		inputs := map[string]any{"test": "value"}
		info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

		key1 := generateModuleCacheKey(inputs, info)
		key2 := generateModuleCacheKey(inputs, info)

		// Same inputs should generate same key
		if key1 != key2 {
			t.Error("Expected same inputs to generate same cache key")
		}

		// Different inputs should generate different key
		differentInputs := map[string]any{"test": "different"}
		key3 := generateModuleCacheKey(differentInputs, info)

		if key1 == key3 {
			t.Error("Expected different inputs to generate different cache key")
		}

		// Should start with "module:" prefix
		if !strings.HasPrefix(key1, "module:") {
			t.Error("Expected cache key to start with 'module:' prefix")
		}
	})

	t.Run("generateToolCacheKey", func(t *testing.T) {
		args := map[string]interface{}{"test": "value"}
		info := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})

		key1 := generateToolCacheKey(args, info)
		key2 := generateToolCacheKey(args, info)

		// Same args should generate same key
		if key1 != key2 {
			t.Error("Expected same args to generate same cache key")
		}

		// Should start with "tool:" prefix
		if !strings.HasPrefix(key1, "tool:") {
			t.Error("Expected cache key to start with 'tool:' prefix")
		}
	})
}

// Benchmark tests.
func BenchmarkMemoryCache(b *testing.B) {
	cache := NewMemoryCache()

	b.Run("Set", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			cache.Set("key", "value", time.Second)
		}
	})

	b.Run("Get", func(b *testing.B) {
		cache.Set("key", "value", time.Second)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cache.Get("key")
		}
	})
}

func BenchmarkCircuitBreaker(b *testing.B) {
	cb := NewCircuitBreaker(100, time.Second, 10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cb.Allow()
		if i%10 == 0 {
			cb.RecordSuccess()
		}
	}
}

func BenchmarkCachingModuleInterceptor(b *testing.B) {
	cache := NewMemoryCache()
	interceptor := CachingModuleInterceptor(cache, time.Second)

	ctx := context.Background()
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

func BenchmarkTimeoutModuleInterceptor(b *testing.B) {
	interceptor := TimeoutModuleInterceptor(time.Second)

	ctx := context.Background()
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
func TestCacheExpiration(t *testing.T) {
	cache := NewMemoryCache()
	interceptor := CachingModuleInterceptor(cache, 50*time.Millisecond)

	ctx := context.Background()
	inputs := map[string]any{"test": "value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	callCount := 0
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		callCount++
		return map[string]any{"result": "success", "call_count": callCount}, nil
	}

	// First call
	result1, _ := interceptor(ctx, inputs, info, handler)
	if result1["call_count"] != 1 {
		t.Errorf("Expected call count 1, got %v", result1["call_count"])
	}

	// Wait for expiration
	time.Sleep(100 * time.Millisecond)

	// Call after expiration should execute handler again
	result2, _ := interceptor(ctx, inputs, info, handler)
	if result2["call_count"] != 2 {
		t.Errorf("Expected call count 2 after expiration, got %v", result2["call_count"])
	}
}
