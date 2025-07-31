package interceptors

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// CacheEntry represents a cached result with expiration.
type CacheEntry struct {
	Result    interface{}
	ExpiresAt time.Time
}

// Cache interface allows for different caching implementations.
type Cache interface {
	Get(key string) (interface{}, bool)
	Set(key string, value interface{}, ttl time.Duration)
	Delete(key string)
	Clear()
}

// MemoryCache implements an in-memory cache with TTL support.
type MemoryCache struct {
	mu       sync.RWMutex
	items    map[string]*CacheEntry
	stopChan chan struct{}
	stopped  bool
}

// NewMemoryCache creates a new in-memory cache.
func NewMemoryCache() *MemoryCache {
	cache := &MemoryCache{
		items:    make(map[string]*CacheEntry),
		stopChan: make(chan struct{}),
		stopped:  false,
	}
	
	// Start cleanup goroutine
	go cache.cleanup()
	
	return cache
}

// Get retrieves a value from the cache.
func (mc *MemoryCache) Get(key string) (interface{}, bool) {
	mc.mu.RLock()
	entry, exists := mc.items[key]
	if !exists {
		mc.mu.RUnlock()
		return nil, false
	}

	if time.Now().After(entry.ExpiresAt) {
		mc.mu.RUnlock()
		// Upgrade to write lock for deletion
		mc.mu.Lock()
		// Check again in case another goroutine already deleted it
		if entry, stillExists := mc.items[key]; stillExists && time.Now().After(entry.ExpiresAt) {
			delete(mc.items, key)
		}
		mc.mu.Unlock()
		return nil, false
	}

	result := entry.Result
	mc.mu.RUnlock()
	return result, true
}

// Set stores a value in the cache with TTL.
func (mc *MemoryCache) Set(key string, value interface{}, ttl time.Duration) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.items[key] = &CacheEntry{
		Result:    value,
		ExpiresAt: time.Now().Add(ttl),
	}
}

// Delete removes a value from the cache.
func (mc *MemoryCache) Delete(key string) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	delete(mc.items, key)
}

// Clear removes all items from the cache.
func (mc *MemoryCache) Clear() {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.items = make(map[string]*CacheEntry)
}

// Stop terminates the cleanup goroutine and marks cache as stopped.
func (mc *MemoryCache) Stop() {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	
	if !mc.stopped {
		mc.stopped = true
		close(mc.stopChan)
	}
}

// cleanup removes expired entries periodically.
func (mc *MemoryCache) cleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			mc.mu.Lock()
			if mc.stopped {
				mc.mu.Unlock()
				return
			}
			now := time.Now()
			for key, entry := range mc.items {
				if now.After(entry.ExpiresAt) {
					delete(mc.items, key)
				}
			}
			mc.mu.Unlock()
		case <-mc.stopChan:
			return
		}
	}
}

// CircuitBreaker implements the circuit breaker pattern.
type CircuitBreaker struct {
	mu           sync.RWMutex
	state        CircuitState
	failures     int
	requests     int
	lastFailTime time.Time
	
	// Configuration
	failureThreshold int
	resetTimeout     time.Duration
	requestThreshold int
}

// CircuitState represents the current state of the circuit breaker.
type CircuitState int

const (
	CircuitClosed CircuitState = iota
	CircuitOpen
	CircuitHalfOpen
)

// NewCircuitBreaker creates a new circuit breaker.
func NewCircuitBreaker(failureThreshold int, resetTimeout time.Duration, requestThreshold int) *CircuitBreaker {
	return &CircuitBreaker{
		state:            CircuitClosed,
		failureThreshold: failureThreshold,
		resetTimeout:     resetTimeout,
		requestThreshold: requestThreshold,
	}
}

// Allow checks if a request should be allowed through the circuit breaker.
func (cb *CircuitBreaker) Allow() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()

	switch cb.state {
	case CircuitClosed:
		return true
	case CircuitOpen:
		if now.Sub(cb.lastFailTime) > cb.resetTimeout {
			cb.state = CircuitHalfOpen
			cb.requests = 0
			cb.failures = 0
			return true
		}
		return false
	case CircuitHalfOpen:
		return cb.requests < cb.requestThreshold
	}

	return false
}

// RecordSuccess records a successful request.
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.requests++
	
	if cb.state == CircuitHalfOpen && cb.requests >= cb.requestThreshold {
		cb.state = CircuitClosed
		cb.failures = 0
		cb.requests = 0
	}
}

// RecordFailure records a failed request.
func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failures++
	cb.requests++
	cb.lastFailTime = time.Now()

	if cb.failures >= cb.failureThreshold {
		cb.state = CircuitOpen
	}
}

// GetState returns the current circuit breaker state.
func (cb *CircuitBreaker) GetState() CircuitState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// CachingModuleInterceptor creates an interceptor that caches module results.
// It uses input data and module info to generate cache keys.
func CachingModuleInterceptor(cache Cache, ttl time.Duration) core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		// Generate cache key from inputs and module info
		cacheKey := generateModuleCacheKey(inputs, info)

		// Try to get from cache first
		if cached, found := cache.Get(cacheKey); found {
			if result, ok := cached.(map[string]any); ok {
				return result, nil
			}
		}

		// Cache miss - execute the handler
		result, err := handler(ctx, inputs, opts...)
		if err != nil {
			return result, err
		}

		// Cache the successful result
		cache.Set(cacheKey, result, ttl)
		return result, nil
	}
}

// CachingToolInterceptor creates an interceptor that caches tool results.
func CachingToolInterceptor(cache Cache, ttl time.Duration) core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		// Generate cache key from args and tool info
		cacheKey := generateToolCacheKey(args, info)

		// Try to get from cache first
		if cached, found := cache.Get(cacheKey); found {
			if result, ok := cached.(core.ToolResult); ok {
				return result, nil
			}
		}

		// Cache miss - execute the handler
		result, err := handler(ctx, args)
		if err != nil {
			return result, err
		}

		// Cache the successful result
		cache.Set(cacheKey, result, ttl)
		return result, nil
	}
}

// TimeoutModuleInterceptor creates an interceptor that enforces timeouts on module execution.
func TimeoutModuleInterceptor(timeout time.Duration) core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		// Create context with timeout
		timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()

		// Channel to receive result
		type result struct {
			output map[string]any
			err    error
		}
		resultChan := make(chan result, 1)

		// Execute handler in goroutine
		go func() {
			output, err := handler(timeoutCtx, inputs, opts...)
			resultChan <- result{output: output, err: err}
		}()

		// Wait for result or timeout
		select {
		case res := <-resultChan:
			return res.output, res.err
		case <-timeoutCtx.Done():
			return nil, fmt.Errorf("module %s execution timed out after %v", info.ModuleName, timeout)
		}
	}
}

// TimeoutAgentInterceptor creates an interceptor that enforces timeouts on agent execution.
func TimeoutAgentInterceptor(timeout time.Duration) core.AgentInterceptor {
	return func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
		// Create context with timeout
		timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()

		// Channel to receive result
		type result struct {
			output map[string]interface{}
			err    error
		}
		resultChan := make(chan result, 1)

		// Execute handler in goroutine
		go func() {
			output, err := handler(timeoutCtx, input)
			resultChan <- result{output: output, err: err}
		}()

		// Wait for result or timeout
		select {
		case res := <-resultChan:
			return res.output, res.err
		case <-timeoutCtx.Done():
			return nil, fmt.Errorf("agent %s execution timed out after %v", info.AgentID, timeout)
		}
	}
}

// TimeoutToolInterceptor creates an interceptor that enforces timeouts on tool execution.
func TimeoutToolInterceptor(timeout time.Duration) core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		// Create context with timeout
		timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()

		// Channel to receive result
		type result struct {
			output core.ToolResult
			err    error
		}
		resultChan := make(chan result, 1)

		// Execute handler in goroutine
		go func() {
			output, err := handler(timeoutCtx, args)
			resultChan <- result{output: output, err: err}
		}()

		// Wait for result or timeout
		select {
		case res := <-resultChan:
			return res.output, res.err
		case <-timeoutCtx.Done():
			return core.ToolResult{}, fmt.Errorf("tool %s execution timed out after %v", info.Name, timeout)
		}
	}
}

// CircuitBreakerModuleInterceptor creates an interceptor that implements circuit breaker pattern for modules.
func CircuitBreakerModuleInterceptor(cb *CircuitBreaker) core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		if !cb.Allow() {
			return nil, fmt.Errorf("circuit breaker is open for module %s", info.ModuleName)
		}

		result, err := handler(ctx, inputs, opts...)
		
		if err != nil {
			cb.RecordFailure()
			return result, err
		}

		cb.RecordSuccess()
		return result, nil
	}
}

// CircuitBreakerAgentInterceptor creates an interceptor that implements circuit breaker pattern for agents.
func CircuitBreakerAgentInterceptor(cb *CircuitBreaker) core.AgentInterceptor {
	return func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
		if !cb.Allow() {
			return nil, fmt.Errorf("circuit breaker is open for agent %s", info.AgentID)
		}

		result, err := handler(ctx, input)
		
		if err != nil {
			cb.RecordFailure()
			return result, err
		}

		cb.RecordSuccess()
		return result, nil
	}
}

// CircuitBreakerToolInterceptor creates an interceptor that implements circuit breaker pattern for tools.
func CircuitBreakerToolInterceptor(cb *CircuitBreaker) core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		if !cb.Allow() {
			return core.ToolResult{}, fmt.Errorf("circuit breaker is open for tool %s", info.Name)
		}

		result, err := handler(ctx, args)
		
		if err != nil {
			cb.RecordFailure()
			return result, err
		}

		cb.RecordSuccess()
		return result, nil
	}
}

// NOTE: BatchingModuleInterceptor was removed due to fundamental design flaws.
// 
// True batching requires module-specific knowledge of how to:
// 1. Combine multiple input sets into a single batch request
// 2. Split batch responses back to individual results  
// 3. Handle partial failures within a batch
//
// A generic batching interceptor cannot provide these capabilities without
// understanding the specific module's batching semantics. Instead, modules
// that support batching should implement it internally or provide
// batch-specific interfaces.
//
// For future implementation, consider:
// - Module-specific batch interfaces (e.g., BatchableModule)
// - Batch request/response types in module signatures
// - Explicit batch configuration per module type

// Helper functions

// generateModuleCacheKey creates a cache key for module results.
func generateModuleCacheKey(inputs map[string]any, info *core.ModuleInfo) string {
	hasher := sha256.New()
	
	// Add module info
	hasher.Write([]byte(info.ModuleName))
	hasher.Write([]byte(info.ModuleType))
	hasher.Write([]byte(info.Version))
	
	// Add inputs (in a deterministic way)
	inputsJSON, err := json.Marshal(inputs)
	if err != nil {
		// Fallback to a deterministic representation if JSON marshaling fails
		fmt.Fprintf(hasher, "%+v", inputs)
	} else {
		hasher.Write(inputsJSON)
	}
	
	return "module:" + hex.EncodeToString(hasher.Sum(nil))
}

// generateToolCacheKey creates a cache key for tool results.
func generateToolCacheKey(args map[string]interface{}, info *core.ToolInfo) string {
	hasher := sha256.New()
	
	// Add tool info
	hasher.Write([]byte(info.Name))
	hasher.Write([]byte(info.ToolType))
	hasher.Write([]byte(info.Version))
	
	// Add arguments (in a deterministic way)
	argsJSON, err := json.Marshal(args)
	if err != nil {
		// Fallback to a deterministic representation if JSON marshaling fails
		fmt.Fprintf(hasher, "%+v", args)
	} else {
		hasher.Write(argsJSON)
	}
	
	return "tool:" + hex.EncodeToString(hasher.Sum(nil))
}

// RetryConfig holds configuration for retry operations.
type RetryConfig struct {
	MaxAttempts int
	Delay       time.Duration
	Backoff     float64 // Multiplier for delay between retries
}

// RetryModuleInterceptor creates an interceptor that retries failed module executions.
func RetryModuleInterceptor(config RetryConfig) core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		var lastErr error
		delay := config.Delay

		for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
			result, err := handler(ctx, inputs, opts...)
			if err == nil {
				return result, nil
			}

			lastErr = err

			// Don't retry on the last attempt
			if attempt == config.MaxAttempts {
				break
			}

			// Wait before retrying
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}

			// Apply backoff
			delay = time.Duration(float64(delay) * config.Backoff)
		}

		return nil, fmt.Errorf("module %s failed after %d attempts: %w", info.ModuleName, config.MaxAttempts, lastErr)
	}
}

// RetryAgentInterceptor creates an interceptor that retries failed agent executions.
func RetryAgentInterceptor(config RetryConfig) core.AgentInterceptor {
	return func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
		var lastErr error
		delay := config.Delay

		for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
			result, err := handler(ctx, input)
			if err == nil {
				return result, nil
			}

			lastErr = err

			// Don't retry on the last attempt
			if attempt == config.MaxAttempts {
				break
			}

			// Wait before retrying
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}

			// Apply backoff
			delay = time.Duration(float64(delay) * config.Backoff)
		}

		return nil, fmt.Errorf("agent %s failed after %d attempts: %w", info.AgentID, config.MaxAttempts, lastErr)
	}
}

// RetryToolInterceptor creates an interceptor that retries failed tool executions.
func RetryToolInterceptor(config RetryConfig) core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		var lastErr error
		delay := config.Delay

		for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
			result, err := handler(ctx, args)
			if err == nil {
				return result, nil
			}

			lastErr = err

			// Don't retry on the last attempt
			if attempt == config.MaxAttempts {
				break
			}

			// Wait before retrying
			select {
			case <-ctx.Done():
				return core.ToolResult{}, ctx.Err()
			case <-time.After(delay):
			}

			// Apply backoff
			delay = time.Duration(float64(delay) * config.Backoff)
		}

		return core.ToolResult{}, fmt.Errorf("tool %s failed after %d attempts: %w", info.Name, config.MaxAttempts, lastErr)
	}
}