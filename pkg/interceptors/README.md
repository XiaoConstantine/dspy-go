# Interceptors

This package provides a comprehensive set of interceptors for dspy-go modules, agents, and tools. Interceptors follow the middleware pattern, allowing you to add cross-cutting concerns like logging, caching, security, and structured output parsing without modifying your core logic.

## Architecture

Interceptors in dspy-go follow the gRPC interceptor pattern, providing a clean and composable way to add functionality to module execution. Each interceptor can:

- Inspect and modify inputs before execution
- Inspect and modify outputs after execution
- Handle errors and implement fallback logic
- Collect metrics and perform logging
- Implement security policies

## Available Interceptors

### Standard Interceptors (`standard.go`)

**Logging Interceptors**
- `LoggingModuleInterceptor()` - Logs module execution with timing
- `LoggingAgentInterceptor()` - Logs agent execution with timing
- `LoggingToolInterceptor()` - Logs tool execution with arguments

**Tracing Interceptors**
- `TracingModuleInterceptor()` - Adds distributed tracing spans
- `TracingAgentInterceptor()` - Traces agent execution
- `TracingToolInterceptor()` - Traces tool calls

**Metrics Interceptors**
- `MetricsModuleInterceptor()` - Collects performance metrics
- `MetricsAgentInterceptor()` - Tracks agent metrics
- `MetricsToolInterceptor()` - Monitors tool usage

### Performance Interceptors (`performance.go`)

**Caching**
- `CachingModuleInterceptor(cache, ttl)` - Caches module results
- `CachingToolInterceptor(cache, ttl)` - Caches tool results
- `MemoryCache` - In-memory cache implementation with TTL

**Timeouts**
- `TimeoutModuleInterceptor(duration)` - Enforces execution timeouts
- `TimeoutAgentInterceptor(duration)` - Agent timeout protection
- `TimeoutToolInterceptor(duration)` - Tool execution limits

**Circuit Breakers**
- `CircuitBreakerModuleInterceptor(cb)` - Prevents cascade failures
- `CircuitBreakerAgentInterceptor(cb)` - Agent circuit protection
- `CircuitBreakerToolInterceptor(cb)` - Tool failure isolation

**Retry Logic**
- `RetryModuleInterceptor(config)` - Retry failed executions
- `RetryAgentInterceptor(config)` - Agent retry with backoff
- `RetryToolInterceptor(config)` - Tool call retries

### Security Interceptors (`security.go`)

**Rate Limiting**
- `RateLimitingAgentInterceptor(limit, window)` - Agent rate limits
- `RateLimitingToolInterceptor(limit, window)` - Tool call limits

**Input Validation**
- `ValidationModuleInterceptor(config)` - Validates module inputs
- `ValidationAgentInterceptor(config)` - Agent input validation
- `ValidationToolInterceptor(config)` - Tool argument validation

**Authorization**
- `AuthorizationInterceptor` - Policy-based access control
- `ModuleAuthorizationInterceptor()` - Module access control
- `AgentAuthorizationInterceptor()` - Agent authorization
- `ToolAuthorizationInterceptor()` - Tool permission checks

**Input Sanitization**
- `SanitizingModuleInterceptor()` - Sanitizes module inputs
- `SanitizingAgentInterceptor()` - Agent input sanitization
- `SanitizingToolInterceptor()` - Tool argument sanitization

### XML Interceptors (`xml.go`, `xml_config.go`)

**Structured Output Processing**
- `XMLFormatModuleInterceptor(config)` - Injects XML formatting instructions
- `XMLParseModuleInterceptor(config)` - Parses XML responses to structured data
- `XMLModuleInterceptor(config)` - Combined format + parse interceptor

**Configuration Options**
- `DefaultXMLConfig()` - Balanced settings for general use
- `StrictXMLConfig()` - Strict parsing, no fallback
- `FlexibleXMLConfig()` - Allows missing fields, has fallback
- `PerformantXMLConfig()` - Optimized for speed
- `SecureXMLConfig()` - Enhanced security restrictions

**Helper Functions**
- `ApplyXMLInterceptors(module, config)` - Apply to InterceptableModule
- `CreateXMLInterceptorChain(config, additional...)` - Build interceptor chains

## Usage Patterns

### Basic Interceptor Application

```go
import "github.com/XiaoConstantine/dspy-go/pkg/interceptors"

// Apply single interceptor
module.SetInterceptors([]core.ModuleInterceptor{
    interceptors.LoggingModuleInterceptor(),
})

// Apply multiple interceptors
module.SetInterceptors([]core.ModuleInterceptor{
    interceptors.LoggingModuleInterceptor(),
    interceptors.MetricsModuleInterceptor(),
    interceptors.XMLModuleInterceptor(interceptors.DefaultXMLConfig()),
})
```

### Interceptor Chaining

```go
// Build comprehensive processing pipeline
chain := []core.ModuleInterceptor{
    interceptors.LoggingModuleInterceptor(),                    // Log requests
    interceptors.ValidationModuleInterceptor(validationConfig), // Validate inputs
    interceptors.RetryModuleInterceptor(retryConfig),          // Retry on failure
    interceptors.XMLFormatModuleInterceptor(xmlConfig),        // Format request
    interceptors.CachingModuleInterceptor(cache, 5*time.Minute), // Cache results
    interceptors.XMLParseModuleInterceptor(xmlConfig),         // Parse response
    interceptors.MetricsModuleInterceptor(),                   // Collect metrics
}

module.SetInterceptors(chain)
```

### XML Structured Output

```go
// Configure XML interceptor for structured responses
xmlConfig := interceptors.DefaultXMLConfig().
    WithStrictParsing(true).
    WithCustomTag("user_query", "query").
    WithTypeHints(true)

// Apply XML interceptors
xmlInterceptor := interceptors.XMLModuleInterceptor(xmlConfig)
module.SetInterceptors([]core.ModuleInterceptor{xmlInterceptor})

// Module will automatically:
// 1. Inject XML formatting instructions into prompts
// 2. Parse XML responses into structured fields
```

### Performance Optimization

```go
// Create memory cache
cache := interceptors.NewMemoryCache()

// Create circuit breaker
cb := interceptors.NewCircuitBreaker(5, 30*time.Second, 10)

// Build performance-focused chain
perfChain := []core.ModuleInterceptor{
    interceptors.CachingModuleInterceptor(cache, 10*time.Minute),
    interceptors.TimeoutModuleInterceptor(30*time.Second),
    interceptors.CircuitBreakerModuleInterceptor(cb),
    interceptors.RetryModuleInterceptor(interceptors.RetryConfig{
        MaxAttempts: 3,
        Delay:       100*time.Millisecond,
        Backoff:     2.0,
    }),
}

module.SetInterceptors(perfChain)
```

### Security Configuration

```go
// Configure validation
validationConfig := interceptors.DefaultValidationConfig()
validationConfig.MaxInputSize = 1024 * 1024  // 1MB limit
validationConfig.ForbiddenPatterns = append(
    validationConfig.ForbiddenPatterns,
    `(?i)custom_dangerous_pattern`,
)

// Configure authorization
authInterceptor := interceptors.NewAuthorizationInterceptor()
authInterceptor.SetPolicy("sensitive_module", interceptors.AuthorizationPolicy{
    RequiredRoles: []string{"admin", "power_user"},
    RequireAuth:   true,
})

// Build security chain
secChain := []core.ModuleInterceptor{
    authInterceptor.ModuleAuthorizationInterceptor(),
    interceptors.ValidationModuleInterceptor(validationConfig),
    interceptors.SanitizingModuleInterceptor(),
    interceptors.RateLimitingAgentInterceptor(100, time.Minute),
}

module.SetInterceptors(secChain)
```

## Configuration Examples

### XML Configuration

```go
// Custom XML configuration
xmlConfig := interceptors.DefaultXMLConfig().
    WithStrictParsing(false).           // Allow missing fields
    WithFallback(true).                 // Enable text fallback
    WithValidation(true).               // Validate XML syntax
    WithMaxDepth(10).                   // Limit nesting depth
    WithMaxSize(1024*1024).            // 1MB size limit
    WithTimeout(30*time.Second).        // Parse timeout
    WithCustomTag("question", "q").     // Custom tag mapping
    WithTypeHints(true).               // Include type hints
    WithPreserveWhitespace(false)      // Trim whitespace
```

### Retry Configuration

```go
retryConfig := interceptors.RetryConfig{
    MaxAttempts: 3,                    // Retry up to 3 times
    Delay:       100*time.Millisecond, // Initial delay
    MaxBackoff:  5*time.Second,        // Maximum delay
    Backoff:     2.0,                  // Exponential backoff
}
```

### Validation Configuration

```go
validationConfig := interceptors.ValidationConfig{
    MaxInputSize:     10*1024*1024,    // 10MB limit
    MaxStringLength:  100000,          // 100KB per string
    ForbiddenPatterns: []string{       // Security patterns
        `(?i)<script[^>]*>.*?</script>`,
        `(?i)javascript:`,
        `\$\{.*\}`,
    },
    RequiredFields: []string{"user_id"}, // Required fields
    AllowHTML:      false,               // Block HTML content
}
```

## Best Practices

### 1. Interceptor Ordering

Order interceptors carefully for correct behavior:

```go
// Recommended order:
[]core.ModuleInterceptor{
    // 1. Security (first line of defense)
    authInterceptor,
    validationInterceptor,
    sanitizationInterceptor,

    // 2. Observability
    loggingInterceptor,
    tracingInterceptor,

    // 3. Reliability
    retryInterceptor,
    circuitBreakerInterceptor,

    // 4. Performance
    cachingInterceptor,
    timeoutInterceptor,

    // 5. Content Processing
    xmlFormatInterceptor,
    xmlParseInterceptor,

    // 6. Metrics (capture final state)
    metricsInterceptor,
}
```

### 2. Error Handling

```go
// Use fallback-enabled configurations in production
xmlConfig := interceptors.FlexibleXMLConfig() // Has fallback

// Implement graceful degradation
validationConfig := interceptors.DefaultValidationConfig()
validationConfig.ForbiddenPatterns = []string{} // Start permissive
```

### 3. Performance Monitoring

```go
// Always include metrics for production
chain := []core.ModuleInterceptor{
    interceptors.MetricsModuleInterceptor(),
    // ... other interceptors
}

// Use caching for expensive operations
cache := interceptors.NewMemoryCache()
cachingInterceptor := interceptors.CachingModuleInterceptor(cache, 10*time.Minute)
```

### 4. Security Layers

```go
// Implement defense in depth
securityChain := []core.ModuleInterceptor{
    authInterceptor,      // Authentication/authorization
    validationInterceptor, // Input validation
    sanitizationInterceptor, // Input sanitization
    rateLimitInterceptor, // Rate limiting
}
```

## Performance Characteristics

Based on benchmarks:

- **XML Parsing**: ~3,663 ns/op (327K ops/sec)
- **Full XML Pipeline**: ~5,727 ns/op (209K ops/sec)
- **Memory Usage**: ~3.5KB/op for parsing, ~7.8KB/op for full pipeline
- **Caching**: Near-zero overhead for cache hits
- **Logging**: Minimal overhead with structured logging

## Extension Points

The interceptor system is designed for extensibility:

### Custom Interceptors

```go
func CustomModuleInterceptor(config CustomConfig) core.ModuleInterceptor {
    return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
        // Pre-processing
        modifiedInputs := preprocessInputs(inputs, config)

        // Execute
        outputs, err := handler(ctx, modifiedInputs, opts...)

        // Post-processing
        finalOutputs := postprocessOutputs(outputs, config)

        return finalOutputs, err
    }
}
```

### Custom Configurations

```go
type CustomConfig struct {
    Setting1 string
    Setting2 int
    // ... other settings
}

func DefaultCustomConfig() CustomConfig {
    return CustomConfig{
        Setting1: "default",
        Setting2: 42,
    }
}
```

This interceptor system provides a powerful, composable foundation for building robust, secure, and high-performance language model applications with dspy-go.
