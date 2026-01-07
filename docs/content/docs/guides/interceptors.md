---
title: "Interceptors"
description: "Composable middleware for modules, agents, and tools"
summary: "Add logging, caching, security, XML parsing, and more to your pipelines with the interceptor pattern"
date: 2025-01-06T00:00:00+00:00
lastmod: 2025-01-06T00:00:00+00:00
draft: false
weight: 350
toc: true
seo:
  title: "Interceptors - dspy-go"
  description: "Complete guide to interceptors for adding cross-cutting concerns to dspy-go applications"
  canonical: ""
  noindex: false
---

# Interceptors

Interceptors provide composable middleware for modules, agents, and tools. They follow the gRPC interceptor pattern, allowing you to add cross-cutting concerns like logging, caching, security, and structured output parsing without modifying core logic.

## Architecture

Each interceptor can:
- Inspect and modify inputs before execution
- Inspect and modify outputs after execution
- Handle errors and implement fallback logic
- Collect metrics and perform logging
- Implement security policies

```
Input → [Security] → [Logging] → [Retry] → [Cache] → LLM → [Parse] → [Metrics] → Output
```

---

## Quick Start

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
    interceptors.RetryModuleInterceptor(retryConfig),
})
```

---

## Standard Interceptors

### Logging

```go
// Module logging with timing
interceptors.LoggingModuleInterceptor()

// Agent logging
interceptors.LoggingAgentInterceptor()

// Tool logging with arguments
interceptors.LoggingToolInterceptor()
```

### Tracing

```go
// Distributed tracing spans
interceptors.TracingModuleInterceptor()
interceptors.TracingAgentInterceptor()
interceptors.TracingToolInterceptor()
```

### Metrics

```go
// Performance metrics collection
interceptors.MetricsModuleInterceptor()
interceptors.MetricsAgentInterceptor()
interceptors.MetricsToolInterceptor()
```

---

## Performance Interceptors

### Caching

```go
// Create memory cache with TTL
cache := interceptors.NewMemoryCache()

// Cache module results
interceptors.CachingModuleInterceptor(cache, 10*time.Minute)

// Cache tool results
interceptors.CachingToolInterceptor(cache, 5*time.Minute)
```

### Timeouts

```go
// Enforce execution timeouts
interceptors.TimeoutModuleInterceptor(30*time.Second)
interceptors.TimeoutAgentInterceptor(2*time.Minute)
interceptors.TimeoutToolInterceptor(10*time.Second)
```

### Circuit Breakers

Prevent cascade failures:

```go
// Create circuit breaker
// (failure threshold, recovery timeout, half-open requests)
cb := interceptors.NewCircuitBreaker(5, 30*time.Second, 10)

// Apply to module
interceptors.CircuitBreakerModuleInterceptor(cb)
interceptors.CircuitBreakerAgentInterceptor(cb)
interceptors.CircuitBreakerToolInterceptor(cb)
```

### Retry Logic

```go
retryConfig := interceptors.RetryConfig{
    MaxAttempts: 3,                     // Retry up to 3 times
    Delay:       100*time.Millisecond,  // Initial delay
    MaxBackoff:  5*time.Second,         // Maximum delay
    Backoff:     2.0,                   // Exponential backoff
}

interceptors.RetryModuleInterceptor(retryConfig)
interceptors.RetryAgentInterceptor(retryConfig)
interceptors.RetryToolInterceptor(retryConfig)
```

---

## Security Interceptors

### Rate Limiting

```go
// Limit to 100 requests per minute
interceptors.RateLimitingAgentInterceptor(100, time.Minute)
interceptors.RateLimitingToolInterceptor(50, time.Minute)
```

### Input Validation

```go
validationConfig := interceptors.ValidationConfig{
    MaxInputSize:     10*1024*1024,     // 10MB limit
    MaxStringLength:  100000,           // 100KB per string
    ForbiddenPatterns: []string{
        `(?i)<script[^>]*>.*?</script>`,
        `(?i)javascript:`,
        `\$\{.*\}`,
    },
    RequiredFields: []string{"user_id"},
    AllowHTML:      false,
}

interceptors.ValidationModuleInterceptor(validationConfig)
interceptors.ValidationAgentInterceptor(validationConfig)
interceptors.ValidationToolInterceptor(validationConfig)
```

### Authorization

```go
authInterceptor := interceptors.NewAuthorizationInterceptor()
authInterceptor.SetPolicy("sensitive_module", interceptors.AuthorizationPolicy{
    RequiredRoles: []string{"admin", "power_user"},
    RequireAuth:   true,
})

authInterceptor.ModuleAuthorizationInterceptor()
authInterceptor.AgentAuthorizationInterceptor()
authInterceptor.ToolAuthorizationInterceptor()
```

### Input Sanitization

```go
// Sanitize potentially dangerous inputs
interceptors.SanitizingModuleInterceptor()
interceptors.SanitizingAgentInterceptor()
interceptors.SanitizingToolInterceptor()
```

---

## XML Interceptors

XML interceptors provide structured output parsing as an alternative to JSON. They're useful when you need fine-grained control over parsing behavior, security limits, or want to compose with other interceptors.

### Basic Usage

```go
// Create a module
predict := modules.NewPredict(signature)

// Apply XML interceptors with default config
config := interceptors.DefaultXMLConfig()
err := interceptors.ApplyXMLInterceptors(predict, config)

// Process as normal - outputs are parsed from XML
result, err := predict.Process(ctx, map[string]interface{}{
    "question": "What is machine learning?",
})

// Structured output fields extracted reliably
fmt.Println(result["answer"])
fmt.Println(result["confidence"])
```

### Preset Configurations

```go
// Balanced configuration for general use
config := interceptors.DefaultXMLConfig()

// Strict parsing - requires all fields, no fallback
config := interceptors.StrictXMLConfig()

// Flexible - allows missing fields, has fallback
config := interceptors.FlexibleXMLConfig()

// Performance optimized
config := interceptors.PerformantXMLConfig()

// Enhanced security restrictions
config := interceptors.SecureXMLConfig()
```

### Custom Configuration

```go
config := &interceptors.XMLConfig{
    // Parsing behavior
    StrictParsing:   true,           // Require all output fields
    FallbackToText:  true,           // Use text parsing if XML fails
    ValidateXML:     true,           // Validate XML syntax

    // Security limits
    MaxDepth:        10,             // Maximum nesting depth
    MaxSize:         1024 * 1024,    // Maximum XML size (1MB)
    Timeout:         30 * time.Second,

    // Customization
    CustomTags: map[string]string{
        "answer": "response",        // Use <response> instead of <answer>
    },
    TypeHints:       true,           // Include type info in instructions
}
```

### XML vs Structured Output

| Feature | XML Interceptor | `.WithStructuredOutput()` |
|---------|-----------------|---------------------------|
| **Output Format** | XML tags | Native JSON |
| **Security Controls** | Depth/size limits | Provider-dependent |
| **Fallback** | Configurable | Built-in |
| **Interceptor Pattern** | Yes - composable | No |
| **Provider Support** | All providers | Providers with JSON mode |

---

## Interceptor Chaining

### Recommended Order

```go
chain := []core.ModuleInterceptor{
    // 1. Security (first line of defense)
    authInterceptor.ModuleAuthorizationInterceptor(),
    interceptors.ValidationModuleInterceptor(validationConfig),
    interceptors.SanitizingModuleInterceptor(),

    // 2. Observability
    interceptors.LoggingModuleInterceptor(),
    interceptors.TracingModuleInterceptor(),

    // 3. Reliability
    interceptors.RetryModuleInterceptor(retryConfig),
    interceptors.CircuitBreakerModuleInterceptor(cb),

    // 4. Performance
    interceptors.CachingModuleInterceptor(cache, 5*time.Minute),
    interceptors.TimeoutModuleInterceptor(30*time.Second),

    // 5. Content Processing
    interceptors.XMLFormatModuleInterceptor(xmlConfig),
    interceptors.XMLParseModuleInterceptor(xmlConfig),

    // 6. Metrics (capture final state)
    interceptors.MetricsModuleInterceptor(),
}

module.SetInterceptors(chain)
```

### Production Pipeline Example

```go
// Create caching
cache := interceptors.NewMemoryCache()

// Create circuit breaker
cb := interceptors.NewCircuitBreaker(5, 30*time.Second, 10)

// Create retry config
retryConfig := interceptors.RetryConfig{
    MaxAttempts: 3,
    Delay:       100*time.Millisecond,
    Backoff:     2.0,
}

// Build production chain
chain := []core.ModuleInterceptor{
    interceptors.LoggingModuleInterceptor(),
    interceptors.ValidationModuleInterceptor(interceptors.DefaultValidationConfig()),
    interceptors.RetryModuleInterceptor(retryConfig),
    interceptors.CircuitBreakerModuleInterceptor(cb),
    interceptors.CachingModuleInterceptor(cache, 10*time.Minute),
    interceptors.TimeoutModuleInterceptor(30*time.Second),
    interceptors.MetricsModuleInterceptor(),
}

predict := modules.NewPredict(signature)
predict.SetInterceptors(chain)
```

---

## Custom Interceptors

Create your own interceptors:

```go
func CustomModuleInterceptor(config CustomConfig) core.ModuleInterceptor {
    return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
        // Pre-processing
        modifiedInputs := preprocessInputs(inputs, config)

        // Execute handler
        outputs, err := handler(ctx, modifiedInputs, opts...)
        if err != nil {
            return nil, err
        }

        // Post-processing
        finalOutputs := postprocessOutputs(outputs, config)

        return finalOutputs, nil
    }
}
```

---

## Performance

Based on benchmarks:

| Operation | Performance | Throughput |
|-----------|-------------|------------|
| XML Parsing | ~3,663 ns/op | 273K ops/sec |
| Full Pipeline | ~5,727 ns/op | 175K ops/sec |
| Cache Hit | Near-zero | - |
| Logging | Minimal overhead | - |

---

## Best Practices

**1. Order matters** - Security first, metrics last

**2. Use fallbacks in production**
```go
xmlConfig := interceptors.FlexibleXMLConfig() // Has fallback
```

**3. Always include metrics**
```go
chain := []core.ModuleInterceptor{
    interceptors.MetricsModuleInterceptor(),
    // ... other interceptors
}
```

**4. Defense in depth**
```go
securityChain := []core.ModuleInterceptor{
    authInterceptor,
    validationInterceptor,
    sanitizationInterceptor,
    rateLimitInterceptor,
}
```

---

## Next Steps

- **[Core Concepts](core-concepts/)** - Understand modules and signatures
- **[Building Agents](agents/)** - Use interceptors with agents
- **[Tool Management](tools/)** - Apply interceptors to tools
