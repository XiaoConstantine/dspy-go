---
title: "XML Adapters"
description: "Structured output parsing with XML interceptors"
summary: "Reliable multi-field extraction using XML-based output parsing with security controls"
date: 2025-01-06T00:00:00+00:00
lastmod: 2025-01-06T00:00:00+00:00
draft: false
weight: 350
toc: true
seo:
  title: "XML Adapters - dspy-go"
  description: "Complete guide to XML-based structured output parsing in dspy-go"
  canonical: ""
  noindex: false
---

# XML Adapters

XML adapters provide an alternative to JSON structured output for reliable multi-field extraction. They use XML-based parsing with configurable security controls and preset configurations.

## Why XML Adapters?

While `.WithStructuredOutput()` uses native JSON generation, XML adapters offer:

- **Fine-grained Control**: Configure parsing behavior, validation, and fallbacks
- **Security Controls**: Depth limits, size limits, and validation
- **Custom Tags**: Override default XML tag names
- **Interceptor Pattern**: Compose with logging, metrics, and other interceptors
- **Graceful Fallback**: Configurable behavior for malformed responses

---

## Quick Start

### Basic Usage

```go
import (
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
    "github.com/XiaoConstantine/dspy-go/pkg/modules/interceptors"
)

// Create a module
predict := modules.NewPredict(signature)

// Apply XML interceptors with default config
config := interceptors.DefaultXMLConfig()
err := interceptors.ApplyXMLInterceptors(predict, config)
if err != nil {
    log.Fatal(err)
}

// Process as normal - outputs are parsed from XML
result, err := predict.Process(ctx, map[string]interface{}{
    "question": "What is machine learning?",
})

// Structured output fields extracted reliably
fmt.Println(result["answer"])
fmt.Println(result["confidence"])
```

### Using Preset Configurations

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

---

## Configuration Options

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
        "reasoning": "thought",      // Use <thought> instead of <reasoning>
    },
    TypeHints:       true,           // Include type info in instructions
}
```

### Configuration Reference

| Option | Description | Default |
|--------|-------------|---------|
| `StrictParsing` | Require all output fields present | `false` |
| `FallbackToText` | Fall back to text parsing on XML failure | `true` |
| `ValidateXML` | Perform XML syntax validation | `true` |
| `MaxDepth` | Maximum nesting depth | `10` |
| `MaxSize` | Maximum XML size in bytes | `1048576` (1MB) |
| `CustomTags` | Override default tag names | `nil` |
| `TypeHints` | Include type info in instructions | `false` |

---

## Preset Configurations

### Default Configuration

Balanced for general use:

```go
config := interceptors.DefaultXMLConfig()
// StrictParsing: false
// FallbackToText: true
// ValidateXML: true
// MaxDepth: 10
// MaxSize: 1MB
```

### Strict Configuration

For production systems requiring reliability:

```go
config := interceptors.StrictXMLConfig()
// StrictParsing: true   - All fields required
// FallbackToText: false - No fallback
// ValidateXML: true     - Full validation
```

### Flexible Configuration

For experimental or development:

```go
config := interceptors.FlexibleXMLConfig()
// StrictParsing: false  - Missing fields OK
// FallbackToText: true  - Has fallback
// ValidateXML: false    - Minimal validation
```

### Performant Configuration

Optimized for speed:

```go
config := interceptors.PerformantXMLConfig()
// StrictParsing: false
// ValidateXML: false    - Skip validation
// MaxDepth: 5           - Reduced depth
```

### Secure Configuration

Maximum security restrictions:

```go
config := interceptors.SecureXMLConfig()
// StrictParsing: true
// ValidateXML: true
// MaxDepth: 5           - Limited depth
// MaxSize: 102400       - 100KB limit
```

---

## Composing with Other Interceptors

### Interceptor Chain

Combine XML interceptors with logging, metrics, and more:

```go
import "github.com/XiaoConstantine/dspy-go/pkg/modules/interceptors"

// Create comprehensive interceptor chain
chain := interceptors.CreateXMLInterceptorChain(config,
    interceptors.LoggingModuleInterceptor(),
    interceptors.MetricsModuleInterceptor(),
)

// Apply to module
module.SetInterceptors(chain)
```

### Manual Chain Construction

```go
// Build chain manually for fine-grained control
chain := []core.ModuleInterceptor{
    interceptors.LoggingModuleInterceptor(),      // 1. Logging
    interceptors.XMLFormatModuleInterceptor(config), // 2. XML format injection
    interceptors.MetricsModuleInterceptor(),      // 3. Metrics collection
    interceptors.XMLParseModuleInterceptor(config),  // 4. XML response parsing
}

module.SetInterceptors(chain)
```

### How Interceptors Work

1. **XMLFormatModuleInterceptor**: Injects XML formatting instructions into the prompt
2. **XMLParseModuleInterceptor**: Parses XML responses into structured fields

```
Input → [Logging] → [XML Format] → [Metrics] → LLM → [XML Parse] → Output
```

---

## XML vs Structured Output

| Feature | XML Adapter | `.WithStructuredOutput()` |
|---------|-------------|---------------------------|
| **Output Format** | XML tags | Native JSON |
| **Security Controls** | Depth/size limits, validation | Provider-dependent |
| **Fallback** | Configurable text fallback | Built-in graceful fallback |
| **Custom Tags** | Supported | N/A |
| **Interceptor Pattern** | Yes - composable | No |
| **Performance** | ~3,663 ns/op parsing | Provider-optimized |
| **Provider Support** | All providers | Providers with JSON mode |

### When to Use XML Adapters

- Need fine-grained control over parsing behavior
- Want to compose with other interceptors (logging, metrics)
- Need custom XML tag names
- Require enhanced security controls
- Working with providers without native JSON mode

### When to Use Structured Output

- Simpler use cases
- Provider has good native JSON support
- Don't need interceptor composition
- Want provider-optimized performance

---

## Use Cases

### Multi-Field Extraction

```go
signature := core.NewSignature(
    []core.InputField{
        {Field: core.NewField("document")},
    },
    []core.OutputField{
        {Field: core.NewField("summary")},
        {Field: core.NewField("sentiment")},
        {Field: core.NewField("key_points")},
        {Field: core.NewField("confidence")},
    },
)

predict := modules.NewPredict(signature)
interceptors.ApplyXMLInterceptors(predict, interceptors.DefaultXMLConfig())

result, _ := predict.Process(ctx, map[string]interface{}{
    "document": longDocument,
})

// All fields reliably extracted
fmt.Println(result["summary"])
fmt.Println(result["sentiment"])
fmt.Println(result["key_points"])
fmt.Println(result["confidence"])
```

### With ReAct Module

```go
react := modules.NewReAct(signature, registry, 5)

// Enable XML parsing for ReAct responses
config := interceptors.SecureXMLConfig()
react.WithXMLParsing(config)

result, _ := react.Process(ctx, inputs)
```

### Production Pipeline

```go
// Production configuration with logging and metrics
config := interceptors.StrictXMLConfig()

chain := interceptors.CreateXMLInterceptorChain(config,
    interceptors.LoggingModuleInterceptor(),
    interceptors.MetricsModuleInterceptor(),
    interceptors.CachingModuleInterceptor(cacheConfig),
)

predict := modules.NewPredict(signature)
predict.SetInterceptors(chain)
```

---

## Performance

Benchmarks for XML interceptor operations:

| Operation | Performance | Throughput |
|-----------|-------------|------------|
| XML Parsing | ~3,663 ns/op | 273K ops/sec |
| Full Pipeline | ~5,727 ns/op | 175K ops/sec |
| Memory Usage | ~3.5KB/op | - |
| Pipeline Memory | ~7.8KB/op | - |

XML adapters are designed for high-performance production use while maintaining security and flexibility.

---

## Examples

Complete examples are available in the repository:

```bash
# Basic XML usage
cd examples/xml_adapter/basic_usage
go run basic_usage.go

# Composable interceptors
cd examples/xml_adapter/composable
go run composable.go

# Custom configurations
cd examples/xml_adapter/custom_config
go run custom_config.go

# Predict with XML
cd examples/xml_adapter/predict_xml
export GEMINI_API_KEY="your-key"
go run predict_xml_example.go

# ReAct with XML
cd examples/xml_adapter/react_xml
export GEMINI_API_KEY="your-key"
go run react_xml_example.go
```

**[XML Adapter Examples](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/xml_adapter)**

---

## Next Steps

- **[Core Concepts](core-concepts/)** - Understand modules and signatures
- **[Building Agents](agents/)** - Use XML adapters with agents
- **[Tool Management](tools/)** - Integrate with tool pipelines
