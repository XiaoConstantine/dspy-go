# XML Interceptor Examples

This directory contains examples demonstrating how to use XML interceptors in dspy-go. The XML interceptors provide structured XML output parsing using the interceptor pattern.

## Examples

### 1. Basic Usage (`basic_usage/`)

Demonstrates the fundamental usage of XML interceptors:
- Creating XML interceptor with default configuration
- Applying interceptor to a module
- Automatic XML instruction injection and response parsing

```bash
cd basic_usage && go run basic_usage.go
```

### 2. Composable Interceptors (`composable/`)

Shows how to combine XML interceptors with other interceptors:
- Chaining logging, metrics, and XML interceptors
- Creating comprehensive processing pipelines
- Order of interceptor execution

```bash
cd composable && go run composable.go
```

### 3. Custom Configuration (`custom_config/`)

Demonstrates various XML configuration options:
- Strict vs flexible parsing modes
- Performance-optimized configurations
- Custom XML tags and type hints
- Security settings

```bash
cd custom_config && go run custom_config.go
```

### 4. Predict with XML Output (`predict_xml/`)

**NEW**: Demonstrates enhanced Predict module with XML output formatting:
- Traditional Predict with prefix-based output (backward compatibility)
- Predict with XML structured output (multi-field responses)
- Performance-optimized XML configuration
- Secure XML configuration with enhanced security features

**Requirements:**
- Set `GEMINI_API_KEY` environment variable
- Valid Gemini API access

```bash
export GEMINI_API_KEY="your-api-key-here"
cd predict_xml && go run predict_xml_example.go
```

**Key Features Demonstrated:**
- ✅ **Structured Multi-field Output**: Generate multiple output fields in structured XML format
- ✅ **Enhanced Security**: Size limits, depth limits, timeout protection
- ✅ **Performance Optimization**: Custom XML tags and performance-tuned configurations
- ✅ **Backward Compatibility**: Existing Predict code works unchanged
- ✅ **Real LLM Integration**: Works with actual Gemini responses

### 5. ReAct with XML Interceptors (`react_xml/`)

Demonstrates enhanced ReAct module with XML interceptor integration:
- Traditional ReAct with hardcoded XML parsing (backward compatibility)
- ReAct with XML interceptors (enhanced parsing with security features)
- Real-world usage with Gemini LLM and actual tool execution
- Error handling and configuration examples

**Requirements:**
- Set `GEMINI_API_KEY` environment variable
- Valid Gemini API access

```bash
export GEMINI_API_KEY="your-api-key-here"
cd ../react_xml && go run react_xml_example.go
```

**Key Features Demonstrated:**
- ✅ **Backward Compatibility**: Existing ReAct code works unchanged
- ✅ **Enhanced Security**: Size limits, validation, timeout protection
- ✅ **Robust Error Handling**: Graceful fallback for malformed XML
- ✅ **Opt-in Enhancement**: Use `WithXMLParsing(config)` to enable
- ✅ **Real LLM Integration**: Works with actual Gemini responses

## Key Concepts

### XML Interceptor Chain

The XML interceptor system consists of two main components:

1. **Format Interceptor** (`XMLFormatModuleInterceptor`): Injects XML formatting instructions into the input
2. **Parse Interceptor** (`XMLParseModuleInterceptor`): Parses XML responses into structured fields

### Configuration Options

- **StrictParsing**: Requires all output fields to be present
- **FallbackToText**: Enables graceful fallback for malformed XML
- **ValidateXML**: Performs XML syntax validation
- **MaxDepth/MaxSize**: Security limits for XML parsing
- **CustomTags**: Override default XML tag names
- **TypeHints**: Include type information in XML instructions

### Preset Configurations

- `DefaultXMLConfig()`: Balanced configuration for general use
- `StrictXMLConfig()`: Strict parsing, no fallback
- `FlexibleXMLConfig()`: Allows missing fields, has fallback
- `PerformantXMLConfig()`: Optimized for speed
- `SecureXMLConfig()`: Enhanced security restrictions

## Integration Patterns

### With Existing Modules

```go
// Apply to any interceptable module
module := yourModule // Must implement core.InterceptableModule
xmlInterceptor := interceptors.XMLModuleInterceptor(config)
module.SetInterceptors([]core.ModuleInterceptor{xmlInterceptor})
```

### With Other Interceptors

```go
// Create comprehensive interceptor chain
chain := []core.ModuleInterceptor{
    interceptors.LoggingModuleInterceptor(),        // Logging
    interceptors.XMLFormatModuleInterceptor(config), // XML format
    interceptors.MetricsModuleInterceptor(),        // Metrics
    interceptors.XMLParseModuleInterceptor(config), // XML parse
}
module.SetInterceptors(chain)
```

### Helper Functions

```go
// Apply XML interceptors to a module
err := interceptors.ApplyXMLInterceptors(module, config)

// Create interceptor chain with additional interceptors
chain := interceptors.CreateXMLInterceptorChain(config,
    interceptors.LoggingModuleInterceptor(),
    interceptors.MetricsModuleInterceptor(),
)
```

## Running the Examples

1. Ensure you have Go 1.21+ installed
2. From this directory, run any example:
   ```bash
   cd basic_usage && go run basic_usage.go
   cd ../composable && go run composable.go
   cd ../custom_config && go run custom_config.go
   cd ../predict_xml && go run predict_xml_example.go
   cd ../react_xml && go run react_xml_example.go
   ```

## Real-World Usage

In production environments, you would:

1. **Replace mock modules** with actual dspy-go modules (ChainOfThought, React, etc.)
2. **Configure appropriate settings** based on your security and performance requirements
3. **Combine with other interceptors** for logging, caching, and monitoring
4. **Handle errors appropriately** with proper fallback strategies

## Performance Considerations

- XML parsing: ~3,663 ns/op (327K ops/sec)
- Full pipeline: ~5,727 ns/op (209K ops/sec)
- Memory usage: ~3.5KB/op for parsing, ~7.8KB/op for full pipeline

The XML interceptors are designed for high-performance production use while maintaining security and flexibility.
