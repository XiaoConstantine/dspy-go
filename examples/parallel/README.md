# Parallel Processing Examples

This directory contains examples demonstrating the Parallel module in DSPy-Go, which enables concurrent execution of any module across multiple inputs for improved performance.

## Overview

The Parallel module wraps any DSPy module to provide:
- **Concurrent Execution**: Process multiple inputs simultaneously using configurable worker pools
- **Error Handling**: Options for handling failures (continue, stop, or collect failures)
- **Performance**: Significant speedup for batch processing tasks
- **Flexibility**: Works with any DSPy module (Predict, ChainOfThought, ReAct, etc.)

## Running the Examples

1. **Run the examples with your API key**:
   ```bash
   cd examples/parallel
   go run main.go -api-key="your-google-api-key-here"
   ```

2. **Get a Google API key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Use it with the `-api-key` flag

## Examples Included

### 1. Basic Batch Text Summarization
- **Purpose**: Demonstrates basic parallel processing for text summarization
- **Features**: Default worker configuration, simple batch processing
- **Use Case**: Processing multiple documents simultaneously

### 2. Sentiment Analysis with Error Handling
- **Purpose**: Shows error handling and failure collection
- **Features**: Custom worker count, return failures option
- **Use Case**: Robust batch processing where some inputs might fail

### 3. Multi-language Translation
- **Purpose**: Demonstrates concurrent translation tasks
- **Features**: Custom worker pool size
- **Use Case**: Processing multiple translation requests efficiently

### 4. Question Answering with Stop-on-First-Error
- **Purpose**: Shows fail-fast behavior for critical applications
- **Features**: Stop-on-first-error configuration
- **Use Case**: When any failure should halt the entire batch

## Configuration Options

The Parallel module supports several configuration options:

```go
parallel := modules.NewParallel(module,
    modules.WithMaxWorkers(4),          // Set worker pool size
    modules.WithReturnFailures(true),   // Include failed results
    modules.WithStopOnFirstError(true), // Stop on first failure
)
```

### Worker Pool Configuration
- **Default**: `runtime.NumCPU()` workers
- **Custom**: Set any positive number of workers
- **Recommendation**: Start with CPU count, adjust based on I/O vs CPU workload

### Error Handling Strategies

1. **Continue Processing** (default):
   ```go
   // Failed inputs are skipped, successful ones are returned
   parallel := modules.NewParallel(module)
   ```

2. **Collect Failures**:
   ```go
   // Failed inputs are included in the output with error details
   parallel := modules.NewParallel(module, 
       modules.WithReturnFailures(true))
   ```

3. **Stop on First Error**:
   ```go
   // Processing stops immediately when any input fails
   parallel := modules.NewParallel(module, 
       modules.WithStopOnFirstError(true))
   ```

## Input Format

The Parallel module expects a special input format:

```go
inputs := map[string]interface{}{
    "batch_inputs": []map[string]interface{}{
        {"field1": "value1", "field2": "value2"},
        {"field1": "value3", "field2": "value4"},
        // ... more inputs
    },
}
```

For single inputs, the module automatically falls back to normal processing.

## Output Format

```go
result := map[string]interface{}{
    "results": []map[string]interface{}{
        {"output_field": "result1"},
        {"output_field": "result2"},
        // ... results in same order as inputs
    },
    // Optional: only if WithReturnFailures(true)
    "failures": []map[string]interface{}{
        {"index": 2, "error": "error message"},
    },
}
```

## Performance Considerations

- **I/O Bound Tasks**: Use more workers than CPU cores (e.g., 2-4x CPU count)
- **CPU Bound Tasks**: Use workers ≈ CPU cores
- **Memory Usage**: More workers = more concurrent LLM calls = higher memory usage
- **Rate Limits**: Consider LLM provider rate limits when setting worker count

## Best Practices

1. **Start Small**: Begin with default settings and measure performance
2. **Monitor Resources**: Watch memory usage and API rate limits
3. **Handle Errors**: Always consider what should happen when some inputs fail
4. **Batch Size**: Larger batches improve throughput but increase latency
5. **Retry Logic**: For transient failures, implement retry at the application level

## Integration with Other Modules

The Parallel module works with any DSPy module:

```go
// Parallel Chain of Thought reasoning
cot := modules.NewChainOfThought(signature)
parallelCoT := modules.NewParallel(cot)

// Parallel ReAct agents
react := modules.NewReAct(signature, tools)
parallelReAct := modules.NewParallel(react)

// Parallel Refine for quality improvement
refine := modules.NewRefine(predict, refineConfig)
parallelRefine := modules.NewParallel(refine)
```

## Common Use Cases

- **Document Processing**: Summarization, extraction, classification
- **Data Analysis**: Sentiment analysis, entity recognition, categorization
- **Content Generation**: Translation, rewriting, formatting
- **Question Answering**: FAQ processing, knowledge base queries
- **Evaluation**: Running test suites against multiple examples

## Troubleshooting

**Issue**: Workers not improving performance
- **Solution**: Check if task is I/O bound vs CPU bound, adjust worker count

**Issue**: Out of memory errors
- **Solution**: Reduce worker count or batch size

**Issue**: Rate limit errors
- **Solution**: Reduce worker count or implement backoff/retry logic

**Issue**: Results in wrong order
- **Solution**: Results maintain input order automatically - check input preparation

For more examples and documentation, see the [main DSPy-Go repository](https://github.com/XiaoConstantine/dspy-go).