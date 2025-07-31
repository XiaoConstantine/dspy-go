# DSPy-Go Optimizer Benchmarks

This directory contains comprehensive benchmark tests for all DSPy-Go optimizers, following Go's standard benchmarking conventions.

## Available Benchmarks

### 1. COPRO (Collaborative Prompt Optimization)
- `BenchmarkCOPRO` - Main benchmark suite
- `BenchmarkCOPRODatasetScaling` - Performance across dataset sizes
- `BenchmarkCOPROParameterTuning` - Different parameter configurations
- `BenchmarkCOPROConcurrency` - Concurrency level testing

### 2. SIMBA (Simplified Multi-step Bidirectional Approach)
- `BenchmarkSIMBA` - Main benchmark suite
- `BenchmarkSIMBADatasetScaling` - Performance across dataset sizes
- `BenchmarkSIMBAParameterTuning` - Different parameter configurations
- `BenchmarkSIMBAFastMode` - Fast vs standard mode comparison

### 3. MIPRO (Multi-step Interactive Prompt Optimization)
- `BenchmarkMIPRO` - Main benchmark suite
- `BenchmarkMIPRODatasetScaling` - Performance across dataset sizes
- `BenchmarkMIPROParameterTuning` - Different parameter configurations
- `BenchmarkMIPROMode` - Light vs medium mode comparison
- `BenchmarkMIPROTrials` - Number of trials scaling

### 4. BootstrapFewShot (Bootstrap Few-Shot Learning)
- `BenchmarkBootstrapFewShot` - Main benchmark suite
- `BenchmarkBootstrapFewShotDatasetScaling` - Performance across dataset sizes
- `BenchmarkBootstrapFewShotParameterTuning` - Different parameter configurations
- `BenchmarkBootstrapFewShotMetricVariations` - Different metric behaviors
- `BenchmarkBootstrapFewShotConcurrency` - Concurrency level testing

## Running Benchmarks

### Run All Benchmarks
```bash
go test -bench=Benchmark ./pkg/optimizers
```

### Run Specific Optimizer Benchmarks
```bash
# COPRO benchmarks only
go test -bench=BenchmarkCOPRO ./pkg/optimizers

# SIMBA benchmarks only
go test -bench=BenchmarkSIMBA ./pkg/optimizers

# MIPRO benchmarks only
go test -bench=BenchmarkMIPRO ./pkg/optimizers

# BootstrapFewShot benchmarks only
go test -bench=BenchmarkBootstrapFewShot ./pkg/optimizers
```

### Run with Custom Duration
```bash
# Run for 5 seconds per benchmark
go test -bench=Benchmark -benchtime=5s ./pkg/optimizers

# Run 10 iterations per benchmark
go test -bench=Benchmark -benchtime=10x ./pkg/optimizers
```

### Memory Profiling
```bash
go test -bench=Benchmark -benchmem ./pkg/optimizers
```

### CPU Profiling
```bash
go test -bench=Benchmark -cpuprofile=cpu.prof ./pkg/optimizers
```

## Benchmark Configuration

### Dataset Sizes
- **Tiny**: 10 examples
- **Small**: 20 examples
- **Medium**: 50 examples
- **Large**: 100 examples

### Configuration Presets
- **Fast**: Quick benchmarks for development (3 trials, small datasets)
- **Standard**: Balanced benchmarks for CI (5 trials, medium datasets)
- **Comprehensive**: Thorough benchmarks for performance analysis (8+ trials, large datasets)

## Shared Utilities

The `benchmark_utils.go` file provides:

- **CreateBenchmarkDatasets()**: Standard datasets for consistent testing
- **CreateBenchmarkProgram()**: Standard program structure
- **BenchmarkAccuracyMetric()**: Consistent accuracy measurement
- **StandardBenchmarkConfigs()**: Predefined configuration sets
- **BenchmarkDatasetFromExamples()**: Dataset conversion utilities

## Performance Expectations

Typical performance ranges (on Apple M3 Pro):

- **BootstrapFewShot**: 85-280μs per operation
- **COPRO**: 100-500ms per operation (includes LLM-assisted optimization)
- **SIMBA**: 50-200ms per operation (depends on mini-batch processing)
- **MIPRO**: 200-800ms per operation (includes teacher-student optimization)

## Best Practices

1. **Development**: Use fast benchmarks (`-benchtime=100ms`)
2. **CI/CD**: Use standard benchmarks with consistent environment
3. **Performance Analysis**: Use comprehensive benchmarks with profiling
4. **Comparison**: Always benchmark on the same hardware and Go version
5. **Trending**: Run benchmarks regularly to detect performance regressions

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce dataset size or concurrency levels
2. **Slow Benchmarks**: Use shorter benchmark time for development
3. **Inconsistent Results**: Ensure stable system load and thermal conditions

### Environment Variables

```bash
# Disable logging for cleaner benchmark output
export LOG_LEVEL=ERROR

# Set concurrency for reproducible results
export GOMAXPROCS=4
```

## Contributing

When adding new optimizers or modifying existing ones:

1. Add corresponding benchmark tests following the same patterns
2. Use the shared benchmark utilities for consistency
3. Include parameter tuning and scaling benchmarks
4. Test with different dataset sizes and configurations
5. Update this documentation

## Integration with CI/CD

Example GitHub Actions workflow:
```yaml
- name: Run Optimizer Benchmarks
  run: |
    go test -bench=Benchmark -benchtime=1s -benchmem ./pkg/optimizers > benchmarks.txt

- name: Compare Performance
  run: |
    benchstat baseline.txt benchmarks.txt
```
