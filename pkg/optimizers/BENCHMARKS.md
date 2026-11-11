# DSPy-Go Optimizer Benchmarks

This directory contains comprehensive benchmark tests for all DSPy-Go optimizers, following Go's standard benchmarking conventions.

## Offline optimizer score gate

`TestOfflineOptimizerImprovement` is a quality regression gate, not a throughput benchmark. It runs in `.github/workflows/go.yml` on every pull request, and is also included in ordinary `go test ./pkg/...` runs. It needs no API keys, model downloads, Python, or network access after Go dependencies are installed.

```bash
go test -tags skip -race -count=1 -timeout=2m -run '^TestOffline' -v ./pkg/optimizers
```

On the Linux orb with Go 1.27.4, the race-instrumented suite took about **4.4 seconds** excluding compilation. A compiled test binary also passed inside an isolated network namespace with an empty environment (only `PATH` and `HOME` supplied), demonstrating that neither connectivity nor API keys are required. Ten consecutive `-race` suite runs passed with the MIPRO quarantine below.

The scripted LLM processes real `Predict` prompts through clone-safe `Program` forward factories, including GEPA's installed interceptors. It interprets a four-symbol routing codebook in instructions and demonstrations. The metric scores the four output labels independently; explanatory prose is not a valid four-label answer. Every example contains every symbol, so different minibatches cannot inflate accuracy by sampling only easy cases.

- There are 12 training inputs and 12 held-out inputs with disjoint ticket IDs and separately seeded symbol permutations. Held-out outputs never enter `Compile`.
- The baseline scores **0.25**. Every seed must improve by **at least 0.625**; a partial codebook scoring 0.75 fails. The complete codebook scores 1.00. These are exact scores with no measurement noise.
- Seeds 1 through 8 each run twice with fresh optimizer/model instances. The test compares held-out predictions and scores between replays and verifies that compilation leaves the original program unchanged.
- Proposal responses include wrong and useful instructions. COPRO and GEPA must refine a useful parent rather than merely call the LLM again. SIMBA must incorporate trajectory-derived rules. MIPRO receives a sequential menu containing a correct instruction. TPE searches an explicit categorical instruction domain. Bootstrap's teacher alternates wrong and correct demos.
- The fixture rejects unknown prompts, including errors swallowed by optimizer fallback paths. It never chooses a response based on elapsed time, optimizer scores, or held-out labels.

### What “default” means

MIPRO, COPRO, SIMBA, and GEPA use constructor defaults without smaller search budgets or different strategies. Only seeds are set. SIMBA has no public seed option, so this same-package test replaces its private RNG before compilation. The GEPA comparison changes only `SelectionStrategy` to `"tournament"`.

BootstrapFewShot requires a capacity argument and has no default: the fixture supplies four. TPE is a `SearchStrategy`, not a `core.Optimizer`; a test-only adapter supplies its required domain and 25-trial budget, evaluates real programs, and applies `GetBestParams`. TPE hyperparameters remain at defaults. These two required caller choices are not claimed to be library defaults.

### Observed results

Measured on f2b0583 plus the GEPA metric-selection fix. Each optimized score below was the same for all eight seeds. These numbers describe this synthetic task, not real-model accuracy or algorithm rankings.

| Optimizer | Baseline | Optimized | CI status | Random final choice rejected |
| --- | ---: | ---: | --- | ---: |
| BootstrapFewShot (capacity 4) | 0.25 | 1.00 | Enforced | 6/8 seeds |
| MIPRO default | 0.25 | 0.00 | Measured, quarantined failure | 8/8 seeds |
| COPRO default | 0.25 | 1.00 | Enforced | 7/8 seeds |
| SIMBA default | 0.25 | 1.00 | Enforced | 6/8 seeds |
| GEPA default (`adaptive_pareto`) | 0.25 | 1.00 | Enforced | 7/8 seeds |
| GEPA `tournament` | 0.25 | 1.00 | Enforced | 7/8 seeds |
| TPE (25 trials) | 0.25 | 1.00 | Enforced | 8/8 seeds |

The no-op is rejected on all eight seeds for every optimizer. Before the GEPA fix, both GEPA cases scored 0.00 on every seed: interceptor heuristics overrode the metric, marked the run converged at generation zero, and raced under default concurrency. After removing those, the default `MutationRate` of 0.3 still ended some seeds at 0.75 by skipping proposals, so the default is now 1.0.

### Demonstrating that the gate rejects broken selection

Every normal run scores a random-choice control using the actual emitted candidates and checks that it fails the same gate. The control gets the real optimizer's entire generation/evaluation budget, then ignores scores and uniformly chooses a distinct emitted instruction or the original. Bootstrap instead chooses an observed teacher demo without checking its metric. This tests score-blind **final selection**, not random search that evaluates candidates and correctly retains the best.

These mutation commands replace the actual returned program before held-out evaluation. Both must exit nonzero; neither changes the metric, margin, or expected scores:

```bash
DSPY_OFFLINE_MUTANT=noop go test -tags skip -race -count=1 -run '^TestOfflineOptimizerImprovement$' -v ./pkg/optimizers
DSPY_OFFLINE_MUTANT=random go test -tags skip -race -count=1 -run '^TestOfflineOptimizerImprovement$' -v ./pkg/optimizers
```

### Known failures and enabling the strict gate

**MIPRO:** `NewMIPRO` leaves `NumTrials=0`. The main loop runs no trials. TPE returns an empty best-parameter map and `-MaxFloat64`, which compares greater than MIPRO's initial `-Inf`; the final-candidate fallback therefore applies instruction index zero without a metric comparison. It selects this instruction with **no demos**:

```text
Explain routing with a=blue b=coral c=dusk d=amber. Include a metaphor and a detailed analytical justification.
```

For `ticket=12345 symbols=abcd`, the selected program produces prose containing the wrong labels `blue coral dusk amber`, rather than the expected `amber blue coral dusk`. Its score is 0.00. Rebuilding the selected instruction and demos in a fresh program gives identical output; there are no unknown prompts or parse errors. Independent partial/correct instruction controls score 0.75/1.00. This is an optimizer defect, not an instruction-parsing failure.

Run the unchanged strict assertion explicitly (currently expected to fail):

```bash
DSPY_OFFLINE_STRICT=1 go test -tags skip -race -count=1 -run '^TestOfflineOptimizerImprovement$/MIPRO$' -v ./pkg/optimizers
```

After a fix, run the relevant strict command first. Once it passes, remove that case's `quarantine` text from `offlineCases`. Do not lower the margin, change the script, or change optimizer defaults to activate it. MIPRO continues to run and report its failure in ordinary CI, and an unexpected pass tells the maintainer to remove its quarantine.

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
