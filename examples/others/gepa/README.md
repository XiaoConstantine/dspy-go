# GEPA (Generative Evolutionary Prompt Adaptation) Example

This example demonstrates metric-driven prompt optimization based on ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/pdf/2507.19457).

## Overview

GEPA starts from the original program and retains accepted children alongside their parents. It selects parents using per-example user-metric scores, not synthetic quality or style objectives.

## Key Features Demonstrated

### Minibatch acceptance
- Draw an epoch-shuffled training minibatch each iteration.
- Evaluate parent and proposed child on the same examples.
- Accept only strict improvement in the sum of metric scores.

### Evidence-based reflection
- Make one reflection-model proposal call from the selected parent's minibatch inputs, outputs, scores, feedback, and module traces.
- Skip already-solved batches and failed, empty, or unchanged proposals.
- Never replace a failed proposal with random style edits.

### Validation and budget
- Fully validate only the seed and accepted children; cache equivalent candidate/example evaluations.
- Keep parents so later proposals can revisit useful alternatives.
- Choose the final program by aggregate validation score.
- Stop at the metric budget, a perfect validation result, or an enabled safety cap.

The seven `MultiObjectiveFitness` dimensions remain available as agent diagnostics; they do not drive this optimizer's acceptance or selection.

## Usage

### Basic Usage

```bash
# Run with Gemini API key
go run main.go -api-key YOUR_GEMINI_API_KEY

# Use different dataset
go run main.go -api-key YOUR_KEY -dataset hotpotqa

# Customize evolution parameters
go run main.go -api-key YOUR_KEY -generations 10

# Enable verbose logging to see evolution details
go run main.go -api-key YOUR_KEY -verbose
```

### Command Line Options

- `-api-key`: **Required** - Your Google Gemini API key
- `-dataset`: Dataset to use (`gsm8k` or `hotpotqa`, default: `gsm8k`)
- `-generations`: Safety cap including seed validation (default: 8)
- `-verbose`: Enable detailed logging of the evolutionary process

The example keeps the default 1000 metric-call budget. Progress output reports the best metric score, metric-call count, and retained archive; scores and runtime depend on the model and examples.

## Datasets Supported

### GSM8K (Grade School Math 8K)
- **Task**: Mathematical reasoning and problem solving
- **Metric**: Exact answer matching with numerical extraction
- **Signature**: Question → Reasoning + Answer

### HotPotQA
- **Task**: Multi-hop question answering
- **Metric**: String similarity with partial credit
- **Signature**: Question → Reasoning + Answer

## Configuration Options

Start with defaults rather than a partial configuration literal:

```go
config := optimizers.DefaultGEPAConfig()
config.MaxMetricCalls = 1000
config.MaxGenerations = 8
config.MutationRate = 1.0 // Propose each iteration (the default).
config.EvaluationBatchSize = 5
config.RandomSeed = 42
```

The unused `PopulationSize`, `ReflectionFreq`, `ValidationFrequency`, `CrossoverRate`, `ElitismRate`, `Temperature`, `MaxTokens`, and `SelfCritiqueTemp` fields were removed. Configure sampling on the supplied LM, not on GEPA. `SkipPerfectScore` defaults to true with `PerfectScore=1`; change that threshold or disable skipping for other metric scales.

## Performance Considerations

- **Metric budget**: Counts uncached training and validation evaluations, not reflection calls. Complete evaluations must fit the remaining budget.
- **Generation cap**: Bounds failed or skipped proposals that may consume no new metric calls.
- **Validation set**: Use `ValidationExamples` or `ValidationSplit` for a holdout; otherwise each accepted child is validated on the full training set.
- **Dataset size**: Full validation cost still scales with the validation set; it no longer scales with an upfront paraphrase population.

## Comparison with Other Optimizers

| Aspect | GEPA | MIPRO | SIMBA | BootstrapFewShot |
|--------|------|-------|-------|------------------|
| **Approach** | Reflective prompt evolution | TPE search | Introspective learning | Example selection |
| **Objectives** | Per-example user metric | Single objective | Self-analysis | Few-shot quality |
| **Reflection** | LLM-based critique | None | Self-introspection | None |
| **Diversity** | Retained parents + validation archive | Limited | Batch-based | Example-based |
| **Complexity** | High (comprehensive) | Medium | Medium-High | Low |
| **Quality** | Highest | High | High | Medium |

## Requirements

- Go 1.27 or later
- Google Gemini API key
- Internet connection for dataset loading and LLM calls
- Sufficient computation time (GEPA is thorough but takes longer than simpler optimizers)

## Tips for Best Results

1. **Start Small**: Use a budget sufficient for seed validation and a few proposals
2. **Monitor Progress**: Use `-verbose` flag to understand the evolutionary process
3. **Dataset Quality**: Higher quality training examples lead to better optimization
4. **Patience**: GEPA's comprehensive approach takes time but delivers superior results
5. **Archive Analysis**: Examine which validation examples each candidate solves

## Paper Reference

This implementation is based on:
- **Paper**: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
- **URL**: https://arxiv.org/pdf/2507.19457
- **Key Innovation**: Natural language reflection for prompt optimization
- **Performance**: 10% average improvement over RL methods, 35x fewer rollouts

The example demonstrates all major paper contributions while providing a practical, usable implementation for real-world prompt optimization tasks.
