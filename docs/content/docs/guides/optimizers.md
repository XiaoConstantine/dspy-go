---
title: "Optimizers"
description: "Automatically improve your LLM applications"
summary: "Master GEPA, MIPRO, SIMBA, and other advanced optimizers to systematically enhance prompt performance"
date: 2025-10-13T00:00:00+00:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 300
toc: true
seo:
  title: "Optimizers - Automatic Prompt Improvement in dspy-go"
  description: "Learn about GEPA, MIPRO, SIMBA and other state-of-the-art optimizers for systematic prompt optimization"
  canonical: ""
  noindex: false
---

# Optimizers

**Optimizers** are one of dspy-go's most powerful features. Instead of manually tweaking prompts, optimizers automatically improve your signatures, instructions, and examples based on your dataset and metrics.

## Why Optimize?

Manual prompt engineering is:
- **Time-consuming**: Hours of trial and error
- **Subjective**: What works for you might not work universally
- **Brittle**: Small changes can break everything
- **Un-scalable**: Hard to improve systematically

**Optimizers solve this** by:
- ✅ Automatically testing variations
- ✅ Using data to find what works
- ✅ Systematically improving performance
- ✅ Producing reproducible results

---

## Quick Optimizer Comparison

| Optimizer | Best For | Speed | Quality | Complexity |
|-----------|----------|-------|---------|-----------|
| **BootstrapFewShot** | Quick wins, simple tasks | ⚡⚡⚡ | ⭐⭐⭐ | Low |
| **COPRO** | Multi-module systems | ⚡⚡ | ⭐⭐⭐⭐ | Medium |
| **MIPRO** | Systematic optimization | ⚡⚡ | ⭐⭐⭐⭐ | Medium |
| **SIMBA** | Introspective learning | ⚡ | ⭐⭐⭐⭐⭐ | High |
| **GEPA** | Multi-objective evolution | ⚡ | ⭐⭐⭐⭐⭐ | High |

---

## GEPA - Generative Evolutionary Prompt Adaptation

**The most advanced optimizer** in dspy-go. Uses evolutionary algorithms with multi-objective Pareto optimization and LLM-based self-reflection.

### What Makes GEPA Special?

- 🧬 **Evolutionary Optimization**: Genetic algorithms for prompt evolution
- 🎯 **Multi-Objective**: Optimizes across 7 dimensions simultaneously
- 🏆 **Pareto Selection**: Maintains diverse solutions for different trade-offs
- 🧠 **LLM Self-Reflection**: Uses LLMs to analyze and critique prompts
- 📊 **Elite Archive**: Preserves high-quality solutions across generations

### The 7 Optimization Dimensions

1. **Success Rate**: How often the program succeeds
2. **Quality**: Answer accuracy and completeness
3. **Efficiency**: Token usage and speed
4. **Robustness**: Performance across edge cases
5. **Generalization**: How well it handles new inputs
6. **Diversity**: Variety in reasoning approaches
7. **Innovation**: Novel problem-solving strategies

### When to Use GEPA

✅ **Perfect for:**
- Complex, multi-faceted problems
- When you need multiple solution strategies
- Production systems requiring robustness
- Trade-offs between speed and quality

❌ **Avoid for:**
- Simple, single-objective tasks
- Very tight time constraints
- Limited compute resources

### GEPA Example

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/optimizers"
    "github.com/XiaoConstantine/dspy-go/pkg/datasets"
)

func main() {
    // 1. Configure LLM
    llm, _ := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
    core.SetDefaultLLM(llm)

    // 2. Create your program
    program := createMyProgram() // Your application

    // 3. Load dataset
    dataset, _ := datasets.LoadGSM8K("path/to/gsm8k")

    // 4. Define metric
    metricFunc := func(example, prediction map[string]interface{}, trace *core.Trace) (float64, map[string]interface{}) {
        expected := example["answer"].(string)
        actual := prediction["answer"].(string)

        if expected == actual {
            return 1.0, map[string]interface{}{"correct": true}
        }
        return 0.0, map[string]interface{}{"correct": false}
    }

    // 5. Configure GEPA
    config := &optimizers.GEPAConfig{
        PopulationSize:    20,                // Size of population
        MaxGenerations:    10,                // Number of generations
        SelectionStrategy: "adaptive_pareto", // Multi-objective Pareto
        MutationRate:      0.3,               // Mutation probability
        CrossoverRate:     0.7,               // Crossover probability
        ReflectionFreq:    2,                 // LLM reflection every 2 generations
        ElitismRate:       0.1,               // Preserve top 10%
    }

    gepa, err := optimizers.NewGEPA(config)
    if err != nil {
        log.Fatal(err)
    }

    // 6. Optimize
    ctx := context.Background()
    optimizedProgram, err := gepa.Compile(ctx, program, dataset, metricFunc)
    if err != nil {
        log.Fatal(err)
    }

    // 7. Access results
    state := gepa.GetOptimizationState()
    fmt.Printf("Best fitness: %.3f\n", state.BestFitness)
    fmt.Printf("Generations: %d\n", state.CurrentGeneration)

    // 8. Get Pareto archive (multiple solutions optimized for different trade-offs)
    archive := state.GetParetoArchive()
    fmt.Printf("Elite solutions: %d\n", len(archive))

    // Each solution in archive excels in different ways:
    // - Some optimized for speed
    // - Some optimized for accuracy
    // - Some balanced between both
}
```

### GEPA Configuration Guide

```go
// Quick optimization (5-10 minutes)
config := &optimizers.GEPAConfig{
    PopulationSize:    10,
    MaxGenerations:    5,
    SelectionStrategy: "adaptive_pareto",
    MutationRate:      0.3,
    CrossoverRate:     0.7,
    ReflectionFreq:    3,
}

// Balanced optimization (15-30 minutes)
config := &optimizers.GEPAConfig{
    PopulationSize:    20,
    MaxGenerations:    10,
    SelectionStrategy: "adaptive_pareto",
    MutationRate:      0.3,
    CrossoverRate:     0.7,
    ReflectionFreq:    2,
}

// Deep optimization (1-2 hours)
config := &optimizers.GEPAConfig{
    PopulationSize:    40,
    MaxGenerations:    20,
    SelectionStrategy: "adaptive_pareto",
    MutationRate:      0.2,
    CrossoverRate:     0.8,
    ReflectionFreq:    1,
    ElitismRate:       0.15,
}
```

---

## Agent Optimization Workflows

GEPA is no longer just for prompt modules. Native, ReAct, and RLM-backed agents can now export a shared optimized-program envelope, save it, restore it, and replay it on held-out tasks.

### Baseline -> Optimize -> Save -> Restore -> Replay

```go
package main

import (
    "context"
    "time"

    agentrlm "github.com/XiaoConstantine/dspy-go/pkg/agents/rlm"
    "github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
    "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
)

func main() {
    baseAgent := agentrlm.NewAgent(
        "demo-rlm",
        rlm.NewFromLLM(
            llm,
            rlm.WithMaxIterations(8),
            rlm.WithContextPolicyPreset(rlm.ContextPolicyAdaptive),
        ),
    )

    workflow, err := optimize.RunGEPAWorkflow(ctx, baseAgent, optimize.GEPAWorkflowRequest{
        Evaluator:        evaluator,
        TrainingExamples: trainExamples,
        ValidationExamples: validationExamples,
        ReplayExamples:   heldOutExamples,
        PassThreshold:    0.9,
        ApplyBest:        true,
        ArtifactPath:     "optimized_program.json",
        Config: optimize.GEPAAdapterConfig{
            PopulationSize:             4,
            MaxGenerations:             2,
            ValidationFrequency:        1,
            MaxMetricCalls:             50,
            ScoreThreshold:             0.95,
            MaxRuntime:                 2 * time.Minute,
            AddFormatFailureAsFeedback: true,
            PrimaryArtifact:            optimize.ArtifactRLMIterationPrompt,
        },
    })
    if err != nil {
        panic(err)
    }

    restored, err := optimize.ReadOptimizedAgentProgram("optimized_program.json")
    if err != nil {
        panic(err)
    }

    replayAgent, _ := baseAgent.Clone()
    _ = optimize.ApplyOptimizedAgentProgram(replayAgent, restored)

    _ = workflow
}
```

`RunGEPAWorkflow` gives you:

- a baseline harness run
- GEPA optimization over agent artifacts
- a persisted `optimized_program.json` envelope
- a replay run on held-out examples

### Named Targets And Stable IDs

Agents can expose stable target IDs instead of one opaque prompt blob. That means GEPA can optimize surfaces like:

- `root.rlm.iteration`
- `root.rlm.max_iterations`
- `root.rlm.adaptive.confidence_threshold`
- `root.react.tool_policy`

These target IDs are what get persisted into the optimized-program envelope.

### Forward-Compatible Restore

`ApplyOptimizedAgentProgram` skips unknown target IDs instead of failing hard. That matters when you restore an older saved program onto a newer agent shape that dropped or renamed a target.

In practice:

- known target IDs are applied
- obsolete target IDs are ignored
- missing values do not wipe unrelated defaults

That makes saved optimized programs much safer to carry across agent revisions.

### GEPA Adapter Controls

For agent workflows, the most useful GEPA controls are:

- `ValidationFrequency`
- `MaxMetricCalls`
- `ScoreThreshold`
- `MaxRuntime`
- `FeedbackEvaluator`
- `AddFormatFailureAsFeedback`

The `examples/rlm_oolong_gepa` directory shows the full persisted agent workflow in a live RLM setting.

---

## MIPRO - Multi-step Interactive Prompt Optimization

**Systematic optimization** using TPE (Tree-structured Parzen Estimator) search. MIPRO is ideal for methodical, data-driven prompt improvement.

### What Makes MIPRO Special?

- 🎯 **TPE Search**: Bayesian optimization for efficient exploration
- 📊 **Systematic**: Methodical testing of variations
- ⚡ **Multiple Modes**: Light, Medium, Heavy optimization
- 🔍 **Interpretable**: Clear insights into what works

### When to Use MIPRO

✅ **Perfect for:**
- Systematic, reproducible optimization
- When you have a good dataset (50+ examples)
- Medium-complexity tasks
- Need explainable improvements

❌ **Avoid for:**
- Very simple tasks (Bootstrap is faster)
- Extremely complex multi-objective problems (use GEPA)
- Limited datasets (< 20 examples)

### MIPRO Example

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

func main() {
    // Configure LLM
    llm, _ := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
    core.SetDefaultLLM(llm)

    // Create MIPRO optimizer
    mipro := optimizers.NewMIPRO(
        metricFunc,
        optimizers.WithMode(optimizers.LightMode),      // Fast optimization
        optimizers.WithNumTrials(10),                   // Number of trials
        optimizers.WithTPEGamma(0.25),                  // Exploration parameter
        optimizers.WithMinExamplesPerModule(5),         // Min examples needed
    )

    // Optimize program
    optimizedProgram, err := mipro.Compile(ctx, program, dataset, nil)
    if err != nil {
        log.Fatal(err)
    }

    // Use optimized program
    result, _ := optimizedProgram.Execute(ctx, inputs)
}
```

### MIPRO Modes

```go
// Light Mode - Quick optimization (5-10 minutes)
// - Fewer trials
// - Faster convergence
// - Good for iteration
optimizers.WithMode(optimizers.LightMode)

// Medium Mode - Balanced (15-30 minutes)
// - Moderate trials
// - Better quality
// - Production-ready
optimizers.WithMode(optimizers.MediumMode)

// Heavy Mode - Thorough (30-60 minutes)
// - Many trials
// - Highest quality
// - Critical systems
optimizers.WithMode(optimizers.HeavyMode)
```

---

## SIMBA - Stochastic Introspective Mini-Batch Ascent

**Introspective learning** with self-analysis. SIMBA learns from its own optimization process.

### What Makes SIMBA Special?

- 🧠 **Introspection**: Analyzes its own optimization progress
- 📦 **Mini-Batch**: Stochastic optimization for efficiency
- 📈 **Adaptive**: Adjusts strategy based on progress
- 💡 **Insights**: Provides detailed learning analysis

### When to Use SIMBA

✅ **Perfect for:**
- Complex reasoning tasks
- When you want insights into the optimization process
- Iterative improvement workflows
- Research and experimentation

❌ **Avoid for:**
- Simple tasks
- Very limited compute
- Need for speed over quality

### SIMBA Example

```go
package main

import (
    "context"
    "fmt"
    "github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

func main() {
    // Create SIMBA optimizer
    simba := optimizers.NewSIMBA(
        optimizers.WithSIMBABatchSize(8),            // Mini-batch size
        optimizers.WithSIMBAMaxSteps(12),            // Max optimization steps
        optimizers.WithSIMBANumCandidates(6),        // Candidates per iteration
        optimizers.WithSamplingTemperature(0.2),     // Exploration vs exploitation
    )

    // Optimize
    optimizedProgram, err := simba.Compile(ctx, program, dataset, metricFunc)
    if err != nil {
        log.Fatal(err)
    }

    // Get introspective insights
    state := simba.GetState()
    fmt.Printf("Steps completed: %d\n", state.CurrentStep)
    fmt.Printf("Best score: %.3f\n", state.BestScore)

    // View detailed analysis
    for i, insight := range state.IntrospectionLog {
        fmt.Printf("Insight %d: %s\n", i+1, insight)
    }

    // Example insights SIMBA might provide:
    // - "Longer instructions improved performance by 15%"
    // - "Adding context examples reduced errors in edge cases"
    // - "Temperature 0.7 balanced creativity and accuracy"
}
```

---

## BootstrapFewShot - Quick Example Selection

**Fast and effective** for most tasks. Automatically selects high-quality few-shot examples.

### What Makes Bootstrap Special?

- ⚡ **Fast**: Quickest optimizer
- 🎯 **Effective**: Works well for most tasks
- 📚 **Few-Shot Learning**: Automatic example selection
- 🔄 **Simple**: Easy to use and understand

### When to Use Bootstrap

✅ **Perfect for:**
- Getting started with optimization
- Simple to medium complexity tasks
- Quick iterations
- Proof of concepts

❌ **Avoid for:**
- Very complex multi-step reasoning
- Need for multi-objective optimization
- Research on optimization strategies

### Bootstrap Example

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/optimizers"
    "github.com/XiaoConstantine/dspy-go/pkg/metrics"
)

func main() {
    // Create Bootstrap optimizer
    bootstrap := optimizers.NewBootstrapFewShot(
        dataset,
        metrics.NewExactMatchMetric("answer"),
        optimizers.WithMaxBootstrappedDemos(5),      // Max examples
        optimizers.WithMaxLabeledDemos(3),           // Max labeled examples
    )

    // Optimize module
    optimizedModule, err := bootstrap.Optimize(ctx, originalModule)
    if err != nil {
        log.Fatal(err)
    }

    // Use optimized module
    result, _ := optimizedModule.Process(ctx, inputs)
}
```

---

## COPRO - Collaborative Prompt Optimization

**Multi-module optimization** for complex systems with multiple components.

### When to Use COPRO

✅ **Perfect for:**
- Systems with multiple modules
- RAG pipelines
- Multi-step workflows
- Coordinated optimization

### COPRO Example

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/optimizers"
    "github.com/XiaoConstantine/dspy-go/pkg/metrics"
)

func main() {
    // Create COPRO optimizer
    copro := optimizers.NewCopro(
        dataset,
        metrics.NewRougeMetric("answer"),
    )

    // Optimize module (automatically handles multi-module systems)
    optimizedModule, err := copro.Optimize(ctx, originalModule)
    if err != nil {
        log.Fatal(err)
    }
}
```

---

## Choosing the Right Optimizer

### Decision Tree

```
Start here
    ↓
Is it a simple task? → YES → Use Bootstrap
    ↓ NO
Multiple modules? → YES → Use COPRO
    ↓ NO
Need multi-objective? → YES → Use GEPA
    ↓ NO
Want introspection? → YES → Use SIMBA
    ↓ NO
Use MIPRO (default choice)
```

### By Use Case

| Use Case | Recommended Optimizer |
|----------|----------------------|
| **Getting Started** | BootstrapFewShot |
| **Simple Q&A** | BootstrapFewShot |
| **Math/Logic** | MIPRO (Medium Mode) |
| **RAG Pipelines** | COPRO or MIPRO |
| **Complex Reasoning** | SIMBA or GEPA |
| **Production Critical** | GEPA (Deep optimization) |
| **Research** | SIMBA or GEPA |
| **Multi-Module Systems** | COPRO |

---

## Metrics - Measuring Success

Every optimizer needs a metric to optimize for:

### Built-in Metrics

```go
import "github.com/XiaoConstantine/dspy-go/pkg/metrics"

// Exact match
exactMatch := metrics.NewExactMatchMetric("answer")

// ROUGE score (for summarization)
rouge := metrics.NewRougeMetric("summary")

// F1 score
f1 := metrics.NewF1Metric("prediction", "label")
```

### Custom Metrics

```go
// Define custom metric function
metricFunc := func(example, prediction map[string]interface{}, trace *core.Trace) (float64, map[string]interface{}) {
    expected := example["answer"].(string)
    actual := prediction["answer"].(string)

    // Simple exact match
    if expected == actual {
        return 1.0, map[string]interface{}{"correct": true}
    }

    // Partial credit for close answers
    similarity := calculateSimilarity(expected, actual)
    return similarity, map[string]interface{}{
        "correct": false,
        "similarity": similarity,
    }
}
```

---

## Best Practices

### Dataset Preparation

✅ **DO:**
- Use diverse, representative examples
- Include edge cases
- Aim for 50+ examples for MIPRO/SIMBA
- Balance positive and negative cases

❌ **DON'T:**
- Use only simple examples
- Ignore data quality
- Optimize on test set
- Mix different task types

### Optimization Strategy

1. **Start Small**: Use Bootstrap first
2. **Measure**: Establish baseline performance
3. **Iterate**: Try MIPRO for systematic improvement
4. **Specialize**: Use SIMBA/GEPA for complex needs
5. **Validate**: Test on held-out data

### Avoiding Overfitting

- Use train/validation split
- Don't over-optimize (stop when validation plateaus)
- Test on diverse examples
- Monitor generalization metrics

---

## CLI Tool - Zero Code Optimization

Try all optimizers without writing code:

```bash
# Build CLI
cd cmd/dspy-cli && go build

# Try Bootstrap
./dspy-cli try bootstrap --dataset gsm8k --max-examples 10

# Try MIPRO
./dspy-cli try mipro --dataset gsm8k --verbose

# Try GEPA
./dspy-cli try gepa --dataset hotpotqa --max-examples 20

# Compare optimizers
./dspy-cli compare --dataset gsm8k --optimizers bootstrap,mipro,gepa
```

---

## What's Next?

- **[Core Concepts](core-concepts/)** - Understand what optimizers improve
- **[Examples](https://github.com/XiaoConstantine/dspy-go/tree/main/examples)** - See optimizers in action
- **[Compatibility Testing](https://github.com/XiaoConstantine/dspy-go/tree/main/compatibility_test)** - Verify optimizer behavior

### Example Applications

- [MIPRO Example](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/others/mipro)
- [SIMBA Example](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/others/simba)
- [GEPA Example](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/others/gepa)
- [GSM8K with Bootstrap](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/gsm8k)
