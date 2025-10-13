---
title: "Core Concepts"
description: "Understanding the building blocks of dspy-go"
summary: "Learn about Signatures, Modules, and Programs - the foundation of every dspy-go application"
date: 2025-10-13T00:00:00+00:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 200
toc: true
seo:
  title: "Core Concepts - dspy-go"
  description: "Master Signatures, Modules, and Programs in dspy-go for building robust LLM applications"
  canonical: ""
  noindex: false
---

# Core Concepts

dspy-go is built around three fundamental concepts that work together to create powerful LLM applications: **Signatures**, **Modules**, and **Programs**. Understanding these building blocks will help you build reliable, maintainable AI systems.

## Signatures

**Signatures** define the contract between your application and the LLM. They specify what inputs the LLM needs and what outputs it should produce.

### Why Signatures Matter

Instead of crafting prompts manually, signatures let you:
- **Define clear expectations** - What goes in, what comes out
- **Type safety** - Strong typing ensures correctness
- **Reusability** - Use the same signature across different modules
- **Optimization** - Optimizers can improve signatures automatically

### Creating a Signature

There are two ways to create signatures in dspy-go:

#### Method 1: Struct-based Signatures (Type-Safe)

Perfect for production code with compile-time safety:

```go
type QuestionAnswerSignature struct {
    core.Signature
}

func (s QuestionAnswerSignature) Inputs() []core.InputField {
    return []core.InputField{
        {Field: core.NewField("question",
            core.WithDescription("The question to answer"))},
        {Field: core.NewField("context",
            core.WithDescription("Background information"))},
    }
}

func (s QuestionAnswerSignature) Outputs() []core.OutputField {
    return []core.OutputField{
        {Field: core.NewField("answer",
            core.WithDescription("A concise, accurate answer"))},
        {Field: core.NewField("confidence",
            core.WithDescription("Confidence level: high/medium/low"))},
    }
}

func (s QuestionAnswerSignature) Instruction() string {
    return "You are a helpful assistant. Answer the question accurately using the provided context."
}
```

#### Method 2: Functional Signatures (Quick & Flexible)

Perfect for prototyping and simple use cases:

```go
signature := core.NewSignature(
    []core.InputField{
        {Field: core.NewField("question")},
    },
    []core.OutputField{
        {Field: core.NewField("answer")},
    },
)
```

### Field Descriptions Matter

**Important**: Field descriptions aren't just documentation—they directly influence prompt quality and LLM behavior.

```go
// ❌ Weak description
{Field: core.NewField("sentiment")}

// ✅ Strong description
{Field: core.NewField("sentiment",
    core.WithDescription("The emotional tone: Positive, Negative, or Neutral"))}

// ✅ Even better - guides the LLM's reasoning
{Field: core.NewField("sentiment",
    core.WithDescription("Analyze the emotional tone. Consider word choice, context, and intensity. Return: Positive, Negative, or Neutral"))}
```

---

## Modules

**Modules** are the execution engines that take your signatures and turn them into working LLM applications. They encapsulate different reasoning patterns and behaviors.

### Core Modules

#### 1. Predict - Direct Prediction

The simplest module. Makes a single prediction based on your signature.

```go
predictor := modules.NewPredict(QuestionAnswerSignature{})

result, err := predictor.Process(ctx, map[string]interface{}{
    "question": "What is the capital of France?",
    "context": "France is a country in Western Europe.",
})

// result["answer"] = "Paris"
// result["confidence"] = "high"
```

**When to use**: Simple, single-step tasks where you need direct answers.

#### 2. ChainOfThought - Step-by-Step Reasoning

Implements chain-of-thought reasoning, breaking down complex problems into steps.

```go
cot := modules.NewChainOfThought(signature)

result, err := cot.Process(ctx, map[string]interface{}{
    "question": "If a train travels 120 miles in 2 hours, how far will it go in 5 hours?",
})

// result includes both:
// - "rationale": "Speed = 120/2 = 60 mph. Distance = 60 * 5 = 300 miles"
// - "answer": "300 miles"
```

**When to use**: Math problems, logical reasoning, multi-step analysis.

**Key insight**: ChainOfThought automatically adds a "rationale" field to guide the LLM's thinking process.

#### 3. ReAct - Reasoning + Acting

Combines reasoning with tool use. The LLM can call tools to gather information before answering.

```go
// Create tools
calculator := tools.NewCalculatorTool()
searchTool := tools.NewSearchTool()

// Create registry
registry := tools.NewInMemoryToolRegistry()
registry.Register(calculator)
registry.Register(searchTool)

// Create ReAct module
react := modules.NewReAct(signature, registry, 5) // max 5 iterations

result, err := react.Process(ctx, map[string]interface{}{
    "question": "What is the population of Tokyo divided by 1000?",
})

// ReAct will:
// 1. Reason: "I need to find Tokyo's population"
// 2. Act: Call search tool
// 3. Reason: "Now I need to divide by 1000"
// 4. Act: Call calculator
// 5. Answer: "14,000 (approximately)"
```

**When to use**: Questions requiring external data, calculations, or API calls.

#### 4. MultiChainComparison - Multi-Perspective Analysis

Compares multiple reasoning attempts and synthesizes a comprehensive answer.

```go
multiChain := modules.NewMultiChainComparison(signature, 3, 0.7)

completions := []map[string]interface{}{
    {"rationale": "Cost-focused approach...", "solution": "Reduce expenses"},
    {"rationale": "Growth-focused approach...", "solution": "Invest in marketing"},
    {"rationale": "Balanced approach...", "solution": "Optimize both"},
}

result, err := multiChain.Process(ctx, map[string]interface{}{
    "problem": "How should we improve business performance?",
    "completions": completions,
})

// result contains synthesized recommendation considering all perspectives
```

**When to use**: Complex decisions requiring multiple viewpoints.

#### 5. Refine - Quality Improvement

Runs multiple attempts with different parameters and selects the best result.

```go
rewardFn := func(inputs, outputs map[string]interface{}) float64 {
    // Score based on answer length, completeness, etc.
    answer := outputs["answer"].(string)
    return calculateQualityScore(answer)
}

refine := modules.NewRefine(
    modules.NewPredict(signature),
    modules.RefineConfig{
        N:         5,          // 5 attempts
        RewardFn:  rewardFn,
        Threshold: 0.8,        // Stop if quality > 0.8
    },
)

result, err := refine.Process(ctx, inputs)
// Returns the highest-quality result
```

**When to use**: When quality is critical and you want the best possible answer.

#### 6. Parallel - Batch Processing

Wraps any module for concurrent execution across multiple inputs.

```go
baseModule := modules.NewPredict(signature)

parallel := modules.NewParallel(baseModule,
    modules.WithMaxWorkers(4),              // 4 concurrent workers
    modules.WithReturnFailures(false),      // Skip failures
)

batchInputs := []map[string]interface{}{
    {"question": "What is 2+2?"},
    {"question": "What is the capital of France?"},
    {"question": "What is the speed of light?"},
}

result, err := parallel.Process(ctx, map[string]interface{}{
    "batch_inputs": batchInputs,
})

// Process all inputs concurrently
results := result["results"].([]map[string]interface{})
```

**When to use**: Batch processing, bulk operations, performance optimization.

### Module Composition

**Modules can be composed** to create sophisticated workflows:

```go
// Combine ChainOfThought with Refine for high-quality reasoning
cotModule := modules.NewChainOfThought(signature)
refinedCot := modules.NewRefine(cotModule, refineConfig)

// Wrap with Parallel for batch high-quality reasoning
parallelRefinedCot := modules.NewParallel(refinedCot, parallelConfig)
```

---

## Programs

**Programs** orchestrate multiple modules into complete workflows. They define how data flows through your system.

### Creating a Program

```go
program := core.NewProgram(
    map[string]core.Module{
        "retriever": retrieverModule,
        "generator": generatorModule,
    },
    func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
        // Step 1: Retrieve relevant documents
        retrieverResult, err := retrieverModule.Process(ctx, inputs)
        if err != nil {
            return nil, err
        }

        // Step 2: Generate answer using retrieved documents
        generatorInputs := map[string]interface{}{
            "question": inputs["question"],
            "documents": retrieverResult["documents"],
        }

        return generatorModule.Process(ctx, generatorInputs)
    },
)

// Execute the program
result, err := program.Execute(ctx, map[string]interface{}{
    "question": "What are the benefits of Go for LLM applications?",
})
```

### RAG (Retrieval-Augmented Generation) Example

A complete RAG pipeline demonstrating program composition:

```go
// Define signatures
retrievalSig := core.NewSignature(
    []core.InputField{
        {Field: core.NewField("query")},
    },
    []core.OutputField{
        {Field: core.NewField("documents")},
    },
)

generationSig := core.NewSignature(
    []core.InputField{
        {Field: core.NewField("question")},
        {Field: core.NewField("documents")},
    },
    []core.OutputField{
        {Field: core.NewField("answer")},
    },
)

// Create modules
retriever := modules.NewPredict(retrievalSig)
generator := modules.NewChainOfThought(generationSig)

// Compose into RAG program
ragProgram := core.NewProgram(
    map[string]core.Module{
        "retriever": retriever,
        "generator": generator,
    },
    func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
        // Retrieval phase
        docs, err := retriever.Process(ctx, map[string]interface{}{
            "query": inputs["question"],
        })
        if err != nil {
            return nil, err
        }

        // Generation phase
        return generator.Process(ctx, map[string]interface{}{
            "question": inputs["question"],
            "documents": docs["documents"],
        })
    },
)

// Use the program
answer, err := ragProgram.Execute(ctx, map[string]interface{}{
    "question": "How does dspy-go handle tool management?",
})
```

### Program Optimization

**Key feature**: Programs can be optimized automatically:

```go
// Create MIPRO optimizer
optimizer := optimizers.NewMIPRO(
    metricFunc,
    optimizers.WithNumTrials(10),
)

// Optimize the entire program
optimizedProgram, err := optimizer.Compile(ctx, ragProgram, dataset, nil)

// optimizedProgram now has improved prompts and parameters
```

---

## Putting It All Together

Here's a complete example showing Signatures, Modules, and Programs working together:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// 1. Define Signature
type AnalysisSignature struct {
    core.Signature
}

func (s AnalysisSignature) Inputs() []core.InputField {
    return []core.InputField{
        {Field: core.NewField("text",
            core.WithDescription("The text to analyze"))},
    }
}

func (s AnalysisSignature) Outputs() []core.OutputField {
    return []core.OutputField{
        {Field: core.NewField("summary",
            core.WithDescription("A concise summary"))},
        {Field: core.NewField("sentiment",
            core.WithDescription("Overall sentiment: Positive/Negative/Neutral"))},
        {Field: core.NewField("key_points",
            core.WithDescription("Main takeaways as bullet points"))},
    }
}

func (s AnalysisSignature) Instruction() string {
    return "Analyze the provided text thoroughly and extract key insights."
}

func main() {
    // 2. Configure LLM
    llm, _ := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
    core.SetDefaultLLM(llm)

    // 3. Create Modules
    analyzer := modules.NewChainOfThought(AnalysisSignature{})

    // 4. Create Program
    program := core.NewProgram(
        map[string]core.Module{"analyzer": analyzer},
        func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
            return analyzer.Process(ctx, inputs)
        },
    )

    // 5. Execute
    result, err := program.Execute(context.Background(), map[string]interface{}{
        "text": "dspy-go provides a systematic approach to building LLM applications...",
    })
    if err != nil {
        log.Fatal(err)
    }

    // 6. Use Results
    fmt.Printf("Summary: %s\n", result["summary"])
    fmt.Printf("Sentiment: %s\n", result["sentiment"])
    fmt.Printf("Key Points: %s\n", result["key_points"])
}
```

---

## Best Practices

### Signature Design

✅ **DO:**
- Write detailed, specific field descriptions
- Include examples in descriptions when helpful
- Use meaningful field names
- Specify expected output formats

❌ **DON'T:**
- Leave descriptions empty
- Use vague instructions
- Change signatures frequently (breaks optimization)
- Over-complicate with too many fields

### Module Selection

| Use Case | Recommended Module |
|----------|-------------------|
| Simple Q&A | `Predict` |
| Math/Logic | `ChainOfThought` |
| Tool Use | `ReAct` |
| Quality-Critical | `Refine` |
| Batch Processing | `Parallel` |
| Complex Decisions | `MultiChainComparison` |

### Program Structure

✅ **DO:**
- Keep workflows simple and linear when possible
- Handle errors at each step
- Log intermediate results for debugging
- Use context for cancellation and timeouts

❌ **DON'T:**
- Create circular dependencies
- Ignore error handling
- Make programs too deeply nested
- Forget to pass context through

---

## What's Next?

Now that you understand the core concepts, explore:

- **[Optimizers →](optimizers/)** - Automatically improve your signatures and programs
- **[Tool Management →](tools/)** - Extend ReAct with smart tool selection
- **[Examples](https://github.com/XiaoConstantine/dspy-go/tree/main/examples)** - See these concepts in real applications

### Deep Dives

- **Signatures**: [Field Options](../reference/signatures/), [Advanced Patterns](../reference/signatures/#advanced)
- **Modules**: [Custom Modules](../reference/modules/), [Module Configuration](../reference/modules/#config)
- **Programs**: [Workflows](../reference/programs/), [Error Handling](../reference/programs/#errors)
