---
title: "Introducing RLM: Recursive Language Models in DSPy-Go"
description: "Process near-infinite context with 40% token savings using Recursive Language Models"
summary: "Learn how RLM enables LLMs to programmatically explore large contexts through iterative REPL-based execution, achieving significant token efficiency gains."
date: 2025-01-07T10:00:00+00:00
lastmod: 2025-01-07T10:00:00+00:00
draft: false
weight: 50
categories: ["Features", "Advanced"]
tags: ["rlm", "modules", "token-efficiency", "large-context"]
contributors: ["Xiao Constantine"]
pinned: true
homepage: true
seo:
  title: "RLM: Recursive Language Models for Large Context Processing"
  description: "Deep dive into how DSPy-Go implements Recursive Language Models for efficient large context processing with 40% token savings"
  canonical: ""
  noindex: false
---

## The Context Window Problem

Large Language Models have revolutionized how we process text, but they face a fundamental constraint: **context window limits**. Even with models supporting 100K+ tokens, processing massive documents, codebases, or datasets often requires either:

1. **Truncation** — losing potentially critical information
2. **Chunking with summarization** — introducing lossy compression
3. **Retrieval-Augmented Generation (RAG)** — adding infrastructure complexity

What if there was a way to give LLMs the ability to *programmatically explore* context of any size, querying exactly what they need, when they need it?

Enter **Recursive Language Models (RLM)**.

## What is RLM?

RLM is a novel inference paradigm developed by researchers at MIT's OASYS Lab ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)). Instead of stuffing everything into a single prompt, RLM provides the LLM with a **REPL (Read-Eval-Print Loop) environment** where:

1. The full context is stored as a variable the LLM can query
2. The LLM writes code to explore and process the context
3. Sub-LLM calls can be made from within the REPL
4. Iteration continues until the LLM signals completion

Think of it as giving the LLM a programming environment where it can methodically work through large contexts instead of trying to process everything at once.

### The Core Pattern: Query() and FINAL()

RLM revolves around two key primitives:

```go
// The LLM can query sub-sections or make sub-LLM calls
result := Query("Summarize the key findings in section 3")

// When done, the LLM signals completion
FINAL(finalAnswer)
```

This simple pattern enables sophisticated exploration strategies:

- **Divide and conquer** — Process sections independently, then synthesize
- **Iterative refinement** — Build understanding progressively
- **Targeted extraction** — Query only relevant portions
- **Hierarchical analysis** — Drill down into specific areas

## RLM in DSPy-Go

We've implemented RLM as a first-class module in DSPy-Go, fully integrated with our module system, interceptors, and tracing infrastructure.

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "time"

    "github.com/XiaoConstantine/dspy-go/pkg/llms"
    "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
)

func main() {
    ctx := context.Background()

    // Initialize your LLM
    llm, _ := llms.NewAnthropicLLM(ctx, "claude-sonnet-4-20250514",
        llms.WithAnthropicAPIKey("your-api-key"))

    // Create RLM module with configuration
    rlmModule := rlm.NewFromLLM(llm,
        rlm.WithMaxIterations(30),
        rlm.WithTimeout(5*time.Minute),
        rlm.WithVerbose(true),
        rlm.WithTraceDir("./rlm_logs"),
    )

    // Process a large document
    largeDocument := loadDocument("research_paper.txt") // 50K+ tokens

    result, err := rlmModule.Process(ctx, map[string]any{
        "context": largeDocument,
        "query":   "What are the three main contributions of this paper?",
    })

    if err != nil {
        panic(err)
    }

    fmt.Printf("Answer: %s\n", result["response"])
    fmt.Printf("Iterations used: %d\n", result["iterations"])
    fmt.Printf("Total tokens: %d\n", result["total_tokens"])
}
```

### How It Works Under the Hood

The RLM module orchestrates an iterative loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                        RLM Execution Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Initialize                                                   │
│     └── Load context into REPL environment                       │
│     └── Set up Query() and FINAL() functions                     │
│                                                                  │
│  2. Iteration Loop (max 30 iterations)                           │
│     ┌──────────────────────────────────────────────────┐        │
│     │  a) LLM receives:                                 │        │
│     │     • System prompt with RLM instructions         │        │
│     │     • Current query/task                          │        │
│     │     • Iteration history                           │        │
│     │     • Available variables in REPL                 │        │
│     │                                                   │        │
│     │  b) LLM responds with:                            │        │
│     │     • Reasoning (exploration strategy)            │        │
│     │     • Code block (Go code to execute)             │        │
│     │     • Action (explore/analyze/final)              │        │
│     │                                                   │        │
│     │  c) System executes code in sandboxed REPL        │        │
│     │     • Captures stdout/stderr                      │        │
│     │     • Processes any Query() sub-calls             │        │
│     │     • Updates variable state                      │        │
│     │                                                   │        │
│     │  d) Check completion:                             │        │
│     │     • FINAL() called? → Return answer             │        │
│     │     • Otherwise → Continue loop                   │        │
│     └──────────────────────────────────────────────────┘        │
│                                                                  │
│  3. Return Result                                                │
│     └── Final answer + metadata (tokens, iterations, time)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The REPL Environment

DSPy-Go's RLM uses [Yaegi](https://github.com/traefik/yaegi), a Go interpreter, to execute code within a sandboxed environment:

```go
// Available in the REPL environment:

// The full context as a variable
context string  // Your large document/data

// Query function for sub-LLM calls
Query(prompt string) string

// Batch queries for efficiency
QueryBatched(prompts []string) []string

// Signal completion
FINAL(answer interface{})
FINAL_VAR(variableName string)  // Return a variable's value
```

**Security considerations:**
- Restricted imports (no `os`, `sys`, `subprocess`, etc.)
- Sandboxed execution environment
- Configurable timeouts
- Size limits on outputs

## Token Efficiency: The 40% Advantage

RLM achieves approximately **40% token savings** compared to naive approaches. Here's how:

### Traditional Approach
```
Prompt = Full Document (50K tokens) + Question
        ───────────────────────────────────────
        Every query sends the entire document
```

### RLM Approach
```
Iteration 1: Question + "What sections exist?"     (~500 tokens)
Iteration 2: Question + "Read section 3"           (~2K tokens)
Iteration 3: Question + "Analyze finding X"        (~1K tokens)
Iteration 4: FINAL(synthesized answer)             (~500 tokens)
            ───────────────────────────────────────
            Total: ~4K tokens vs 50K+ traditional
```

The savings come from:

1. **Programmatic chunking** — Only load what's needed
2. **Incremental exploration** — Build understanding progressively
3. **Sub-call optimization** — Batch related queries
4. **History compression** — Summarize previous iterations

### Token Tracking

DSPy-Go tracks token usage across the entire RLM execution:

```go
result, _ := rlmModule.Process(ctx, inputs)

usage := result["usage"].(rlm.TokenUsage)
fmt.Printf("Root LLM tokens: %d\n", usage.RootTokens)
fmt.Printf("Sub-call tokens: %d\n", usage.SubCallTokens)
fmt.Printf("Total tokens: %d\n", usage.TotalTokens)
fmt.Printf("Iterations: %d\n", result["iterations"])
```

## Advanced Configuration

### Custom Iteration Limits

```go
// For complex analysis tasks
rlm.NewFromLLM(llm,
    rlm.WithMaxIterations(50),  // Allow more exploration
    rlm.WithTimeout(10*time.Minute),
)

// For quick extraction tasks
rlm.NewFromLLM(llm,
    rlm.WithMaxIterations(5),   // Faster completion
    rlm.WithTimeout(1*time.Minute),
)
```

### Execution Tracing

Enable detailed tracing for debugging and analysis:

```go
rlm.NewFromLLM(llm,
    rlm.WithVerbose(true),
    rlm.WithTraceDir("./rlm_traces"),
)
```

This produces JSONL trace files containing:
- Each iteration's LLM response
- Code executed and outputs
- Sub-LLM call details
- Token usage per step
- Timing information

### Integration with DSPy-Go Modules

RLM works seamlessly with other DSPy-Go components:

```go
// Use with interceptors
rlmModule := rlm.NewFromLLM(llm)
rlmModule.SetInterceptors([]core.ModuleInterceptor{
    interceptors.LoggingModuleInterceptor(),
    interceptors.TracingModuleInterceptor(),
})

// Compose with other modules
program := core.NewProgram(
    map[string]core.Module{
        "extract":   rlmModule,
        "summarize": modules.NewPredict(summarySignature),
    },
    func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
        // RLM extracts key points
        extracted, _ := program.Modules["extract"].Process(ctx, inputs)

        // Then summarize
        return program.Modules["summarize"].Process(ctx, map[string]any{
            "content": extracted["response"],
        })
    },
)
```

## Use Cases

### 1. Codebase Analysis

```go
result, _ := rlmModule.Process(ctx, map[string]any{
    "context": entireCodebase,  // Thousands of files
    "query":   "Find all security vulnerabilities related to SQL injection",
})
```

The LLM can:
- List files and directories
- Read specific files
- Search for patterns
- Analyze dependencies
- Synthesize findings

### 2. Research Paper Analysis

```go
result, _ := rlmModule.Process(ctx, map[string]any{
    "context": researchPaper,
    "query":   "Compare the methodology with Smith et al. (2023) and identify gaps",
})
```

### 3. Log Analysis

```go
result, _ := rlmModule.Process(ctx, map[string]any{
    "context": serverLogs,  // Gigabytes of logs
    "query":   "Find the root cause of the outage on January 5th",
})
```

### 4. Document Q&A

```go
result, _ := rlmModule.Process(ctx, map[string]any{
    "context": legalContract,
    "query":   "What are all the termination clauses and their conditions?",
})
```

## Best Practices

### 1. Clear, Specific Queries

```go
// Good: Specific and actionable
"query": "List all API endpoints that don't have authentication middleware"

// Less effective: Vague
"query": "Tell me about the API"
```

### 2. Appropriate Iteration Limits

| Task Type | Recommended Max Iterations |
|-----------|---------------------------|
| Simple extraction | 5-10 |
| Analysis | 15-25 |
| Complex synthesis | 30-50 |

### 3. Monitor Token Usage

```go
if usage.TotalTokens > expectedBudget {
    log.Warn("Token usage exceeded expectations",
        "expected", expectedBudget,
        "actual", usage.TotalTokens)
}
```

### 4. Use Tracing for Debugging

When results are unexpected, enable tracing to understand the LLM's exploration path:

```go
rlm.NewFromLLM(llm,
    rlm.WithVerbose(true),
    rlm.WithTraceDir("./debug_traces"),
)
```

## Architecture Deep Dive

### Module Structure

```
pkg/modules/rlm/
├── rlm.go           # Core RLM module implementation
├── config.go        # Configuration options
├── repl_yaegi.go    # Yaegi-based Go REPL
├── signature.go     # DSPy signature definitions
├── token_tracker.go # Token usage tracking
└── rlm_test.go      # Comprehensive tests
```

### Key Components

**RLM Module** — Orchestrates the iteration loop, manages state, handles completion detection.

**SubLLMClient** — Interface for making sub-LLM calls from within the REPL:

```go
type SubLLMClient interface {
    Query(ctx context.Context, prompt string) (string, error)
    QueryBatched(ctx context.Context, prompts []string) ([]string, error)
}
```

**TokenTracker** — Accumulates token usage across all calls:

```go
type TokenTracker struct {
    RootInputTokens  int
    RootOutputTokens int
    SubInputTokens   int
    SubOutputTokens  int
}
```

**YaegiREPL** — Sandboxed Go interpreter for code execution with security restrictions.

## Comparison with Alternatives

| Approach | Context Limit | Token Efficiency | Accuracy | Complexity |
|----------|--------------|------------------|----------|------------|
| Full Context | Model limit | Low | High | Low |
| Chunking | Unlimited | Medium | Medium | Medium |
| RAG | Unlimited | High | Variable | High |
| **RLM** | Unlimited | High | High | Medium |

RLM combines the accuracy advantages of full-context processing with the efficiency of selective retrieval, without requiring external vector databases or retrieval infrastructure.

## Getting Started

1. **Install DSPy-Go:**
   ```bash
   go get github.com/XiaoConstantine/dspy-go
   ```

2. **Import the RLM module:**
   ```go
   import "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
   ```

3. **Create and use:**
   ```go
   rlmModule := rlm.NewFromLLM(yourLLM)
   result, _ := rlmModule.Process(ctx, map[string]any{
       "context": yourLargeDocument,
       "query":   "Your question here",
   })
   ```

## What's Next

We're actively developing RLM support in DSPy-Go with plans for:

- **Nested RLM calls** — Recursive sub-RLM invocations for hierarchical analysis
- **Custom REPL environments** — Support for Python and other languages
- **Streaming results** — Real-time iteration updates
- **Caching layer** — Avoid redundant sub-calls across queries
- **Visual debugger** — Web UI for exploring RLM execution traces

## Conclusion

RLM represents a paradigm shift in how we approach large context processing. By giving LLMs the ability to programmatically explore data through iterative REPL execution, we unlock:

- **Near-infinite context handling** without model limits
- **40% token savings** through intelligent exploration
- **Higher accuracy** by focusing on relevant information
- **Transparent reasoning** via execution traces

The integration in DSPy-Go makes it easy to add RLM capabilities to your existing workflows while maintaining compatibility with our module, interceptor, and tracing systems.

Try RLM today and experience a new way of working with large contexts.

---

*For questions and feedback, open an issue on [GitHub](https://github.com/XiaoConstantine/dspy-go) or join our community discussions.*

## References

- [Recursive Language Models: A New Inference Paradigm](https://arxiv.org/abs/2512.24601) — Zhang, Kraska, Khattab (MIT OASYS Lab)
- [DSPy-Go Documentation](/docs/)
- [RLM Example Code](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/rlm)
