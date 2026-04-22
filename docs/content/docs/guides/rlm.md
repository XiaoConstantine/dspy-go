---
title: "RLM Module"
description: "Recursive Language Model workflows for large-context exploration"
summary: "Use RLM for iterative long-context reasoning, structured replay policies, and budgeted sub-RLM delegation"
date: 2026-04-16T00:00:00+00:00
lastmod: 2026-04-16T00:00:00+00:00
draft: false
weight: 450
toc: true
seo:
  title: "RLM Module - dspy-go"
  description: "Guide to Recursive Language Models in dspy-go, including context-policy presets, checkpointed replay, and sub-RLM budgets"
  canonical: ""
  noindex: false
---

# RLM Module

The **Recursive Language Model** (`pkg/modules/rlm`) is dspy-go's long-context execution loop. Instead of asking the model to answer from one giant prompt, `RLM` gives it a Go REPL, lets it inspect the context iteratively, call sub-LLMs, and finish only when it has enough evidence.

Use it when you need:

- iterative reasoning over large context windows
- code-assisted extraction, counting, filtering, or aggregation
- recursive decomposition with nested sub-RLM calls
- better control over replay cost than raw "append the whole transcript" prompting

## Basic Usage

```go
package main

import (
    "context"
    "time"

    "github.com/XiaoConstantine/dspy-go/pkg/llms"
    "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
)

func main() {
    llm, _ := llms.NewGeminiLLM("", "gemini-2.0-flash")

    module := rlm.NewFromLLM(
        llm,
        rlm.WithMaxIterations(8),
        rlm.WithTimeout(3*time.Minute),
    )

    result, _ := module.Complete(
        context.Background(),
        largeDocument,
        "What percentage of reviews are positive versus negative?",
    )

    println(result.Response)
}
```

The `examples/rlm` and `examples/rlm_context_policy` directories show live end-to-end usage.

## Structured Replay Policies

Every RLM iteration needs some representation of prior work. dspy-go now supports three replay presets:

| Policy | Behavior | Best For |
|---|---|---|
| `full` | replay every prior entry verbatim | shortest runs, debugging exact step history |
| `checkpointed` | summarize older entries, keep recent ones verbatim | long runs where prompt growth matters |
| `adaptive` | start full, switch to checkpointed once history grows | sensible default when run length varies |

### Full Replay

```go
module := rlm.NewFromLLM(
    llm,
    rlm.WithContextPolicyPreset(rlm.ContextPolicyFull),
)
```

This keeps the raw step history intact. It is the easiest mode to reason about, but it grows prompt size linearly with each iteration.

### Checkpointed Replay

```go
module := rlm.NewFromLLM(
    llm,
    rlm.WithContextPolicyPreset(rlm.ContextPolicyCheckpointed),
    rlm.WithHistoryCompression(2, 300),
)
```

`WithHistoryCompression(verbatimIterations, maxSummaryTokens)` controls how much recent history stays verbatim and how much budget is allowed for the checkpoint summary.

### Adaptive Replay

```go
module := rlm.NewFromLLM(
    llm,
    rlm.WithContextPolicyPreset(rlm.ContextPolicyAdaptive),
    rlm.WithAdaptiveCheckpointThreshold(5),
)
```

`adaptive` stays on full replay until the number of history entries reaches the configured threshold, then it switches to checkpointed replay. If you do not set a threshold, the default is `5`.

## Inspecting Replay Behavior

If you wrap the module in `pkg/agents/rlm`, the execution trace exposes the replay metadata directly:

```go
trace := agent.LastExecutionTrace()
fmt.Println(trace.ContextMetadata["context_policy_preset"])
fmt.Println(trace.ContextMetadata["history_compressions"])
fmt.Println(trace.ContextMetadata["root_prompt_mean_tokens"])
fmt.Println(trace.ContextMetadata["root_prompt_max_tokens"])
```

The `examples/rlm_context_policy` example prints exactly these fields for `full`, `checkpointed`, and `adaptive` runs on the same task.

## Adaptive Iteration

RLM can also scale the iteration budget with context size:

```go
module := rlm.NewFromLLM(
    llm,
    rlm.WithAdaptiveIterationConfig(rlm.AdaptiveIterationConfig{
        Enabled:                true,
        BaseIterations:         3,
        MaxIterations:          12,
        ContextScaleFactor:     1 << 19,
        EnableEarlyTermination: true,
        ConfidenceThreshold:    2,
    }),
)
```

This is useful when short tasks should complete quickly, but long tasks need room to explore.

## Sub-RLM Budgets

Nested sub-RLM calls are powerful, but they can fan out aggressively. Use `WithSubRLMConfig` to bound that tree:

```go
module := rlm.NewFromLLM(
    llm,
    rlm.WithSubRLMConfig(rlm.SubRLMConfig{
        MaxDepth:               3,
        MaxIterationsPerSubRLM: 6,
        MaxDirectSubRLMCalls:   2,
        MaxTotalSubRLMCalls:    8,
    }),
)
```

The budgets mean:

- `MaxDepth`: maximum nesting depth
- `MaxIterationsPerSubRLM`: cap per child RLM
- `MaxDirectSubRLMCalls`: how many direct child sub-RLM calls one node may spawn
- `MaxTotalSubRLMCalls`: total sub-RLM calls across the whole request tree

Budget enforcement happens before depth escalation, so direct/total limits fail fast even when further nesting is still technically possible.

## Tracing And Debugging

Useful options when you are tuning long-context behavior:

```go
module := rlm.NewFromLLM(
    llm,
    rlm.WithVerbose(true),
    rlm.WithTraceDir("./rlm_logs"),
    rlm.WithOutputTruncationConfig(rlm.OutputTruncationConfig{
        Enabled:          true,
        MaxOutputLen:     4000,
        MaxVarPreviewLen: 120,
        MaxHistoryEntryLen: 800,
    }),
)
```

- `WithVerbose(true)` prints iteration progress
- `WithTraceDir(...)` writes JSONL traces
- output truncation keeps execution logs readable without exploding prompt replay size

## Related Examples

- `examples/rlm`: basic live RLM run
- `examples/rlm_context_policy`: compare `full`, `checkpointed`, and `adaptive`
- `examples/rlm_subrlm_budgets`: deterministic direct/total delegation budget demo
- `examples/rlm_oolong_gepa`: optimize an RLM agent, save the optimized program, restore it, and replay it
