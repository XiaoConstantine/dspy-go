---
title: "API Reference"
description: "Reference links and current integration patterns for dspy-go"
summary: "Find package documentation and the current provider, configuration, and CLI references"
date: 2025-10-13T00:00:00+00:00
lastmod: 2026-08-26T00:00:00+00:00
draft: false
weight: 900
toc: true
sidebar:
  collapsed: false
seo:
  title: "API Reference - dspy-go"
  description: "API reference and integration patterns for dspy-go"
  canonical: ""
  noindex: false
---

The authoritative symbol-level API documentation is on
[pkg.go.dev](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go). The guides
in this section explain the configuration and integration boundaries around
those APIs.

## Reference Guides

| Guide | Description |
|---|---|
| **[Configuration Reference →](configuration/)** | Credentials, model selection, generation options, runtimes, and YAML settings |
| **[LLM Providers →](providers/)** | llm-go-backed provider constructors, streaming, embeddings, and Codex subscription access |
| **[CLI Reference →](cli/)** | `dspy-cli` commands and flags |

## Package Documentation

### Core Framework

- [`pkg/core`](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/core):
  signatures, modules, LLM capability interfaces, execution state, and runtime
  model resolution
- [`pkg/modules`](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/modules):
  Predict, ChainOfThought, ReAct, Refine, Parallel, and other DSPy modules
- [`pkg/optimizers`](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/optimizers):
  GEPA, MIPRO, SIMBA, BootstrapFewShot, COPRO, and optimizer utilities
- [`pkg/config`](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/config):
  typed file and environment configuration loading

### Models and Agents

- [`pkg/llms`](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/llms):
  adapters from llm-go generators to dspy-go's compatibility interfaces, plus
  dspy-go-owned embedding clients
- [`pkg/agents`](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/agents):
  provider-neutral execution loops, harnesses, typed events, and traces
- [`pkg/agents/native`](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/agents/native):
  provider-native tool-calling agents and sessions
- [`pkg/tools`](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/tools):
  tool contracts, registries, composition, and built-in toolsets

## Current LLM Boundary

Provider generation protocols and model metadata live in
[`llm-go`](https://github.com/XiaoConstantine/llm-go). `pkg/llms` preserves the
`core.LLM` integration used by existing dspy-go modules and agents. dspy-go
continues to own Gemini and OpenAI-compatible embeddings because embedding is
part of that compatibility contract.

New consumers should depend on the smallest core capability interface they
need, such as `core.TextGenerator`, `core.StreamGenerator`, `core.Embedder`, or
`core.ToolCallingChatLLM`, rather than requiring the full `core.LLM` interface
unnecessarily.

## Common Patterns

### Configure a Default Model

```go
llm, err := llms.NewLLM(apiKey, core.ModelGoogleGeminiFlash)
if err != nil {
    return err
}
core.SetDefaultLLM(llm)
```

### Configure One Module

```go
predictor := modules.NewPredict(signature)
predictor.SetLLM(llm)
```

### Configure One Request

```go
ctx := core.WithRuntime(context.Background(), &core.Runtime{
    DefaultLLM: llm,
})
result, err := predictor.Process(ctx, inputs)
```

### Configure One Generation Call

```go
response, err := llm.Generate(ctx, prompt,
    core.WithMaxTokens(1024),
    core.WithTemperature(0.2),
)
```

Use `context.WithTimeout` or `context.WithCancel` to control an operation's
lifetime. Streaming calls return a `core.StreamResponse`; consume
`ChunkChannel` and call `Cancel` if you stop before the stream completes.

## Model IDs and Capabilities

Use the exported `core.ModelID` constants instead of copying model strings when
possible. Model catalogs change more frequently than this documentation, so
consult the current `pkg/core` GoDoc and provider documentation for available
models and limits.

Capability support is model-specific. Check `llm.Capabilities()` at runtime
when your application requires streaming, JSON, tool calling, vision, audio,
or embeddings.

## Next Steps

- **[Browse all packages on pkg.go.dev →](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go)**
- **[Configuration Reference →](configuration/)**
- **[LLM Providers →](providers/)**
- **[Getting Started →](../guides/getting-started/)**
