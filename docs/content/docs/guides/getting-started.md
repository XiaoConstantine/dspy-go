---
title: "Getting Started"
description: "Install dspy-go and run your first module"
summary: "Build a first llm-go-backed dspy-go application or explore optimizers with the CLI"
date: 2025-10-13T00:00:00+00:00
lastmod: 2026-08-26T00:00:00+00:00
draft: false
weight: 100
toc: true
seo:
  title: "Getting Started with dspy-go"
  description: "Build your first dspy-go LLM application"
  canonical: ""
  noindex: false
---

# Getting Started with dspy-go

dspy-go provides composable modules, agents, and optimizers. Provider
generation is supplied by llm-go through the compatibility adapters in
`pkg/llms`.

## Programming Quick Start

### 1. Install dspy-go

```bash
go get github.com/XiaoConstantine/dspy-go
```

### 2. Set a Provider Credential

This example uses Gemini:

```bash
export GEMINI_API_KEY="your-api-key"
```

Other supported credential variables include `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, and `ANTHROPIC_OAUTH_TOKEN`. Selecting a model is still
explicit; dspy-go does not choose a provider based on whichever key happens to
be present.

### 3. Run a Prediction

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func main() {
    llm, err := llms.NewGeminiLLM("", core.ModelGoogleGeminiFlash)
    if err != nil {
        log.Fatal(err)
    }
    core.SetDefaultLLM(llm)

    signature := core.NewSignature(
        []core.InputField{
            {Field: core.NewTextField("sentence",
                core.WithDescription("Sentence to classify"))},
        },
        []core.OutputField{
            {Field: core.NewTextField("sentiment",
                core.WithDescription("Positive, Negative, or Neutral"))},
        },
    ).WithInstruction("Classify the sentiment of the sentence.")

    predictor := modules.NewPredict(signature).WithStructuredOutput()
    result, err := predictor.Process(context.Background(), map[string]any{
        "sentence": "dspy-go makes LLM workflows composable.",
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(result["sentiment"])
}
```

Save the program as `main.go`, then run:

```bash
go run main.go
```

## Choosing Another Provider

`llms.NewLLM` infers the registered provider from a known model ID:

```go
// Anthropic
llm, err := llms.NewLLM(apiKey, core.ModelAnthropicSonnet)

// OpenAI (the registry path uses llm-go's Responses API)
llm, err := llms.NewLLM(apiKey, core.ModelOpenAIGPT4o)

// Ollama through its OpenAI-compatible API
llm, err := llms.NewLLM("", core.ModelOllamaLlama3_1_8B)
```

Use `llms.NewOpenAICompatible` for LiteLLM, LocalAI, LM Studio, or another
compatible endpoint:

```go
llm, err := llms.NewOpenAICompatible(
    "local",
    core.ModelID("local-model"),
    "http://localhost:1234/v1",
)
```

See the [provider reference](../../reference/providers/) for dedicated
constructors, streaming, embeddings, and OpenAI Codex subscription access.

## Model Scope

The package default is a compatibility convenience:

```go
core.SetDefaultLLM(llm)
```

You can instead pin a model to one module:

```go
predictor.SetLLM(llm)
```

Or select defaults for one request:

```go
ctx := core.WithRuntime(context.Background(), &core.Runtime{
    DefaultLLM: llm,
})
result, err := predictor.Process(ctx, inputs)
```

Resolution order is module-local, request-local runtime, then package default.
Prefer the helper functions over mutating `core.GlobalConfig` directly.

## Per-Call Generation Settings

```go
result, err := predictor.Process(ctx, inputs,
    core.WithGenerateOptions(
        core.WithMaxTokens(512),
        core.WithTemperature(0.2),
    ),
)
```

These options apply to the call. They are not arguments to Gemini or Anthropic
constructors.

## CLI Quick Start

The CLI can list and run optimizers with built-in sample datasets:

```bash
cd cmd/dspy-cli
go build -o dspy-cli

export GEMINI_API_KEY="your-api-key"

./dspy-cli list
./dspy-cli recommend --use-case balanced
./dspy-cli try bootstrap --dataset gsm8k --max-examples 5
./dspy-cli try mipro --dataset gsm8k --max-examples 5 --verbose
```

Run `./dspy-cli --help` for the full current command list, or see the
[CLI reference](../../reference/cli/).

## Troubleshooting

### Provider creation reports a missing API key

Pass a key explicitly or set the variable used by the selected constructor:

- Gemini: `GEMINI_API_KEY`
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_OAUTH_TOKEN` or `ANTHROPIC_API_KEY`

Ollama and llama.cpp require a running OpenAI-compatible server rather than a
hosted-provider API key.

### A generation call times out

Use a context deadline appropriate for the operation:

```go
ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
defer cancel()

result, err := predictor.Process(ctx, inputs)
```

### Next steps

- **[Core Concepts →](../core-concepts/)**
- **[Optimizers →](../optimizers/)**
- **[Agents →](../agents/)**
- **[Multimodal →](../multimodal/)**
- **[Examples →](https://github.com/XiaoConstantine/dspy-go/tree/main/examples)**
