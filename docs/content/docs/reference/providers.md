---
title: "LLM Providers"
description: "Current provider constructors, capabilities, streaming, and embeddings"
summary: "Configure llm-go-backed generation providers through dspy-go's compatibility layer"
date: 2025-10-13T00:00:00+00:00
lastmod: 2026-08-26T00:00:00+00:00
draft: false
weight: 930
toc: true
seo:
  title: "LLM Providers - dspy-go"
  description: "Configure generation and embedding providers in dspy-go"
  canonical: ""
  noindex: false
---

dspy-go uses [`llm-go`](https://github.com/XiaoConstantine/llm-go) for model
generation protocols and model metadata. `pkg/llms` adapts an llm-go
`Generator` to dspy-go's `core.LLM` interface so existing modules, agents, and
optimizers keep the same integration point.

## Recommended Setup

`llms.NewLLM` selects the registered provider from the model ID and returns a
cached `core.LLM`:

```go
import (
    "os"

    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
)

llm, err := llms.NewLLM(
    os.Getenv("GEMINI_API_KEY"),
    core.ModelGoogleGeminiFlash,
)
if err != nil {
    return err
}
core.SetDefaultLLM(llm)
```

You can also initialize the factory and use the core convenience function:

```go
llms.EnsureFactory()
if err := core.ConfigureDefaultLLM(apiKey, core.ModelOpenAIGPT4o); err != nil {
    return err
}
```

Modules resolve an LLM in this order: a module-local model set with `SetLLM`,
a request-local runtime supplied by `core.WithRuntime`, then the package
default set with `core.SetDefaultLLM`.

## Provider Constructors

Use a dedicated constructor when you need provider-specific endpoint or
credential behavior:

| Provider | Constructor | Notes |
|---|---|---|
| Anthropic | `NewAnthropicLLM(apiKey, anthropic.Model)` | Uses `ANTHROPIC_OAUTH_TOKEN` first, then `apiKey`, then `ANTHROPIC_API_KEY` |
| Google Gemini | `NewGeminiLLM(apiKey, modelID)` | Reads `GEMINI_API_KEY` when `apiKey` is empty |
| OpenAI-compatible chat | `NewOpenAILLM(modelID, options...)` | Chat Completions constructor; reads `OPENAI_API_KEY` if `WithAPIKey` is omitted |
| OpenAI API key | `NewOpenAI(modelID, apiKey)` | Chat Completions convenience wrapper; requires an explicit key |
| OpenAI Codex subscription | `NewOpenAICodexLLM(modelID, options...)` | Requires an application-owned credential resolver |
| Ollama | `NewOllamaLLM(modelID, options...)` | Requires Ollama's OpenAI-compatible API |
| llama.cpp | `NewLlamacppLLM(endpoint)` | Requires an OpenAI-compatible server |
| Generic compatible endpoint | `NewOpenAICompatible(provider, modelID, baseURL, options...)` | LiteLLM, LocalAI, FastChat, LM Studio, and similar servers |

The generic `NewLLM`/registry path uses llm-go's current native API selection.
In particular, official OpenAI models use the Responses API. The
`NewOpenAI` and `NewOpenAILLM` compatibility constructors continue to use Chat
Completions.

The exported model constants in `pkg/core` and llm-go's catalog are the source
of truth. Provider model availability, context limits, pricing, and rate limits
change independently of dspy-go, so verify those details in the provider's
current documentation.

### OpenAI-Compatible Endpoints

```go
llm, err := llms.NewOpenAICompatible(
    "litellm",
    core.ModelID("my-proxy-model"),
    "http://localhost:4000",
    llms.WithAPIKey(os.Getenv("LITELLM_API_KEY")),
    llms.WithOpenAITimeout(90*time.Second),
    llms.WithHeader("X-Tenant", "example"),
)
```

Available OpenAI-compatible options are `WithAPIKey`,
`WithOpenAIBaseURL`, `WithOpenAIPath`, `WithOpenAITimeout`, `WithHeader`,
and `WithHTTPClient`.

For Ollama, use `WithBaseURL`, `WithAuth`, `WithTimeout`, and
`WithOpenAIAPI`. `WithNativeAPI` returns an unsupported-operation error because
llm-go intentionally targets Ollama's OpenAI-compatible surface.

## Generation Options

Generation settings are call options in `pkg/core`; they are not provider
constructor options:

```go
response, err := llm.Generate(
    ctx,
    "Summarize this incident report.",
    core.WithMaxTokens(800),
    core.WithTemperature(0.2),
    core.WithTopP(0.9),
    core.WithPresencePenalty(0.1),
    core.WithFrequencyPenalty(0.1),
    core.WithStopSequences("END"),
)
```

Pass the same settings through a module with `core.WithGenerateOptions`:

```go
result, err := predictor.Process(
    ctx,
    inputs,
    core.WithGenerateOptions(
        core.WithMaxTokens(800),
        core.WithTemperature(0.2),
    ),
)
```

Support for an individual option remains model- and provider-dependent.

## Streaming

Streaming is explicit. Consume the returned channel and call `Cancel` when the
consumer stops early:

```go
stream, err := llm.StreamGenerate(ctx, "Explain the result.")
if err != nil {
    return err
}
defer stream.Cancel()

for chunk := range stream.ChunkChannel {
    if chunk.Error != nil {
        return chunk.Error
    }
    fmt.Print(chunk.Content)
}
```

Modules can stream with `core.WithStreamHandler`:

```go
result, err := predictor.Process(ctx, inputs,
    core.WithStreamHandler(func(chunk core.StreamChunk) error {
        fmt.Print(chunk.Content)
        return chunk.Error
    }),
)
```

## Embeddings

llm-go is the generation layer. Embeddings remain in dspy-go because they are
part of the compatibility `core.LLM` contract.

| LLM adapter | Embedding client retained by dspy-go | Default embedding model |
|---|---|---|
| Gemini | Google Gen AI SDK | `gemini-embedding-2` |
| OpenAI and OpenAI-compatible | OpenAI-compatible `/v1/embeddings` | `text-embedding-3-small` |
| Ollama and llama.cpp | OpenAI-compatible `/v1/embeddings` | Configured model ID |
| Anthropic and OpenAI Codex | Not supported | — |
| `llms.Adapt(generator)` | Not attached automatically | — |

```go
embedding, err := llm.CreateEmbedding(
    ctx,
    "text to embed",
    core.WithModel("gemini-embedding-2"),
)

batch, err := llm.CreateEmbeddings(
    ctx,
    []string{"first", "second"},
    core.WithBatchSize(32),
)
```

Calling an embedding method on an adapter without an embedding client returns
an unsupported-operation error.

## OpenAI Codex Subscription

ChatGPT subscription access is separate from OpenAI API-key access. Select the
`openai-codex` provider explicitly and send subscription traffic only to the
Codex backend.

```go
llm, err := llms.NewOpenAICodexLLM(
    core.ModelOpenAIGPT54,
    llms.WithOpenAICodexCredentials(
        func(ctx context.Context, rejectedToken string) (llms.OpenAICodexCredentials, error) {
            credential, err := credentialStore.Resolve(ctx, rejectedToken)
            if err != nil {
                return llms.OpenAICodexCredentials{}, err
            }
            return llms.OpenAICodexCredentials{
                AccessToken: credential.AccessToken,
                AccountID:   credential.AccountID,
            }, nil
        },
    ),
)
```

The resolver runs before each request. After a `401`, llm-go invokes it once
with the rejected token so the application can coordinate refresh safely. The
application owns OAuth login, refresh-token rotation, secure persistence, and
cross-process synchronization. dspy-go does not ship interactive OAuth or a
credential store.

For static registry/CLI use, select a model such as
`openai-codex:gpt-5.4` and provide the OAuth access token as the API key or
`OPENAI_OAUTH_TOKEN`. Account identity can come from `account_id`, an ID token,
or compatible access-token claims.

## Timeouts and Capabilities

Use context deadlines for operation-scoped timeouts:

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

response, err := llm.Generate(ctx, prompt)
```

Use `llm.Capabilities()` instead of assuming that every model offered by a
provider supports vision, tools, JSON, or streaming.

## Next Steps

- **[Configuration Reference →](../configuration/)**
- **[Getting Started →](../../guides/getting-started/)**
- **[Multimodal Guide →](../../guides/multimodal/)**
