---
title: "Configuration Reference"
description: "Configure LLMs, generation settings, runtimes, and YAML application settings"
summary: "Current environment, programmatic, and file-based configuration patterns"
date: 2025-10-13T00:00:00+00:00
lastmod: 2026-08-26T00:00:00+00:00
draft: false
weight: 910
toc: true
seo:
  title: "Configuration Reference - dspy-go"
  description: "Current configuration reference for dspy-go"
  canonical: ""
  noindex: false
---

dspy-go separates model construction, request settings, and application config:

- `pkg/llms` constructs `core.LLM` adapters backed by llm-go generators.
- `core.GenerateOption` values configure one generation call.
- `core.Runtime` selects request-scoped default and teacher models.
- `pkg/config` loads typed application settings. It does not automatically
  construct a provider or apply generation settings.

## Provider Credentials

Credential resolution depends on the constructor:

| Path | Credential behavior |
|---|---|
| Gemini through `NewLLM` or `NewGeminiLLM` | Uses the explicit key, then `GEMINI_API_KEY` |
| OpenAI through `NewLLM` or `NewOpenAILLM` | Uses the explicit key or `WithAPIKey`, then `OPENAI_API_KEY` |
| Anthropic through `NewLLM` or `NewAnthropicLLM` | Uses `ANTHROPIC_OAUTH_TOKEN` first, then the explicit key, then `ANTHROPIC_API_KEY` |
| OpenAI Codex through `NewLLM` | Uses the explicit OAuth access token, then `OPENAI_OAUTH_TOKEN` |

`NewOpenAI` requires a non-empty explicit API key. Direct
`NewOpenAICodexLLM` construction requires an application-owned credential
resolver and does not read the environment.

The model is always selected explicitly; setting an unrelated provider key
does not cause automatic provider selection.

```go
llm, err := llms.NewGeminiLLM("", core.ModelGoogleGeminiFlash)
if err != nil {
    return err
}
core.SetDefaultLLM(llm)
```

Pass secrets directly when your application uses a secret manager:

```go
llm, err := llms.NewLLM(secret.Value, core.ModelOpenAIGPT4o)
```

See the [provider reference](../providers/) for OpenAI-compatible endpoints,
Ollama, llama.cpp, and application-owned Codex OAuth credentials.

## Model Selection

Use exported `core.ModelID` constants when one is available:

```go
llm, err := llms.NewLLM(apiKey, core.ModelGoogleGeminiFlash)
```

Use `core.ModelID` for a model exposed by a custom compatible server:

```go
llm, err := llms.NewOpenAICompatible(
    "local",
    core.ModelID("my-local-model"),
    "http://localhost:1234/v1",
)
```

## Package, Request, and Module Scope

Set a package-wide compatibility default with `core.SetDefaultLLM`:

```go
core.SetDefaultLLM(llm)
```

Override it for one request without mutating global state:

```go
ctx := core.WithRuntime(context.Background(), &core.Runtime{
    DefaultLLM: requestLLM,
    TeacherLLM: teacherLLM,
})

result, err := predictor.Process(ctx, inputs)
```

Pin a model to one module with `SetLLM`:

```go
predictor.SetLLM(moduleLLM)
result, err := predictor.Process(ctx, inputs)
```

Resolution order is module-local, request-local runtime, then package default.

## Generation Settings

Generation options are passed to `Generate`:

```go
response, err := llm.Generate(
    ctx,
    prompt,
    core.WithMaxTokens(2048),
    core.WithTemperature(0.7),
    core.WithTopP(0.9),
    core.WithPresencePenalty(0.1),
    core.WithFrequencyPenalty(0.1),
    core.WithStopSequences("END"),
)
```

Pass them through a module with `core.WithGenerateOptions`:

```go
result, err := predictor.Process(ctx, inputs,
    core.WithGenerateOptions(
        core.WithMaxTokens(2048),
        core.WithTemperature(0.7),
        core.WithTopP(0.9),
    ),
)
```

`core.GenerateOptions` contains `MaxTokens`, `Temperature`, `TopP`,
`PresencePenalty`, `FrequencyPenalty`, and `Stop`. Streaming is a separate API,
not a field in this struct.

## Timeouts and Cancellation

Use a context deadline for an operation-scoped timeout:

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

result, err := predictor.Process(ctx, inputs)
```

OpenAI-compatible constructors also accept `llms.WithOpenAITimeout` for their
HTTP client. Ollama accepts a timeout in seconds through `llms.WithTimeout`.

dspy-go does not expose mutable retry setters on `core.LLM`. If an application
needs retries, keep them bounded and operation-aware outside the adapter, or
wrap the underlying llm-go generator before calling `llms.Adapt`.

## Typed YAML Configuration

`pkg/config` loads application settings from files and `DSPY_`-prefixed
environment overrides. Its LLM shape is:

```yaml
llm:
  default:
    provider: google
    model_id: gemini-2.5-flash
    generation:
      max_tokens: 8192
      temperature: 0.7
      top_p: 0.9
    embedding:
      model: gemini-embedding-2
      batch_size: 32
```

Load the file, then explicitly construct and install the provider:

```go
manager, err := config.NewManager(config.WithConfigPath("config.yaml"))
if err != nil {
    return err
}
if err := manager.Load(); err != nil {
    return err
}

provider := manager.Get().LLM.Default
llm, err := llms.NewLLM(provider.APIKey, core.ModelID(provider.ModelID))
if err != nil {
    return err
}
core.SetDefaultLLM(llm)
```

Apply loaded generation settings at the call site:

```go
generation := provider.Generation
result, err := predictor.Process(ctx, inputs,
    core.WithGenerateOptions(
        core.WithMaxTokens(generation.MaxTokens),
        core.WithTemperature(generation.Temperature),
        core.WithTopP(generation.TopP),
        core.WithPresencePenalty(generation.PresencePenalty),
        core.WithFrequencyPenalty(generation.FrequencyPenalty),
        core.WithStopSequences(generation.StopSequences...),
    ),
)
```

The config package's environment source converts names such as
`DSPY_LLM_DEFAULT_MODEL_ID` to `llm.default.model.id` and overlays them on the
loaded file. Provider-native variables such as `GEMINI_API_KEY` are consumed by
provider constructors, not by this typed overlay.

## Advanced Provider Configuration

Use `core.ProviderConfig` when loading a registered provider with a custom
endpoint or headers:

```go
llms.EnsureFactory()

llm, err := core.LoadLLMFromConfig(ctx, core.ProviderConfig{
    Name:   "openai",
    APIKey: apiKey,
    Endpoint: &core.EndpointConfig{
        BaseURL:   "https://gateway.example.com",
        Path:      "/v1/chat/completions",
        Headers:   map[string]string{"X-Tenant": "example"},
        TimeoutSec: 60,
    },
}, core.ModelID("gateway-model"))
```

For an arbitrary OpenAI-compatible endpoint, `llms.NewOpenAICompatible` is
usually simpler than registry configuration.

## Embedding Settings

Embedding settings are separate from generation settings:

```go
embedding, err := llm.CreateEmbedding(
    ctx,
    input,
    core.WithModel(provider.Embedding.Model),
)

batch, err := llm.CreateEmbeddings(
    ctx,
    inputs,
    core.WithModel(provider.Embedding.Model),
    core.WithBatchSize(provider.Embedding.BatchSize),
)
```

Gemini and OpenAI-compatible adapters retain embedding support in dspy-go.
Anthropic, OpenAI Codex, and a plain `llms.Adapt(generator)` do not attach an
embedding client.

## Next Steps

- **[LLM Providers →](../providers/)**
- **[API Reference →](../)**
- **[Getting Started →](../../guides/getting-started/)**
