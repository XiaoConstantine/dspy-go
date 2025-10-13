---
title: "API Reference"
description: "Complete API reference for dspy-go packages, modules, and configuration"
summary: "Comprehensive technical reference for all dspy-go components"
date: 2025-10-13T00:00:00+00:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 900
toc: true
sidebar:
  collapsed: false
seo:
  title: "API Reference - dspy-go"
  description: "Complete API reference documentation for dspy-go framework"
  canonical: ""
  noindex: false
---

Complete technical reference for all dspy-go packages, modules, and configuration options.

## Quick Links

### Core Packages
- **[core](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/core)** - Signatures, fields, and base abstractions
- **[modules](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/modules)** - Predict, ChainOfThought, ReAct, and more
- **[llms](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/llms)** - LLM provider integrations
- **[optimizers](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/optimizers)** - GEPA, MIPRO, SIMBA, Bootstrap, COPRO

### Advanced Packages
- **[tools](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/tools)** - Smart tool registry and management
- **[agents](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/agents)** - Agent orchestration and memory
- **[agents/memory](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go/pkg/agents/memory)** - Conversation memory systems

---

## Package Documentation

### GoDoc Reference

For complete API documentation with all types, interfaces, and functions, visit:

**[pkg.go.dev/github.com/XiaoConstantine/dspy-go](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go)**

The GoDoc provides:
- ✅ Full type definitions
- ✅ Function signatures
- ✅ Method documentation
- ✅ Code examples
- ✅ Source code navigation

---

## Reference Guides

### By Topic

| Guide | Description |
|-------|-------------|
| **[Configuration Reference →](configuration/)** | Environment variables, LLM setup, provider options |
| **[CLI Reference →](cli/)** | Command-line tool usage and flags |
| **[LLM Providers →](providers/)** | Supported providers and model configurations |

---

## Common Patterns

### Initialization Patterns

```go
// Zero-config (recommended)
llm, _ := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
core.SetDefaultLLM(llm)

// Explicit configuration
llm, _ := llms.NewGeminiLLM("api-key", core.ModelGoogleGeminiPro)
core.SetDefaultLLM(llm)

// Per-module override
module := modules.NewPredict(signature)
module.SetLLM(customLLM)
```

### Error Handling

```go
import "github.com/XiaoConstantine/dspy-go/pkg/core"

// Check for specific error types
if err != nil {
    switch {
    case errors.Is(err, core.ErrNoLLMConfigured):
        // Handle missing LLM configuration
    case errors.Is(err, core.ErrInvalidSignature):
        // Handle invalid signature
    default:
        // Handle other errors
    }
}
```

### Context Management

```go
import "context"

// Always pass context
ctx := context.Background()

// With timeout
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

// With cancellation
ctx, cancel := context.WithCancel(context.Background())
defer cancel()
```

---

## Type System

### Core Types

#### Signature
```go
type Signature interface {
    Inputs() []InputField
    Outputs() []OutputField
    Instruction() string
}
```

#### Module
```go
type Module interface {
    Process(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error)
}
```

#### LLM
```go
type LLM interface {
    Generate(ctx context.Context, input string) (string, error)
    GenerateWithOptions(ctx context.Context, input string, opts GenerateOptions) (string, error)
}
```

---

## Constants

### Model Identifiers

```go
// OpenAI Models
core.ModelOpenAIGPT4 = "gpt-4"
core.ModelOpenAIGPT4Turbo = "gpt-4-turbo-preview"
core.ModelOpenAIGPT35Turbo = "gpt-3.5-turbo"

// Anthropic Models
core.ModelAnthropicOpus = "claude-3-opus-20240229"
core.ModelAnthropicSonnet = "claude-3-sonnet-20240229"
core.ModelAnthropicHaiku = "claude-3-haiku-20240307"
```

### Error Constants

```go
var (
    ErrNoLLMConfigured = errors.New("no LLM configured")
    ErrInvalidSignature = errors.New("invalid signature")
    ErrEmptyInput = errors.New("empty input")
    ErrMaxRetriesExceeded = errors.New("max retries exceeded")
)
```

---

## Environment Variables

### LLM Provider Keys

```bash
# Google Gemini
GEMINI_API_KEY="your-api-key"

# OpenAI
OPENAI_API_KEY="your-api-key"
OPENAI_BASE_URL="https://api.openai.com/v1"  # optional

# Anthropic Claude
ANTHROPIC_API_KEY="your-api-key"

# Ollama (local)
OLLAMA_BASE_URL="http://localhost:11434"
```

### Optimization Settings

```bash
# Enable debug logging
DSPY_DEBUG=true

# Set default timeout (seconds)
DSPY_TIMEOUT=30

# Enable caching
DSPY_CACHE_ENABLED=true
```

---

## Best Practices

### Memory Management

```go
// Use context with timeout for long-running operations
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()

// Clean up resources
defer module.Close()
```

### Concurrent Usage

```go
// Modules are NOT goroutine-safe by default
// Create separate instances for concurrent use
module1 := modules.NewPredict(signature)
module2 := modules.NewPredict(signature)

go func() {
    module1.Process(ctx, input1)
}()

go func() {
    module2.Process(ctx, input2)
}()
```

### Rate Limiting

```go
// Use built-in retry logic
llm, _ := llms.NewGeminiLLM(apiKey, model)
llm.SetMaxRetries(3)
llm.SetRetryDelay(time.Second)

// Or implement custom rate limiting
limiter := rate.NewLimiter(rate.Every(time.Second), 10)
limiter.Wait(ctx)
```

---

## Next Steps

- **[Browse Full API on pkg.go.dev →](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go)**
- **[Configuration Reference →](configuration/)**
- **[CLI Reference →](cli/)**
- **[LLM Providers →](providers/)**
