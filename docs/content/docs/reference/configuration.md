---
title: "Configuration Reference"
description: "Complete configuration guide for dspy-go"
summary: "Environment variables, LLM setup, and all configuration options"
date: 2025-10-13T00:00:00+00:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 910
toc: true
seo:
  title: "Configuration Reference - dspy-go"
  description: "Complete configuration reference for dspy-go framework"
  canonical: ""
  noindex: false
---

Complete guide to configuring dspy-go for development and production environments.

---

## Environment Variables

### LLM Provider Configuration

#### Google Gemini

```bash
# Required
GEMINI_API_KEY="your-api-key-here"

# Optional - specify model (defaults to gemini-pro)
GEMINI_MODEL="gemini-1.5-pro"
```

**Get your API key:** [Google AI Studio](https://makersuite.google.com/app/apikey)

#### OpenAI

```bash
# Required
OPENAI_API_KEY="your-api-key-here"

# Optional - custom base URL (for proxies or Azure)
OPENAI_BASE_URL="https://api.openai.com/v1"

# Optional - specify organization
OPENAI_ORG_ID="your-org-id"
```

**Get your API key:** [OpenAI Platform](https://platform.openai.com/api-keys)

#### Anthropic Claude

```bash
# Required
ANTHROPIC_API_KEY="your-api-key-here"

# Optional - API version (defaults to latest)
ANTHROPIC_API_VERSION="2023-06-01"
```

**Get your API key:** [Anthropic Console](https://console.anthropic.com/)

#### Ollama (Local)

```bash
# Required - Ollama server URL
OLLAMA_BASE_URL="http://localhost:11434"

# Optional - specify model
OLLAMA_MODEL="llama2"
```

**Install Ollama:** [ollama.com](https://ollama.com/)

---

## Application Settings

### General Configuration

```bash
# Enable debug logging
DSPY_DEBUG=true

# Set log level (debug, info, warn, error)
DSPY_LOG_LEVEL="info"

# Default timeout for LLM calls (seconds)
DSPY_TIMEOUT=30

# Maximum retries for failed requests
DSPY_MAX_RETRIES=3

# Retry delay (milliseconds)
DSPY_RETRY_DELAY=1000
```

### Caching

```bash
# Enable response caching
DSPY_CACHE_ENABLED=true

# Cache directory
DSPY_CACHE_DIR="~/.cache/dspy-go"

# Cache TTL (time-to-live in seconds)
DSPY_CACHE_TTL=3600

# Maximum cache size (MB)
DSPY_CACHE_MAX_SIZE=1000
```

### Performance

```bash
# Enable parallel execution
DSPY_PARALLEL_ENABLED=true

# Maximum parallel workers
DSPY_MAX_WORKERS=4

# Request rate limit (requests per second)
DSPY_RATE_LIMIT=10

# Enable request batching
DSPY_BATCH_ENABLED=true

# Batch size
DSPY_BATCH_SIZE=5
```

---

## Programmatic Configuration

### Zero-Config Setup

Automatically configures based on environment variables:

```go
import "github.com/XiaoConstantine/dspy-go/pkg/core"

func main() {
    // Detects and configures LLM from environment
    llm, err := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
    if err != nil {
        log.Fatal(err)
    }
    core.SetDefaultLLM(llm)

    // Ready to use modules
    predictor := modules.NewPredict(signature)
}
```

**Priority order:**
1. `GEMINI_API_KEY` → Gemini
2. `OPENAI_API_KEY` → OpenAI
3. `ANTHROPIC_API_KEY` → Anthropic
4. `OLLAMA_BASE_URL` → Ollama

---

### Explicit LLM Configuration

#### Gemini

```go
import (
    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
)

// Basic setup
llm, err := llms.NewGeminiLLM("api-key", core.ModelGoogleGeminiPro)
if err != nil {
    log.Fatal(err)
}
core.SetDefaultLLM(llm)

// With options
llm, err := llms.NewGeminiLLM("api-key", core.ModelGoogleGeminiPro,
    llms.WithTemperature(0.7),
    llms.WithMaxTokens(2048),
    llms.WithTopP(0.9),
)
```

#### OpenAI

```go
// Basic setup
llm, err := llms.NewOpenAI(core.ModelOpenAIGPT4, "api-key")
core.SetDefaultLLM(llm)

// With custom base URL (e.g., Azure)
llm, err := llms.NewOpenAI(
    "gpt-4",
    "api-key",
    llms.WithBaseURL("https://your-azure-endpoint.openai.azure.com"),
    llms.WithAPIVersion("2024-02-15-preview"),
)

// With options
llm, err := llms.NewOpenAI(core.ModelOpenAIGPT4Turbo, "api-key",
    llms.WithTemperature(0.8),
    llms.WithMaxTokens(4096),
    llms.WithPresencePenalty(0.1),
    llms.WithFrequencyPenalty(0.1),
)
```

#### Anthropic Claude

```go
// Basic setup
llm, err := llms.NewAnthropicLLM("api-key", core.ModelAnthropicSonnet)
core.SetDefaultLLM(llm)

// With options
llm, err := llms.NewAnthropicLLM("api-key", core.ModelAnthropicOpus,
    llms.WithTemperature(0.7),
    llms.WithMaxTokens(4096),
    llms.WithTopP(0.9),
    llms.WithTopK(40),
)
```

#### Ollama (Local)

```go
// Basic setup
llm, err := llms.NewOllamaLLM("llama2")
core.SetDefaultLLM(llm)

// With custom server
llm, err := llms.NewOllamaLLM("llama2",
    llms.WithBaseURL("http://192.168.1.100:11434"),
)

// With options
llm, err := llms.NewOllamaLLM("mistral",
    llms.WithTemperature(0.8),
    llms.WithNumCtx(4096),      // Context window
    llms.WithNumPredict(2048),  // Max tokens to predict
)
```

---

### Per-Module Configuration

Override LLM for specific modules:

```go
// Create a module that will use the current default LLM at execution time
predictor := modules.NewPredict(signature)

// Pin the module to a specific LLM
customLLM, _ := llms.NewAnthropicLLM("key", core.ModelAnthropicOpus)
predictor.SetLLM(customLLM)

// Now this module uses Claude Opus
result, _ := predictor.Process(ctx, inputs)
```

If you do not call `SetLLM`, modules resolve models in this order when they execute:

1. module-local LLM set with `SetLLM`
2. request-local runtime from `core.WithRuntime`
3. package default configured with `core.SetDefaultLLM`

### Per-Request Runtime Overrides

Use a runtime to override the model for a single request without mutating package globals or permanently changing the module:

```go
predictor := modules.NewPredict(signature)

runtimeLLM, _ := llms.NewOpenAI(core.ModelOpenAIGPT4, "api-key")
ctx := core.WithRuntime(context.Background(), &core.Runtime{
    DefaultLLM: runtimeLLM,
})

result, err := predictor.Process(ctx, inputs)
```

This runtime override is also respected by interceptors that invoke the model directly, such as structured-output and native function-calling interceptors.

---

## Generation Options

### Common Options

Available for all LLM providers:

```go
type GenerateOptions struct {
    Temperature      float64  // Randomness (0.0 - 1.0)
    MaxTokens        int      // Maximum tokens to generate
    TopP             float64  // Nucleus sampling (0.0 - 1.0)
    StopSequences    []string // Stop generation at these strings
    PresencePenalty  float64  // Penalize new topics (-2.0 - 2.0)
    FrequencyPenalty float64  // Penalize repetition (-2.0 - 2.0)
    Stream           bool     // Enable streaming
}
```

### Temperature Guidelines

| Temperature | Use Case | Example |
|-------------|----------|---------|
| **0.0 - 0.3** | Factual, deterministic | Classification, extraction |
| **0.4 - 0.6** | Balanced | Question answering, analysis |
| **0.7 - 0.9** | Creative | Writing, brainstorming |
| **0.9 - 1.0** | Highly creative | Fiction, poetry |

### Example Usage

```go
opts := core.GenerateOptions{
    Temperature:   0.7,
    MaxTokens:     2048,
    TopP:          0.9,
    StopSequences: []string{"\n\n", "END"},
}

result, err := llm.GenerateWithOptions(ctx, prompt, opts)
```

---

## Advanced Configuration

### Retry Configuration

```go
import "github.com/XiaoConstantine/dspy-go/pkg/core"

llm.SetMaxRetries(5)
llm.SetRetryDelay(2 * time.Second)
llm.SetRetryBackoff(true) // Exponential backoff
```

### Timeout Configuration

```go
// Global timeout
core.SetDefaultTimeout(30 * time.Second)

// Per-request timeout
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()

result, err := predictor.Process(ctx, inputs)
```

### Rate Limiting

```go
import "golang.org/x/time/rate"

// Create rate limiter (10 requests per second)
limiter := rate.NewLimiter(rate.Every(time.Second), 10)

// Wait before making request
if err := limiter.Wait(ctx); err != nil {
    log.Fatal(err)
}

result, err := predictor.Process(ctx, inputs)
```

---

## Configuration Files

### .env File Support

```bash
# .env
GEMINI_API_KEY=your-api-key
DSPY_DEBUG=true
DSPY_CACHE_ENABLED=true
```

Load with:

```go
import "github.com/joho/godotenv"

func init() {
    if err := godotenv.Load(); err != nil {
        log.Println("No .env file found")
    }
}
```

### Config File (YAML)

```yaml
# config.yaml
llm:
  provider: gemini
  model: gemini-1.5-pro
  api_key: ${GEMINI_API_KEY}
  temperature: 0.7
  max_tokens: 2048

cache:
  enabled: true
  directory: ~/.cache/dspy-go
  ttl: 3600

performance:
  max_workers: 4
  rate_limit: 10
  batch_size: 5
```

---

## Production Best Practices

### Security

```go
// ✅ DO: Use environment variables
apiKey := os.Getenv("GEMINI_API_KEY")

// ❌ DON'T: Hardcode API keys
apiKey := "hardcoded-key" // Never do this!

// ✅ DO: Validate configuration
if apiKey == "" {
    log.Fatal("GEMINI_API_KEY not set")
}
```

### Error Handling

```go
// Configure with error handling
llm, err := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
if err != nil {
    log.Fatal("Failed to create LLM:", err)
}
core.SetDefaultLLM(llm)
```

### Monitoring

```go
// When calling an LLM directly, attach execution state for tracing
llm := core.GetDefaultLLM()
if llm == nil {
    log.Fatal("no default LLM configured")
}

ctx := core.WithExecutionState(context.Background())

core.RecordLLMCall(ctx, llm)
resp, err := llm.Generate(ctx, "Summarize the request", core.WithMaxTokens(256))
if err != nil {
    log.Fatal(err)
}

if state := core.GetExecutionState(ctx); state != nil {
    log.Printf("LLM call completed (model=%s, content_len=%d)",
        state.GetModelID(), len(resp.Content))
}
```

If you only depend on a narrower interface that exposes `ModelID()`, use `core.RecordModelCall(ctx, model)` instead.

---

## Troubleshooting

### Common Issues

#### "No LLM configured" Error

**Cause:** No API key found in environment

**Solution:**
```bash
export GEMINI_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"
```

#### Rate Limit Errors

**Cause:** Too many requests to API

**Solution:**
```go
llm.SetMaxRetries(5)
llm.SetRetryDelay(2 * time.Second)
llm.SetRetryBackoff(true)
```

#### Timeout Errors

**Cause:** Requests taking too long

**Solution:**
```go
core.SetDefaultTimeout(60 * time.Second)
// or
ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
```

---

## Next Steps

- **[CLI Reference →](cli/)** - Command-line tool configuration
- **[LLM Providers →](providers/)** - Provider-specific details
- **[API Reference →](../)** - Full API documentation
