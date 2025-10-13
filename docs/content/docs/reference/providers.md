---
title: "LLM Providers"
description: "Complete guide to LLM providers supported by dspy-go"
summary: "Provider-specific configurations, capabilities, and best practices"
date: 2025-10-13T00:00:00+00:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 930
toc: true
seo:
  title: "LLM Providers - dspy-go"
  description: "Complete guide to LLM providers in dspy-go"
  canonical: ""
  noindex: false
---

dspy-go supports multiple LLM providers with native integrations. Each provider has unique capabilities and configuration options.

---

## Supported Providers

| Provider | Streaming | Multimodal | Function Calling | Local | Best For |
|----------|-----------|------------|------------------|-------|----------|
| **Google Gemini** | ✅ | ✅ | ✅ | ❌ | Multimodal, long context (2M tokens!) |
| **OpenAI** | ✅ | ✅ | ✅ | ❌ | Latest GPT-5 models, reliability |
| **Anthropic Claude** | ✅ | ✅ | ✅ | ❌ | Long context, reasoning |
| **Ollama** | ✅ | ❌ | ❌ | ✅ | Local Llama 3.2, Qwen 2.5, privacy |
| **LlamaCpp** | ✅ | ❌ | ❌ | ✅ | Local GGUF models, quantization |
| **LiteLLM** | ✅ | ✅ | ✅ | ❌ | Unified API for 100+ models |

---

## Google Gemini

**Best for:** Multimodal applications, 2M token context, cost-effective

### Setup

```go
import (
    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
)

// Basic setup
llm, err := llms.NewGeminiLLM("your-api-key", core.ModelGoogleGeminiPro)
if err != nil {
    log.Fatal(err)
}
core.SetDefaultLLM(llm)
```

### Available Models

| Model | Context Window | Features | Best For |
|-------|----------------|----------|----------|
| **gemini-2.5-pro** | 2M tokens | Multimodal, function calling, best reasoning | Complex tasks, entire codebases |
| **gemini-2.5-flash** | 1M tokens | Fast, cost-effective, multimodal | Quick responses, high volume |
| **gemini-2.5-flash-lite** | 1M tokens | Ultra-fast, efficient | Lightweight tasks, batch processing |

### Configuration

```go
llm, err := llms.NewGeminiLLM("api-key", core.ModelGoogleGeminiPro,
    llms.WithTemperature(0.7),      // Creativity (0.0-1.0)
    llms.WithMaxTokens(2048),        // Max output tokens
    llms.WithTopP(0.9),              // Nucleus sampling
    llms.WithTopK(40),               // Top-K sampling
    llms.WithStopSequences([]string{"END", "\n\n"}),
)
```

### Multimodal Support

```go
// Analyze images
imageData, _ := os.ReadFile("image.jpg")
result, err := predictor.Process(ctx, map[string]interface{}{
    "image": core.NewImageContent(imageData, "image/jpeg"),
    "question": "What's in this image?",
})

// Multiple images
result, err := predictor.Process(ctx, map[string]interface{}{
    "image1": core.NewImageContent(data1, "image/jpeg"),
    "image2": core.NewImageContent(data2, "image/jpeg"),
    "question": "What changed between these images?",
})
```

### Streaming

```go
llm.SetStreaming(true)

// Handle streaming chunks
llm.SetStreamHandler(func(chunk string) {
    fmt.Print(chunk)
})

result, err := llm.Generate(ctx, prompt)
```

### Rate Limits & Pricing

| Model | RPM (Free) | RPM (Paid) | Cost (Input/Output) |
|-------|------------|------------|---------------------|
| gemini-2.5-pro | 2 | 360 | $0.00125 / $0.005 per 1K tokens |
| gemini-2.5-flash | 15 | 1000 | $0.00004 / $0.00015 per 1K tokens |
| gemini-2.5-flash-lite | 30 | 2000 | $0.00002 / $0.00006 per 1K tokens |

### Best Practices

```go
// ✅ Use 2.5-flash for speed and cost
llm, _ := llms.NewGeminiLLM(key, core.ModelGoogleGeminiFlash)

// ✅ Leverage 2M token context for RAG
// No need to chunk! Can handle entire codebases

// ✅ Use for multimodal tasks
llm, _ := llms.NewGeminiLLM(key, core.ModelGoogleGeminiPro)
```

**Get API Key:** [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## OpenAI

**Best for:** GPT-5 models, reliability, ecosystem

### Setup

```go
import (
    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
)

// Basic setup
llm, err := llms.NewOpenAI(core.ModelOpenAIGPT5, "your-api-key")
core.SetDefaultLLM(llm)
```

### Available Models

| Model | Context Window | Features | Best For |
|-------|----------------|----------|----------|
| **gpt-5** | 256K | Flagship model, multimodal, superior reasoning | Most complex tasks |
| **gpt-5-mini** | 256K | Efficient, fast, multimodal | Balanced tasks |
| **gpt-5-nano** | 128K | Ultra-efficient, fast | High-volume, quick tasks |
| **gpt-4o** | 128K | Optimized, fast, multimodal | General purpose |
| **gpt-4o-mini** | 128K | Affordable, fast | High-volume tasks |
| **gpt-4-turbo** | 128K | Latest GPT-4, multimodal | Complex reasoning |
| **gpt-4** | 8K | Proven, reliable | Production apps |
| **gpt-3.5-turbo** | 16K | Fast, cheap | Quick tasks, chat |

### Configuration

```go
llm, err := llms.NewOpenAI(core.ModelOpenAIGPT5, "api-key",
    llms.WithTemperature(0.7),           // Creativity
    llms.WithMaxTokens(4096),             // Max output
    llms.WithTopP(0.9),                   // Nucleus sampling
    llms.WithPresencePenalty(0.1),        // Discourage repetition
    llms.WithFrequencyPenalty(0.1),       // Penalize frequent words
    llms.WithStopSequences([]string{"\n\n"}),
)
```

### Function Calling

```go
// Define functions
functions := []core.Function{
    {
        Name:        "get_weather",
        Description: "Get current weather for a location",
        Parameters: map[string]interface{}{
            "type": "object",
            "properties": map[string]interface{}{
                "location": map[string]interface{}{
                    "type": "string",
                    "description": "City name",
                },
            },
            "required": []string{"location"},
        },
    },
}

llm.SetFunctions(functions)
```

### Azure OpenAI

```go
llm, err := llms.NewOpenAI("gpt-5", "api-key",
    llms.WithBaseURL("https://your-resource.openai.azure.com"),
    llms.WithAPIVersion("2024-02-15-preview"),
    llms.WithAPIType("azure"),
)
```

### Streaming

```go
llm.SetStreaming(true)
llm.SetStreamHandler(func(chunk string) {
    fmt.Print(chunk)
})

result, err := llm.Generate(ctx, prompt)
```

### Rate Limits & Pricing

| Model | TPM (Tier 1) | Cost (Input/Output) |
|-------|--------------|---------------------|
| gpt-5 | 500K | $0.005 / $0.015 per 1K tokens (estimated) |
| gpt-5-mini | 1M | $0.0015 / $0.004 per 1K tokens (estimated) |
| gpt-5-nano | 2M | $0.0005 / $0.001 per 1K tokens (estimated) |
| gpt-4o | 500K | $0.0025 / $0.01 per 1K tokens |
| gpt-4o-mini | 2M | $0.00015 / $0.0006 per 1K tokens |
| gpt-4-turbo | 300K | $0.01 / $0.03 per 1K tokens |
| gpt-4 | 40K | $0.03 / $0.06 per 1K tokens |
| gpt-3.5-turbo | 200K | $0.0005 / $0.0015 per 1K tokens |

### Best Practices

```go
// ✅ Use GPT-5 for most complex reasoning
llm, _ := llms.NewOpenAI(core.ModelOpenAIGPT5, key)

// ✅ Use gpt-5-nano for high-volume tasks
llm, _ := llms.NewOpenAI(core.ModelOpenAIGPT5Nano, key)

// ✅ Use gpt-4o for production balance
llm, _ := llms.NewOpenAI(core.ModelOpenAIGPT4o, key)

// ✅ Implement retry logic
llm.SetMaxRetries(3)
llm.SetRetryDelay(time.Second)
```

**Get API Key:** [OpenAI Platform](https://platform.openai.com/api-keys)

---

## Anthropic Claude

**Best for:** Long context, detailed reasoning, safety

### Setup

```go
import (
    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
)

// Basic setup
llm, err := llms.NewAnthropicLLM("your-api-key", core.ModelAnthropicSonnet)
core.SetDefaultLLM(llm)
```

### Available Models

| Model | Context Window | Features | Best For |
|-------|----------------|----------|----------|
| **claude-3.5-sonnet** | 200K | Latest, balanced, multimodal | General purpose, production |
| **claude-3-opus** | 200K | Most capable, best reasoning | Complex analysis, research |
| **claude-3-haiku** | 200K | Fast, efficient | Quick tasks, high volume |

### Configuration

```go
llm, err := llms.NewAnthropicLLM("api-key", core.ModelAnthropicSonnet,
    llms.WithTemperature(0.7),
    llms.WithMaxTokens(4096),
    llms.WithTopP(0.9),
    llms.WithTopK(40),
)
```

### Multimodal Support

```go
// Analyze images with Claude
imageData, _ := os.ReadFile("document.jpg")
result, err := predictor.Process(ctx, map[string]interface{}{
    "image": core.NewImageContent(imageData, "image/jpeg"),
    "question": "Extract all text from this document",
})
```

### Streaming

```go
llm.SetStreaming(true)
llm.SetStreamHandler(func(chunk string) {
    fmt.Print(chunk)
})

result, err := llm.Generate(ctx, prompt)
```

### Rate Limits & Pricing

| Model | TPM | Cost (Input/Output) |
|-------|-----|---------------------|
| claude-3.5-sonnet | 400K | $0.003 / $0.015 per 1K tokens |
| claude-3-opus | 400K | $0.015 / $0.075 per 1K tokens |
| claude-3-haiku | 400K | $0.00025 / $0.00125 per 1K tokens |

### Best Practices

```go
// ✅ Use 3.5 Sonnet for production
llm, _ := llms.NewAnthropicLLM(key, core.ModelAnthropicSonnet)

// ✅ Use Haiku for fast, cheap tasks
llm, _ := llms.NewAnthropicLLM(key, core.ModelAnthropicHaiku)

// ✅ Leverage 200K context for documents
// Can analyze entire books!
```

**Get API Key:** [Anthropic Console](https://console.anthropic.com/)

---

## Ollama (Local)

**Best for:** Privacy, offline use, no API costs, Llama 3.2 & Qwen 2.5

### Setup

```go
import (
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
)

// Basic setup (assumes Ollama running on localhost:11434)
llm, err := llms.NewOllamaLLM("llama3:8b")

// Custom server
llm, err := llms.NewOllamaLLM("qwen2.5:7b",
    llms.WithBaseURL("http://192.168.1.100:11434"),
)
```

### Available Models

Latest models supported by dspy-go:

| Model | Size | Context | Best For |
|-------|------|---------|----------|
| **llama3.2:3b** | 3B | 8K | Efficient, fast, latest Llama |
| **llama3.1:8b** | 8B | 128K | Latest Llama 3.1, long context |
| **llama3.1:70b** | 70B | 128K | Most capable Llama |
| **qwen2.5:7b** | 7B | 32K | Latest Qwen, excellent reasoning |
| **qwen2.5:14b** | 14B | 32K | Best Qwen, superior performance |
| **codellama:13b** | 13B | 16K | Code generation |
| **codellama:34b** | 34B | 16K | Advanced code tasks |
| **mistral:7b** | 7B | 32K | Fast, efficient |
| **gemma:2b** | 2B | 8K | Ultra-efficient |
| **gemma:7b** | 7B | 8K | Balanced efficiency |

### Installation

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull latest models
ollama pull llama3.2:3b
ollama pull qwen2.5:7b

# Run Ollama server
ollama serve
```

### Configuration

```go
llm, err := llms.NewOllamaLLM("llama3.1:8b",
    llms.WithTemperature(0.8),
    llms.WithNumCtx(8192),          // Context window
    llms.WithNumPredict(2048),      // Max tokens
    llms.WithNumGPU(1),             // GPU layers
    llms.WithRepeatPenalty(1.1),    // Repetition penalty
)
```

### Streaming

```go
llm.SetStreaming(true)
llm.SetStreamHandler(func(chunk string) {
    fmt.Print(chunk)
})

result, err := llm.Generate(ctx, prompt)
```

### Embedding Models

```go
// Use Ollama for embeddings
llm, err := llms.NewOllamaLLM("nomic-embed-text")
embeddings, err := llm.CreateEmbedding(ctx, "text to embed")
```

Available embedding models:
- **nomic-embed-text** - 768 dimensions, best quality
- **mxbai-embed-large** - 1024 dimensions, large
- **all-minilm** - 384 dimensions, fast

### Performance Tips

```bash
# Use quantized models for speed
ollama pull llama3.2:3b-q4_K_M

# Enable GPU acceleration
export OLLAMA_NUM_GPU=1

# Increase context for long documents
ollama run llama3.1:8b --ctx-size 16384
```

### Best Practices

```go
// ✅ Use Llama 3.2 for latest capabilities
llm, _ := llms.NewOllamaLLM("llama3.2:3b")

// ✅ Use Qwen 2.5 for best reasoning
llm, _ := llms.NewOllamaLLM("qwen2.5:7b")

// ✅ Use CodeLlama for code tasks
llm, _ := llms.NewOllamaLLM("codellama:13b")

// ✅ Batch requests for efficiency
```

**Get Started:** [ollama.com](https://ollama.com/)

---

## LlamaCpp (Local GGUF)

**Best for:** Running quantized models locally, maximum control, GGUF format

### Setup

```go
// Basic setup (assumes llama.cpp server on localhost:8080)
llm, err := llms.NewLlamacppLLM("http://localhost:8080")
if err != nil {
    log.Fatal(err)
}
core.SetDefaultLLM(llm)

// Custom configuration
llm, err := llms.NewLlamacppLLM("http://localhost:8080",
    llms.WithTemperature(0.7),
    llms.WithMaxTokens(2048),
)
```

### Installation

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with GPU support (optional)
make LLAMA_CUBLAS=1

# Download a GGUF model from Hugging Face
# Example: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
wget https://huggingface.co/.../llama-2-7b-chat.Q4_K_M.gguf

# Start server
./server -m llama-2-7b-chat.Q4_K_M.gguf --port 8080
```

### Available Models

Any GGUF quantized model from Hugging Face:

| Quantization | Size | Quality | Use Case |
|--------------|------|---------|----------|
| **Q2_K** | Smallest | Lower | Testing, memory-constrained |
| **Q4_K_M** | Medium | Good | Balanced performance |
| **Q5_K_M** | Larger | Better | Recommended for most |
| **Q8_0** | Largest | Best | Maximum quality |
| **F16** | Full | Native | Best quality, large memory |

### Configuration

```go
llm, err := llms.NewLlamacppLLM("http://localhost:8080",
    llms.WithTemperature(0.8),
    llms.WithTopK(40),
    llms.WithTopP(0.9),
    llms.WithRepeatPenalty(1.1),
)
```

### Popular GGUF Models

- **Llama 3.1 8B** - Latest Meta Llama
- **Qwen 2.5 7B** - Excellent reasoning
- **Mistral 7B** - Fast, efficient
- **CodeLlama 13B** - Code generation
- **Yi 34B** - Strong general purpose

Find more: [Hugging Face GGUF Models](https://huggingface.co/models?search=gguf)

### Streaming

```go
llm.SetStreaming(true)
llm.SetStreamHandler(func(chunk string) {
    fmt.Print(chunk)
})

result, err := llm.Generate(ctx, prompt)
```

### Best Practices

```go
// ✅ Use Q4_K_M for balance
// Good quality, reasonable size

// ✅ Use Q5_K_M for better quality
// Slightly larger, better output

// ✅ Monitor GPU memory usage
// Adjust context size if needed

// ✅ Use --ctx-size for long contexts
./server -m model.gguf --ctx-size 8192
```

---

## LiteLLM (Unified API)

**Best for:** Supporting 100+ models through one API, multi-provider flexibility

### Setup

```go
// Basic setup (assumes LiteLLM proxy running)
config := core.ProviderConfig{
    Name:    "litellm",
    BaseURL: "http://localhost:4000",
}

llm, err := llms.LiteLLMProviderFactory(ctx, config, "gpt-4")
core.SetDefaultLLM(llm)

// With API key
llm, err := llms.LiteLLMProviderFactory(ctx, config, "claude-3-sonnet",
    llms.WithAPIKey("your-litellm-key"),
)
```

### Supported Providers

LiteLLM provides unified access to **100+ models**:

| Category | Providers |
|----------|-----------|
| **Major APIs** | OpenAI, Anthropic, Google, Cohere |
| **Cloud** | AWS Bedrock, Azure OpenAI, Vertex AI |
| **Open Source** | Hugging Face, Replicate, Together AI |
| **Local** | Ollama, LlamaCpp, LocalAI |

### Installation

```bash
# Install LiteLLM
pip install litellm[proxy]

# Create config file
cat > litellm_config.yaml <<EOF
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: llama-3-70b
    litellm_params:
      model: together_ai/meta-llama/Llama-3-70b-chat-hf
      api_key: os.environ/TOGETHER_API_KEY
EOF

# Start proxy server
litellm --config litellm_config.yaml --port 4000
```

### Configuration

```go
// Use any provider through LiteLLM
config := core.ProviderConfig{
    Name:    "litellm",
    BaseURL: "http://localhost:4000",
}

// OpenAI GPT-4
llmGPT4, _ := llms.LiteLLMProviderFactory(ctx, config, "gpt-4")

// Anthropic Claude
llmClaude, _ := llms.LiteLLMProviderFactory(ctx, config, "claude-3-sonnet")

// Together AI Llama
llmLlama, _ := llms.LiteLLMProviderFactory(ctx, config, "llama-3-70b")
```

### Model Routing

```yaml
# litellm_config.yaml - Advanced routing
router_settings:
  routing_strategy: least-busy

model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_ENDPOINT
```

### Load Balancing

```go
// LiteLLM automatically load balances
// Just configure multiple instances in litellm_config.yaml
llm, err := llms.LiteLLMProviderFactory(ctx, config, "gpt-4")
// Requests automatically distributed across providers
```

### Cost Tracking

LiteLLM provides built-in cost tracking:

```bash
# View costs
curl http://localhost:4000/spend/logs

# Set budget limits in config
general_settings:
  master_key: sk-1234
  budget_duration: 30d
  max_budget: 100
```

### Best Practices

```go
// ✅ Use for multi-provider applications
// Switch providers without code changes

// ✅ Implement fallback logic
// LiteLLM can auto-fallback to backup models

// ✅ Monitor costs centrally
// Single dashboard for all providers

// ✅ Use for A/B testing
// Easy to compare different models
```

**Get Started:** [LiteLLM Docs](https://docs.litellm.ai/)

---

## Provider Comparison

### Performance Benchmarks

| Provider | Latency (P50) | Throughput | Cost Efficiency |
|----------|---------------|------------|-----------------|
| Gemini 2.5 Flash | 200ms | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GPT-5 Nano | 300ms | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Claude Haiku | 250ms | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Ollama (local) | 50ms | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Feature Matrix

| Feature | Gemini | OpenAI | Claude | Ollama | LlamaCpp | LiteLLM |
|---------|--------|--------|--------|--------|----------|---------|
| **Streaming** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Multimodal** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Function Calling** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Embeddings** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| **JSON Mode** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Long Context** | 2M tokens | 256K tokens | 200K tokens | 128K | Varies | Varies |

### Context Window Comparison

```
Gemini 2.5 Pro     ██████████████████████████████████████████████████ 2M tokens
GPT-5              ████████████████ 256K tokens
Claude 3.5         ██████████ 200K tokens
Llama 3.1 70B      ████████ 128K tokens
GPT-4o             ████████ 128K tokens
Mistral 7B         ████ 32K tokens
GPT-4              █ 8K tokens
```

---

## Environment Variables

Quick reference for all providers:

```bash
# Google Gemini
export GEMINI_API_KEY="your-api-key"

# OpenAI
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional

# Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key"

# Ollama (local)
export OLLAMA_BASE_URL="http://localhost:11434"

# LiteLLM
export LITELLM_BASE_URL="http://localhost:4000"
export LITELLM_API_KEY="optional-key"
```

---

## Troubleshooting

### Rate Limit Errors

```go
// Implement exponential backoff
llm.SetMaxRetries(5)
llm.SetRetryDelay(2 * time.Second)
llm.SetRetryBackoff(true)
```

### Context Length Errors

```go
// Check model's context window
maxContext := llm.GetContextWindow()

// Truncate if needed
if len(prompt) > maxContext {
    prompt = prompt[:maxContext]
}
```

### API Key Issues

```go
// Verify API key is set
apiKey := os.Getenv("OPENAI_API_KEY")
if apiKey == "" {
    log.Fatal("API key not found")
}

// Test with simple request
result, err := llm.Generate(ctx, "Hello, world!")
```

### Local Model Issues

```bash
# Ollama - Check if running
curl http://localhost:11434/api/tags

# LlamaCpp - Check server
curl http://localhost:8080/health
```

---

## Next Steps

- **[Configuration Reference →](configuration/)** - Detailed configuration options
- **[Getting Started →](../../guides/getting-started/)** - Quick start guide
- **[Multimodal Guide →](../../guides/multimodal/)** - Work with images and vision
