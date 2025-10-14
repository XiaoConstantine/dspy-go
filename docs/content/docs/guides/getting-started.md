---
title: "Getting Started"
description: "Get started with dspy-go in minutes"
summary: "Install dspy-go and build your first LLM application"
date: 2025-10-13T00:00:00+00:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 100
toc: true
seo:
  title: "Getting Started with dspy-go"
  description: "Quick start guide to building LLM applications with dspy-go - zero configuration required"
  canonical: ""
  noindex: false
---

# Getting Started with dspy-go

This guide will walk you through the basics of setting up `dspy-go` and running your first prediction. You can get started in two ways: with our CLI tool (no code required) or by writing Go code.

## Choose Your Path

### 🚀 Option A: CLI Tool (Recommended for Beginners)

Perfect for exploring dspy-go without writing code. Try optimizers, test with sample datasets, and see results instantly.

**[Jump to CLI Quick Start →](#cli-quick-start)**

### 💻 Option B: Programming with Go

Build custom applications with full control. Perfect for production applications and custom workflows.

**[Jump to Programming Quick Start →](#programming-quick-start)**

---

## CLI Quick Start

The dspy-go CLI eliminates 60+ lines of boilerplate and lets you test all optimizers instantly.

### 1. Build the CLI

```bash
cd cmd/dspy-cli
go build -o dspy-cli
```

### 2. Set Your API Key

```bash
export GEMINI_API_KEY="your-api-key-here"
# or
export OPENAI_API_KEY="your-api-key-here"
# or
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 3. Explore Optimizers

```bash
# See all available optimizers
./dspy-cli list

# Get optimizer recommendations
./dspy-cli recommend balanced

# Try Bootstrap optimizer with GSM8K math problems
./dspy-cli try bootstrap --dataset gsm8k --max-examples 5

# Test advanced MIPRO optimizer
./dspy-cli try mipro --dataset gsm8k --verbose

# Experiment with evolutionary GEPA
./dspy-cli try gepa --dataset gsm8k --max-examples 10
```

### What You Can Do with the CLI

- **List Optimizers**: See all available optimization algorithms
- **Try Optimizers**: Test any optimizer with built-in datasets
- **Get Recommendations**: Get suggestions based on your needs (speed/quality trade-off)
- **Compare Results**: Run multiple optimizers and compare performance
- **No Code Required**: Everything works out of the box

[Full CLI Documentation →](https://github.com/XiaoConstantine/dspy-go/tree/main/cmd/dspy-cli/README.md)

---

## Programming Quick Start

Build custom LLM applications with full programmatic control.

### 1. Installation

Add `dspy-go` to your project using `go get`:

```bash
go get github.com/XiaoConstantine/dspy-go
```

### 2. Set Up Your API Key

`dspy-go` supports multiple LLM providers. Set the appropriate environment variable:

```bash
# Google Gemini (multimodal support)
export GEMINI_API_KEY="your-api-key-here"

# OpenAI
export OPENAI_API_KEY="your-api-key-here"

# Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key-here"

# Ollama (local)
export OLLAMA_BASE_URL="http://localhost:11434"
```

### 3. Your First Program

Here's a simple sentiment analysis application using **zero-configuration setup**:

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
	// 1. Zero-Config Setup
	// Pass empty string - automatically uses API key from environment variable
	llm, err := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}
	core.SetDefaultLLM(llm)

	// 2. Define a Signature
	// A signature describes the task inputs, outputs, and instructions
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("sentence", core.WithDescription("The sentence to classify."))},
		},
		[]core.OutputField{
			{Field: core.NewField("sentiment", core.WithDescription("The sentiment (Positive, Negative, Neutral)."))},
		},
	).WithInstruction("You are a helpful sentiment analysis expert. Classify the sentiment of the given sentence.")

	// 3. Create a Predict Module
	predictor := modules.NewPredict(signature)

	// 4. Execute the Predictor
	ctx := context.Background()
	input := map[string]interface{}{
		"sentence": "dspy-go makes building AI applications easy and fun!",
	}

	result, err := predictor.Process(ctx, input)
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}

	// 5. Print the Result
	fmt.Printf("Sentence: %s\n", input["sentence"])
	fmt.Printf("Sentiment: %s\n", result["sentiment"])
}
```

### Running the Code

Save the code above as `main.go`, and run it:

```bash
go run main.go
```

**Output:**
```
Sentence: dspy-go makes building AI applications easy and fun!
Sentiment: Positive
```

Congratulations! You've just built your first LLM application with dspy-go. 🎉

---

## What's Next?

Now that you have dspy-go running, explore these next steps:

### Core Concepts
Learn about the building blocks of dspy-go:
- **[Signatures →](core-concepts/#signatures)** - Define task inputs and outputs
- **[Modules →](core-concepts/#modules)** - Chain-of-Thought, ReAct, and more
- **[Programs →](core-concepts/#programs)** - Compose modules into workflows

### Advanced Features
- **[Optimizers →](optimizers/)** - Automatically improve prompts with GEPA, MIPRO, SIMBA
- **[Tool Management →](tools/)** - Smart tool selection, chaining, and MCP integration
- **[Multimodal →](multimodal/)** - Work with images, vision Q&A, and streaming

### Examples
Check out real-world implementations:
- [Question Answering](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/hotpotqa)
- [Math Problem Solving](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/gsm8k)
- [Smart Tool Registry](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/smart_tool_registry)
- [Tool Chaining](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/tool_chaining)
- [Multimodal Processing](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/multimodal)

### Example Application
See dspy-go in production:
- **[Maestro](https://github.com/XiaoConstantine/maestro)** - A code review and Q&A agent built with dspy-go

---

## Alternative Configuration Methods

While using empty string "" for zero-config is the easiest way to get started, you can also pass API keys explicitly:

### Explicit Configuration

```go
import (
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
)

// Anthropic Claude
llm, err := llms.NewAnthropicLLM("api-key", core.ModelAnthropicSonnet)
core.SetDefaultLLM(llm)

// Google Gemini
llm, err := llms.NewGeminiLLM("api-key", core.ModelGoogleGeminiPro)
core.SetDefaultLLM(llm)

// OpenAI
llm, err := llms.NewOpenAI(core.ModelOpenAIGPT4, "api-key")
core.SetDefaultLLM(llm)

// Ollama (local)
llm, err := llms.NewOllamaLLM(core.ModelOllamaLlama3_8B)
core.SetDefaultLLM(llm)
```

### Per-Module Configuration

```go
// Use a specific LLM for a specific module
llm, _ := llms.NewAnthropicLLM("api-key", core.ModelAnthropicSonnet)
predictor := modules.NewPredict(signature)
predictor.SetLLM(llm)
```

---

## Troubleshooting

### "Failed to configure LLM" Error

**Problem**: LLM creation fails with "API key required" error.

**Solution**: Make sure you've set the appropriate environment variable or pass the API key explicitly:
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OLLAMA_BASE_URL`

### API Rate Limits

**Problem**: Getting rate limit errors from your LLM provider.

**Solution**: dspy-go includes built-in retry logic with exponential backoff. For heavy workloads, consider:
- Using local models with Ollama
- Implementing request throttling
- Caching results with custom middleware

### Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/XiaoConstantine/dspy-go/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/XiaoConstantine/dspy-go/discussions)
- **Examples**: [Browse working examples](https://github.com/XiaoConstantine/dspy-go/tree/main/examples)
