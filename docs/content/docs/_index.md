---
title: "Documentation"
description: "Comprehensive guide to building LLM applications with dspy-go"
summary: "Learn how to build reliable, production-ready LLM applications using systematic prompt engineering"
date: 2023-09-07T16:12:03+02:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 999
toc: true
seo:
  title: "dspy-go Documentation - Build Reliable LLM Applications"
  description: "Complete guide to dspy-go: modular architecture, advanced optimizers, tool management, and multimodal AI"
  canonical: ""
  noindex: false
---

# Welcome to dspy-go

**dspy-go** is a native Go implementation of the DSPy framework that brings systematic prompt engineering and automated reasoning capabilities to Go applications. Build reliable and effective Language Model (LLM) applications through composable modules, powerful optimizers, and intelligent tool management.

## Why dspy-go?

### 🎯 **Go-Native Architecture**
Built from the ground up for Go—not just a port. Idiomatic Go patterns, strong typing, and seamless integration with your existing Go applications.

### 🚀 **Zero to Production in Minutes**
Get started with zero configuration. One-line setup automatically detects your environment and configures the appropriate LLM provider.

```go
llm, _ := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
core.SetDefaultLLM(llm)
// Ready to go!
```

### 🧠 **Advanced Optimizers**
State-of-the-art prompt optimization algorithms that automatically improve your prompts:
- **GEPA**: Multi-objective evolutionary optimization with Pareto selection
- **MIPRO**: TPE-based systematic prompt optimization
- **SIMBA**: Introspective mini-batch learning
- **BootstrapFewShot**: Automated few-shot example selection
- **COPRO**: Collaborative multi-module optimization

### 🔧 **Intelligent Tool Management**
Smart Tool Registry uses Bayesian inference for intelligent tool selection. Automatic performance tracking, capability analysis, and MCP (Model Context Protocol) integration.

### ⚡ **Built for Performance**
- Parallel module execution out of the box
- Concurrent batch processing
- Efficient streaming support
- Optimized for production workloads

### 🎨 **Multimodal from Day One**
Native support for:
- Image analysis and vision Q&A
- Multimodal chat conversations
- Streaming multimodal content
- Content block system for flexible media handling

## What Can You Build?

- **Question Answering Systems**: RAG pipelines with retrieval and generation
- **Code Analysis Tools**: Automated code review and refactoring suggestions
- **Data Processing Pipelines**: Extract, transform, and analyze data with LLMs
- **Intelligent Agents**: ReAct patterns with tool use and reasoning
- **Multi-Step Workflows**: Chain modules for complex task decomposition
- **Optimization Systems**: Automatically improve prompt performance

## Key Features at a Glance

| Feature | Description |
|---------|-------------|
| **Modular Design** | Compose simple, reusable components into complex applications |
| **Multiple LLMs** | Anthropic, OpenAI, Google Gemini, Ollama, LlamaCPP, and more |
| **Tool Chaining** | Sequential pipelines with data transformation and conditional logic |
| **Dependency Resolution** | Automatic execution planning with parallel optimization |
| **CLI Tool** | Explore optimizers without writing code |
| **Compatibility Testing** | Framework ensures Python DSPy parity |
| **Dataset Management** | Auto-download GSM8K, HotPotQA, and more |

## Quick Links

- **[Getting Started →](guides/getting-started/)** - Install and run your first program
- **[Core Concepts →](guides/core-concepts/)** - Understand Signatures, Modules, and Programs
- **[Optimizers →](guides/optimizers/)** - Master GEPA, MIPRO, SIMBA and more
- **[Tool Management →](guides/tools/)** - Smart Registry, MCP, and tool chaining
- **[Examples →](https://github.com/XiaoConstantine/dspy-go/tree/main/examples)** - Real-world implementations

## Community & Support

- **GitHub**: [XiaoConstantine/dspy-go](https://github.com/XiaoConstantine/dspy-go)
- **Go Docs**: [pkg.go.dev](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go)
- **Example App**: [Maestro](https://github.com/XiaoConstantine/maestro) - Code review agent built with dspy-go

Ready to get started? Head to [Getting Started](guides/getting-started/) →
