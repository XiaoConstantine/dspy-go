# DSPy-Go

[![Go Report Card](https://goreportcard.com/badge/github.com/XiaoConstantine/dspy-go)](https://goreportcard.com/report/github.com/XiaoConstantine/dspy-go)
[![codecov](https://codecov.io/gh/XiaoConstantine/dspy-go/graph/badge.svg?token=GGKRLMLXJ9)](https://codecov.io/gh/XiaoConstantine/dspy-go)
[![Go Reference](https://pkg.go.dev/badge/github.com/XiaoConstantine/dspy-go)](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go)

## What is DSPy-Go?

DSPy-Go is a native Go implementation of the DSPy framework, bringing systematic prompt engineering and automated reasoning capabilities to Go applications. Build reliable LLM applications through composable modules and workflows.

**[Full Documentation](https://xiaocui.me/dspy-go/)** | **[API Reference](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go)** | **[Examples](examples/)**

### Key Features

| Feature | Description |
|---------|-------------|
| **Modular Architecture** | Compose simple, reusable components into complex applications |
| **Multiple LLM Providers** | Anthropic, OpenAI, Google Gemini, Ollama, LlamaCPP, and more |
| **Advanced Modules** | Predict, ChainOfThought, ReAct, RLM, Refine, Parallel |
| **Intelligent Agents** | ReAct patterns, ACE framework for self-improving agents |
| **A2A Protocol** | Multi-agent orchestration with hierarchical composition |
| **Smart Tool Management** | Bayesian selection, chaining, composition, MCP integration |
| **Quality Optimizers** | GEPA, MIPRO, SIMBA, BootstrapFewShot, COPRO |
| **Structured Output** | JSON structured output and XML adapters with security controls |

## Installation

```bash
go get github.com/XiaoConstantine/dspy-go
```

## Quick Start

### CLI (Zero Code)

```bash
cd cmd/dspy-cli && go build -o dspy-cli
export GEMINI_API_KEY="your-api-key"

./dspy-cli list                           # See all optimizers
./dspy-cli try mipro --dataset gsm8k      # Test optimizer instantly
./dspy-cli view session.jsonl --stats     # View RLM session logs
```

**[CLI Documentation](cmd/dspy-cli/README.md)**

### Programming

```go
package main

import (
    "context"
    "fmt"
    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func main() {
    // Configure LLM
    llms.EnsureFactory()
    core.ConfigureDefaultLLM("your-api-key", core.ModelGoogleGeminiPro)

    // Create signature and module
    signature := core.NewSignature(
        []core.InputField{{Field: core.NewField("question")}},
        []core.OutputField{{Field: core.NewField("answer")}},
    )
    cot := modules.NewChainOfThought(signature)

    // Execute
    result, _ := cot.Process(context.Background(), map[string]interface{}{
        "question": "What is the capital of France?",
    })
    fmt.Println(result["answer"])
}
```

## Core Concepts

### Signatures
Define input/output contracts for modules:
```go
signature := core.NewSignature(
    []core.InputField{{Field: core.NewField("question", core.WithDescription("Question to answer"))}},
    []core.OutputField{{Field: core.NewField("answer", core.WithDescription("Detailed answer"))}},
).WithInstruction("Answer accurately and concisely.")
```

### Modules

| Module | Description |
|--------|-------------|
| `Predict` | Direct prediction |
| `ChainOfThought` | Step-by-step reasoning |
| `ReAct` | Reasoning + tool use |
| `RLM` | Large context exploration via REPL |
| `Refine` | Quality improvement through iteration |
| `Parallel` | Concurrent batch processing |

### Structured Output
```go
// JSON structured output
cot := modules.NewChainOfThought(signature).WithStructuredOutput()

// XML adapter (alternative)
interceptors.ApplyXMLInterceptors(predict, interceptors.DefaultXMLConfig())
```

**[Core Concepts Guide](https://xiaocui.me/dspy-go/docs/guides/core-concepts/)**

## Documentation

| Guide | Description |
|-------|-------------|
| **[Getting Started](https://xiaocui.me/dspy-go/docs/guides/getting-started/)** | Installation and first program |
| **[Core Concepts](https://xiaocui.me/dspy-go/docs/guides/core-concepts/)** | Signatures, Modules, Programs |
| **[Building Agents](https://xiaocui.me/dspy-go/docs/guides/agents/)** | ReAct, ACE framework, memory |
| **[A2A Protocol](https://xiaocui.me/dspy-go/docs/guides/a2a-protocol/)** | Multi-agent orchestration |
| **[RLM Module](https://xiaocui.me/dspy-go/docs/guides/rlm/)** | Large context exploration |
| **[XML Adapters](https://xiaocui.me/dspy-go/docs/guides/xml-adapters/)** | Structured output parsing |
| **[Tool Management](https://xiaocui.me/dspy-go/docs/guides/tools/)** | Smart registry, chaining, MCP |
| **[Optimizers](https://xiaocui.me/dspy-go/docs/guides/optimizers/)** | GEPA, MIPRO, SIMBA, Bootstrap |

## Examples

### Agent Frameworks
- **[ace_basic](examples/ace_basic/)** - Self-improving agents with ACE
- **[a2a_composition](examples/a2a_composition/)** - Multi-agent deep research
- **[agents](examples/agents/)** - ReAct patterns and orchestration

### Modules
- **[rlm](examples/rlm/)** - Large context exploration
- **[xml_adapter](examples/xml_adapter/)** - XML structured output
- **[parallel](examples/parallel/)** - Batch processing
- **[refine](examples/refine/)** - Quality improvement

### Tools
- **[smart_tool_registry](examples/smart_tool_registry/)** - Intelligent tool selection
- **[tool_chaining](examples/tool_chaining/)** - Pipeline building
- **[tool_composition](examples/tool_composition/)** - Composite tools

### Optimizers
- **[mipro](examples/others/mipro/)** - TPE-based optimization
- **[simba](examples/others/simba/)** - Introspective learning
- **[gepa](examples/others/gepa/)** - Evolutionary optimization

## LLM Providers

```go
// Anthropic Claude
llm, _ := llms.NewAnthropicLLM("api-key", core.ModelAnthropicSonnet)

// Google Gemini
llm, _ := llms.NewGeminiLLM("api-key", core.ModelGoogleGeminiPro)

// OpenAI
llm, _ := llms.NewOpenAI(core.ModelOpenAIGPT4, "api-key")

// Ollama (local)
llm, _ := llms.NewOllamaLLM(core.ModelOllamaLlama3_8B)

// OpenAI-compatible (LiteLLM, LocalAI, etc.)
llm, _ := llms.NewOpenAILLM(core.ModelOpenAIGPT4,
    llms.WithAPIKey("api-key"),
    llms.WithOpenAIBaseURL("http://localhost:4000"))
```

**[Providers Reference](https://xiaocui.me/dspy-go/docs/reference/providers/)**

## Community

- **Documentation**: [xiaocui.me/dspy-go](https://xiaocui.me/dspy-go/)
- **API Reference**: [pkg.go.dev](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go)
- **Example App**: [Maestro](https://github.com/XiaoConstantine/maestro) - Code review agent

## License

DSPy-Go is released under the MIT License. See the [LICENSE](LICENSE) file for details.
