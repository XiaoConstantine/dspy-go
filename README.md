dspy-go
-------
[![Go Report Card](https://goreportcard.com/badge/github.com/XiaoConstantine/dspy-go)](https://goreportcard.com/report/github.com/XiaoConstantine/dspy-go)
[![codecov](https://codecov.io/gh/XiaoConstantine/dspy-go/graph/badge.svg?token=GGKRLMLXJ9)](https://codecov.io/gh/XiaoConstantine/dspy-go)

DSPy-Go is a Go implementation of DSPy, bringing systematic prompt engineering and automated reasoning capabilities to Go applications. It provides a flexible framework for building reliable and effective Language Model (LLM) applications through composable modules and workflows.


### Installation
```go
go get github.com/XiaoConstantine/dspy-go
```

### Quick Start

Here's a simple example to get you started with DSPy-Go:

```go
import (
    "context"
    "fmt"
    "log"

    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
    "github.com/XiaoConstantine/dspy-go/pkg/config"
)

func main() {
    // Configure the default LLM
    err := config.ConfigureDefaultLLM("your-api-key", core.ModelAnthropicSonnet)
    if err != nil {
        log.Fatalf("Failed to configure LLM: %v", err)
    }

    // Create a signature for question answering
    signature := core.NewSignature(
        []core.InputField{{Field: core.Field{Name: "question"}}},
        []core.OutputField{{Field: core.Field{Name: "answer"}}},
    )

    // Create a ChainOfThought module
    cot := modules.NewChainOfThought(signature)

    // Create a program
    program := core.NewProgram(
        map[string]core.Module{"cot": cot},
        func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
            return cot.Process(ctx, inputs)
        },
    )

    // Execute the program
    result, err := program.Execute(context.Background(), map[string]interface{}{
        "question": "What is the capital of France?",
    })
    if err != nil {
        log.Fatalf("Error executing program: %v", err)
    }

    fmt.Printf("Answer: %s\n", result["answer"])
}
```

### Core Concepts

#### Signatures
Signatures define the input and output fields for modules. They help in creating type-safe and well-defined interfaces for your AI components.

#### Modules
Modules are the building blocks of DSPy-Go programs. They encapsulate specific functionalities and can be composed to create complex pipelines. Some key modules include:

* Predict: Basic prediction module
* ChainOfThought: Implements chain-of-thought reasoning
* ReAct: Implements the ReAct (Reasoning and Acting) paradigm


#### Optimizers
Optimizers help improve the performance of your DSPy-Go programs by automatically tuning prompts and module parameters. Including:
* BootstrapFewShot: Automatic few-shot example selection
* MIPRO: Multi-step interactive prompt optimization
* Copro: Collaborative prompt optimization


#### Agents
Use dspy's core concepts as building blocks, impl [Building Effective Agents](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents)


* Chain Workflow: Sequential execution of steps
* Parallel Workflow: Concurrent execution with controlled parallelism
* Router Workflow: Dynamic routing based on classification
* Orchestrator: Flexible task decomposition and execution

See [agent examples](/examples/agents/main.go)


```go
// Chain
workflow := workflows.NewChainWorkflow(store)
workflow.AddStep(&workflows.Step{
    ID: "step1",
    Module: modules.NewPredict(signature1),
})
workflow.AddStep(&workflows.Step{
    ID: "step2", 
    Module: modules.NewPredict(signature2),
})
```
Each workflow step can be configured with:
* Retry logic with exponential backoff
* Conditional execution based on workflow state
* Custom error handling

```go

step := &workflows.Step{
    ID: "retry_example",
    Module: myModule,
    RetryConfig: &workflows.RetryConfig{
        MaxAttempts: 3,
        BackoffMultiplier: 2.0,
    },
    Condition: func(state map[string]interface{}) bool {
        return someCondition(state)
    },
}
```

### Advanced Features

#### Tracing and Logging
```go
// Enable detailed tracing
ctx = core.WithExecutionState(context.Background())

// Configure logging
logger := logging.NewLogger(logging.Config{
    Severity: logging.DEBUG,
    Outputs:  []logging.Output{logging.NewConsoleOutput(true)},
})
logging.SetLogger(logger)
```

#### Custom Tools
You can extend ReAct modules with custom tools:
```go

func (t *CustomTool) CanHandle(action string) bool {
    return strings.HasPrefix(action, "custom_")
}

func (t *CustomTool) Execute(ctx context.Context, action string) (string, error) {
    // Implement tool logic
    return "Tool result", nil
}
```

#### Working with Different LLM Providers
```go
// Using Anthropic Claude
llm, _ := llms.NewAnthropicLLM("api-key", anthropic.ModelSonnet)

// Using Ollama
llm, _ := llms.NewOllamaLLM("http://localhost:11434", "llama2")

// Using LlamaCPP
llm, _ := llms.NewLlamacppLLM("http://localhost:8080")
```


### Examples
Check the examples directory for complete implementations:

* examples/agents: Demonstrates different agent patterns
* examples/hotpotqa: Question-answering implementation
* examples/gsm8k: Math problem solving


### License
DSPy-Go is released under the MIT License. See the LICENSE file for details.
