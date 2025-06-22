# DSPy-Go

[![Go Report Card](https://goreportcard.com/badge/github.com/XiaoConstantine/dspy-go)](https://goreportcard.com/report/github.com/XiaoConstantine/dspy-go)
[![codecov](https://codecov.io/gh/XiaoConstantine/dspy-go/graph/badge.svg?token=GGKRLMLXJ9)](https://codecov.io/gh/XiaoConstantine/dspy-go)
[![Go Reference](https://pkg.go.dev/badge/github.com/XiaoConstantine/dspy-go)](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go)

## What is DSPy-Go?

DSPy-Go is a native Go implementation of the DSPy framework, bringing systematic prompt engineering and automated reasoning capabilities to Go applications. It provides a flexible and idiomatic framework for building reliable and effective Language Model (LLM) applications through composable modules and workflows.

### Key Features

- **Modular Architecture**: Build complex LLM applications by composing simple, reusable components
- **Systematic Prompt Engineering**: Optimize prompts automatically based on examples and feedback
- **Flexible Workflows**: Chain, branch, and orchestrate LLM operations with powerful workflow abstractions
- **Multiple LLM Providers**: Support for Anthropic Claude, Google Gemini, Ollama, and LlamaCPP
- **Advanced Reasoning Patterns**: Implement chain-of-thought, ReAct, and other reasoning techniques

## Installation

```go
go get github.com/XiaoConstantine/dspy-go
```

## Quick Start

Here's a simple example to get you started with DSPy-Go:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
    "github.com/XiaoConstantine/dspy-go/pkg/config"
)

func main() {
    // Configure the default LLM
    llms.EnsureFactory()
    err := config.ConfigureDefaultLLM("your-api-key", core.ModelAnthropicSonnet)
    if err != nil {
        log.Fatalf("Failed to configure LLM: %v", err)
    }

    // Create a signature for question answering
    signature := core.NewSignature(
        []core.InputField{{Field: core.Field{Name: "question"}}},
        []core.OutputField{{Field: core.Field{Name: "answer"}}},
    )

    // Create a ChainOfThought module that implements step-by-step reasoning
    cot := modules.NewChainOfThought(signature)

    // Create a program that executes the module
    program := core.NewProgram(
        map[string]core.Module{"cot": cot},
        func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
            return cot.Process(ctx, inputs)
        },
    )

    // Execute the program with a question
    result, err := program.Execute(context.Background(), map[string]interface{}{
        "question": "What is the capital of France?",
    })
    if err != nil {
        log.Fatalf("Error executing program: %v", err)
    }

    fmt.Printf("Answer: %s\n", result["answer"])
}
```

## Core Concepts

DSPy-Go is built around several key concepts that work together to create powerful LLM applications:

### Signatures

Signatures define the input and output fields for modules, creating a clear contract for what a module expects and produces.

```go
// Create a signature for a summarization task
signature := core.NewSignature(
    []core.InputField{
        {Field: core.Field{Name: "document", Description: "The document to summarize"}},
    },
    []core.OutputField{
        {Field: core.Field{Name: "summary", Description: "A concise summary of the document"}},
        {Field: core.Field{Name: "key_points", Description: "The main points from the document"}},
    },
)
```

Signatures can include field descriptions that enhance prompt clarity and improve LLM performance.

### Modules

Modules are the building blocks of DSPy-Go programs. They encapsulate specific functionalities and can be composed to create complex pipelines. Some key modules include:

#### Predict

The simplest module that makes direct predictions using an LLM.

```go
predict := modules.NewPredict(signature)
result, err := predict.Process(ctx, map[string]interface{}{
    "document": "Long document text here...",
})
// result contains "summary" and "key_points"
```

#### ChainOfThought

Implements chain-of-thought reasoning, which guides the LLM to break down complex problems into intermediate steps.

```go
cot := modules.NewChainOfThought(signature)
result, err := cot.Process(ctx, map[string]interface{}{
    "question": "Solve 25 × 16 step by step.",
})
// result contains both the reasoning steps and the final answer
```

#### ReAct

Implements the Reasoning and Acting paradigm, allowing LLMs to use tools to solve problems.

```go
// Create tools
calculator := tools.NewCalculatorTool()
searchTool := tools.NewSearchTool()

// Create ReAct module with tools
react := modules.NewReAct(signature, []core.Tool{calculator, searchTool})
result, err := react.Process(ctx, map[string]interface{}{
    "question": "What is the population of France divided by 1000?",
})
// ReAct will use the search tool to find the population and the calculator to divide it
```

#### MultiChainComparison

Compares multiple reasoning attempts and synthesizes a holistic evaluation, useful for improving decision quality through multiple perspectives.

```go
// Create a signature for problem solving
signature := core.NewSignature(
    []core.InputField{
        {Field: core.NewField("problem", core.WithDescription("The problem to solve"))},
    },
    []core.OutputField{
        {Field: core.NewField("solution", core.WithDescription("The recommended solution"))},
    },
)

// Create MultiChainComparison module with 3 reasoning attempts
multiChain := modules.NewMultiChainComparison(signature, 3, 0.7)

// Provide multiple reasoning attempts for comparison
completions := []map[string]interface{}{
    {
        "rationale": "focus on cost reduction approach",
        "solution": "Implement automation to reduce operational costs",
    },
    {
        "reasoning": "prioritize customer satisfaction strategy", 
        "solution": "Invest in customer service improvements",
    },
    {
        "rationale": "balance short-term and long-term objectives",
        "solution": "Gradual optimization with phased implementation",
    },
}

result, err := multiChain.Process(ctx, map[string]interface{}{
    "problem": "How should we address declining business performance?",
    "completions": completions,
})
// result contains "rationale" with holistic analysis and "solution" with synthesized recommendation
```

### Programs

Programs combine modules into executable workflows. They define how inputs flow through the system and how outputs are produced.

```go
program := core.NewProgram(
    map[string]core.Module{
        "retriever": retriever,
        "generator": generator,
    },
    func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
        // First retrieve relevant documents
        retrieverResult, err := retriever.Process(ctx, inputs)
        if err != nil {
            return nil, err
        }
        
        // Then generate an answer using the retrieved documents
        generatorInputs := map[string]interface{}{
            "question": inputs["question"],
            "documents": retrieverResult["documents"],
        }
        return generator.Process(ctx, generatorInputs)
    },
)
```

### Optimizers

Optimizers help improve the performance of your DSPy-Go programs by automatically tuning prompts and module parameters.

#### BootstrapFewShot

Automatically selects high-quality examples for few-shot learning.

```go
// Create a dataset of examples
dataset := datasets.NewInMemoryDataset()
dataset.AddExample(map[string]interface{}{
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
})
// Add more examples...

// Create and apply the optimizer
optimizer := optimizers.NewBootstrapFewShot(dataset, metrics.NewExactMatchMetric("answer"))
optimizedModule, err := optimizer.Optimize(ctx, originalModule)
```

#### MIPRO (Multi-step Interactive Prompt Optimization)

Advanced optimizer that uses TPE (Tree-structured Parzen Estimator) search for systematic prompt optimization.

```go
// Create a MIPRO optimizer with configuration
mipro := optimizers.NewMIPRO(
    metricFunc,
    optimizers.WithMode(optimizers.LightMode),      // Fast optimization mode
    optimizers.WithNumTrials(10),                   // Number of optimization trials
    optimizers.WithTPEGamma(0.25),                  // TPE exploration parameter
)

optimizedProgram, err := mipro.Compile(ctx, program, dataset, nil)
```

#### SIMBA (Stochastic Introspective Mini-Batch Ascent)

Cutting-edge optimizer with introspective learning capabilities that analyzes its own optimization progress.

```go
// Create a SIMBA optimizer with introspective features
simba := optimizers.NewSIMBA(
    optimizers.WithSIMBABatchSize(8),            // Mini-batch size for stochastic optimization
    optimizers.WithSIMBAMaxSteps(12),            // Maximum optimization steps
    optimizers.WithSIMBANumCandidates(6),        // Candidate programs per iteration
    optimizers.WithSamplingTemperature(0.2),     // Temperature for exploration vs exploitation
)

optimizedProgram, err := simba.Compile(ctx, program, dataset, metricFunc)

// Access SIMBA's introspective insights
state := simba.GetState()
fmt.Printf("Optimization completed in %d steps with score %.3f\n", 
    state.CurrentStep, state.BestScore)

// View detailed introspective analysis
for _, insight := range state.IntrospectionLog {
    fmt.Printf("Analysis: %s\n", insight)
}
```

#### COPRO (Collaborative Prompt Optimization)

Collaborative optimizer for multi-module prompt optimization.

```go
// Create a COPRO optimizer
copro := optimizers.NewCopro(dataset, metrics.NewRougeMetric("answer"))
optimizedModule, err := copro.Optimize(ctx, originalModule)
```

## Agents and Workflows

DSPy-Go provides powerful abstractions for building more complex agent systems.

### Memory

Different memory implementations for tracking conversation history.

```go
// Create a buffer memory for conversation history
memory := memory.NewBufferMemory(10) // Keep last 10 exchanges
memory.Add(context.Background(), "user", "Hello, how can you help me?")
memory.Add(context.Background(), "assistant", "I can answer questions and help with tasks. What do you need?")

// Retrieve conversation history
history, err := memory.Get(context.Background())
```

### Workflows

#### Chain Workflow

Sequential execution of steps:

```go
// Create a chain workflow
workflow := workflows.NewChainWorkflow(store)

// Add steps to the workflow
workflow.AddStep(&workflows.Step{
    ID: "step1",
    Module: modules.NewPredict(signature1),
})

workflow.AddStep(&workflows.Step{
    ID: "step2", 
    Module: modules.NewPredict(signature2),
})

// Execute the workflow
result, err := workflow.Execute(ctx, inputs)
```

#### Configurable Retry Logic

Each workflow step can be configured with retry logic:

```go
step := &workflows.Step{
    ID: "retry_example",
    Module: myModule,
    RetryConfig: &workflows.RetryConfig{
        MaxAttempts: 3,
        BackoffMultiplier: 2.0,
        InitialBackoff: time.Second,
    },
    Condition: func(state map[string]interface{}) bool {
        return someCondition(state)
    },
}
```

### Orchestrator

Flexible task decomposition and execution:

```go
// Create an orchestrator with subtasks
orchestrator := agents.NewOrchestrator()

// Define and add subtasks
researchTask := agents.NewTask("research", researchModule)
summarizeTask := agents.NewTask("summarize", summarizeModule)

orchestrator.AddTask(researchTask)
orchestrator.AddTask(summarizeTask)

// Execute the orchestration
result, err := orchestrator.Execute(ctx, map[string]interface{}{
    "topic": "Climate change impacts",
})
```

## Working with Different LLM Providers

DSPy-Go supports multiple LLM providers out of the box:

```go
// Using Anthropic Claude
llm, err := llms.NewAnthropicLLM("api-key", core.ModelAnthropicSonnet)

// Using Google Gemini
llm, err := llms.NewGeminiLLM("api-key", "gemini-pro")

// Using Ollama (local)
llm, err := llms.NewOllamaLLM("http://localhost:11434", "ollama:llama2")

// Using LlamaCPP (local)
llm, err := llms.NewLlamacppLLM("http://localhost:8080")

// Set as default LLM
llms.SetDefaultLLM(llm)

// Or use with a specific module
myModule.SetLLM(llm)
```

## Advanced Features

### Tracing and Logging

DSPy-Go includes detailed tracing and structured logging for debugging and optimization:

```go
// Enable detailed tracing
ctx = core.WithExecutionState(context.Background())

// Configure logging
logger := logging.NewLogger(logging.Config{
    Severity: logging.DEBUG,
    Outputs:  []logging.Output{logging.NewConsoleOutput(true)},
})
logging.SetLogger(logger)

// After execution, inspect trace
executionState := core.GetExecutionState(ctx)
steps := executionState.GetSteps("moduleId")
for _, step := range steps {
    fmt.Printf("Step: %s, Duration: %s\n", step.Name, step.Duration)
    fmt.Printf("Prompt: %s\n", step.Prompt)
    fmt.Printf("Response: %s\n", step.Response)
}
```

### Custom Tools

You can extend ReAct modules with custom tools:

```go
// Define a custom tool
type WeatherTool struct{}

func (t *WeatherTool) GetName() string {
    return "weather"
}

func (t *WeatherTool) GetDescription() string {
    return "Get the current weather for a location"
}

func (t *WeatherTool) CanHandle(action string) bool {
    return strings.HasPrefix(action, "weather(")
}

func (t *WeatherTool) Execute(ctx context.Context, action string) (string, error) {
    // Parse location from action
    location := parseLocation(action)
    
    // Fetch weather data (implementation detail)
    weather, err := fetchWeather(location)
    if err != nil {
        return "", err
    }
    
    return fmt.Sprintf("Weather in %s: %s, %d°C", location, weather.Condition, weather.Temperature), nil
}

// Use the custom tool with ReAct
react := modules.NewReAct(signature, []core.Tool{&WeatherTool{}})
```

### Streaming Support

Process LLM outputs incrementally as they're generated:

```go
// Create a streaming handler
handler := func(chunk string) {
    fmt.Print(chunk)
}

// Enable streaming on the module
module.SetStreamingHandler(handler)

// Process with streaming enabled
result, err := module.Process(ctx, inputs)
```

## Examples

Check the examples directory for complete implementations:

* [examples/agents](examples/agents): Demonstrates different agent patterns
* [examples/hotpotqa](examples/hotpotqa): Question-answering implementation
* [examples/gsm8k](examples/gsm8k): Math problem solving
* [examples/parallel](examples/parallel): Parallel processing with batch operations
* [examples/refine](examples/refine): Quality improvement through iterative refinement
* [examples/multi_chain_comparison](examples/multi_chain_comparison): Multi-perspective reasoning and decision synthesis
* [examples/others/mipro](examples/others/mipro): MIPRO optimizer demonstration with GSM8K
* [examples/others/simba](examples/others/simba): SIMBA optimizer with introspective learning showcase

## Documentation

For more detailed documentation:

* [GoDoc Reference](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go): Full API documentation
* [Example Apps: Maestro](https://github.com/XiaoConstantine/maestro): A local code review & question answering agent built on top of dspy-go

## License

DSPy-Go is released under the MIT License. See the [LICENSE](LICENSE) file for details.
