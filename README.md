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
- **Multiple LLM Providers**: Support for Anthropic Claude, Google Gemini (with multimodal support), OpenAI, Ollama, LlamaCPP, and OpenAI-compatible APIs (LiteLLM, LocalAI, etc.)
- **Multimodal Processing**: Native support for image analysis, vision Q&A, and multimodal chat with streaming capabilities
- **Advanced Reasoning Patterns**: Implement chain-of-thought, ReAct, refinement, and multi-chain comparison techniques
- **Parallel Processing**: Built-in support for concurrent execution to improve performance
- **Dataset Management**: Automatic downloading and management of popular datasets like GSM8K and HotPotQA
- **Smart Tool Management**: Intelligent tool selection, performance tracking, and auto-discovery from MCP servers
- **Tool Integration**: Native support for custom tools and MCP (Model Context Protocol) servers
- **Tool Chaining**: Sequential execution of tools in pipelines with data transformation and conditional logic
- **Tool Composition**: Create reusable composite tools by combining multiple tools into single units
- **Advanced Parallel Execution**: High-performance parallel tool execution with intelligent scheduling algorithms
- **Dependency Resolution**: Automatic execution planning based on tool dependencies with parallel optimization
- **Quality Optimization**: Advanced optimizers including GEPA (Generative Evolutionary Prompt Adaptation), MIPRO, SIMBA, and BootstrapFewShot for systematic improvement
- **Compatibility Testing**: Comprehensive compatibility testing framework for validating optimizer behavior against Python DSPy implementations

## Installation

```go
go get github.com/XiaoConstantine/dspy-go
```

## 🚀 Quick Start with CLI (Recommended)

**New to DSPy-Go?** Try our interactive CLI tool to explore optimizers instantly:

```bash
# Build and try the CLI
cd cmd/dspy-cli
go build -o dspy-cli

# Set your API key
export GEMINI_API_KEY="your-api-key-here"

# Explore optimizers without writing any code
./dspy-cli list                                    # See all optimizers
./dspy-cli try bootstrap --dataset gsm8k          # Test Bootstrap with math problems
./dspy-cli try mipro --dataset gsm8k --verbose    # Advanced systematic optimization
./dspy-cli recommend balanced                      # Get optimizer recommendations
```

The CLI eliminates 60+ lines of boilerplate code and lets you test all optimizers (Bootstrap, MIPRO, SIMBA, GEPA, COPRO) with sample datasets instantly. **[→ Full CLI Documentation](cmd/dspy-cli/README.md)**

## Programming Quick Start

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
    err := core.ConfigureDefaultLLM("your-api-key", core.ModelAnthropicSonnet)
    if err != nil {
        log.Fatalf("Failed to configure LLM: %v", err)
    }

    // Create a signature for question answering
    signature := core.NewSignature(
        []core.InputField{{Field: core.NewField("question", core.WithDescription("Question to answer"))}},
        []core.OutputField{{Field: core.NewField("answer", core.WithDescription("Answer to the question"))}},
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
        {Field: core.NewField("document", core.WithDescription("The document to summarize"))},
    },
    []core.OutputField{
        {Field: core.NewField("summary", core.WithDescription("A concise summary of the document"))},
        {Field: core.NewField("key_points", core.WithDescription("The main points from the document"))},
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

#### Refine

Improves prediction quality by running multiple attempts with different temperatures and selecting the best result based on a reward function.

```go
// Define a reward function that evaluates prediction quality
rewardFn := func(inputs, outputs map[string]interface{}) float64 {
    // Custom logic to evaluate the quality of the prediction
    // Return a score between 0.0 and 1.0
    answer := outputs["answer"].(string)
    // Example: longer answers might be better for some tasks
    return math.Min(1.0, float64(len(answer))/100.0)
}

// Create a Refine module
refine := modules.NewRefine(
    modules.NewPredict(signature),
    modules.RefineConfig{
        N:         5,       // Number of refinement attempts
        RewardFn:  rewardFn,
        Threshold: 0.8,     // Stop early if this threshold is reached
    },
)

result, err := refine.Process(ctx, map[string]interface{}{
    "question": "Explain quantum computing in detail.",
})
// Refine will try multiple times with different temperatures and return the best result
```

#### Parallel

Wraps any module to enable concurrent execution across multiple inputs, providing significant performance improvements for batch processing.

```go
// Create any module (e.g., Predict, ChainOfThought, etc.)
baseModule := modules.NewPredict(signature)

// Wrap it with Parallel for concurrent execution
parallel := modules.NewParallel(baseModule,
    modules.WithMaxWorkers(4),              // Use 4 concurrent workers
    modules.WithReturnFailures(true),       // Include failed results in output
    modules.WithStopOnFirstError(false),   // Continue processing even if some fail
)

// Prepare batch inputs
batchInputs := []map[string]interface{}{
    {"question": "What is 2+2?"},
    {"question": "What is 3+3?"},
    {"question": "What is 4+4?"},
}

// Process all inputs in parallel
result, err := parallel.Process(ctx, map[string]interface{}{
    "batch_inputs": batchInputs,
})

// Extract results (maintains input order)
results := result["results"].([]map[string]interface{})
for i, res := range results {
    if res != nil {
        fmt.Printf("Input %d: %s\n", i, res["answer"])
    }
}
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

#### GEPA (Generative Evolutionary Prompt Adaptation)

State-of-the-art evolutionary optimizer that combines multi-objective Pareto optimization with LLM-based self-reflection for comprehensive prompt evolution.

```go
// Create GEPA optimizer with advanced configuration
config := &optimizers.GEPAConfig{
    PopulationSize:    20,                // Size of evolutionary population
    MaxGenerations:    10,                // Maximum number of generations
    SelectionStrategy: "adaptive_pareto", // Multi-objective Pareto selection
    MutationRate:      0.3,               // Probability of mutation
    CrossoverRate:     0.7,               // Probability of crossover
    ReflectionFreq:    2,                 // LLM-based reflection every 2 generations
    ElitismRate:       0.1,               // Elite solution preservation rate
}

gepa, err := optimizers.NewGEPA(config)
if err != nil {
    log.Fatalf("Failed to create GEPA optimizer: %v", err)
}

// Optimize program with multi-objective evolutionary approach
optimizedProgram, err := gepa.Compile(ctx, program, dataset, metricFunc)

// Access comprehensive optimization results
state := gepa.GetOptimizationState()
fmt.Printf("Optimization completed in %d generations\n", state.CurrentGeneration)
fmt.Printf("Best fitness: %.3f\n", state.BestFitness)

// Access Pareto archive of elite solutions optimized for different trade-offs
archive := state.GetParetoArchive()
fmt.Printf("Elite solutions preserved: %d\n", len(archive))

// Each solution in archive excels in different objectives:
// - Success rate vs efficiency trade-offs
// - Quality vs speed optimizations
// - Robustness vs generalization balance
```

**GEPA Key Features:**

- **Multi-Objective Optimization**: Optimizes across 7 dimensions (success rate, quality, efficiency, robustness, generalization, diversity, innovation)
- **Pareto-Based Selection**: Maintains diverse solutions optimized for different trade-offs
- **LLM-Based Self-Reflection**: Uses language models to analyze and critique prompt performance
- **Semantic Diversity Metrics**: Employs LLM-based similarity for true semantic diversity assessment
- **Elite Archive Management**: Preserves high-quality solutions across generations with crowding distance
- **Real-Time System Monitoring**: Context-aware performance tracking and adaptive parameter adjustment
- **Advanced Genetic Operators**: Semantic crossover and mutation using LLMs for meaningful prompt evolution

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

DSPy-Go supports multiple LLM providers with flexible configuration options:

```go
// Using Anthropic Claude
llm, err := llms.NewAnthropicLLM("api-key", core.ModelAnthropicSonnet)

// Using Google Gemini
llm, err := llms.NewGeminiLLM("api-key", "gemini-pro")

// Using OpenAI (standard)
llm, err := llms.NewOpenAI(core.ModelOpenAIGPT4, "api-key")

// Using LiteLLM (OpenAI-compatible)
llm, err := llms.NewOpenAILLM(core.ModelOpenAIGPT4,
    llms.WithAPIKey("api-key"),
    llms.WithOpenAIBaseURL("http://localhost:4000"))

// Using LocalAI (OpenAI-compatible)
llm, err := llms.NewOpenAILLM(core.ModelOpenAIGPT4,
    llms.WithAPIKey("api-key"),
    llms.WithOpenAIBaseURL("http://localhost:8080"),
    llms.WithOpenAIPath("/v1/chat/completions"))

// Using custom OpenAI-compatible API
llm, err := llms.NewOpenAILLM(core.ModelOpenAIGPT4,
    llms.WithAPIKey("custom-key"),
    llms.WithOpenAIBaseURL("https://api.custom-provider.com"),
    llms.WithHeader("Custom-Header", "value"),
    llms.WithOpenAITimeout(30*time.Second))

// Using Ollama (local)
llm, err := llms.NewOllamaLLM("http://localhost:11434", "ollama:llama2")

// Using LlamaCPP (local)
llm, err := llms.NewLlamacppLLM("http://localhost:8080")

// Set as default LLM
llms.SetDefaultLLM(llm)

// Or use with a specific module
myModule.SetLLM(llm)
```

### OpenAI-Compatible APIs

DSPy-Go's OpenAI provider supports any OpenAI-compatible API through functional options, making it easy to work with various providers:

#### LiteLLM
[LiteLLM](https://docs.litellm.ai/) provides a unified interface to 100+ LLMs:

```go
// Basic LiteLLM setup
llm, err := llms.NewOpenAILLM(core.ModelOpenAIGPT4,
    llms.WithAPIKey("your-api-key"),
    llms.WithOpenAIBaseURL("http://localhost:4000"))

// LiteLLM with custom configuration
llm, err := llms.NewOpenAILLM(core.ModelOpenAIGPT4,
    llms.WithAPIKey("litellm-key"),
    llms.WithOpenAIBaseURL("https://your-litellm-instance.com"),
    llms.WithOpenAITimeout(60*time.Second),
    llms.WithHeader("X-Custom-Header", "value"))
```

#### LocalAI
[LocalAI](https://localai.io/) for running local models:

```go
llm, err := llms.NewOpenAILLM(core.ModelOpenAIGPT4,
    llms.WithOpenAIBaseURL("http://localhost:8080"),
    llms.WithOpenAIPath("/v1/chat/completions"))
```

#### FastChat
[FastChat](https://github.com/lm-sys/FastChat) for serving local models:

```go
llm, err := llms.NewOpenAILLM(core.ModelOpenAIGPT4,
    llms.WithOpenAIBaseURL("http://localhost:8000"),
    llms.WithOpenAIPath("/v1/chat/completions"))
```

#### Azure OpenAI
Configure for Azure OpenAI Service:

```go
llm, err := llms.NewOpenAILLM(core.ModelOpenAIGPT4,
    llms.WithAPIKey("azure-api-key"),
    llms.WithOpenAIBaseURL("https://your-resource.openai.azure.com"),
    llms.WithOpenAIPath("/openai/deployments/your-deployment/chat/completions"),
    llms.WithHeader("api-version", "2024-02-15-preview"))
```

#### Custom Providers
Any API that follows the OpenAI chat completions format:

```go
llm, err := llms.NewOpenAILLM(core.ModelOpenAIGPT4,
    llms.WithAPIKey("custom-key"),
    llms.WithOpenAIBaseURL("https://api.custom-provider.com"),
    llms.WithOpenAIPath("/v1/chat/completions"),
    llms.WithOpenAITimeout(30*time.Second),
    llms.WithHeader("Authorization", "Bearer custom-token"),
    llms.WithHeader("X-API-Version", "v1"))
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

### Smart Tool Registry

DSPy-Go includes an intelligent tool management system that uses Bayesian inference for optimal tool selection:

```go
import "github.com/XiaoConstantine/dspy-go/pkg/tools"

// Create intelligent tool registry
config := &tools.SmartToolRegistryConfig{
    AutoDiscoveryEnabled:       true,  // Auto-discover from MCP servers
    PerformanceTrackingEnabled: true,  // Track tool performance metrics
    FallbackEnabled:           true,   // Intelligent fallback selection
}
registry := tools.NewSmartToolRegistry(config)

// Register tools
registry.Register(mySearchTool)
registry.Register(myAnalysisTool)

// Intelligent tool selection based on intent
tool, err := registry.SelectBest(ctx, "find user information")
if err != nil {
    log.Fatal(err)
}

// Execute with performance tracking
result, err := registry.ExecuteWithTracking(ctx, tool.Name(), params)
```

**Key Features:**
- 🧠 **Bayesian Tool Selection**: Multi-factor scoring with configurable weights
- 📊 **Performance Tracking**: Real-time metrics and reliability scoring
- 🔍 **Capability Analysis**: Automatic capability extraction and matching
- 🔄 **Auto-Discovery**: Dynamic tool registration from MCP servers
- 🛡️ **Fallback Mechanisms**: Intelligent fallback when tools fail

### Tool Chaining and Composition

DSPy-Go provides powerful capabilities for chaining and composing tools to build complex workflows:

#### Tool Chaining

Create sequential pipelines with data transformation and conditional execution:

```go
import "github.com/XiaoConstantine/dspy-go/pkg/tools"

// Create a tool pipeline with fluent API
pipeline, err := tools.NewPipelineBuilder("data_processing", registry).
    Step("data_extractor").
    StepWithTransformer("data_validator", tools.TransformExtractField("result")).
    ConditionalStep("data_enricher",
        tools.ConditionExists("validation_result"),
        tools.ConditionEquals("status", "validated")).
    StepWithRetries("data_transformer", 3).
    FailFast().
    EnableCaching().
    Build()

// Execute the pipeline
result, err := pipeline.Execute(ctx, map[string]interface{}{
    "raw_data": "input data to process",
})
```

#### Data Transformations

Transform data between pipeline steps:

```go
// Extract specific fields
transformer := tools.TransformExtractField("important_field")

// Rename fields
transformer := tools.TransformRename(map[string]string{
    "old_name": "new_name",
})

// Chain multiple transformations
transformer := tools.TransformChain(
    tools.TransformRename(map[string]string{"status": "processing_status"}),
    tools.TransformAddConstant(map[string]interface{}{"pipeline_id": "001"}),
    tools.TransformFilter([]string{"result", "pipeline_id", "processing_status"}),
)
```

#### Dependency Resolution

Automatic execution planning with parallel optimization:

```go
// Create dependency graph
graph := tools.NewDependencyGraph()

// Define tool dependencies
graph.AddNode(&tools.DependencyNode{
    ToolName:     "data_extractor",
    Dependencies: []string{},
    Outputs:      []string{"raw_data"},
    Priority:     1,
})

graph.AddNode(&tools.DependencyNode{
    ToolName:     "data_validator",
    Dependencies: []string{"data_extractor"},
    Inputs:       []string{"raw_data"},
    Outputs:      []string{"validated_data"},
    Priority:     2,
})

// Create dependency-aware pipeline
depPipeline, err := tools.NewDependencyPipeline("smart_pipeline", registry, graph, options)

// Execute with automatic parallelization
result, err := depPipeline.ExecuteWithDependencies(ctx, input)
```

#### Parallel Execution

High-performance parallel tool execution with advanced scheduling:

```go
// Create parallel executor
executor := tools.NewParallelExecutor(registry, 4) // 4 workers

// Define parallel tasks
tasks := []*tools.ParallelTask{
    {
        ID:       "task1",
        ToolName: "analyzer",
        Input:    data1,
        Priority: 1,
    },
    {
        ID:       "task2",
        ToolName: "processor",
        Input:    data2,
        Priority: 2,
    },
}

// Execute with priority scheduling
results, err := executor.ExecuteParallel(ctx, tasks, &tools.PriorityScheduler{})

// Or use fair share scheduling
results, err := executor.ExecuteParallel(ctx, tasks, tools.NewFairShareScheduler())
```

#### Tool Composition

Create reusable composite tools by combining multiple tools:

```go
// Define a composite tool
type CompositeTool struct {
    name     string
    pipeline *tools.ToolPipeline
}

// Helper function to create composite tools
func NewCompositeTool(name string, registry core.ToolRegistry,
    builder func(*tools.PipelineBuilder) *tools.PipelineBuilder) (*CompositeTool, error) {

    pipeline, err := builder(tools.NewPipelineBuilder(name+"_pipeline", registry)).Build()
    if err != nil {
        return nil, err
    }

    return &CompositeTool{
        name:     name,
        pipeline: pipeline,
    }, nil
}

// Create a composite tool
textProcessor, err := NewCompositeTool("text_processor", registry,
    func(builder *tools.PipelineBuilder) *tools.PipelineBuilder {
        return builder.
            Step("text_uppercase").
            Step("text_reverse").
            Step("text_length")
    })

// Register and use like any other tool
registry.Register(textProcessor)
result, err := textProcessor.Execute(ctx, input)

// Use in other pipelines or compositions
complexPipeline, err := tools.NewPipelineBuilder("complex", registry).
    Step("text_processor").  // Using our composite tool
    Step("final_formatter").
    Build()
```

**Key Features:**
- 🔗 **Sequential Chaining**: Build complex workflows by chaining tools together
- 🔄 **Data Transformation**: Transform data between steps with built-in transformers
- ⚡ **Conditional Execution**: Execute steps based on previous results
- 🕸️ **Dependency Resolution**: Automatic execution planning with topological sorting
- 🚀 **Parallel Optimization**: Execute independent tools concurrently for performance
- 🧩 **Tool Composition**: Create reusable composite tools as building blocks
- 💾 **Result Caching**: Cache intermediate results for improved performance
- 🔄 **Retry Logic**: Configurable retry mechanisms for reliability
- 📊 **Performance Tracking**: Monitor execution metrics and worker utilization

### MCP (Model Context Protocol) Integration

DSPy-Go supports integration with MCP servers for accessing external tools and services:

```go
import (
    "github.com/XiaoConstantine/dspy-go/pkg/tools"
    "github.com/XiaoConstantine/mcp-go/pkg/client"
)

// Connect to an MCP server
mcpClient, err := client.NewStdioClient("path/to/mcp-server")
if err != nil {
    log.Fatal(err)
}

// Create MCP tools from server capabilities
mcpTools, err := tools.CreateMCPToolsFromServer(mcpClient)
if err != nil {
    log.Fatal(err)
}

// Use MCP tools with ReAct
react := modules.NewReAct(signature, mcpTools)

// Or use with Smart Tool Registry for intelligent selection
registry.Register(mcpTools...)
selectedTool, err := registry.SelectBest(ctx, "analyze financial data")
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

### Dataset Management

DSPy-Go provides built-in support for downloading and managing common datasets:

```go
import "github.com/XiaoConstantine/dspy-go/pkg/datasets"

// Automatically download and get path to GSM8K dataset
gsm8kPath, err := datasets.EnsureDataset("gsm8k")
if err != nil {
    log.Fatal(err)
}

// Load GSM8K dataset
gsm8kDataset, err := datasets.LoadGSM8K(gsm8kPath)
if err != nil {
    log.Fatal(err)
}

// Similarly for HotPotQA dataset
hotpotPath, err := datasets.EnsureDataset("hotpotqa")
if err != nil {
    log.Fatal(err)
}

hotpotDataset, err := datasets.LoadHotPotQA(hotpotPath)
if err != nil {
    log.Fatal(err)
}

// Use datasets with optimizers
optimizer := optimizers.NewBootstrapFewShot(gsm8kDataset, metricFunc)
optimizedModule, err := optimizer.Optimize(ctx, module)
```

## Compatibility Testing

DSPy-Go includes a comprehensive compatibility testing framework to ensure that optimizer implementations match the behavior of Python DSPy:

```bash
cd compatibility_test

# Test all optimizers (BootstrapFewShot, MIPRO, SIMBA, GEPA)
./run_experiment.sh

# Test specific optimizer
./run_experiment.sh --optimizer gepa --dataset-size 10

# Manual execution
python dspy_comparison.py --optimizer gepa
go run go_comparison.go --optimizer gepa
python compare_results.py
```

The framework tests:
- **API Compatibility**: Parameter names, types, and behavior
- **Performance Parity**: Score differences within acceptable thresholds
- **Behavioral Consistency**: Similar optimization patterns and convergence
- **All Four Optimizers**: BootstrapFewShot, MIPRO, SIMBA, and GEPA

Results are saved as JSON files with detailed compatibility analysis and recommendations.

## Examples

### 🎯 CLI Tool (Zero Code Required)

**[DSPy-CLI](cmd/dspy-cli/README.md)**: Interactive command-line tool for exploring optimizers without writing code. Perfect for getting started, testing optimizers, and rapid experimentation.

```bash
cd cmd/dspy-cli && go build -o dspy-cli
./dspy-cli try mipro --dataset gsm8k --max-examples 5 --verbose
```

### 📚 Code Examples

Check the examples directory for complete implementations:

* **[examples/smart_tool_registry](examples/smart_tool_registry)**: Intelligent tool management with Bayesian selection
* **[examples/tool_chaining](examples/tool_chaining)**: Sequential tool execution with data transformation, conditional logic, dependency resolution, and parallel execution
* **[examples/tool_composition](examples/tool_composition)**: Creating reusable composite tools by combining multiple tools into single units
* [examples/agents](examples/agents): Demonstrates different agent patterns
* [examples/hotpotqa](examples/hotpotqa): Question-answering implementation
* [examples/gsm8k](examples/gsm8k): Math problem solving
* [examples/parallel](examples/parallel): Parallel processing with batch operations
* [examples/refine](examples/refine): Quality improvement through iterative refinement
* [examples/multi_chain_comparison](examples/multi_chain_comparison): Multi-perspective reasoning and decision synthesis
* [examples/others/mipro](examples/others/mipro): MIPRO optimizer demonstration with GSM8K
* [examples/others/simba](examples/others/simba): SIMBA optimizer with introspective learning showcase
* [examples/others/gepa](examples/others/gepa): GEPA evolutionary optimizer with multi-objective Pareto optimization

### Smart Tool Registry Examples

```bash
# Run basic Smart Tool Registry example
cd examples/smart_tool_registry
go run main.go

# Run advanced features demonstration
# Edit advanced_example.go to uncomment main() function
go run advanced_example.go
```

The Smart Tool Registry examples demonstrate:
- Intelligent tool selection using Bayesian inference
- Performance tracking and metrics collection
- Auto-discovery from MCP servers
- Capability analysis and matching
- Fallback mechanisms and error handling
- Custom selector configuration

### Tool Chaining and Composition Examples

```bash
# Run tool chaining examples
cd examples/tool_chaining
go run main.go

# Run tool composition examples
cd examples/tool_composition
go run main.go
```

The Tool Chaining and Composition examples demonstrate:

**Tool Chaining:**
- Sequential pipeline execution with fluent API
- Data transformation between pipeline steps
- Conditional step execution based on previous results
- Dependency-aware execution with automatic parallelization
- Advanced parallel tool execution with intelligent scheduling
- Batch processing with fair share and priority scheduling
- Result caching and performance optimization
- Comprehensive error handling and retry mechanisms

**Tool Composition:**
- Creating reusable composite tools by combining multiple tools
- Nested composition (composites using other composites)
- Using composite tools as building blocks in larger pipelines
- Tool composition with data transformations
- Registry integration for seamless tool management

### Multimodal Processing Example

```bash
# Run multimodal example (requires GEMINI_API_KEY)
cd examples/multimodal
export GEMINI_API_KEY="your-api-key-here"
go run main.go
```

The Multimodal Processing example demonstrates:
- **Image Analysis**: Basic image analysis with natural language questions
- **Vision Question Answering**: Structured visual analysis with detailed observations
- **Multimodal Chat**: Interactive conversations with images
- **Streaming Multimodal**: Real-time processing of multimodal content
- **Multiple Image Analysis**: Comparing and analyzing multiple images simultaneously
- **Content Block System**: Flexible handling of text, image, and future audio content

## Documentation

For more detailed documentation:

* [GoDoc Reference](https://pkg.go.dev/github.com/XiaoConstantine/dspy-go): Full API documentation
* [Example Apps: Maestro](https://github.com/XiaoConstantine/maestro): A local code review & question answering agent built on top of dspy-go

## License

DSPy-Go is released under the MIT License. See the [LICENSE](LICENSE) file for details.
