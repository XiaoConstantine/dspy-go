---
title: "Building Agents"
description: "ReAct patterns, orchestration, and memory management"
summary: "Create intelligent agents with reasoning, tool use, and conversation memory"
date: 2025-10-13T00:00:00+00:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 500
toc: true
seo:
  title: "Building Agents - dspy-go"
  description: "Complete guide to building agents with ReAct, orchestration, and memory in dspy-go"
  canonical: ""
  noindex: false
---

# Building Agents

dspy-go's **agent package** provides powerful abstractions for building intelligent agents that can reason, use tools, maintain conversation history, and orchestrate complex workflows.

## Agent Architecture

An agent in dspy-go combines:
- **ReAct Module**: Reasoning + Acting pattern
- **Tool Registry**: Available tools the agent can use
- **Memory**: Conversation history and context
- **Orchestrator**: Task decomposition and coordination

---

## ReAct Pattern

**Reasoning and Acting** - The foundation of intelligent agents.

### What is ReAct?

ReAct combines:
1. **Thought**: The agent reasons about what to do
2. **Action**: The agent uses a tool
3. **Observation**: The agent sees the tool's result
4. **Repeat**: Until the task is complete

### Basic ReAct Agent

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
    "github.com/XiaoConstantine/dspy-go/pkg/tools"
)

func main() {
    // Configure LLM
    llm, _ := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
    core.SetDefaultLLM(llm)

    // Create tools
    calculator := tools.NewCalculatorTool()
    searchTool := tools.NewSearchTool()
    weatherTool := tools.NewWeatherTool()

    // Create tool registry
    registry := tools.NewInMemoryToolRegistry()
    registry.Register(calculator)
    registry.Register(searchTool)
    registry.Register(weatherTool)

    // Define signature for the agent's task
    signature := core.NewSignature(
        []core.InputField{
            {Field: core.NewField("question",
                core.WithDescription("The question to answer"))},
        },
        []core.OutputField{
            {Field: core.NewField("answer",
                core.WithDescription("The final answer"))},
        },
    )

    // Create ReAct module
    react := modules.NewReAct(
        signature,
        registry,
        5, // max iterations
    )

    // Execute
    ctx := context.Background()
    result, err := react.Process(ctx, map[string]interface{}{
        "question": "What is the population of Tokyo divided by 1000?",
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Answer: %s\n", result["answer"])
}
```

### How ReAct Works

Given the question "What is the population of Tokyo divided by 1000?":

**Iteration 1:**
- **Thought**: "I need to find the population of Tokyo"
- **Action**: `search("Tokyo population")`
- **Observation**: "Tokyo has a population of approximately 14 million"

**Iteration 2:**
- **Thought**: "Now I need to divide 14,000,000 by 1000"
- **Action**: `calculator("14000000 / 1000")`
- **Observation**: "14000"

**Iteration 3:**
- **Thought**: "I have the answer"
- **Action**: `finish("14,000")`

---

## Custom Tools

**Extend agents** with domain-specific tools.

### Creating a Custom Tool

```go
package main

import (
    "context"
    "fmt"
    "strings"
)

// Custom Weather Tool
type WeatherTool struct{}

func (t *WeatherTool) GetName() string {
    return "weather"
}

func (t *WeatherTool) GetDescription() string {
    return "Get the current weather for a location. Usage: weather(location)"
}

func (t *WeatherTool) CanHandle(action string) bool {
    return strings.HasPrefix(action, "weather(")
}

func (t *WeatherTool) Execute(ctx context.Context, action string) (string, error) {
    // Parse location from action string
    location := parseLocation(action)

    // Fetch weather data (your implementation)
    weather, err := fetchWeather(location)
    if err != nil {
        return "", err
    }

    return fmt.Sprintf("Weather in %s: %s, %d°C",
        location, weather.Condition, weather.Temperature), nil
}

func parseLocation(action string) string {
    // Extract "Paris" from "weather(Paris)"
    start := strings.Index(action, "(") + 1
    end := strings.Index(action, ")")
    return action[start:end]
}
```

### Using Custom Tools

```go
// Register custom tool
registry := tools.NewInMemoryToolRegistry()
registry.Register(&WeatherTool{})
registry.Register(&DatabaseTool{})
registry.Register(&EmailTool{})

// Create ReAct agent with custom tools
react := modules.NewReAct(signature, registry, 10)
```

**[Full Agents Example →](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/agents)**

---

## Memory Management

**Conversation history** and context tracking.

### Buffer Memory

Store recent conversation turns:

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/agents/memory"
)

func main() {
    // Create buffer memory (keeps last 10 exchanges)
    mem := memory.NewBufferMemory(10)

    ctx := context.Background()

    // Add messages
    mem.Add(ctx, "user", "Hello, how can you help me?")
    mem.Add(ctx, "assistant", "I can answer questions and help with tasks.")
    mem.Add(ctx, "user", "What's the weather in Paris?")
    mem.Add(ctx, "assistant", "The weather in Paris is sunny, 22°C.")

    // Retrieve conversation history
    history, err := mem.Get(ctx)
    if err != nil {
        log.Fatal(err)
    }

    // Use history in prompts
    for _, msg := range history {
        fmt.Printf("%s: %s\n", msg.Role, msg.Content)
    }
}
```

### Summary Memory

**Compress long conversations** into summaries:

```go
// Create summary memory
summaryMem := memory.NewSummaryMemory(
    core.GetDefaultLLM(),
    100, // summarize after 100 messages
)

// Add messages (automatically summarizes when threshold is reached)
summaryMem.Add(ctx, "user", "Long conversation...")
summaryMem.Add(ctx, "assistant", "Response...")

// Get summarized history
history, err := summaryMem.Get(ctx)
```

### Using Memory with Agents

```go
// Create agent with memory
mem := memory.NewBufferMemory(20)

// In your agent loop
for {
    userInput := getUserInput()

    // Add user message to memory
    mem.Add(ctx, "user", userInput)

    // Get conversation history
    history, _ := mem.Get(ctx)

    // Include history in agent context
    result, err := react.Process(ctx, map[string]interface{}{
        "question": userInput,
        "history": history,
    })

    // Add assistant response to memory
    mem.Add(ctx, "assistant", result["answer"].(string))

    fmt.Println(result["answer"])
}
```

---

## Orchestrator

**Task decomposition** and multi-agent coordination.

### Basic Orchestrator

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/agents"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func main() {
    // Create orchestrator
    orchestrator := agents.NewOrchestrator()

    // Define subtasks with different modules
    researchTask := agents.NewTask("research", researchModule)
    analyzeTask := agents.NewTask("analyze", analyzeModule)
    summarizeTask := agents.NewTask("summarize", summarizeModule)

    // Add tasks to orchestrator
    orchestrator.AddTask(researchTask)
    orchestrator.AddTask(analyzeTask)
    orchestrator.AddTask(summarizeTask)

    // Execute orchestration
    ctx := context.Background()
    result, err := orchestrator.Execute(ctx, map[string]interface{}{
        "topic": "Impact of AI on healthcare",
    })

    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Final Result: %v\n", result)
}
```

### Task Dependencies

```go
// Create tasks with dependencies
task1 := agents.NewTask("fetch_data", fetchModule)
task2 := agents.NewTask("process_data", processModule)
task3 := agents.NewTask("analyze", analyzeModule)

// Set dependencies (task2 depends on task1, task3 depends on task2)
task2.DependsOn(task1)
task3.DependsOn(task2)

// Orchestrator automatically handles execution order
orchestrator.AddTask(task1)
orchestrator.AddTask(task2)
orchestrator.AddTask(task3)
```

---

## Advanced Agent Patterns

### Multi-Agent System

**Specialized agents** working together:

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/agents"
)

type MultiAgentSystem struct {
    researcher  *Agent
    analyst     *Agent
    writer      *Agent
    coordinator *agents.Orchestrator
}

func NewMultiAgentSystem() *MultiAgentSystem {
    return &MultiAgentSystem{
        researcher:  NewResearchAgent(),
        analyst:     NewAnalystAgent(),
        writer:      NewWriterAgent(),
        coordinator: agents.NewOrchestrator(),
    }
}

func (m *MultiAgentSystem) Execute(ctx context.Context, task string) (string, error) {
    // 1. Research phase
    researchTask := agents.NewTask("research", m.researcher.module)
    research, err := researchTask.Execute(ctx, map[string]interface{}{
        "task": task,
    })

    // 2. Analysis phase
    analysisTask := agents.NewTask("analyze", m.analyst.module)
    analysis, err := analysisTask.Execute(ctx, map[string]interface{}{
        "research": research,
    })

    // 3. Writing phase
    writingTask := agents.NewTask("write", m.writer.module)
    result, err := writingTask.Execute(ctx, map[string]interface{}{
        "analysis": analysis,
    })

    return result["output"].(string), nil
}
```

### Agent with Reflection

**Self-improvement** through reflection:

```go
type ReflectiveAgent struct {
    react      *modules.ReAct
    memory     memory.Memory
    reflection *modules.ChainOfThought
}

func (a *ReflectiveAgent) ExecuteWithReflection(ctx context.Context, task string) (string, error) {
    // 1. Execute task
    result, err := a.react.Process(ctx, map[string]interface{}{
        "question": task,
    })

    // 2. Reflect on performance
    reflection, err := a.reflection.Process(ctx, map[string]interface{}{
        "action": "reflect on previous attempt",
        "result": result,
        "task": task,
    })

    // 3. Store reflection for future improvement
    a.memory.Add(ctx, "reflection", reflection["rationale"].(string))

    // 4. Retry if needed based on reflection
    if shouldRetry(reflection) {
        return a.react.Process(ctx, map[string]interface{}{
            "question": task,
            "reflection": reflection,
        })
    }

    return result["answer"].(string), nil
}
```

---

## ACE Framework (Agentic Context Engineering)

**Self-improving agents** that learn from execution trajectories.

Based on the [ACE paper (arXiv:2510.04618)](https://arxiv.org/abs/2510.04618), ACE enables agents to:
- Record execution trajectories (steps, tool calls, reasoning)
- Extract patterns from successes and failures
- Persist learnings across sessions
- Inject learnings into future prompts

### Quick Start with ACE

Enable ACE on a ReAct agent:

```go
import (
    "github.com/XiaoConstantine/dspy-go/pkg/agents/ace"
    "github.com/XiaoConstantine/dspy-go/pkg/agents/react"
)

// Configure ACE
aceConfig := ace.Config{
    Enabled:           true,
    LearningsPath:     "./learnings/agent.md",  // Persistent storage
    AsyncReflection:   true,                    // Process in background
    CurationFrequency: 10,                      // Curate every 10 trajectories
    MinConfidence:     0.7,                     // Threshold for new learnings
    MaxTokens:         80000,                   // Token budget for learnings
}

// Create agent with ACE
agent := react.NewReActAgent(
    "my-agent",
    "Research Assistant",
    react.WithACE(aceConfig),          // Enable ACE!
    react.WithReflection(true, 3),     // Also enable reflection
    react.WithMaxIterations(10),
)
```

### How ACE Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Execution                          │
├─────────────────────────────────────────────────────────────┤
│  1. StartTrajectory() - Begin recording                     │
│  2. RecordStep() - Capture each action/observation          │
│  3. EndTrajectory() - Finalize with outcome                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Reflection                               │
├─────────────────────────────────────────────────────────────┤
│  • UnifiedReflector combines multiple insight sources       │
│  • SimpleReflector extracts basic patterns (no LLM)         │
│  • Adapters bridge existing systems (SelfReflector, etc.)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Curation                                 │
├─────────────────────────────────────────────────────────────┤
│  • Add new learnings (strategies, mistakes)                 │
│  • Update existing learnings (helpful/harmful counts)       │
│  • Prune ineffective learnings                              │
│  • Merge similar learnings                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage (learnings.md)                   │
├─────────────────────────────────────────────────────────────┤
│  ## STRATEGIES                                              │
│  [strategies-00001] helpful=5 harmful=0 :: Use calculator   │
│  [strategies-00002] helpful=3 harmful=1 :: Search once      │
│                                                             │
│  ## MISTAKES                                                │
│  [mistakes-00001] helpful=0 harmful=4 :: Avoid broken_db    │
└─────────────────────────────────────────────────────────────┘
```

### Standalone ACE Usage

Use ACE components directly without a ReAct agent:

```go
import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/agents/ace"
)

func main() {
    // Configure
    config := ace.Config{
        Enabled:           true,
        LearningsPath:     "./learnings.md",
        AsyncReflection:   false,
        CurationFrequency: 5,
        MinConfidence:     0.6,
        MaxTokens:         80000,
    }

    // Create reflector and manager
    reflector := ace.NewUnifiedReflector(nil, ace.NewSimpleReflector())
    manager, _ := ace.NewManager(config, reflector)
    defer manager.Close()

    ctx := context.Background()

    // Record a trajectory
    recorder := manager.StartTrajectory("agent-1", "research", "Find weather in NYC")

    recorder.RecordStep(
        "search",           // action
        "web_search",       // tool
        "Searching for NYC weather",  // reasoning
        map[string]any{"query": "NYC weather"},  // input
        map[string]any{"result": "Sunny, 72F"},  // output
        nil,                // error (nil = success)
    )

    manager.EndTrajectory(ctx, recorder, ace.OutcomeSuccess)

    // Get learnings for context injection
    contextStr := manager.LearningsContext()
    fmt.Println(contextStr)

    // Check metrics
    metrics := manager.Metrics()
    fmt.Printf("Trajectories: %d, Learnings: %d\n",
        metrics["trajectories_processed"],
        metrics["learnings_added"])
}
```

### Citation Tracking

ACE tracks when the agent cites learnings in its reasoning:

```go
// Agent reasoning that cites a learning
recorder.RecordStep(
    "search",
    "web_search",
    "Using [L001] efficient search strategy, I'll search once",  // Cites L001!
    input, output, nil,
)

// After successful execution, L001 gets a "helpful" vote
// After failure, L001 gets a "harmful" vote
// Learnings with low success rates get pruned
```

### Learnings File Format

ACE stores learnings in a human-readable markdown format:

```markdown
## STRATEGIES
[strategies-00001] helpful=5 harmful=0 :: Use calculator for arithmetic
[strategies-00002] helpful=3 harmful=1 :: Search once, then respond

## MISTAKES
[mistakes-00001] helpful=0 harmful=4 :: Avoid broken_database tool
```

### Context Injection

Learnings are formatted for injection into agent prompts:

```go
contextStr := manager.LearningsContext()
// Returns:
// ## Learned Strategies (cite by ID if using)
// [L001] Use calculator for arithmetic (100% success)
// [L002] Search once, then respond (75% success)
//
// ## Mistakes to Avoid (cite by ID if avoiding)
// [M001] Avoid broken_database tool
```

### ACE Examples

Two complete examples are available:

```bash
# Basic ACE usage (no LLM required)
go run ./examples/ace_basic/...

# ACE integrated with ReAct agent
GEMINI_API_KEY=your-key go run ./examples/ace_react/...

# Persist learnings across runs
go run ./examples/ace_react/... --learnings-dir=./my_learnings
```

**[ACE Examples →](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/ace_basic)**

---

## Production Agent Example

**Complete production-ready agent** with all features:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
    "github.com/XiaoConstantine/dspy-go/pkg/tools"
    "github.com/XiaoConstantine/dspy-go/pkg/agents/memory"
)

type ProductionAgent struct {
    react    *modules.ReAct
    memory   memory.Memory
    registry core.ToolRegistry
}

func NewProductionAgent() *ProductionAgent {
    // Configure LLM
    llm, _ := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
    core.SetDefaultLLM(llm)

    // Create Smart Tool Registry
    registry := tools.NewSmartToolRegistry(&tools.SmartToolRegistryConfig{
        AutoDiscoveryEnabled:       true,
        PerformanceTrackingEnabled: true,
        FallbackEnabled:           true,
    })

    // Register tools
    registry.Register(tools.NewCalculatorTool())
    registry.Register(tools.NewSearchTool())
    registry.Register(tools.NewDatabaseTool())
    registry.Register(tools.NewAPITool())

    // Create signature
    signature := core.NewSignature(
        []core.InputField{
            {Field: core.NewField("question")},
            {Field: core.NewField("history")},
        },
        []core.OutputField{
            {Field: core.NewField("answer")},
        },
    )

    // Create ReAct module
    react := modules.NewReAct(signature, registry, 10)

    // Create memory
    memory := memory.NewBufferMemory(50)

    return &ProductionAgent{
        react:    react,
        memory:   memory,
        registry: registry,
    }
}

func (a *ProductionAgent) Chat(ctx context.Context, userInput string) (string, error) {
    // Add user message to memory
    a.memory.Add(ctx, "user", userInput)

    // Get conversation history
    history, err := a.memory.Get(ctx)
    if err != nil {
        return "", err
    }

    // Execute with history
    result, err := a.react.Process(ctx, map[string]interface{}{
        "question": userInput,
        "history":  formatHistory(history),
    })
    if err != nil {
        return "", err
    }

    answer := result["answer"].(string)

    // Add assistant response to memory
    a.memory.Add(ctx, "assistant", answer)

    return answer, nil
}

func formatHistory(history []memory.Message) string {
    var formatted string
    for _, msg := range history {
        formatted += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
    }
    return formatted
}

func main() {
    agent := NewProductionAgent()

    ctx := context.Background()

    // Interactive loop
    for {
        fmt.Print("You: ")
        var input string
        fmt.Scanln(&input)

        if input == "exit" {
            break
        }

        response, err := agent.Chat(ctx, input)
        if err != nil {
            log.Printf("Error: %v\n", err)
            continue
        }

        fmt.Printf("Agent: %s\n\n", response)
    }
}
```

---

## Key Agent Features

| Feature | Description | Example |
|---------|-------------|---------|
| **ReAct Pattern** | Reasoning + tool use | Research agents, Q&A bots |
| **Custom Tools** | Domain-specific actions | Database queries, API calls |
| **Memory** | Conversation history | Multi-turn chat |
| **Orchestration** | Task decomposition | Complex workflows |
| **Multi-Agent** | Specialized agents | Research + analysis + writing |
| **Reflection** | Self-improvement | Iterative refinement |
| **ACE Framework** | Self-improving agents | Learn from trajectories |

---

## Examples

### Complete Agent Examples
- **[Agents Package Examples](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/agents)** - ReAct, orchestrator, memory
- **[ACE Basic Example](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/ace_basic)** - Standalone ACE usage (no LLM)
- **[ACE + ReAct Example](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/ace_react)** - Self-improving ReAct agent
- **[Maestro](https://github.com/XiaoConstantine/maestro)** - Production code review agent
- **[Smart Tool Registry](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/smart_tool_registry)** - Advanced tool management

### Running the Examples

```bash
# Basic agent examples
cd examples/agents && go run main.go

# ACE examples
go run ./examples/ace_basic/...
GEMINI_API_KEY=your-key go run ./examples/ace_react/...

# Production agent (Maestro)
git clone https://github.com/XiaoConstantine/maestro
cd maestro && go run main.go
```

---

## Next Steps

- **[Tool Management →](tools/)** - Build sophisticated tool systems
- **[Core Concepts →](core-concepts/)** - Understand modules and signatures
- **[Optimizers →](optimizers/)** - Improve agent performance automatically
- **[Examples →](https://github.com/XiaoConstantine/dspy-go/tree/main/examples)** - More agent patterns
