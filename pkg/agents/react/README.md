# ReAct Agents

A modern implementation of ReAct (Reasoning and Acting) agents with advanced patterns including self-reflection, planning, and memory optimization.

## Features

### Core Capabilities
- **ReAct Loop**: Classic Thought → Action → Observation cycle
- **Multiple Execution Modes**: ReAct, ReWOO (plan-then-execute), and Hybrid
- **Tool Integration**: Dynamic tool registration and management
- **Interceptor Support**: Compatible with the existing interceptor pattern

### Advanced Patterns (2024-2025)
- **Self-Reflection**: Learns from past actions and improves performance
- **Task Planning**: Decomposes complex tasks into manageable subtasks
- **Memory Optimization**: Uses Ebbinghaus forgetting curve for efficient memory management
- **Adaptive Execution**: Switches between execution modes based on task complexity

## Quick Start

```go
import "github.com/XiaoConstantine/dspy-go/pkg/agents/react"

// Create agent with modern patterns
agent := react.NewReActAgent("my-agent", "Research Assistant",
    react.WithExecutionMode(react.ModeHybrid),
    react.WithReflection(true, 3),
    react.WithPlanning(react.Interleaved, 5),
    react.WithMemoryOptimization(24*time.Hour, 0.3),
)

// Initialize with LLM and signature
signature := core.NewSignature(
    []core.InputField{{Field: core.Field{Name: "task"}}},
    []core.OutputField{{Field: core.Field{Name: "result"}}},
)
agent.Initialize(llm, signature)

// Register tools
agent.RegisterTool(searchTool)
agent.RegisterTool(calculatorTool)

// Execute tasks
result, err := agent.Execute(ctx, map[string]interface{}{
    "task": "Research AI developments and calculate market size",
})
```

## Execution Modes

### ReAct Mode (`ModeReAct`)
Classic ReAct execution with iterative reasoning and action cycles.
- **Best for**: Interactive, exploratory tasks
- **Characteristics**: Flexible, adaptive, higher token usage
- **Use case**: Document Q&A, research tasks

### ReWOO Mode (`ModeReWOO`)
Plan-then-execute approach that reduces token consumption.
- **Best for**: Structured, predictable workflows
- **Characteristics**: Efficient, lower token usage, faster execution
- **Use case**: Batch processing, data analysis pipelines

### Hybrid Mode (`ModeHybrid`)
Automatically chooses between ReAct and ReWOO based on task complexity.
- **Best for**: General-purpose applications
- **Characteristics**: Adaptive, balanced performance
- **Use case**: Multi-purpose agents, varying task complexity

## Advanced Features

### Self-Reflection
The agent automatically analyzes its performance and learns from experience:

```go
// Enable reflection with depth of 3 analysis levels
agent := react.NewReActAgent("agent", "name",
    react.WithReflection(true, 3))

// Get insights after execution
reflections := agent.Reflector.GetTopReflections(5)
for _, ref := range reflections {
    fmt.Printf("Insight: %s (confidence: %.2f)\n",
        ref.Insight, ref.Confidence)
}
```

### Task Planning
Decomposes complex tasks using proven patterns:

```go
// Configure planning strategy
agent := react.NewReActAgent("agent", "name",
    react.WithPlanning(react.Interleaved, 5))

// Plans are automatically created for complex tasks
// The agent uses templates and decomposition patterns
```

### Memory Optimization
Implements Ebbinghaus forgetting curve for intelligent memory management:

```go
// Configure memory with 24h retention and 0.3 forget threshold
agent := react.NewReActAgent("agent", "name",
    react.WithMemoryOptimization(24*time.Hour, 0.3))

// Memory automatically:
// - Retains important information longer
// - Compresses similar memories
// - Forgets low-importance items
```

## Configuration Options

```go
config := react.ReActAgentConfig{
    // Core settings
    MaxIterations: 10,
    ExecutionMode: react.ModeHybrid,
    Timeout:       5 * time.Minute,

    // Memory settings
    MemoryRetention: 24 * time.Hour,
    ForgetThreshold: 0.3,
    EnableMemoryOpt: true,

    // Reflection settings
    EnableReflection: true,
    ReflectionDepth:  3,
    ReflectionDelay:  100 * time.Millisecond,

    // Planning settings
    PlanningStrategy: react.Interleaved,
    MaxPlanDepth:     5,
    EnablePlanning:   true,

    // Tool settings
    ToolTimeout:    30 * time.Second,
    ParallelTools:  true,
    MaxToolRetries: 3,
}
```

## Performance Characteristics

### Token Efficiency
- **ReAct**: Standard token usage with full reasoning traces
- **ReWOO**: ~65% reduction in token usage vs ReAct
- **Hybrid**: Adaptive usage based on task complexity

### Success Rates (benchmarks)
- **ReAct**: Flexible but higher token cost
- **ReWOO**: 42.4% vs 40.8% on HotpotQA (GPT-3.5) with 80% fewer tokens
- **Hybrid**: Best of both modes

### Memory Optimization
- Implements Ebbinghaus forgetting curve
- Selective retention based on importance and access patterns
- Compression strategies: summarization, merging, pruning

## Tool Integration

The agent works with the existing tool ecosystem:

```go
// Function-based tools
searchTool := tools.NewFuncTool("search", "Search information",
    schema, searchFunction)
agent.RegisterTool(searchTool)

// MCP tools
mcpTool := tools.NewMCPTool(mcpClient, "mcp_tool")
agent.RegisterTool(mcpTool)

// Custom tools implementing core.Tool interface
agent.RegisterTool(customTool)
```

## Architecture

```
ReActAgent
├── ReAct Module (core reasoning loop)
├── Tool Registry (dynamic tool management)
├── Memory Store (with optimization)
├── Self-Reflector (learning and improvement)
├── Task Planner (decomposition and planning)
└── Memory Optimizer (forgetting curve + compression)
```

## Examples

See `examples/react_agent/main.go` for a comprehensive demonstration including:
- Multi-modal tool usage (search, calculate, summarize)
- All execution modes
- Performance monitoring
- Reflection insights
- Memory statistics

## Integration with Existing Systems

The ReAct agent implements the standard `agents.Agent` interface and supports:
- **Workflows**: Can be used in workflow systems
- **Orchestration**: Compatible with orchestrator patterns
- **Communication**: Works with agent networks
- **Interceptors**: Full interceptor support for monitoring/logging

## Benchmarking

Run benchmarks to compare execution modes:

```bash
go test -bench=. ./pkg/agents/react/
```

## Future Enhancements

- [ ] Multi-agent collaboration patterns
- [ ] Advanced planning algorithms (MCTS, A*)
- [ ] Learned tool selection
- [ ] Dynamic strategy adaptation
- [ ] Integration with vector databases
- [ ] Streaming execution support
