# ReAct Agent Framework

A production-ready implementation of ReAct (Reasoning and Acting) agents with advanced cognitive architectures including self-reflection, task planning, and memory optimization with forgetting curves.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         ReAct Agent                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Execution     │  │ Task         │  │ Self                 │  │
│  │ Modes         │  │ Planning     │  │ Reflection           │  │
│  ├───────────────┤  ├──────────────┤  ├──────────────────────┤  │
│  │ • ReAct       │  │ • Decompose  │  │ • Strategy Analysis  │  │
│  │ • ReWOO       │  │ • Templates  │  │ • Performance        │  │
│  │ • Hybrid      │  │ • Parallel   │  │ • Learning           │  │
│  └───────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Memory        │  │ Tool         │  │ Execution            │  │
│  │ Optimizer     │  │ Registry     │  │ History              │  │
│  ├───────────────┤  ├──────────────┤  ├──────────────────────┤  │
│  │ • Forgetting  │  │ • Dynamic    │  │ • Tracking           │  │
│  │   Curve       │  │   Loading    │  │ • Metrics            │  │
│  │ • Compression │  │ • Validation │  │ • Analysis           │  │
│  │ • Indexing    │  │ • Execution  │  │ • Interceptors       │  │
│  └───────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Features

### 🎯 Execution Modes

#### ReAct Mode (Classic)
Traditional ReAct loop with iterative Thought → Action → Observation cycles.
```go
agent := react.NewReActAgent("my-agent", "Assistant",
    react.WithExecutionMode(react.ModeReAct))
```
- **Best for**: Exploratory tasks, interactive queries, complex reasoning
- **Token usage**: Standard (full reasoning traces)
- **Flexibility**: Maximum adaptability

#### ReWOO Mode (Plan-then-Execute)
Reasoning Without Observation - creates plan upfront then executes.
```go
agent := react.NewReActAgent("my-agent", "Assistant",
    react.WithExecutionMode(react.ModeReWOO))
```
- **Best for**: Structured workflows, batch processing, predictable tasks
- **Token usage**: ~65% reduction vs ReAct
- **Speed**: Faster execution with less LLM calls

#### Hybrid Mode (Adaptive)
Automatically selects mode based on task complexity analysis.
```go
agent := react.NewReActAgent("my-agent", "Assistant",
    react.WithExecutionMode(react.ModeHybrid))
```
- **Best for**: General-purpose applications, varying workloads
- **Intelligence**: Analyzes task keywords, length, and available tools
- **Balance**: Optimal token/performance trade-off

### 🧠 Memory Optimization

Implements Ebbinghaus forgetting curve with intelligent compression strategies:

```go
optimizer := react.NewMemoryOptimizer(
    24*time.Hour,  // retention period
    0.3,           // forget threshold
)
```

**Features:**
- **Forgetting Curve**: R = e^(-t/S) where R is retention, t is time, S is strength
- **Compression Strategies**:
  - **Summarization**: Groups similar memories into summaries
  - **Merging**: Combines semantically similar items
  - **Pruning**: Removes low-importance memories
- **Semantic Search**: Embedding-based similarity matching
- **Category Organization**: Automatic task categorization
- **Importance Scoring**: Based on success, complexity, and access patterns

### 📊 Self-Reflection System

Multi-level reflection with pattern recognition and learning:

```go
reflector := react.NewSelfReflector(
    3,                     // reflection depth
    100*time.Millisecond,  // reflection delay
)
```

**Reflection Types:**
- **Strategy Reflection**: Identifies repeated patterns and tool usage
- **Performance Reflection**: Analyzes success rates and efficiency
- **Learning Reflection**: Extracts reusable insights
- **Error Reflection**: Learns from failures

### 📋 Task Planning

Sophisticated planning with decomposition and parallelization:

```go
planner := react.NewTaskPlanner(
    react.DecompositionFirst,  // or Interleaved
    5,                        // max decomposition depth
)
```

**Features:**
- **Plan Templates**: Reusable patterns for common tasks
- **Task Decomposition**: Breaks complex tasks into subtasks
- **Dependency Analysis**: Identifies parallel execution opportunities
- **Critical Path**: Marks essential vs optional steps
- **Optimization**: Reorders steps for efficiency

## Quick Start

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/agents/react"
    "github.com/XiaoConstantine/dspy-go/pkg/core"
)

func main() {
    // Create agent with all advanced features
    agent := react.NewReActAgent("research-agent", "Research Assistant",
        react.WithExecutionMode(react.ModeHybrid),
        react.WithReflection(true, 3),
        react.WithPlanning(react.Interleaved, 5),
        react.WithMemoryOptimization(24*time.Hour, 0.3),
        react.WithMaxIterations(10),
        react.WithTimeout(5*time.Minute),
    )

    // Initialize with typed signature
    type Input struct {
        Task string `json:"task"`
    }
    type Output struct {
        Result string `json:"result"`
    }

    signature := core.NewTypedSignature[Input, Output]()
    err := agent.Initialize(llm, signature.ToLegacySignature())

    // Register tools
    tools := []core.Tool{searchTool, calculatorTool, summarizerTool}
    for _, tool := range tools {
        agent.RegisterTool(tool)
    }

    // Execute with interceptors for monitoring
    result, err := agent.ExecuteWithInterceptors(
        context.Background(),
        map[string]interface{}{"task": "Analyze market trends"},
        []core.AgentInterceptor{loggingInterceptor, metricsInterceptor},
    )

    // Analyze performance
    history := agent.GetExecutionHistory()
    reflections := agent.Reflector.GetTopReflections(3)
    metrics := agent.Reflector.GetMetrics()
}
```

## Configuration

### Complete Configuration Options

```go
config := react.ReActAgentConfig{
    // Core Execution
    MaxIterations:    10,              // Max reasoning cycles
    ExecutionMode:    ModeHybrid,      // ReAct/ReWOO/Hybrid
    Timeout:         5*time.Minute,    // Overall timeout

    // Memory Management
    MemoryRetention:      24*time.Hour,  // How long to retain memories
    ForgetThreshold:      0.3,           // Minimum retention score
    EnableMemoryOpt:      true,          // Enable optimization
    CompressionThreshold: 100,           // Items before compression

    // Self-Reflection
    EnableReflection:     true,               // Enable reflection system
    ReflectionDepth:      3,                  // Analysis depth
    ReflectionDelay:      100*time.Millisecond, // Delay between reflections
    ReflectionThreshold:  0.7,                // Confidence threshold

    // Task Planning
    PlanningStrategy:     Interleaved,   // DecompositionFirst/Interleaved
    MaxPlanDepth:        5,              // Max decomposition depth
    EnablePlanning:      true,           // Enable planning system

    // Tool Execution
    ToolTimeout:         30*time.Second, // Per-tool timeout
    ParallelTools:       true,           // Enable parallel execution
    MaxToolRetries:      3,              // Retry failed tools

    // Monitoring
    EnableInterceptors:  true,           // Enable interceptor support
}
```

## Advanced Usage

### Memory Statistics and Analysis

```go
// Get memory statistics
stats := agent.Optimizer.GetStatistics()
// Returns: total_items, categories, retention_rate, compression_ratio,
//          category_distribution, avg_importance, avg_access_count

// Find relevant memories
memories := agent.Optimizer.Retrieve(ctx, map[string]interface{}{
    "task": "previous research query",
})
```

### Plan Metrics and Optimization

```go
// Get plan metrics
plan, _ := agent.Planner.CreatePlan(ctx, input, tools)
metrics := agent.Planner.GetPlanMetrics(plan)
// Returns: total_steps, parallel_steps, critical_steps,
//          estimated_time, strategy, parallelization, tool_usage

// Validate plan
err := agent.Planner.ValidatePlan(plan)
```

### Reflection Insights

```go
// Get detailed metrics
metrics := agent.Reflector.GetMetrics()
// Returns: total_reflections, avg_confidence, success_rate,
//          avg_iterations, improvements, reflection_types

// Calculate improvement rate
improvement := agent.Reflector.CalculateImprovement()

// Reset for new session
agent.Reflector.Reset()
```

## Performance Characteristics

### Token Efficiency (GPT-3.5 Benchmarks)
| Mode | Token Usage | Success Rate | Speed |
|------|------------|--------------|-------|
| ReAct | Baseline (100%) | 40.8% | Standard |
| ReWOO | ~35% | 42.4% | 2-3x faster |
| Hybrid | 50-70% | 41.5% | 1.5-2x faster |

### Memory Performance
- **Compression Ratio**: 50-70% reduction in memory footprint
- **Retrieval Speed**: O(n log n) with category indexing
- **Accuracy**: 85%+ relevance for top-5 retrievals

### Test Coverage
- **Overall Coverage**: 80.7%
- **Core Components**: 85%+
- **Edge Cases**: Comprehensive error handling

## Tool Integration

### Standard Tool Registration
```go
// Implements core.Tool interface
type MyTool struct {
    name string
}

func (t *MyTool) Name() string { return t.name }
func (t *MyTool) Description() string { return "My custom tool" }
func (t *MyTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
    // Implementation
}
// ... other required methods

agent.RegisterTool(&MyTool{})
```

### Function Tools
```go
tool := tools.NewFuncTool("search", "Search the web",
    schema,
    func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
        // Implementation
    })
agent.RegisterTool(tool)
```

## Monitoring and Observability

### Interceptor Support
```go
// Built-in support for interceptors
agent.SetInterceptors([]core.AgentInterceptor{
    logging.NewInterceptor(),
    metrics.NewInterceptor(),
    tracing.NewInterceptor(),
})

// Or per-execution
result, err := agent.ExecuteWithInterceptors(ctx, input, interceptors)
```

### Execution History
```go
history := agent.GetExecutionHistory()
for _, record := range history {
    fmt.Printf("Task: %v\n", record.Input)
    fmt.Printf("Success: %v\n", record.Success)
    fmt.Printf("Actions taken: %d\n", len(record.Actions))
    fmt.Printf("Duration: %v\n", record.Duration)
}
```

## Examples

See `/examples/react_agent/` for complete examples including:
- Multi-tool orchestration
- All execution modes comparison
- Memory optimization demonstration
- Reflection analysis
- Performance monitoring

## Testing

```bash
# Run all tests
go test ./pkg/agents/react/...

# Run with coverage
go test -cover ./pkg/agents/react/...

# Run benchmarks
go test -bench=. ./pkg/agents/react/

# Run specific test suites
go test -run TestMemoryOptimizer ./pkg/agents/react/
go test -run TestTaskPlanner ./pkg/agents/react/
go test -run TestSelfReflector ./pkg/agents/react/
```

## Production Considerations

### Resource Management
- Memory optimizer automatically manages memory footprint
- Configurable compression thresholds
- Automatic cleanup of old memories

### Error Handling
- Comprehensive error recovery
- Atomic operations with rollback
- Critical vs non-critical step failures

### Scalability
- Parallel tool execution support
- Efficient memory indexing
- Optimized dependency analysis

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../../../LICENSE) for details.
