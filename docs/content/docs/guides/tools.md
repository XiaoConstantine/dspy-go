---
title: "Tool Management"
description: "Smart tool selection, chaining, and MCP integration"
summary: "Build sophisticated agent workflows with intelligent tool management"
date: 2025-10-13T00:00:00+00:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 400
toc: true
seo:
  title: "Tool Management - dspy-go"
  description: "Smart Tool Registry, tool chaining, dependency resolution, and MCP integration in dspy-go"
  canonical: ""
  noindex: false
---

# Tool Management

dspy-go provides a sophisticated tool management system that goes far beyond basic function calling. Build intelligent agents with Bayesian tool selection, automatic dependency resolution, and seamless MCP integration.

## Smart Tool Registry

**Intelligent tool selection** using Bayesian inference and performance tracking.

### Why Smart Tool Registry?

Traditional tool systems pick tools randomly or use simple rules. Smart Tool Registry:
- 🧠 **Bayesian Inference**: Multi-factor scoring for optimal tool selection
- 📊 **Performance Tracking**: Real-time metrics and reliability scoring
- 🔍 **Capability Analysis**: Automatic capability extraction and matching
- 🔄 **Auto-Discovery**: Dynamic tool registration from MCP servers
- 🛡️ **Fallback Mechanisms**: Intelligent fallback when tools fail

### Basic Usage

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/tools"
)

func main() {
    // Create Smart Tool Registry
    config := &tools.SmartToolRegistryConfig{
        AutoDiscoveryEnabled:       true,  // Auto-discover from MCP
        PerformanceTrackingEnabled: true,  // Track tool metrics
        FallbackEnabled:           true,   // Intelligent fallback
    }
    registry := tools.NewSmartToolRegistry(config)

    // Register tools
    registry.Register(mySearchTool)
    registry.Register(myAnalysisTool)

    // Intelligent tool selection based on intent
    ctx := context.Background()
    tool, err := registry.SelectBest(ctx, "find user information")
    if err != nil {
        log.Fatal(err)
    }

    // Execute with performance tracking
    result, err := registry.ExecuteWithTracking(ctx, tool.Name(), params)
}
```

**[Full Smart Tool Registry Example →](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/smart_tool_registry)**

---

## Tool Chaining

**Sequential pipelines** with data transformation and conditional execution.

### Pipeline Builder

Create sophisticated workflows with a fluent API:

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/tools"
)

func main() {
    // Create a tool pipeline
    pipeline, err := tools.NewPipelineBuilder("data_processing", registry).
        Step("data_extractor").                                    // Extract data
        StepWithTransformer("data_validator",                      // Validate
            tools.TransformExtractField("result")).
        ConditionalStep("data_enricher",                           // Conditional enrichment
            tools.ConditionExists("validation_result"),
            tools.ConditionEquals("status", "validated")).
        StepWithRetries("data_transformer", 3).                    // Transform with retries
        FailFast().                                                // Stop on first error
        EnableCaching().                                           // Cache results
        Build()

    if err != nil {
        log.Fatal(err)
    }

    // Execute the pipeline
    ctx := context.Background()
    result, err := pipeline.Execute(ctx, map[string]interface{}{
        "raw_data": "input data to process",
    })
}
```

### Data Transformations

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

**[Full Tool Chaining Example →](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/tool_chaining)**

---

## Dependency Resolution

**Automatic execution planning** with parallel optimization.

### Dependency Graph

Define tool dependencies and let dspy-go optimize execution:

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/tools"
)

func main() {
    // Create dependency graph
    graph := tools.NewDependencyGraph()

    // Define tool dependencies
    graph.AddNode(&tools.DependencyNode{
        ToolName:     "data_extractor",
        Dependencies: []string{},                    // No dependencies
        Outputs:      []string{"raw_data"},
        Priority:     1,
    })

    graph.AddNode(&tools.DependencyNode{
        ToolName:     "data_validator",
        Dependencies: []string{"data_extractor"},    // Depends on extractor
        Inputs:       []string{"raw_data"},
        Outputs:      []string{"validated_data"},
        Priority:     2,
    })

    graph.AddNode(&tools.DependencyNode{
        ToolName:     "data_analyzer",
        Dependencies: []string{"data_validator"},    // Depends on validator
        Inputs:       []string{"validated_data"},
        Outputs:      []string{"analysis"},
        Priority:     3,
    })

    // Create dependency-aware pipeline
    options := &tools.DependencyPipelineOptions{
        MaxParallelism: 4,                           // Run up to 4 tools in parallel
        EnableCaching:  true,
    }

    depPipeline, err := tools.NewDependencyPipeline(
        "smart_pipeline",
        registry,
        graph,
        options,
    )

    // Execute with automatic parallelization
    ctx := context.Background()
    result, err := depPipeline.ExecuteWithDependencies(ctx, input)
}
```

**Benefits:**
- ⚡ Automatic parallel execution of independent tools
- 🎯 Topological sorting ensures correct execution order
- 💾 Result caching prevents redundant work
- 🔄 Retry logic for failed dependencies

---

## Parallel Execution

**High-performance** parallel tool execution with intelligent scheduling.

### Parallel Executor

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/tools"
)

func main() {
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
        {
            ID:       "task3",
            ToolName: "enricher",
            Input:    data3,
            Priority: 1,
        },
    }

    // Execute with priority scheduling
    ctx := context.Background()
    results, err := executor.ExecuteParallel(ctx, tasks, &tools.PriorityScheduler{})

    // Or use fair share scheduling
    results, err := executor.ExecuteParallel(ctx, tasks, tools.NewFairShareScheduler())
}
```

**Scheduling Strategies:**
- **Priority Scheduler**: High-priority tasks first
- **Fair Share Scheduler**: Equal CPU time for all tasks
- **Custom Schedulers**: Implement your own scheduling logic

---

## Tool Composition

**Create reusable composite tools** by combining multiple tools into single units.

### Building Composite Tools

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/tools"
)

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

func main() {
    // Create a composite tool for text processing
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
        Step("text_processor").     // Using our composite tool
        Step("final_formatter").
        Build()
}
```

**[Full Tool Composition Example →](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/tool_composition)**

---

## MCP Integration

**Model Context Protocol** integration for accessing external tools and services.

### Connecting to MCP Servers

```go
package main

import (
    "github.com/XiaoConstantine/dspy-go/pkg/tools"
    "github.com/XiaoConstantine/mcp-go/pkg/client"
)

func main() {
    // Connect to an MCP server
    mcpClient, err := client.NewStdioClient("path/to/mcp-server")
    if err != nil {
        log.Fatal(err)
    }

    // Use with ReAct (requires InMemoryToolRegistry)
    registry := tools.NewInMemoryToolRegistry()
    err = tools.RegisterMCPTools(registry, mcpClient)
    if err != nil {
        log.Fatal(err)
    }

    // Create ReAct module with MCP tools
    react := modules.NewReAct(signature, registry, 5)
}
```

### With Smart Tool Registry

```go
// Use Smart Tool Registry for intelligent tool selection
smartRegistry := tools.NewSmartToolRegistry(&tools.SmartToolRegistryConfig{
    PerformanceTrackingEnabled: true,
    AutoDiscoveryEnabled:       true,
})

// Register MCP tools
err = tools.RegisterMCPTools(smartRegistry, mcpClient)
if err != nil {
    log.Fatal(err)
}

// Intelligent selection of MCP tools
ctx := context.Background()
selectedTool, err := smartRegistry.SelectBest(ctx, "analyze financial data")
```

---

## Key Features Summary

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Smart Registry** | Bayesian tool selection | Optimal tool choice for any task |
| **Tool Chaining** | Sequential pipelines | Multi-step data processing |
| **Dependency Resolution** | Automatic parallelization | Complex workflow optimization |
| **Parallel Execution** | High-performance scheduling | Batch operations |
| **Tool Composition** | Reusable composite tools | Modular tool building |
| **MCP Integration** | External tool access | Extend with any MCP server |

---

## Examples

### Complete Examples
- **[Smart Tool Registry](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/smart_tool_registry)** - Intelligent tool selection showcase
- **[Tool Chaining](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/tool_chaining)** - Pipeline building and transformations
- **[Tool Composition](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/tool_composition)** - Creating composite tools

### Running the Examples

```bash
# Smart Tool Registry
cd examples/smart_tool_registry && go run main.go

# Tool Chaining
cd examples/tool_chaining && go run main.go

# Tool Composition
cd examples/tool_composition && go run main.go
```

---

## Next Steps

- **[Agents Guide →](agents/)** - Build agents using ReAct and tool management
- **[Core Concepts →](core-concepts/)** - Understand modules and programs
- **[Examples →](https://github.com/XiaoConstantine/dspy-go/tree/main/examples)** - More working examples
