---
title: "A2A Protocol"
description: "Agent-to-Agent communication for multi-agent orchestration"
summary: "Build sophisticated multi-agent systems with hierarchical composition and standardized messaging"
date: 2025-01-06T00:00:00+00:00
lastmod: 2025-01-06T00:00:00+00:00
draft: false
weight: 550
toc: true
seo:
  title: "A2A Protocol - dspy-go"
  description: "Complete guide to building multi-agent systems with A2A protocol in dspy-go"
  canonical: ""
  noindex: false
---

# A2A Protocol (Agent-to-Agent Communication)

The **A2A Protocol** enables multi-agent orchestration with hierarchical composition, allowing agents to delegate work to specialized sub-agents. Build sophisticated workflows where multiple agents collaborate to solve complex tasks.

## Why A2A?

Traditional single-agent approaches struggle with:
- **Complex tasks** requiring diverse expertise
- **Long-running workflows** that benefit from decomposition
- **Specialized knowledge** needed for different parts of a task

A2A solves this by enabling:
- **Hierarchical Composition**: Orchestrators manage specialized sub-agents
- **Standardized Messages**: Structured message format with metadata
- **In-Process Communication**: No HTTP overhead for local coordination
- **Capability Discovery**: Agents can advertise and discover capabilities

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                        │
│  (ResearchOrchestrator)                                     │
├─────────────────────────────────────────────────────────────┤
│  1. Receives task: "Research AI in healthcare"              │
│  2. Delegates to sub-agents via CallSubAgent()              │
│  3. Aggregates results into final output                    │
└────────────┬──────────────────┬──────────────────┬──────────┘
             │                  │                  │
             ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ SearchAgent │    │AnalysisAgent│    │SynthesisAgent│
    │  (search)   │    │  (analysis) │    │ (synthesis) │
    ├─────────────┤    ├─────────────┤    ├─────────────┤
    │ Generates   │    │ Extracts    │    │ Creates     │
    │ search      │    │ key findings│    │ final       │
    │ queries     │    │ and patterns│    │ report      │
    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Quick Start

### 1. Create Specialized Agents

Each agent implements the `agents.Agent` interface:

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/agents"
    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// SearchAgent performs web searches
type SearchAgent struct {
    module core.Module
}

func NewSearchAgent() (*SearchAgent, error) {
    signature := core.NewSignature(
        []core.InputField{
            {Field: core.Field{Name: "topic", Description: "Research topic"}},
        },
        []core.OutputField{
            {Field: core.Field{Name: "search_queries", Description: "3-5 search queries"}},
            {Field: core.Field{Name: "search_results", Description: "Search results"}},
        },
    ).WithInstruction(`Generate targeted search queries and find relevant information.`)

    return &SearchAgent{
        module: modules.NewPredict(signature),
    }, nil
}

func (s *SearchAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
    return s.module.Process(ctx, input)
}

func (s *SearchAgent) GetCapabilities() []core.Tool { return nil }
func (s *SearchAgent) GetMemory() agents.Memory     { return nil }
```

### 2. Wrap Agents with A2A Executors

```go
import a2a "github.com/XiaoConstantine/dspy-go/pkg/agents/communication"

// Create specialized agents
searchAgent, _ := NewSearchAgent()
analysisAgent, _ := NewAnalysisAgent()
synthesisAgent, _ := NewSynthesisAgent()

// Wrap with A2A executors
searchExec := a2a.NewExecutorWithConfig(searchAgent, a2a.ExecutorConfig{
    Name: "SearchAgent",
})
analysisExec := a2a.NewExecutorWithConfig(analysisAgent, a2a.ExecutorConfig{
    Name: "AnalysisAgent",
})
synthesisExec := a2a.NewExecutorWithConfig(synthesisAgent, a2a.ExecutorConfig{
    Name: "SynthesisAgent",
})
```

### 3. Create Orchestrator

```go
type ResearchOrchestrator struct {
    executor *a2a.A2AExecutor
}

func NewResearchOrchestrator() (*ResearchOrchestrator, *a2a.A2AExecutor) {
    agent := &ResearchOrchestrator{}
    executor := a2a.NewExecutorWithConfig(agent, a2a.ExecutorConfig{
        Name: "ResearchOrchestrator",
    })
    agent.executor = executor
    return agent, executor
}

func (r *ResearchOrchestrator) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
    topic := input["topic"].(string)

    // Step 1: Search for information
    searchResult, err := r.executor.CallSubAgent(ctx, "search", a2a.NewUserMessage(topic))
    if err != nil {
        return nil, fmt.Errorf("search failed: %w", err)
    }

    // Extract search results from artifact
    searchResults := extractField(searchResult, "search_results")

    // Step 2: Analyze the results
    analysisInput := a2a.NewMessage(a2a.RoleUser,
        a2a.NewTextPartWithMetadata(topic, map[string]interface{}{"field": "topic"}),
        a2a.NewTextPartWithMetadata(searchResults, map[string]interface{}{"field": "search_results"}),
    )
    analysisResult, err := r.executor.CallSubAgent(ctx, "analysis", analysisInput)
    if err != nil {
        return nil, fmt.Errorf("analysis failed: %w", err)
    }

    // Step 3: Synthesize final report
    // ... continue with synthesis agent

    return output, nil
}

func (r *ResearchOrchestrator) GetCapabilities() []core.Tool { return nil }
func (r *ResearchOrchestrator) GetMemory() agents.Memory     { return nil }
```

### 4. Register Sub-Agents and Execute

```go
func main() {
    // Create and configure
    _, orchestratorExec := NewResearchOrchestrator()
    orchestratorExec.
        WithSubAgent("search", searchExec).
        WithSubAgent("analysis", analysisExec).
        WithSubAgent("synthesis", synthesisExec)

    // Execute workflow
    msg := a2a.NewMessage(a2a.RoleUser,
        a2a.NewTextPartWithMetadata("AI in healthcare", map[string]interface{}{"field": "topic"}),
    )

    artifact, err := orchestratorExec.Execute(ctx, msg)
    if err != nil {
        log.Fatal(err)
    }

    // Process results from artifact
    for _, part := range artifact.Parts {
        if field, ok := part.Metadata["field"].(string); ok {
            fmt.Printf("%s: %s\n", field, part.Text)
        }
    }
}
```

---

## Message Protocol

### Creating Messages

```go
// Simple user message
msg := a2a.NewUserMessage("What is the weather?")

// Message with multiple parts and metadata
msg := a2a.NewMessage(a2a.RoleUser,
    a2a.NewTextPartWithMetadata("Paris", map[string]interface{}{"field": "location"}),
    a2a.NewTextPartWithMetadata("today", map[string]interface{}{"field": "date"}),
)

// Message roles
a2a.RoleUser      // User input
a2a.RoleAssistant // Agent response
a2a.RoleSystem    // System instructions
```

### Working with Artifacts

Artifacts are the result of agent execution:

```go
artifact, err := executor.Execute(ctx, msg)

// Iterate over parts
for _, part := range artifact.Parts {
    // Access text content
    fmt.Println(part.Text)

    // Access metadata
    if field, ok := part.Metadata["field"].(string); ok {
        fmt.Printf("Field: %s = %s\n", field, part.Text)
    }
}

// Convert to map
result := make(map[string]interface{})
for _, part := range artifact.Parts {
    if field, ok := part.Metadata["field"].(string); ok {
        result[field] = part.Text
    }
}
```

---

## Advanced Patterns

### Parallel Sub-Agent Execution

Execute multiple sub-agents concurrently:

```go
func (o *Orchestrator) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
    topic := input["topic"].(string)

    // Create channels for results
    type result struct {
        name string
        data *a2a.Artifact
        err  error
    }
    results := make(chan result, 3)

    // Launch parallel searches
    searchTopics := []string{"technical", "business", "regulatory"}
    for _, searchType := range searchTopics {
        go func(st string) {
            msg := a2a.NewMessage(a2a.RoleUser,
                a2a.NewTextPartWithMetadata(topic, map[string]interface{}{"field": "topic"}),
                a2a.NewTextPartWithMetadata(st, map[string]interface{}{"field": "search_type"}),
            )
            artifact, err := o.executor.CallSubAgent(ctx, "search", msg)
            results <- result{st, artifact, err}
        }(searchType)
    }

    // Collect results
    combined := make(map[string]*a2a.Artifact)
    for i := 0; i < len(searchTopics); i++ {
        r := <-results
        if r.err != nil {
            return nil, r.err
        }
        combined[r.name] = r.data
    }

    // Continue with aggregation...
}
```

### Conditional Routing

Route to different agents based on task type:

```go
func (o *Orchestrator) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
    task := input["task"].(string)
    taskType := classifyTask(task)

    var agentName string
    switch taskType {
    case "research":
        agentName = "research_agent"
    case "analysis":
        agentName = "analysis_agent"
    case "writing":
        agentName = "writing_agent"
    default:
        agentName = "general_agent"
    }

    result, err := o.executor.CallSubAgent(ctx, agentName, a2a.NewUserMessage(task))
    // ...
}
```

### Error Handling and Fallbacks

```go
func (o *Orchestrator) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
    // Try primary agent
    result, err := o.executor.CallSubAgent(ctx, "primary_search", msg)
    if err != nil {
        // Fallback to secondary agent
        log.Printf("Primary search failed: %v, trying fallback", err)
        result, err = o.executor.CallSubAgent(ctx, "fallback_search", msg)
        if err != nil {
            return nil, fmt.Errorf("all search agents failed: %w", err)
        }
    }

    return processResult(result), nil
}
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Hierarchical Composition** | Orchestrators manage multiple sub-agents |
| **Standardized Messages** | Consistent message format with metadata |
| **In-Process Communication** | Zero network overhead for local agents |
| **Capability Discovery** | Agents advertise their capabilities |
| **Field Metadata** | Tag data fields for structured passing |
| **Flexible Routing** | Route tasks to appropriate agents |

---

## Complete Example

See the full deep research agent example:

```bash
cd examples/a2a_composition
go run main.go --api-key YOUR_API_KEY
```

This example demonstrates:
- Multi-agent hierarchical composition
- Search -> Analysis -> Synthesis workflow
- Standardized message/artifact protocol
- Real LLM integration with Gemini

**[A2A Composition Example](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/a2a_composition)**

---

## Next Steps

- **[Building Agents](agents/)** - ReAct patterns and agent architecture
- **[Tool Management](tools/)** - Smart tool selection for agents
- **[ACE Framework](agents/#ace-framework-agentic-context-engineering)** - Self-improving agents
