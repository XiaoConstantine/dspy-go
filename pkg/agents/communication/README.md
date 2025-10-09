# A2A (Agent-to-Agent) Package

This package implements Google's Agent-to-Agent (a2a) protocol for dspy-go, enabling **agent composition** and interoperability.

## ✅ Implementation Complete

### Core Features
- ✅ **In-process agent composition** - Parent agents with sub-agents (like ADK Python)
- ✅ **a2a message protocol** - Message, Task, Artifact types
- ✅ **Bidirectional conversion** - dspy-go ↔ a2a format
- ✅ **HTTP server** - Optional JSON-RPC over HTTP (for remote agents)
- ✅ **Comprehensive tests** - 60+ test cases covering all functionality

## Package Structure

```
pkg/a2a/
├── protocol.go           # Core types (Message, Task, Part, AgentCard)
├── converters.go         # Bidirectional message conversion
├── agent_executor.go     # In-process agent composition ⭐
├── server.go             # HTTP server for remote access
├── *_test.go             # Comprehensive test suite
└── README.md             # This file
```

## Quick Start

### 1. Agent-to-Agent Composition (Recommended)

Compose agents hierarchically, similar to ADK Python's `sub_agents`:

```go
import "github.com/XiaoConstantine/dspy-go/pkg/a2a"

// Create sub-agents
searchAgent := // ... your search agent
reasoningAgent := // ... your reasoning agent

searchExec := a2a.NewExecutor(searchAgent)
reasoningExec := a2a.NewExecutor(reasoningAgent)

// Create parent agent with sub-agents
parent := // ... your orchestrator agent
parentExec := a2a.NewExecutor(parent).
    WithSubAgent("search", searchExec).
    WithSubAgent("reasoning", reasoningExec)

// Parent calls sub-agent (in-process via a2a protocol)
msg := a2a.NewUserMessage("What is the capital of France?")
artifact, _ := parentExec.CallSubAgent(ctx, "search", msg)

fmt.Println(a2a.ExtractTextFromArtifact(artifact))
```

### 2. Direct Execution

Execute an agent with a2a messages:

```go
executor := a2a.NewExecutor(myAgent)

msg := a2a.NewUserMessage("Explain quantum physics")
task, _ := executor.SendMessage(ctx, msg)

// Access results
for _, artifact := range task.Artifacts {
    fmt.Println(a2a.ExtractTextFromArtifact(artifact))
}
```

### 3. Multi-Level Hierarchy

Build complex agent systems:

```go
// Level 3: Specialist agents
calculator := a2a.NewExecutor(calculatorAgent)
webSearch := a2a.NewExecutor(searchAgent)

// Level 2: Tool orchestrator
toolAgent := a2a.NewExecutor(toolOrchestratorAgent).
    WithSubAgent("calculator", calculator).
    WithSubAgent("search", webSearch)

// Level 1: Main orchestrator
mainAgent := a2a.NewExecutor(mainOrchestratorAgent).
    WithSubAgent("tools", toolAgent)

// Deep composition works automatically
result, _ := mainAgent.CallSubAgentSimple(ctx, "tools", "Calculate 2+2")
```

### 4. HTTP Server (Optional, for Remote Access)

Expose your agent via HTTP:

```go
server, _ := a2a.NewServer(myAgent, a2a.ServerConfig{
    Host: "localhost",
    Port: 8080,
    Name: "MyAgent",
    Description: "A helpful agent",
})

server.Start(ctx)
// AgentCard: http://localhost:8080/.well-known/agent.json
// JSON-RPC: http://localhost:8080/rpc
```

## Development

```bash
# Run tests
go test ./pkg/a2a/...

# Run with coverage
go test -cover ./pkg/a2a/...

# Build examples
go build ./examples/a2a/...
```

## References

- [Google ADK Python](https://github.com/google/adk-python)
- [a2a-sdk](https://github.com/google/a2a-sdk)
- [JSON-RPC 2.0](https://www.jsonrpc.org/specification)
