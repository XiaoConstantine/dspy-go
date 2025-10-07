# A2A Agent Composition Example

This example demonstrates Google's **Agent-to-Agent (a2a) protocol** implementation in dspy-go, focusing on **in-process agent composition** where parent agents can have sub-agents that communicate via the a2a message protocol.

## What This Demonstrates

### Core Features

1. **Agent Composition** - Creating hierarchical agent structures
2. **In-Process Communication** - Agents communicate via a2a protocol without HTTP
3. **Multi-Level Hierarchies** - Orchestrator → Specialized Agents → Leaf Agents
4. **Message Protocol** - Standardized a2a messages, tasks, and artifacts
5. **Capability Discovery** - Agent cards and capability exposure

### Architecture

```
OrchestratorAgent (Parent)
├── CalculatorAgent (Sub-agent)
│   └── Handles arithmetic operations
├── SearchAgent (Sub-agent)
│   └── Handles information search
└── ReasoningAgent (Sub-agent)
    └── Provides logical reasoning
```

## Running the Example

```bash
cd examples/a2a_composition
go run main.go
```

## What Happens

The example demonstrates:

1. **Basic Agent Testing** - Tests individual agents with a2a messages
2. **Multi-Level Hierarchy** - Creates an orchestrator with multiple sub-agents
3. **Agent Orchestration** - Orchestrator intelligently delegates to sub-agents
4. **Direct Sub-Agent Calls** - Parent calls specific sub-agents via a2a protocol
5. **Capability Discovery** - Lists available sub-agents and their capabilities

## Sample Output

```
╔════════════════════════════════════════════════════════════════╗
║          A2A Agent Composition Example - dspy-go               ║
╚════════════════════════════════════════════════════════════════╝

================================================================
Example 1: Basic Agent-to-Agent Communication
================================================================

📊 Testing Calculator Agent:
   Result: The answer is 4

🔍 Testing Search Agent:
   Result: Current weather: Sunny, 72°F

================================================================
Example 2: Multi-Level Agent Hierarchy
================================================================
[Creates orchestrator with 3 sub-agents]

================================================================
Example 3: Agent Orchestration in Action
================================================================

[Question 1] Can you calculate 2+2 for me?
────────────────────────────────────────────────────────────────

🎯 Orchestrator analyzing: Can you calculate 2+2 for me?
   → Delegating to Calculator agent

🤖 Orchestrator Response:

I delegated to the calculator agent:
The answer is 4
```

## Key Code Patterns

### Creating an A2A Agent Executor

```go
// Wrap your agent with A2AExecutor
executor := a2a.NewExecutorWithConfig(myAgent, a2a.ExecutorConfig{
    Name:        "MyAgent",
    Description: "Does something useful",
})
```

### Composing Agents

```go
// Parent agent with sub-agents
parent := a2a.NewExecutor(parentAgent).
    WithSubAgent("calculator", calcExecutor).
    WithSubAgent("search", searchExecutor)
```

### Calling Sub-Agents

```go
// Method 1: Simple text call
result, err := parent.CallSubAgentSimple(ctx, "calculator", "What is 2+2?")

// Method 2: Full a2a message
msg := a2a.NewUserMessage("What is 2+2?")
artifact, err := parent.CallSubAgent(ctx, "calculator", msg)
text := a2a.ExtractTextFromArtifact(artifact)
```

### Listing Sub-Agents

```go
// Get all sub-agent names
subAgents := executor.ListSubAgents()
for _, name := range subAgents {
    fmt.Println(name)
}
```

### Getting Agent Card

```go
// Retrieve agent metadata and capabilities
card := executor.GetAgentCard()
fmt.Printf("Name: %s\n", card.Name)
fmt.Printf("Capabilities: %d\n", len(card.Capabilities))
```

## Understanding the A2A Protocol

### Message Flow

1. **User Message** → Parent Agent
2. Parent Agent → **Decision Logic** (which sub-agent to call?)
3. Parent → **A2A Message** → Sub-Agent
4. Sub-Agent → **A2A Artifact** → Parent
5. Parent → **Final Response** → User

### Key A2A Types

- **Message** - Communication unit with parts (text, files, data)
- **Task** - Execution tracking with status and artifacts
- **Artifact** - Output from agent execution
- **AgentCard** - Agent metadata and capabilities

## Advanced Usage

### Multi-Level Hierarchies

```go
// Create 3-level hierarchy
leaf := a2a.NewExecutor(leafAgent)
middle := a2a.NewExecutor(middleAgent).WithSubAgent("leaf", leaf)
top := a2a.NewExecutor(topAgent).WithSubAgent("middle", middle)

// Top can call middle, which can call leaf
result, _ := top.CallSubAgentSimple(ctx, "middle", "question")
```

### Context Propagation

```go
// Context flows through all sub-agent calls
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

artifact, err := orchestrator.Execute(ctx, msg)
// Timeout applies to all sub-agent calls
```

## Design Principles

1. **Composability** - Agents can be freely composed without modification
2. **Protocol-First** - All communication uses a2a message format
3. **Type-Safety** - Go interfaces ensure compile-time correctness
4. **Interoperability** - Compatible with Google's ADK Python agents (via HTTP)

## Related Examples

- **HTTP Server** - See `examples/a2a_server/` for remote agent access via JSON-RPC
- **SSE Streaming** - See `examples/a2a_streaming/` for real-time task updates

## Next Steps

1. Try creating your own custom agents
2. Build deeper hierarchies (3+ levels)
3. Add actual LLM-powered agents using dspy-go modules
4. Experiment with agent collaboration patterns
5. Deploy agents as HTTP services for cross-language interop

## References

- [Google Agent Developer Kit (ADK)](https://github.com/google/adk-python)
- [A2A Protocol Specification](https://developers.google.com/agent-developer-kit/docs)
- [dspy-go Documentation](../../README.md)
