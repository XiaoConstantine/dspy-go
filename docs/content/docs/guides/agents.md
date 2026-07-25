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

The current agent stack is split into a reusable execution core plus higher-level adapters:

- **`pkg/agents`**: provider-neutral execution contracts, canonical messages, typed events, `RunLoop`, `Harness`, and `ExecutionTrace`
- **`pkg/agents/native`**: native tool-calling agents with sessions, tool registration, and typed execution/session events
- **`pkg/agents/react`**: higher-level ReAct-style agents that layer planning, context engineering, and ACE-oriented patterns on top of shared execution contracts
- **`pkg/agents/rlm`**: agent wrapper around the RLM module with the same execution/trace surface
- **`pkg/modules.ReAct`**: the signature-oriented text/XML module, separate from native tool-calling execution

A useful mental model is:

```text
RunLoop   = one model/tool execution algorithm
Harness   = transcript, active-run lifecycle, and cancellation
native    = tool-calling adapter with sessions and tools
react     = higher-level ReAct wrapper
rlm       = RLM-backed agent wrapper
modules   = text/XML module layer
```

If you are starting fresh and want a runnable example, begin with `examples/native_agent_session`.
If you need a signature-oriented text/XML loop, use `pkg/modules.ReAct` directly.

---

## Choosing an Agent Style

### `pkg/modules.ReAct`

Use `pkg/modules.ReAct` when you want a signature-driven text/XML module that stays close to the core DSPy module model.
This is the right fit for structured prompt programming and module composition.

### `pkg/agents/native`

Use `pkg/agents/native` when you want an application-facing adapter for provider-native tool calling, typed execution events, sessions, and operation-scoped traces.
The reusable execution core remains `pkg/agents` (`RunLoop` and `Harness`).

### `pkg/agents/react`

Use `pkg/agents/react` when you want a higher-level ReAct agent wrapper with shared execution contracts plus additional planning/context behavior.

### `pkg/agents/rlm`

Use `pkg/agents/rlm` when you want the RLM module exposed as an agent with the same trace and optimization surfaces as other agent families.

---

## Native Tool-Calling Agent

The most current end-to-end agent surface is `pkg/agents/native`.
A minimal setup looks like this:

```go
llms.EnsureFactory()
llm, err := llms.NewLLM(apiKey, core.ModelGoogleGeminiFlash)
if err != nil {
    panic(err)
}

toolset, err := filetools.NewToolset(filetools.Config{
    Root: workspaceDir,
})
if err != nil {
    panic(err)
}

agent, err := native.NewAgent(llm, native.Config{
    MaxTurns:           12,
    SystemPrompt:       "Use the workspace tools to inspect files and finish the task.",
    EventSink:          myExecutionSink,
    SessionEventSink:   mySessionSink,
    SessionEventStore:  store,
    SessionID:          "example-session",
})
if err != nil {
    panic(err)
}
for _, tool := range toolset.Tools() {
    if err := agent.RegisterTool(tool); err != nil {
        panic(err)
    }
}

execution, err := agent.ExecuteWithTrace(ctx, map[string]any{
    "task": "Inspect the workspace and report what you changed.",
})
if err != nil {
    panic(err)
}

fmt.Println(execution.Output["final_answer"])
fmt.Println(execution.Trace.TerminationCause)
```

See `examples/native_agent_session` for a compilable version with SQLite-backed session recall, branching, and typed event printing.

### How Native Execution Works

A native run typically follows this lifecycle:

1. the model receives the task plus current transcript state
2. the model proposes tool calls or a finish action
3. registered tools execute through the shared `RunLoop`
4. typed `ExecutionEvent` values report run, turn, message, and tool outcomes
5. `ExecutionTrace` is projected from those events

When you need the output and trace for the same run, prefer `ExecuteWithTrace(...)` over reading a later mutable trace accessor.

---

## Tools

Tool registration is explicit: create or load tools, then register them on the agent.
For a workspace-safe file baseline, use `pkg/tools/files`, which provides rooted `ls`, `read`, `write`, and `edit` operations.

`pkg/tools/defaults` also includes unrestricted `bash`. Its `Root` config sets the command's initial working directory; it is not an OS sandbox and does not prevent access to absolute paths, the user's home directory, the network, or other process-visible resources. Enable shell access only as an explicit trusted opt-in.

If you need custom tools, implement the current `core.Tool` contract rather than the older string-action examples that appeared in pre-Phase-10 material.
For a current end-to-end example, use the native agent examples and built-in toolsets as the reference surface.

---

## Memory, Sessions, and State

The reusable execution layer now centers state around transcripts, harness lifecycle, and optional native session persistence.
For native agents, `SessionEventStore` performs recall and persistence. `SessionEventSink` only observes typed session lifecycle notifications; it does not store session state.

Use this split:

- **short-lived in-memory execution state**: `Harness` / current transcript
- **persistent native recall and branching**: `native` session event store APIs
- **application-owned long-term memory**: your own storage layer on top of traces, sessions, or exported summaries

Again, `examples/native_agent_session` is the current runnable reference.

---

## Orchestration and Multi-Agent Work

For multi-agent composition, prefer the current A2A and communication/orchestration surfaces documented in the A2A guide instead of the older `NewOrchestrator` / `NewTask` examples that no longer reflect exported APIs.

Good fits are:

- **A2A composition** for explicit agent-to-agent delegation
- **shared `ScopedExecutionAgent` / `ExecuteWithTrace(...)`** when evaluators or orchestrators must keep outputs correlated with traces
- **workflow examples** under `examples/agents` for broader orchestration patterns

---

## Reflection Patterns

Reflection and self-improvement now show up primarily through ACE and the optimizable-agent surfaces below, rather than through the older standalone reflection sketches.
If you want runnable self-improvement examples, jump to the ACE section and the optimization examples.

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

## Stateful Agent Applications

Stateful applications should compose the current execution surfaces rather than copy an application-specific agent loop:

- use `Harness` for transcript ownership, overlap protection, and cancellation
- use `ExecuteWithTrace(...)` when output/trace correlation matters
- use a native `SessionEventStore` when runs need persistent recall or branching
- consume `SessionEventSink` only for typed lifecycle notifications
- keep application-specific memory, workspace ownership, and UI state outside the reusable execution loop

The runnable `examples/native_agent_session` program demonstrates this composition. `Maestro` is a larger application example that adds a workspace session and frontend around the same execution spine.

---

## Reusable Agent Execution

Native tool execution now uses the provider-neutral contracts in `pkg/agents`.
Consume typed execution events through `EventSink`, native-only session events
through `SessionEventSink`, and prefer `ExecuteWithTrace` when you need the
output and trace from the same execution:

```go
var executionEvents []agents.ExecutionEvent
var sessionEvents []native.SessionEvent

agent, err := native.NewAgent(llm, native.Config{
    MaxTurns: 20,
    SessionID: "example-session",
    EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
        executionEvents = append(executionEvents, agents.CloneExecutionEvent(event))
    }),
    SessionEventSink: native.SessionEventSinkFunc(func(_ context.Context, event native.SessionEvent) {
        sessionEvents = append(sessionEvents, event)
    }),
})
if err != nil {
    panic(err)
}

execution, err := agent.ExecuteWithTrace(context.Background(), map[string]any{
    "task": "Inspect the repository and finish.",
})
if err != nil {
    panic(err)
}

fmt.Println(execution.Output["completed"], execution.Trace.TerminationCause)
```

`ExecutionEvent` payloads describe balanced run and turn lifecycles plus
balanced terminal outcomes for proposed tool calls. Message additions are
immutable point-in-time events. `ExecutionTrace` is projected from those
canonical events and is defensively cloned before being returned to callers.
Use `ExecuteWithTrace` when you need operation-scoped correlation instead of a
later mutable trace lookup.

### Pre-1.0 migration notes

The reusable execution-layer cleanup shipped as a pre-1.0 breaking change in
`v0.86.0`:

- replace `native.Config.OnEvent` with `native.Config.EventSink`
- replace `agents.AgentEvent` string/map callbacks with typed
  `agents.ExecutionEvent` payloads
- replace native `session_loaded` and `session_persisted` callback maps with
  `native.SessionEventSink`
- replace `LastNativeTrace` and native-specific trace structs with
  `agents.ExecutionTrace`; prefer `ExecuteWithTrace` when you need the trace
  correlated with one specific execution
- configure native ReAct execution through
  `react.WithNativeFunctionCalling(...)`; direct `modules.ReAct` remains the
  text/XML module and no longer installs a native function-calling interceptor

Users that still need the removed APIs should remain on the previous minor
series while migrating. See `examples/native_agent_session` for a compilable
typed execution/session event example.

## Optimizable Agents

dspy-go now has a shared optimization surface for several agent families:

- native agents in `pkg/agents/native`
- ReAct agents in `pkg/agents/react`
- RLM-backed agents in `pkg/agents/rlm`

These agents expose mutable artifact state through the `OptimizableAgent` contract and can participate in GEPA workflows.

### Persisted Optimized Programs

Optimizable agents can export and apply a shared persisted envelope:

```go
program, err := agent.ExportOptimizedProgram()
if err != nil {
    panic(err)
}

loaded, err := optimize.ReadOptimizedAgentProgram("optimized_program.json")
if err != nil {
    panic(err)
}

if err := agent.ApplyOptimizedProgram(loaded); err != nil {
    panic(err)
}
```

That envelope stores stable target IDs rather than only raw artifact maps, which makes saved optimization state portable across agent types.

### Stable Optimization Targets

Examples of user-facing target IDs include:

- `root.rlm.iteration`
- `root.rlm.max_iterations`
- `root.rlm.adaptive.enabled`
- `root.react.tool_policy`

These stable IDs are what GEPA feedback, persisted optimized programs, and replay workflows operate on.

### Full Workflow

For end-to-end optimization, prefer `optimize.RunGEPAWorkflow(...)` over hand-wiring `Optimize(...)` and `SetArtifacts(...)`.

That workflow gives you:

- baseline evaluation
- GEPA optimization
- saved optimized-program artifacts
- restore + replay on held-out examples

See:

- `examples/rlm_oolong_gepa`
- the [Optimizers](optimizers/) guide

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
- **[Agents Package Examples](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/agents)** - workflow-style orchestration examples
- **[ACE Basic Example](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/ace_basic)** - Standalone ACE usage (no LLM)
- **[ACE + ReAct Example](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/ace_react)** - Self-improving ReAct agent
- **[Maestro](https://github.com/XiaoConstantine/maestro)** - Production code review agent
- **[Smart Tool Registry](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/smart_tool_registry)** - Advanced tool management

### Running the Examples

```bash
# Basic agent examples
cd examples/agents && go run .

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
