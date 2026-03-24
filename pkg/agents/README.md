# Agents Package

`pkg/agents` contains the runtime building blocks for long-lived assistants, specialized worker agents, and orchestration layers.

The package now has three distinct roles:

- `native`: provider-native tool-calling agents for modern models
- `react`: ReAct-based agents for compatibility, research, and benchmarking
- `orchestrator`: control-plane decomposition and task execution across bounded processors

It also includes shared infrastructure:

- `sessionevent`: persistent branchable session storage
- `memory`: memory backends and helpers
- `communication`: cross-agent/subagent execution patterns
- `ace`: trajectory recording and self-improvement hooks
- `rlm`: recursive context-processing strategy components
- shared traces, events, message types, and tool-observation normalization

## Choosing The Right Abstraction

### `native.Agent`

Use the native agent when you want one model to directly drive tools.

Best for:

- interactive assistants
- coding agents
- session-based Q&A
- Pi-style minimal tool packs
- modern tool-calling models

Key properties:

- provider-native tool calling
- tool interceptors
- structured lifecycle events
- session recall and branch-aware resume
- detailed execution traces

Relevant code:

- [native/agent.go](./native/agent.go)
- [native/session.go](./native/session.go)
- [native/session_control.go](./native/session_control.go)

### `react.ReActAgent`

Use ReAct when you explicitly want a text-mediated reasoning loop.

Best for:

- weaker or non-native-tool-calling models
- experimentation
- benchmarking against classic agent patterns
- research on reasoning styles

ReAct is still useful, but it is no longer the default mental model for modern agents in this repo.

Relevant code:

- [react/react_agent.go](./react/react_agent.go)
- [react/README.md](./react/README.md)

### `FlexibleOrchestrator`

Use the orchestrator when the top-level problem is not “let one agent figure it out,” but “split work, route bounded tasks, and enforce policy.”

Best for:

- batch workflows
- multi-stage pipelines
- parallel task execution
- review/control planes that supervise specialized workers

The orchestrator should own:

- task decomposition
- routing
- dependency management
- retries and fallbacks
- validation gates
- final decision authority

It should not be treated as just “another smart agent.”

Relevant code:

- [orchestrator.go](./orchestrator.go)

## Agent vs Subagent vs Orchestrator

These are different roles.

### Main Agent

The main agent owns an end-user task directly.

Example:

- “answer this repository question”
- “edit this file”
- “complete this coding task”

In `dspy-go`, that is usually a `native.Agent`.

### Dedicated Subagent

A subagent is a bounded specialist invoked by something else.

Example:

- a review assistant that inspects one PR chunk
- a reply assistant that drafts one thread response
- a verifier that checks one candidate finding

The important point is not “smaller model” vs “bigger model.”  
The important point is scope:

- narrow input
- narrow tool surface
- strict budgets
- structured result

You can build dedicated subagents with `native.Agent`, `react.ReActAgent`, or the `communication` package.

### Orchestrator

An orchestrator supervises work instead of doing the whole reasoning job itself.

Example:

- split a PR into chunk-review tasks
- invoke a dedicated review subagent per chunk
- dedupe results
- decide what survives to final output

That is why an orchestrator is a control plane, not just a bigger agent.

## Recommended Architecture Patterns

### General Interactive Assistant

Use:

- `native.Agent`
- a small tool pack
- `sessionevent` for persistence

This is the default direction for modern assistants in this repo.

### Specialized Worker Agent

Use:

- `native.Agent` or `react.ReActAgent`
- a restricted tool pack
- low turn/tool budgets
- structured outputs

This is the right pattern for things like code review assistants, verification workers, or bounded repo analyzers.

### Supervisory Workflow

Use:

- `FlexibleOrchestrator`
- task processors
- dedicated worker agents underneath

This is the right pattern when global policy matters more than open-ended autonomy.

## Runtime Strategy vs Optimization

Not every agent concept belongs at the same layer.

### Execution Shape

These define who is responsible for doing work:

- main agent
- dedicated subagent
- orchestrator

This README is primarily about that layer.

### Runtime Strategy

These define how work is executed once you have chosen an agent shape:

- native tool calling
- ReAct
- session recall
- tool policies
- RLM / recursive context reduction

RLM is not an alternative to an agent or orchestrator.  
It is a strategy for handling large context efficiently.

Use RLM when the hard problem is:

- too much context
- recursive summarization
- cost reduction under large inputs
- bounded context windows for specialized workers

### Optimization Layer

These define how an agent or workflow improves over time:

- GEPA
- ACE
- trajectory recording
- artifact and prompt tuning

GEPA is not the execution model.  
It is the optimization loop that can improve:

- a native interactive agent
- a specialized review subagent
- a verifier
- an orchestrator prompt or decomposition strategy

ACE is similar in that it augments learning and trajectory capture rather than replacing the runtime shape.

## Sessions And Continuity

`sessionevent` provides the persistent session model for native agents.

It supports:

- session creation
- active branches
- forked branches
- lineage loading
- out-of-band summaries

Relevant code:

- [sessionevent/store.go](./sessionevent/store.go)
- [sessionevent/sqlite.go](./sessionevent/sqlite.go)

This is the right base for:

- long-lived assistants
- branchable agent sessions
- recall across runs

## Current Direction

The current architectural direction in `dspy-go` is:

1. native tool-calling first
2. session-aware runtimes
3. small default tool packs
4. orchestrators for supervision, not as the default answer to every agent problem
5. ReAct as compatibility and research infrastructure, not the center of the product architecture
6. RLM as a context-processing strategy where scale demands it
7. GEPA and ACE as optimization layers on top of the runtime, not substitutes for agent design

## Practical Rule Of Thumb

If you are deciding what to build:

- Start with `native.Agent` for one modern interactive agent.
- Add `sessionevent` if the agent needs continuity.
- Introduce a dedicated subagent when one bounded task needs a different prompt, budget, or tool surface.
- Introduce `FlexibleOrchestrator` only when you need a real control plane across multiple bounded tasks.
- Add RLM when context size, recursion, or cost becomes the bottleneck.
- Add GEPA when the runtime shape is stable enough to optimize systematically.

That usually produces a cleaner system than starting with a giant autonomous agent or forcing orchestration too early.
