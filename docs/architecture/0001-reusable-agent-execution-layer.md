# ADR 0001: Make `pkg/agents` the reusable execution layer

- Status: Accepted
- Date: 2026-07-23
- Tracking issue: [#207](https://github.com/XiaoConstantine/dspy-go/issues/207)

## Context

DSPy-Go currently has several agent execution paths:

- `pkg/modules.ReAct.ProcessWithTrace` owns the signature-oriented text/XML
  ReAct loop.
- `pkg/agents/react.ReActAgent.Execute` adds context engineering, planning,
  reflection, and ACE around `modules.ReAct`.
- `pkg/agents/native.Agent.Execute` owns a separate provider-native tool loop as
  well as prompt construction, sessions, skills, optimization, tracing, and
  events.
- `pkg/agents/loop.go` declares loop types, but there is no `RunLoop`
  implementation and no caller uses those types.

`pkg/llms` already owns provider implementations, and `pkg/core` already owns
provider-neutral LLM and tool primitives. The missing layer is a small,
reusable execution loop that application concerns can wrap.

## Decision

`pkg/agents` will own the canonical message model, typed model adapter
boundary, typed execution events, a pure sequential `RunLoop`, and a stateful
`Harness`.

The intended dependency direction is:

```text
pkg/llms  -> pkg/core
pkg/agents -> pkg/core
integration adapters -> pkg/agents + pkg/core
pkg/agents/native -> pkg/agents + session/skill/optimization adapters
```

In particular, the portable `pkg/agents` layer must not import:

- `pkg/llms`
- `pkg/agents/native`
- `pkg/agents/sessionevent`
- `pkg/agents/skills`
- `pkg/agents/optimize`
- `pkg/agents/communication`
- CLI or UI packages

Provider implementations remain in `pkg/llms`; this decision does not rename
that package.

### Responsibility split

```text
RunLoop = one model/tool execution algorithm
Harness = transcript, active-run lifecycle, and cancellation
Native Agent = native configuration and compatibility/application adapters
```

The first migration target is `pkg/agents/native`. Text/XML ReAct, ReWOO,
hybrid planning, ACE, and RLM are not required to move in the first slice.
They may consume shared primitives later where the abstraction fits.

### Loop invariants

The reusable loop will enforce these invariants:

1. Every started run has exactly one terminal run event.
2. Every started turn has exactly one terminal turn event.
3. Every proposed tool call reaches exactly one terminal outcome.
4. Unknown tools, invalid arguments, blocked tools, and execution failures
   become model-visible error tool results.
5. Cancellation prevents another model turn.
6. Returned result, error, terminal event, trace, and persisted status agree.
7. The loop owns transcript mutation; wrappers supply initial messages and
   consume results/events.

### Initial scope

The first loop implementation will be sequential. It will not initially add
parallel tool execution, provider streaming deltas, steering queues, or a
large strategy interface. Those features require concrete use cases and tests
before they become part of the portable contract.

## Existing behavior to preserve or migrate deliberately

The native implementation has behavior that compatibility tests now record.
Some behavior is desirable and should be preserved; some is inconsistent and
must change only with an explicit compatibility decision.

| Condition | Current native behavior | Migration direction |
|---|---|---|
| Explicit `Finish` call | Completes successfully; emits `tool_call_proposed` but no tool terminal event | Preserve completion; balance the typed tool lifecycle |
| Multiple calls in one response | Executes sequentially in provider order | Preserve |
| Unknown tool | Adds an error observation and continues; only a proposed event is emitted | Preserve model-visible recovery; add a terminal tool-result event |
| Invalid arguments | Adds an error observation and continues; only a proposed event is emitted | Preserve model-visible recovery; add a terminal tool-result event |
| Blocked tool | Adds a synthetic error observation and continues | Preserve |
| Tool execution error | Adds an error observation and continues | Preserve |
| Provider/model error | Returns a Go error and emits failed plus finished events | Preserve error propagation; replace overlapping terminal events with one typed terminal event |
| Repeated responses without calls | Returns `completed:false` with a nil Go error | Represent explicitly with a typed stop reason; legacy adapter may preserve the map contract temporarily |
| Maximum turns reached | Returns `completed:false` with a nil Go error and does not emit `run_failed` | Represent explicitly with a typed stop reason; legacy adapter may preserve the map contract temporarily |
| Context already canceled | Reaches the model call, returns `context.Canceled`, and emits the normal provider-error lifecycle | Preserve `errors.Is`; the new loop may fail earlier before a model call |
| Tool output channels | Sends model text back to the model while retaining display text/details in traces | Preserve |
| Native tool-call metadata | Preserved across assistant/tool replay | Preserve |

During migration, compatibility projections may retain legacy output maps and
string/map events, but the canonical loop result and events must not encode
failure solely inside `map[string]any`.

## Alternatives considered

### Keep the native loop in `pkg/agents/native`

Rejected because other agents and frontends cannot reuse the execution
lifecycle without depending on sessions, skills, optimization, and native
trace details.

### Move provider implementations into `pkg/agents`

Rejected. `pkg/llms` already has the correct provider responsibility. The loop
needs a narrow consumer-owned interface or adapter, not provider code.

### Force every agent through one loop immediately

Rejected. RLM and ReWOO have different execution semantics, and text/XML ReAct
has compatibility requirements. The initial goal is one authoritative native
tool-calling loop, not one abstraction for every workflow.

### Implement all fields currently declared in `LoopConfig`

Rejected. Several fields and `LoopStrategy` methods have no implementation or
caller. The real loop contract should start with the minimum behavior exercised
by native-agent tests.

## Consequences

Positive:

- Native tool execution becomes independently testable with fake models/tools.
- Frontends, tracing, and persistence can consume one event protocol.
- Native agents become smaller adapters rather than the execution engine.
- Provider-specific result maps stop leaking into loop logic.

Costs:

- A temporary compatibility layer is required for native output maps and old
  events.
- Canonical message and tool-result semantics must be repaired before moving
  the loop.
- Existing native and ReAct implementations will coexist during migration.

## Compatibility removal timing

DSPy-Go is still pre-1.0 (`v0.x`). The Phase 10 removals are therefore targeted
for the next minor release rather than deferred to a `v1` major release. The
release must call these removals out as breaking changes, and users that still
need `native.Config.OnEvent`, `agents.AgentEvent`, legacy native trace models,
or legacy result-map parsing should remain on the preceding minor series while
migrating.

Native-only session lifecycle notifications move first: first-party consumers
must use `native.SessionEventSink`, after which `session_loaded` and
`session_persisted` string/map notifications can be removed. Portable execution
callbacks follow the same sequence: migrate consumers to `agents.EventSink`,
then remove `OnEvent`, `AgentEvent`, `EmitEvent`, and `LegacyEventSink` before
the Phase 10 release. This does not require a separate major release, but it
does require explicit migration notes and compiled examples.

## Verification

Every implementation phase must run:

```bash
go test ./pkg/agents/...
go test -race ./pkg/agents/...
go vet ./pkg/agents/...
```

The migration will also add an architecture test that rejects forbidden
imports from the portable `pkg/agents` layer.
