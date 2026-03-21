# Agent Framework Comparison: Pi-Mono vs Hermes vs DSPy-Go

## Executive Summary

This document compares three AI agent frameworks — **Pi-Mono** (by Mario Zechner/badlogic), **Hermes Agent** (by Nous Research), and **DSPy-Go** — to identify design patterns, strengths, and opportunities for making DSPy-Go more popular and developer-friendly.

---

## 1. Pi-Mono

**Repository:** [github.com/badlogic/pi-mono](https://github.com/badlogic/pi-mono)
**Stars:** ~26.6k | **Language:** TypeScript | **License:** MIT

### Philosophy
Pi takes a "build it yourself" anti-framework approach. Instead of baking in sub-agents, plan modes, and permission gates, it provides minimal primitives (read, write, edit, bash) and lets users extend everything via TypeScript extensions.

### Architecture
A monorepo with 7 packages:

| Package | Purpose |
|---------|---------|
| `pi-ai` | Unified LLM API (OpenAI, Anthropic, Google, etc.) |
| `pi-agent-core` | Agent execution runtime, tool invocation, state persistence |
| `pi-coding-agent` | Interactive CLI coding agent |
| `pi-mom` | Slack bot integration |
| `pi-tui` | Terminal UI with differential rendering |
| `pi-web-ui` | Web components for AI chat |
| `pi-pods` | vLLM deployment management |

### Key Strengths

1. **Radical extensibility** — Extensions can register tools, commands, keyboard shortcuts, event handlers, and custom UI components. The entire lifecycle is hookable (session start, agent start, tool calls, context modification, compaction).

2. **Minimal core, maximum composability** — Only 4 built-in tools. Everything else is user-provided. Sub-agents, plan mode, permission gates, git checkpointing — all implemented as extensions.

3. **Session tree** — JSONL-based session persistence with tree branching. Users can navigate, fork, and compact sessions while preserving full history.

4. **4 operating modes** — Interactive (TUI), print/JSON, RPC (process integration), SDK (embedded). Decouples presentation from agent logic.

5. **Package ecosystem** — Extensions, skills, prompts, and themes bundled as installable packages from npm, git, or URLs.

6. **Developer experience** — Hot reload for extensions, `@` file references, tab completion, image paste, `!bash` shortcuts.

### Key Weaknesses

1. **TypeScript only** — Not suitable for Go-based systems or performance-critical backends.
2. **No optimization/learning** — No built-in prompt optimization, few-shot learning, or self-improvement.
3. **No structured output** — Relies on raw LLM output rather than typed signatures.
4. **Coding agent focus** — Primarily designed for coding tasks, not general-purpose agent orchestration.
5. **Security model** — Extensions run with full system access; no sandboxing.

### Code Example
```typescript
// Extension: custom tool
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "greet",
    description: "Greet someone by name",
    parameters: Type.Object({
      name: Type.String({ description: "Name to greet" }),
    }),
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      return { content: [{ type: "text", text: `Hello, ${params.name}!` }] };
    },
  });
}

// SDK embedding
const { session } = await createAgentSession({
  sessionManager: SessionManager.inMemory(),
  modelRegistry: new ModelRegistry(authStorage),
});
await session.prompt("What files are in the current directory?");
```

---

## 2. Hermes Agent

**Repository:** [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
**Stars:** ~4.5k | **Language:** Python | **License:** MIT

### Philosophy
"The agent that grows with you." Hermes is a self-improving agent with a closed-loop learning system. It creates skills from experience, improves them during use, persists knowledge across sessions, and models user preferences over time.

### Architecture
A Python-based agent with a synchronous conversation loop:

1. Accept user message → build system prompt
2. Call LLM with available tools
3. Execute tool calls → append results
4. Repeat until model stops requesting tools

Key components:
- **AIAgent** — Core loop in `run_agent.py`
- **Tool Registry** — Central tool discovery and dispatch (`tools/registry.py`)
- **SessionDB** — SQLite + FTS5 for session search and memory
- **Skills** — User-created commands in `~/.hermes/skills/`
- **Gateway** — Multi-platform messaging (Telegram, Discord, Slack, WhatsApp, Signal)

### Key Strengths

1. **Self-improvement loop** — Autonomously creates and improves skills from execution traces. Uses DSPy + GEPA for reflective prompt evolution via the [hermes-agent-self-evolution](https://github.com/NousResearch/hermes-agent-self-evolution) companion repo.

2. **Multi-platform** — Single agent accessible from Telegram, Discord, Slack, WhatsApp, Signal, and CLI simultaneously via a unified gateway.

3. **Memory architecture** — FTS5 session search with LLM summarization, Honcho-based dialectic user modeling, agent-curated periodic memory nudges.

4. **Flexible deployment** — 6 terminal backends (local, Docker, SSH, Daytona, Singularity, Modal). Serverless persistence on Daytona/Modal.

5. **Skills ecosystem** — 40+ bundled skills, open agentskills.io format, autonomous creation of new skills.

6. **RL training pipeline** — Batch trajectory generation, Atropos RL environments, trajectory compression for training tool-calling models.

7. **One-line install** — Handles Python, Node.js, dependencies automatically.

### Key Weaknesses

1. **Python only** — Not embeddable in Go services without inter-process communication.
2. **Synchronous loop** — Basic agent execution model; no parallel tool execution or advanced orchestration.
3. **No typed signatures** — Relies on LLM tool-calling conventions rather than typed I/O contracts.
4. **Monolithic design** — Hard to use individual components (memory, tools) without the full agent.
5. **Heavy dependencies** — Requires Python, Node.js, and many packages.

### Code Example
```python
# Tool definition
@register_tool
def my_tool(params: dict) -> dict:
    """Tool description for LLM."""
    return {"result": "..."}

# Skill definition (SKILL.md)
# name: deploy-app
# description: Deploy the current application
# steps:
#   1. Run tests
#   2. Build Docker image
#   3. Push to registry
```

---

## 3. DSPy-Go (This Project)

**Repository:** Current project
**Language:** Go | **License:** MIT

### Philosophy
Port of Python DSPy's programming model to Go — treating LLM calls as optimizable, composable modules with typed signatures. Focus on programmatic prompt optimization rather than manual prompt engineering.

### Architecture

| Package | Purpose |
|---------|---------|
| `core/` | Module, Signature, LLM interfaces |
| `agents/react/` | ReAct agent with multiple execution modes |
| `agents/ace/` | Self-improving agent framework (ACE) |
| `agents/memory/` | In-memory, filesystem, SQLite storage |
| `agents/context/` | Manus-inspired context engineering |
| `agents/workflows/` | Chain, Parallel, Router orchestration |
| `agents/communication/` | A2A (Agent-to-Agent) protocol |
| `modules/` | Predict, ChainOfThought, ReAct, Refine, Parallel, MultiChainComparison |
| `tools/` | Registry, smart selection, MCP, pipelines |
| `optimizers/` | GEPA, MIPRO, SIMBA, COPRO, Bootstrap |
| `llms/` | Anthropic, OpenAI, Gemini, Ollama, LlamaCPP |
| `interceptors/` | Pre/post processing hooks |

### Key Strengths

1. **Typed signatures** — Input/output contracts with field types (text, image, audio). Compile-time safety for LLM interactions.

2. **Prompt optimization** — 5 built-in optimizers (GEPA, MIPRO, SIMBA, COPRO, Bootstrap). No other Go framework offers this.

3. **Advanced agent patterns** — ReAct with multiple modes (ReAct, ReWOO, Hybrid), self-reflection, ACE self-improvement, A2A protocol.

4. **Composable modules** — ChainOfThought, MultiChainComparison, Parallel, Refine — all chainable with interceptors.

5. **Go-native** — Single binary, no runtime dependencies, excellent concurrency, type safety.

6. **Smart tool selection** — Bayesian probabilistic selection, intent matching, performance tracking.

7. **Context engineering** — Manus-inspired compression, KV-cache optimization, error retention.

8. **Comprehensive testing** — 154 test files, CI/CD ready.

9. **Multi-provider** — Anthropic, OpenAI, Gemini, Ollama, LlamaCPP with capability system.

### Key Weaknesses

1. **Steep learning curve for advanced features** — ACE, GEPA, context engineering require deep understanding.
2. **No CLI agent** — Unlike Pi and Hermes, no standalone agent binary for immediate use.
3. **No multi-platform messaging** — No Telegram/Discord/Slack gateway.
4. **No skill/extension ecosystem** — No package marketplace or community extensions.
5. **Documentation gaps** — Missing architecture diagrams, advanced pattern guides.
6. **Map-based I/O** — `map[string]interface{}` everywhere instead of generics-based typed I/O.

---

## Side-by-Side Comparison

| Feature | Pi-Mono | Hermes | DSPy-Go |
|---------|---------|--------|---------|
| **Language** | TypeScript | Python | Go |
| **Stars** | ~26.6k | ~4.5k | — |
| **Primary Use** | Coding agent | Self-improving assistant | LLM programming framework |
| **CLI Agent** | Yes | Yes | No |
| **Extension System** | Excellent | Good (skills) | None |
| **Prompt Optimization** | No | Via DSPy/GEPA addon | Built-in (5 optimizers) |
| **Typed I/O** | No | No | Signatures (partial) |
| **Tool System** | 4 built-in + extensions | Registry + 40+ skills | Smart registry + MCP |
| **Memory** | Session JSONL | SQLite + FTS5 | In-memory/FS/SQLite |
| **Multi-Provider** | 20+ providers | 200+ via OpenRouter | 5+ direct |
| **Multi-Platform** | No | Yes (5 platforms) | No |
| **Agent Patterns** | User-built | Simple loop | ReAct, ReWOO, ACE, A2A |
| **Concurrency** | Single-threaded | Sync loop | Native goroutines |
| **Deployment** | npm install | curl one-liner | `go get` |
| **Self-Improvement** | No | Yes (core feature) | Yes (ACE) |
| **RL Training** | No | Yes | No |
| **Interceptors/Hooks** | Full lifecycle | Limited | Module + Agent + Tool |
| **Orchestration** | User-built | Basic | Chain/Parallel/Router |
| **Package Ecosystem** | npm/git packages | agentskills.io | None |

---

## What Makes Pi and Hermes Popular

### Pi-Mono's Success Factors
1. **Immediate value** — Install, run, start coding. Zero configuration needed.
2. **Famous creator** — Mario Zechner (libGDX) has strong developer following.
3. **Extensibility story** — "Build it yourself" resonates with power users who feel constrained by opinionated tools.
4. **Low barrier** — TypeScript is widely known; npm install is frictionless.
5. **Multiple modes** — CLI, SDK, RPC, and web — meets developers where they are.
6. **Visual polish** — Custom TUI with themes, keyboard shortcuts, and real-time rendering.

### Hermes's Success Factors
1. **Self-improvement narrative** — "The agent that grows with you" is compelling marketing.
2. **Multi-platform** — Access from Telegram/Discord means non-developers can use it.
3. **One-line install** — `curl | bash` handles everything.
4. **Nous Research brand** — Known for Hermes LLM models; built-in community.
5. **Skills ecosystem** — 40+ bundled skills provide immediate utility.
6. **RL training** — Appeals to ML researchers who want to train better models.

---

## Recommendations for DSPy-Go

### High-Impact, Low-Effort

1. **Create a standalone CLI agent** — A `dspy-agent` binary that works out of the box with `go install`. This is the single biggest gap. Both Pi and Hermes succeed because users can start immediately.

2. **Add generics-based typed I/O** — Replace `map[string]interface{}` with `Process[I, O](ctx, input I) (O, error)`. Go 1.18+ generics make this possible and would dramatically improve DX.

3. **One-command quickstart** — `go install github.com/XiaoConstantine/dspy-go/cmd/dspy@latest && dspy init` to scaffold a project with working examples.

4. **Better README with "5-minute" examples** — Show the simplest possible agent in 10 lines, then progressively disclose complexity. Pi's README starts with install + run. Hermes starts with curl + go.

### Medium-Impact, Medium-Effort

5. **Extension/plugin system** — Allow users to register custom tools, modules, and optimizers as Go plugins or via a registry. Follow Pi's model of lifecycle hooks.

6. **MCP-first tool ecosystem** — Lean into Model Context Protocol as the standard for tool interop. This gives DSPy-Go access to the growing MCP tool ecosystem without building custom integrations.

7. **Add a skills/recipe system** — Pre-built compositions (e.g., "research agent", "code review agent", "data analysis agent") that users can use directly or customize.

8. **Interactive documentation site** — Architecture diagrams, pattern guides, and runnable examples. The advanced features (GEPA, ACE, context engineering) need visual explanations.

### High-Impact, High-Effort

9. **Multi-platform messaging gateway** — A gateway package that connects agents to Telegram, Discord, Slack. Hermes shows this dramatically increases adoption by reaching non-developer users.

10. **Agent marketplace / registry** — A place to share and discover agent configurations, optimized prompts, and tool compositions.

11. **Self-evolution pipeline** — Package the ACE + GEPA combination as a turnkey self-improvement system, similar to hermes-agent-self-evolution. This is a unique differentiator.

### DSPy-Go's Unique Advantages to Emphasize

- **Only Go framework with prompt optimization** — No competitor offers GEPA, MIPRO, SIMBA in Go.
- **Production-grade concurrency** — Go's goroutines + channels beat Python/TypeScript for parallel agent execution.
- **Single binary deployment** — No Python/Node.js runtime needed. Critical for cloud-native and edge deployments.
- **Type safety** — Signatures catch errors at compile time, not runtime.
- **ACE self-improvement** — Built-in, not an addon. Combined with GEPA, this is unmatched in the Go ecosystem.

### Priority Roadmap

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P0 | Standalone CLI agent binary | Very High | Medium |
| P0 | Generics-based typed I/O | High | Medium |
| P1 | One-command quickstart + better README | High | Low |
| P1 | Pre-built agent recipes/templates | High | Low |
| P1 | MCP tool ecosystem integration | High | Medium |
| P2 | Extension/plugin system | Medium | Medium |
| P2 | Interactive docs with diagrams | Medium | Medium |
| P2 | Multi-platform gateway | Medium | High |
| P3 | Agent marketplace | Low | High |
| P3 | Self-evolution turnkey pipeline | Medium | High |

---

## Conclusion

Pi-Mono and Hermes succeed through **immediate usability** (install and go), **extensibility** (customize everything), and **ecosystem** (skills, extensions, multi-platform). DSPy-Go has stronger technical foundations (typed signatures, prompt optimization, Go concurrency) but lacks the **on-ramp experience** that makes frameworks go viral.

The path to popularity: **make the first 5 minutes magical** (CLI agent + quickstart), **leverage unique strengths** (optimization + Go performance), and **build ecosystem** (MCP tools + recipes + extensions). DSPy-Go doesn't need to copy Pi or Hermes — it needs to make its existing power accessible to developers who want production-grade AI agents in Go.
