# TypeSafe cheap-gate program

This example composes experimental System One decisions and a generative
DSPy-Go module in one `core.Program`:

```text
support ticket
    |
    v
Decide: answerable + needs_human + category
    |-- human review ----------> stop before generation
    |-- account tools required -> stop before generation
    `-- answerable ------------> ChainOfThought drafts a bounded acknowledgment
```

The built-in batch contains two tickets eligible for a bounded acknowledgment,
one production incident, and one account-specific refund request. The gate
therefore invokes the reply-writing LLM for only two of four tickets. Because
this example supplies no product documentation, generated replies must not
invent navigation steps, product behavior, or policy.

> **Experimental:** The packages and their persisted formats may change or be
> removed in a dspy-go v0 minor release.

## Run entirely offline

Replay mode starts a local `httptest` System One server backed by recorded JSON
responses. It also uses a deterministic stand-in for the generative LLM, so it
requires no API keys or network access:

```bash
go run ./examples/typesafe_decide -replay
```

The summary should report four System One calls, two generative calls, and two
generative calls avoided. Replay probabilities are fixtures for exercising the
program; they are not evidence about live Jev quality or calibration.

## Run live

Set credentials for TypeSafe and the default Gemini generator:

```bash
export TYPESAFE_API_KEY="..."
export GEMINI_API_KEY="..."
go run ./examples/typesafe_decide
```

Select explicit models or provide the generative key directly:

```bash
go run ./examples/typesafe_decide \
  -model YOUR_PINNED_JEV_MODEL \
  -llm-model YOUR_GENERATIVE_MODEL \
  -llm-api-key YOUR_GENERATIVE_KEY
```

Process one custom ticket in live mode:

```bash
go run ./examples/typesafe_decide \
  -ticket "How do I rotate an API key?"
```

Replay mode only recognizes the exact recorded System One requests. A change to
the ticket, model, question instructions, or Choice criteria fails rather than
silently reusing stale evidence.

## What to notice

- `Decide` and `ChainOfThought` are sibling `core.Module` values in one
  `core.Program`.
- Program-wide `SetLLM` is applied to every module. It configures
  `ChainOfThought`, while `Decide.SetLLM` remains a no-op and its explicit
  System One client continues receiving all decision calls.
- `decide.Get[T]` reads typed evidence without unchecked assertions.
- Human and tool routes avoid generative calls. A real application could replace
  the tool-route short circuit with ReAct after defining safe tools and approval
  policy; this example deliberately keeps that branch deterministic.
- `ProcessDecision` creates a trace span containing model, request ID, and token
  usage metadata.
- Ordinary tests and replay runs need no provider credentials. Live mode incurs
  TypeSafe and generative-provider requests.
