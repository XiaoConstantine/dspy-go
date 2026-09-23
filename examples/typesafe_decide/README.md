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
    `-- answerable ------------> ChainOfThought copies an approved acknowledgment
                                  `-- exact local guard releases canonical text
```

The built-in batch contains two tickets eligible for a bounded acknowledgment,
one production incident, and one account-specific refund request. The gate
therefore invokes the reply-writing LLM for only two of four tickets. Ticket
text is untrusted: the model must copy a locally approved acknowledgment
exactly, every other output fails closed, and only the local canonical string is
released. Model-generated procedures are never presented to the user.

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
silently reusing stale evidence. The recorded model is `jev-replay`; explicitly
passing another value with `-replay -model ...` therefore fails.

## What to notice

- `Decide` and `ChainOfThought` are sibling `core.Module` values in one
  `core.Program`.
- Program-wide `SetLLM` is applied to every module. It configures
  `ChainOfThought`, while `Decide.SetLLM` remains a no-op and its explicit
  System One client continues receiving all decision calls.
- `decide.Get[T]` reads typed evidence without unchecked assertions.
- Human and tool routes avoid generative calls. On the draft route, an exact
  allowlist guard rejects any generated wording other than the locally approved
  acknowledgment, including instructions injected through ticket text. A real
  application could replace the tool-route short circuit with ReAct after
  defining safe tools and approval policy; this example deliberately keeps that
  branch deterministic.
- `ProcessDecision` creates a trace span containing model, request ID, and token
  usage metadata.
- Ordinary tests and replay runs need no provider credentials. Live mode incurs
  TypeSafe and generative-provider requests.
