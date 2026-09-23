# TypeSafe Jev-first cascade

This example sends every closed-set support classification to experimental
`Decide` first. It accepts the Jev result directly when Choice provider
confidence is at or above a policy threshold and escalates lower-confidence
items to an ordinary `modules.Predict` using the **same signature**:

```text
support ticket -> Decide -> confidence >= threshold -> accept category
                         `-> confidence < threshold --> Predict category
```

The summary reports direct and escalated path counts, escalation rate,
Jev/Predict agreement on escalated items, provider call and token counts, and
live latency percentiles for each path.

> Choice provider confidence is evidence from the provider, not calibrated
> correctness probability. Select and validate an escalation threshold on a
> labeled calibration split before using this routing policy. Agreement between
> two systems is also not a substitute for human ground truth.

## Run entirely offline

```bash
go run ./examples/typesafe_cascade -replay
```

Replay mode serves six recorded System One responses through a local
`httptest` server and uses a deterministic completion-only LLM. Fixtures match
the exact model, state, question instructions, and Choice criteria. The recorded
model is `jev-replay`, so a different `-model` override fails exact matching. With the default
`0.75` threshold it escalates three of six tickets and the two systems agree on
two of those three. Replay latency is intentionally reported as not meaningful.
Fixtures exercise the harness; they make no claim about live model quality.

Try another illustrative routing threshold:

```bash
go run ./examples/typesafe_cascade -replay -confidence-threshold 0.60
```

## Run live

```bash
export TYPESAFE_API_KEY="..."
export GEMINI_API_KEY="..."
go run ./examples/typesafe_cascade \
  -model YOUR_PINNED_JEV_MODEL \
  -llm-model gemini-2.5-flash
```

Use `-llm-api-key` when passing the generative credential directly. Live mode
makes one System One call per ticket and an LLM call only for escalated tickets.
The measurement path explicitly disables dspy-go's transparent LLM response
cache, so reported calls and latency reach the configured provider. Pin both
models for a reproducible comparison.

> **Experimental:** The TypeSafe client, `Decide`, evidence types, and routing
> policy may change or be removed in a dspy-go v0 minor release.
