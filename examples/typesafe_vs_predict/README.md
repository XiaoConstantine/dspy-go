# TypeSafe Decide vs Predict

This is the runnable harness for the Jev proposal's W7 opt-in comparison with
ordinary `modules.Predict`. A small inline labeled support dataset is classified
by both systems using the same `core.Signature` and closed category labels.

The report includes:

- accuracy against the labels;
- logical provider call counts;
- System One input/output tokens and generative prompt/completion tokens;
- p50/p95 per-call latency in live mode.

It intentionally performs both calls for every item. For selective routing, see
[`typesafe_cascade`](../typesafe_cascade/).

## Run entirely offline

```bash
go run ./examples/typesafe_vs_predict -replay
```

Replay mode serves recorded System One responses from a local `httptest` server
and uses a deterministic completion-only LLM. Fixtures match the exact model,
state, question instructions, and Choice criteria. The recorded model is
`jev-replay`, so a different `-model` override fails exact matching. Replay explicitly reports
that local fixture latency is not meaningful. The replay predictions exist to exercise the
comparison harness; differences in their displayed accuracy are not evidence
about either live system.

## Run live

```bash
export TYPESAFE_API_KEY="..."
export GEMINI_API_KEY="..."
go run ./examples/typesafe_vs_predict \
  -model YOUR_PINNED_JEV_MODEL \
  -llm-model gemini-2.5-flash
```

Use `-llm-api-key` to pass the generative credential directly. The live
measurement explicitly disables dspy-go's transparent LLM response cache so
its call counts and latency reach the configured provider. Pin both models and use
a sufficiently large held-out labeled dataset before drawing quality, latency,
or cost conclusions. Provider token accounting may differ, so the two
token totals are reported separately rather than treated as directly
interchangeable cost units.

> **Experimental:** The TypeSafe client, `Decide`, evidence types, and this
> harness may change or be removed in a dspy-go v0 minor release.
