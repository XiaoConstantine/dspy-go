# TypeSafe Decide example

This example uses the experimental TypeSafe System One client and `Decide`
module to triage a support ticket. One provider request returns three closed-set
outputs:

- `urgent`: a Noul (yes/no) probability, interpreted with a local threshold;
- `severity`: an ordinal distribution converted to the expected value of the
  declared anchors `0`, `2`, and `10`;
- `category`: a typed Go choice (`billing`, `technical`, `account`, or `other`).

`ProcessDecision` returns both ordinary values for program composition and the
provider evidence used to derive them.

> **Experimental:** The packages and their persisted formats may change or be
> removed in a dspy-go v0 minor release.

## Run

Set a TypeSafe API key and run from the repository root:

```bash
export TYPESAFE_API_KEY="..."
go run ./examples/typesafe_decide
```

Classify another ticket or change the local urgency boundary:

```bash
go run ./examples/typesafe_decide \
  -ticket "I cannot sign in after enabling SSO." \
  -urgent-threshold 0.7
```

To choose a specific model rather than the default `jev-latest` alias:

```bash
go run ./examples/typesafe_decide -model YOUR_PINNED_MODEL
```

`TYPESAFE_DEFAULT_MODEL` and `TYPESAFE_BASE_URL` are also supported. A pinned
model is recommended for reproducible experiments.

## What to notice

- No generative `core.LLM` is configured. `Decide` uses its explicit System One
  client, and its `SetLLM` method intentionally does not replace that client.
- All three outputs are batched into one `/v1/systemone` request.
- `result.Outputs` contains native `bool`, `float64`, and `ticketCategory`
  values.
- `result.Decisions` keeps provider probabilities and confidence provenance.
  Noul confidence is threshold-relative boundary distance, not provider
  confidence or a calibrated probability of correctness.
- Local thresholds, Score anchors, and Choice multipliers are not sent to the
  provider.

This is a live example and incurs a TypeSafe API request. The package tests use
recorded fixtures and never require credentials.
