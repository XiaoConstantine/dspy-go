# Two-sided TypeSafe threshold tuning

This example uses a labeled **calibration split** to choose an abstaining policy
for System One's `urgent` probability:

- automatically urgent when `P(urgent) >= high`;
- automatically not urgent when `P(urgent) <= low`;
- route the uncertain middle band to a human.

For each candidate `(low, high)` pair it reports urgent-side precision,
not-urgent-side precision, and **coverage**, the share of all traffic decided
automatically on either side. It selects the highest-coverage calibration row
where both automatic sides meet `-target-precision`.

The example then freezes the band, persists the high threshold through
`Decide.GetTunedParameters`, reloads it with `SetTunedParameters`, and reports
one result on a separate held-out test split. The low threshold lives alongside
the Decide parameters in the example's versioned policy envelope because it is
an abstention policy rather than a `Decide` Noul parameter.

`Decide.Reinterpret` performs every calibration sweep locally over retained
probabilities. It never calls System One again.

> Both included splits are intentionally tiny illustrative fixtures. They do
> not establish quality or calibration. Use adequately sized, disjoint
> calibration and test sets, freeze the policy, and report the test split once.

## Run entirely offline

```bash
go run ./examples/typesafe_threshold_tuning -replay
```

Replay mode serves recorded responses from a local `httptest` server and
matches the exact model, state, and question instructions. The recorded model
is `jev-replay`, so a different `-model` override fails exact matching. It should select
`(low=0.20, high=0.80)` for the default 95% target, report 50% calibration
coverage and 66.7% held-out coverage, and make:

```text
8 calibration + 0 during sweep/reload + 6 held-out provider calls
```

Change the illustrative precision policy:

```bash
go run ./examples/typesafe_threshold_tuning \
  -replay \
  -target-precision 0.75
```

## Run live

```bash
export TYPESAFE_API_KEY="..."
go run ./examples/typesafe_threshold_tuning \
  -model YOUR_PINNED_JEV_MODEL
```

Live results need not select the same band as the replay fixtures. Pin the model
whenever a persisted policy depends on its probability distribution.

> **Experimental:** `Reinterpret`, typed evidence, and the persisted parameter
> schema may change or be removed in a dspy-go v0 minor release.
