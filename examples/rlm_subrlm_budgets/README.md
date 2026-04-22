# RLM Sub-RLM Budget Example

This example is a deterministic scripted demo of the two delegation budgets in `pkg/modules/rlm`:

- `MaxDirectSubRLMCalls`
- `MaxTotalSubRLMCalls`

It forces a root RLM to attempt two `subrlm` actions and shows how the second one is rejected once the configured budget is exhausted.

## Run it

```bash
go run ./examples/rlm_subrlm_budgets
```

## What it prints

- termination cause
- total steps
- successful sub-RLM call count
- the specific budget error that blocked the second delegation
- the final answer returned after the guardrail fired

## Notes

- This example is scripted and does not require an API key.
- The `direct-budget` scenario allows one direct child delegation per node.
- The `total-budget` scenario allows only one sub-RLM call across the whole request tree.
