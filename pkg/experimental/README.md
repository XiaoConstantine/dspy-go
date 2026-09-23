# Experimental packages

APIs below this directory are public experiments. They may change or be removed
in a dspy-go v0 minor release, including persisted-state and cache formats.

Dependency direction is one-way: experimental packages may import stable
packages, but packages outside `pkg/experimental` must not import this tree.
`import_guard_test.go` enforces that rule.

Current packages:

- `typesafe`: a small, direct System One HTTP client.
- `decide`: a closed-set `core.Module` using a narrow System One client.

Neither package performs global registration. Tests use recorded JSON fixtures
and local HTTP servers; no API credential or live service is required.

A client reads `TYPESAFE_API_KEY` by default:

```go
client, err := typesafe.NewClient(
    typesafe.WithDefaultModel("jev-latest"),
)
response, err := client.SystemOne(ctx, typesafe.SystemOneRequest{
    State: "I was charged twice.",
    Questions: map[string]typesafe.Question{
        "billing": typesafe.NoulQuestion{
            Instructions: "Is this about billing?",
        },
    },
})
```

`decide.New` accepts that client through its narrow `SystemOneClient` interface
and batches all declared signature outputs into one request. Call `Process` for
native values or `ProcessDecision` for values plus provider evidence. Use
`decide.Get[T]` to retrieve named evidence without an unchecked assertion.

After changing a Noul threshold, Score anchor, or Choice multiplier,
`Decide.Reinterpret` derives a new local result from an existing result without
calling the provider. It preserves the original model, request ID, token usage,
probabilities, provider selection, and provider confidence, and rejects results
whose captured signature or answer space differs from the current module. `ProcessDecision`
also records a `Decide` span with provider provenance and one `token_usage`
annotation. System One usage remains on the result and span; it does not update
the program's overwrite-only `ExecutionState` LLM counter, where it could erase
a generative module's usage in a mixed program.

Runnable integrations are available in:

- [`examples/typesafe_decide`](../../examples/typesafe_decide): a System One
  gate and generative `ChainOfThought` in one `core.Program`;
- [`examples/typesafe_threshold_tuning`](../../examples/typesafe_threshold_tuning):
  a two-sided abstention band chosen on calibration data, persisted, and
  reported once on a held-out split;
- [`examples/typesafe_cascade`](../../examples/typesafe_cascade): a Jev-first
  classifier that escalates uncertain items to `modules.Predict`;
- [`examples/typesafe_vs_predict`](../../examples/typesafe_vs_predict): the W7
  labeled head-to-head harness for quality, latency, calls, and token usage.

All four examples support `-replay` and need no credentials in that mode.
