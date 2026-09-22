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
native values or `ProcessDecision` for values plus provider evidence.
