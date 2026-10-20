# Local code lookup through MCP

This optional example registers [Sem](https://github.com/Ataraxy-Labs/sem)'s
`sem_find` using the existing MCP registry and prints its result.
It needs no LLM, API key, Node.js, or HTTP bridge.

Install Sem on `PATH` first (`sem --version`; tested with 0.27.0). From this
repository's root, with Go installed:

```sh
go run ./examples/sem_code -repo /absolute/path/to/checkout -query parse_config
go test ./examples/sem_code
```

Replace the query with a symbol in your checkout. The subprocess uses that
checkout as both its working directory and `SEM_REPO`. The example has a
two-minute overall timeout and terminates/reaps the subprocess on exit.

Sem may write a local index. `SEM_REPO` selects a repository, not a sandbox.
This program registers and invokes only `sem_find`. The full server also has
validation tools that can execute commands; tool selection is not a sandbox.
An agent using these results sends source context to its configured model.

## Why register the tool explicitly?

The current `mcp-go` dependency (v0.3.1) models each property type as a string.
Sem 0.27.0 advertises valid nullable types such as `["string", "null"]`, so
`RegisterMCPTools` fails during discovery. This example uses `NewMCPTool` with
only the three non-null inputs it supplies. It does not alter the server's
schema or claim to support general discovery. Revisit this adapter when the
client supports union types; the server still validates each call.
