# Parallel Search MCP

Search the web or fetch a page through dspy-go's existing MCP tool registry.
This example connects to the anonymous
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
endpoint, discovers its tools with `tools.RegisterMCPTools`, selects a tool with
`registry.Get`, and calls `tool.Execute`. It does not require a Parallel API key
or an LLM. The free endpoint has rate limits and is intended for exploration and
light use.

The pinned `mcp-go` client supports stdio. The example uses the documented
`mcp-remote` bridge to connect it to Streamable HTTP, with a
`User-Agent: dspy-go/parallel-search-example` header and no Authorization header.
It does not change the library's provider defaults or existing configuration.

## Run

Install the Go version required by the root `go.mod`, Node.js 20 or newer, and
the bridge:

```sh
npm install --global mcp-remote@0.14.3
```

From the repository root:

```sh
go run ./examples/parallel_search -query 'Go context cancellation documentation'
go run ./examples/parallel_search -url https://go.dev/blog/context
```

`-url` selects `web_fetch` instead of `web_search`. Results are printed as the
text returned by the MCP tool, including source URLs. `-timeout` bounds the
whole run (default: one minute); Ctrl+C also cancels it and stops the bridge.
MCP tool errors produce a nonzero exit status.
The bridge honors `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY` when set.

The example sets `MCP_REMOTE_CONFIG_DIR` to a temporary directory and removes it
on exit, isolating the bridge from any saved OAuth credentials.

With the bridge installed, run the local HTTP fixture tests (no live search
requests) using `go test -race ./examples/parallel_search`. They check tool
registration and execution, anonymous request headers, errors and cancellation.

To use these discovered tools in an agent, pass the same registry to
`modules.NewReAct(signature, registry, maxIterations)` and configure its LLM
separately. Model inference is separate from the free search endpoint.
