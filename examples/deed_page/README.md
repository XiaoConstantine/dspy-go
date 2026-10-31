# deed.page MCP

Publish an HTML page through dspy-go's existing MCP tool registry. This example
connects to the [deed.page](https://deed.page) remote MCP endpoint, discovers its
tools with `tools.RegisterMCPTools`, selects `publish_site` with `registry.Get`,
and calls `tool.Execute`. The result is a public HTTPS URL. It does not need an
API key, an account or an LLM.

By default the example publishes a small sample page that expires after ten
minutes, checks that the URL returns HTTP 200, deletes it with `delete_site`
using the owner token from the publish result, and checks the URL again. The
token is not printed unless you pass `-keep`. Published pages are public, so
don't publish anything private.

The pinned `mcp-go` client supports stdio. Like the
[parallel_search](../parallel_search/) example, this one uses the `mcp-remote`
bridge to reach Streamable HTTP, with a `User-Agent: dspy-go/deed-page-example`
header and no Authorization header.

## Run

Install the Go version required by the root `go.mod`, Node.js 20 or newer, and
the bridge:

```sh
npm install --global mcp-remote@0.14.3
```

From the repository root:

```sh
go run ./examples/deed_page
go run ./examples/deed_page -file report.html -keep
go run ./examples/deed_page -file report.html -keep -ttl 0
```

`-file` publishes your own HTML file instead of the sample page. `-keep` leaves
the page up and prints its owner token once, so you can update or delete it
later; store it somewhere private. `-ttl` sets the expiry in seconds (default
600; `0` keeps the page until it is deleted). `-timeout` bounds the whole run
(default: one minute); Ctrl+C also cancels it and stops the bridge. MCP tool
errors produce a nonzero exit status.

The example sets `MCP_REMOTE_CONFIG_DIR` to a temporary directory and removes it
on exit, isolating the bridge from any saved OAuth credentials.

With the bridge installed, run the local HTTP fixture tests (no live requests)
using `go test -race ./examples/deed_page`. They check tool registration and
execution, request headers, the publish, check and delete flow, `-keep`, `-ttl 0`,
tool errors and cancellation.

To let an agent publish what it writes, pass the same registry to
`modules.NewReAct(signature, registry, maxIterations)` and configure its LLM
separately. Docs: https://deed.page/docs
