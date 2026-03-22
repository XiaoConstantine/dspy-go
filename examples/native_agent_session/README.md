# Native Agent Session Example

This example gives the repo a simple, canonical execution path for the native tool-calling agent with the new session stack:

- `native.Agent`
- SQLite-backed `sessionevent` persistence
- session recall across runs
- branch fork and branch resume

It intentionally stays small. The example registers real workspace tools:

- `list_files`
- `read_file`
- `write_file`
- `edit_file`

The native agent also gets the built-in `finish` tool automatically. On first run, the example seeds the workspace with `project_brief.md` so you can use it immediately.

## Run It

Set the provider API key in the matching environment variable, or pass `-api-key`.

```bash
export GEMINI_API_KEY=...
go run ./examples/native_agent_session \
  -task "Read project_brief.md, create rollout_notes.md with a short rollout plan, then finish."
```

Run a second task against the same session to prove recall works and exercise file editing:

```bash
go run ./examples/native_agent_session \
  -task "Edit rollout_notes.md to add a rollback note, then finish."
```

Then ask a third question that should be answerable from session recall:

```bash
go run ./examples/native_agent_session \
  -task "What file did you create in the previous run? Answer from session recall, then finish."
```

The example prints:

- final answer
- turn/tool-call counts
- active branch ID
- branch list
- current head entry ID
- workspace root path

## Fork A Branch

Fork the active branch after a run:

```bash
go run ./examples/native_agent_session \
  -task "Create an alternate plan in alt_rollout.md based on the current workspace, then finish." \
  -fork-name alt-rollout \
  -activate-fork
```

The printed session state includes branch IDs. Use one of those IDs to continue a specific branch:

```bash
go run ./examples/native_agent_session \
  -branch <branch-id> \
  -task "Continue this branch and summarize the current plan, then finish."
```

## Useful Flags

- `-model` chooses the tool-calling model ID. Default: `gemini-2.0-flash`
- `-session` chooses the logical session ID. Default: `native-agent-demo`
- `-session-db` chooses the SQLite path. Default: `./tmp/native-agent-session/session.db`
- `-workspace` chooses the workspace root. Default: `./tmp/native-agent-session/workspace`
- `-verbose` prints selected lifecycle events

## Why This Example Exists

This example is meant to prove the normal native-agent path works without requiring the CLI session commands or test-only helpers. It is the smallest end-to-end example that exercises:

- provider-backed native execution
- session persistence
- branchable recall
- public session-control methods
- real file read/write/edit tools instead of toy note tools
