---
title: "CLI Reference"
description: "Current command-line interface reference for dspy-cli"
summary: "Optimizer exploration, prompt analysis, log viewing, sessions, agents, and TBLite benchmarks"
date: 2025-10-13T00:00:00+00:00
lastmod: 2026-08-26T00:00:00+00:00
draft: false
weight: 920
toc: true
seo:
  title: "CLI Reference - dspy-go"
  description: "Current dspy-cli commands and usage"
  canonical: ""
  noindex: false
---

`dspy-cli` is a separate Go module for exploring optimizers, analyzing prompts,
viewing traces, managing persisted sessions, and running agent benchmarks.

## Build and Run

```bash
cd cmd/dspy-cli
go build -o dspy-cli
./dspy-cli --help
```

Running `dspy-cli` without a command starts the interactive TUI and therefore
requires a terminal.

## Command Overview

| Command | Purpose |
|---|---|
| `list` | List registered optimizers |
| `describe <optimizer>` | Show details for one optimizer |
| `recommend` | Recommend optimizers by use case or interactive questions |
| `try <optimizer>` | Run an optimizer against a built-in sample dataset |
| `analyze [prompt]` | Analyze and optionally export prompt structure |
| `view <file.jsonl>` | Inspect RLM or native dspy-go JSONL logs |
| `interactive` | Start the guided terminal UI |
| `agent run-terminal-task` | Run one TBLite-style task from JSON stdin |
| `benchmark tblite` | Evaluate or GEPA-optimize the native benchmark agent |
| `session show` | Inspect a persisted native-agent session |
| `session switch` | Change a session's active branch |
| `session fork` | Fork the active branch from its current head |
| `completion` | Generate shell completion scripts |

Use `dspy-cli <command> --help` as the authoritative flag reference.

## Optimizer Discovery

```bash
dspy-cli list
dspy-cli describe mipro
dspy-cli recommend --use-case balanced
dspy-cli recommend --interactive
```

`--use-case` accepts `simple`, `balanced`, `advanced`, or `multi-module`.

## Run an Optimizer

```bash
dspy-cli try <optimizer> [flags]
```

Supported optimizer names are shown by `list`. The current flags are:

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `qa` | `qa`, `gsm8k`, or `hotpotqa` |
| `--max-examples` | `0` | Limit examples; zero uses all examples |
| `--provider` | `google` | `google`, `openai`, or `local` |
| `--model` | `gemini-2.5-flash` | Model ID |
| `--base-url` | empty | OpenAI-compatible endpoint for local/custom use |
| `--api-key` | empty | Explicit API key |
| `--verbose` | false | Enable verbose logging |

Examples:

```bash
export GEMINI_API_KEY="your-api-key"

dspy-cli try bootstrap --dataset qa --max-examples 3
dspy-cli try mipro --dataset gsm8k --max-examples 5 --verbose

# OpenAI: specify an OpenAI model and key explicitly.
dspy-cli try simba \
  --provider openai \
  --model gpt-4o-mini \
  --api-key "$OPENAI_API_KEY"

# Local OpenAI-compatible endpoint, such as LM Studio.
dspy-cli try bootstrap \
  --provider local \
  --model local-model \
  --base-url http://localhost:1234/v1
```

The `try` command reads `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `DSPY_API_KEY`
when `--api-key` is omitted. For other hosted providers, pass `--api-key`
explicitly.

## Analyze a Prompt

```bash
dspy-cli analyze "Answer the user's question clearly."
dspy-cli analyze --interactive
dspy-cli analyze --optimize "Explain this code."
dspy-cli analyze --export signature.yaml "Explain this code."
```

Flags:

- `--interactive`, `-i`: enter a multi-line prompt
- `--optimize`, `-o`: expand toward the analyzer's full prompt structure
- `--export`, `-e`: export YAML or JSON

## View Session Logs

The viewer auto-detects RLM iteration logs and native dspy-go event logs:

```bash
dspy-cli view session.jsonl
dspy-cli view --interactive session.jsonl
dspy-cli view --watch session.jsonl
dspy-cli view --stats session.jsonl
dspy-cli view --iter 3 session.jsonl
dspy-cli view --search error session.jsonl
dspy-cli view --export report.md session.jsonl
```

Additional filters include `--compact`, `--errors`, `--final`, and
`--no-color`.

## Persisted Sessions

Session commands require the SQLite event-store path:

```bash
dspy-cli session --db sessions.db show <session-id>
dspy-cli session --db sessions.db switch <session-id> <branch-id>
dspy-cli session --db sessions.db fork <session-id> --name experiment
dspy-cli session --db sessions.db fork <session-id> --activate
```

## Run One Terminal Task

`agent run-terminal-task` reads a `tblite.TerminalTaskRequest` JSON object from
stdin and writes a `tblite.TerminalTaskResult` JSON object to stdout:

```bash
dspy-cli agent run-terminal-task \
  --provider google \
  --model gemini-2.5-flash \
  --max-turns 20 \
  < request.json > result.json
```

It supports `--api-key`, `--max-tokens`, `--temperature`,
`--tool-output-limit`, and provider/model overrides. This command resolves
provider-specific credentials from `GEMINI_API_KEY`/`GOOGLE_API_KEY`,
`OPENAI_API_KEY`, or `ANTHROPIC_OAUTH_TOKEN`/`ANTHROPIC_API_KEY`; `DSPY_API_KEY`
is the shared fallback.

## TBLite Benchmark

```bash
dspy-cli benchmark tblite --limit 5 --output report.json
```

Use `--tasks-file` for a curated task slice, `--root-dir` for materialized task
workspaces, and `--keep-artifacts` to retain those workspaces. Provider controls
include `--provider`, `--model`, `--api-key`, `--max-turns`, `--max-tokens`,
`--temperature`, and `--tool-output-limit`.

Enable GEPA prompt optimization with:

```bash
dspy-cli benchmark tblite \
  --gepa \
  --population 4 \
  --generations 2 \
  --validation-split 0.2 \
  --test-split 0.2 \
  --output tuned-report.json
```

Run `dspy-cli benchmark tblite --help` for the complete split, shuffle,
concurrency, and stagnation controls.

## Global Flags

The root command exposes only:

- `--help`, `-h`
- `--version`, `-v`

There is no persistent `config` command or CLI configuration file. Configure
each invocation with its flags and documented environment variables.

## Next Steps

- **[Getting Started →](../../guides/getting-started/)**
- **[Configuration Reference →](../configuration/)**
- **[Optimizers Guide →](../../guides/optimizers/)**
