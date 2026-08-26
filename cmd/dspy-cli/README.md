# dspy-cli

`dspy-cli` explores and runs dspy-go optimizers, analyzes prompt structure,
views JSONL traces, manages persisted agent sessions, and runs TBLite agent
benchmarks.

The CLI is a separate Go module. During repository development its `go.mod`
uses a local replacement for the dspy-go module at `../..`.

## Build

Requirements:

- Go 1.27 or newer
- A provider credential for commands that call a hosted model

```bash
cd cmd/dspy-cli
go build -o dspy-cli
./dspy-cli --help
```

Running `./dspy-cli` without a subcommand starts the interactive TUI.

## Quick Start

```bash
export GEMINI_API_KEY="your-api-key"

./dspy-cli list
./dspy-cli describe mipro
./dspy-cli recommend --use-case balanced
./dspy-cli try bootstrap --dataset qa --max-examples 3
./dspy-cli try mipro --dataset gsm8k --max-examples 5 --verbose
```

## Commands

| Command | Purpose |
|---|---|
| `list` | List registered optimizers |
| `describe <optimizer>` | Show one optimizer's details |
| `recommend` | Recommend optimizers by use case or an interactive questionnaire |
| `try <optimizer>` | Run an optimizer on `qa`, `gsm8k`, or `hotpotqa` samples |
| `analyze [prompt]` | Analyze and optionally export prompt structure |
| `view <file.jsonl>` | View RLM or native dspy-go logs |
| `interactive` | Start the guided TUI |
| `agent run-terminal-task` | Read one TBLite-style task from JSON stdin |
| `benchmark tblite` | Run a TBLite evaluation, optionally with GEPA optimization |
| `session show/switch/fork` | Inspect and branch persisted native-agent sessions |
| `completion` | Generate shell completion scripts |

Run `./dspy-cli <command> --help` for current flags.

## Try an Optimizer

```bash
./dspy-cli try <optimizer> [flags]
```

The default provider/model is Google `gemini-2.5-flash`. Useful flags are:

- `--dataset`: `qa`, `gsm8k`, or `hotpotqa` (default `qa`)
- `--max-examples`: zero uses all examples
- `--provider`: `google`, `openai`, or `local`
- `--model`: model ID
- `--base-url`: local or custom OpenAI-compatible endpoint
- `--api-key`: explicit API key
- `--verbose`: detailed logs

The command reads `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `DSPY_API_KEY` when
`--api-key` is empty. Pass the key explicitly for another hosted provider:

```bash
./dspy-cli try simba \
  --provider openai \
  --model gpt-4o-mini \
  --api-key "$OPENAI_API_KEY"
```

Use a local OpenAI-compatible server with:

```bash
./dspy-cli try bootstrap \
  --provider local \
  --model local-model \
  --base-url http://localhost:1234/v1
```

## Prompt Analysis and Log Viewing

```bash
./dspy-cli analyze "Answer the user's question clearly."
./dspy-cli analyze --interactive
./dspy-cli analyze --export signature.yaml "Explain this code."

./dspy-cli view session.jsonl
./dspy-cli view --interactive session.jsonl
./dspy-cli view --watch session.jsonl
./dspy-cli view --stats session.jsonl
./dspy-cli view --export report.md session.jsonl
```

## Persisted Sessions

```bash
./dspy-cli session --db sessions.db show <session-id>
./dspy-cli session --db sessions.db switch <session-id> <branch-id>
./dspy-cli session --db sessions.db fork <session-id> --name experiment
./dspy-cli session --db sessions.db fork <session-id> --activate
```

## Agent and Benchmark Commands

Run one terminal task from JSON stdin:

```bash
./dspy-cli agent run-terminal-task \
  --provider google \
  --model gemini-2.5-flash \
  < request.json > result.json
```

Run a fixed TBLite slice:

```bash
./dspy-cli benchmark tblite --limit 5 --output report.json
```

Enable GEPA optimization before held-out evaluation:

```bash
./dspy-cli benchmark tblite \
  --gepa \
  --population 4 \
  --generations 2 \
  --validation-split 0.2 \
  --test-split 0.2 \
  --output tuned-report.json
```

Use the subcommand help for task-source, artifact, concurrency, and search
controls.

## Development

```bash
cd cmd/dspy-cli
go test ./...
go vet ./...
```

## Documentation

- [CLI reference](../../docs/content/docs/reference/cli.md)
- [Main dspy-go README](../../README.md)
- [Examples](../../examples/)
