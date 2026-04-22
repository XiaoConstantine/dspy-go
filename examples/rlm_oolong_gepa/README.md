# RLM GEPA OOLONG Example

This example runs the persisted `baseline -> optimize -> save -> restore -> replay` workflow against an adaptive RLM-backed agent on OOLONG-style long-context tasks.

It shows:

- a live `RLM`-backed `OptimizableAgent`
- real OOLONG task loading from embedded samples, a local JSON file, or HuggingFace
- answer scoring with the same OOLONG-style matcher used by the benchmark example
- `GEPA` mutating the `rlm_iteration_prompt` artifact on train/validation splits
- writing the optimized program to `optimized_program.json` by default
- loading the saved optimized program back onto a fresh agent for the final replay run

## Run it

Anthropic:

```bash
ANTHROPIC_API_KEY=... go run ./examples/rlm_oolong_gepa -artifact /tmp/oolong-program.json
```

OpenAI:

```bash
OPENAI_API_KEY=... go run ./examples/rlm_oolong_gepa -provider openai -artifact /tmp/oolong-program.json
```

Gemini:

```bash
GOOGLE_API_KEY=... go run ./examples/rlm_oolong_gepa -provider gemini -artifact /tmp/oolong-program.json
```

With HuggingFace tasks:

```bash
GOOGLE_API_KEY=... go run ./examples/rlm_oolong_gepa -provider gemini -hf -task-offset 20 -tasks 10 -artifact /tmp/oolong-program.json -output /tmp/oolong-report.json
```

With a local task file:

```bash
go run ./examples/rlm_oolong_gepa -file ./oolong_tasks.json -tasks 20 -artifact /tmp/oolong-program.json
```

Replay an existing saved optimized program without rerunning GEPA:

```bash
GOOGLE_API_KEY=... go run ./examples/rlm_oolong_gepa -provider gemini -artifact /tmp/oolong-program.json -replay-only
```

## Notes

- This example uses real model calls and may incur provider cost.
- This example executes model-generated Go code through the `RLM` REPL. Only run it on trusted datasets and task files.
- It is intended as a benchmark-oriented optimization example, not a CI test.
- The default embedded tasks are small smoke-test tasks; use `-hf` or `-file` for broader evaluation.
- Use `-task-offset` with `-hf` or `-file` to evaluate a stable task slice across runs.
- Use `-artifact` to control where the optimized program JSON is written or restored from.
- Use `-replay-only` to skip optimization and only compare the baseline agent against a saved optimized program.
- Use `-output` to write a JSON report with aggregate baseline and restored optimized metrics.
