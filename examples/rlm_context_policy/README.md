# RLM Context Policy Example

This example compares the three structured replay modes in `pkg/modules/rlm` on the same long-context task:

- `full`
- `checkpointed`
- `adaptive`

It prints the answer plus the trace metadata that shows the replay effect in practice:

- termination cause
- step count
- history compression count
- sub-LLM calls
- root prompt mean/max tokens

## Run it

Anthropic:

```bash
ANTHROPIC_API_KEY=... go run ./examples/rlm_context_policy
```

OpenAI:

```bash
OPENAI_API_KEY=... go run ./examples/rlm_context_policy -provider openai
```

Gemini:

```bash
GOOGLE_API_KEY=... go run ./examples/rlm_context_policy -provider gemini
```

Tune the adaptive threshold:

```bash
GOOGLE_API_KEY=... go run ./examples/rlm_context_policy -provider gemini -adaptive-threshold 4
```

## Notes

- This is a live example and will make provider calls.
- `checkpointed` uses `WithHistoryCompression(1, 200)` so older iterations are summarized aggressively.
- `adaptive` uses `WithAdaptiveCheckpointThreshold(...)` so it starts with full replay and checkpoints once the history is large enough.
- If the model solves the task in one iteration, you may see zero history compressions even in adaptive/checkpointed mode.
