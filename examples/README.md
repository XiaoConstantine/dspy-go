# DSPy-Go Examples

This directory contains runnable examples for the main module, agent, optimization, and tooling surfaces in DSPy-Go.

## Recommended Starting Points

### Agents

- **[agents](agents/)** - ReAct patterns, orchestration, and memory
- **[ace_basic](ace_basic/)** - Minimal ACE usage without live provider calls
- **[ace_react](ace_react/)** - ACE layered onto a ReAct-style agent
- **[native_agent_session](native_agent_session/)** - Native tool-calling agent with persisted sessions
- **[a2a_composition](a2a_composition/)** - Multi-agent orchestration using A2A

### Long-Context And RLM

- **[rlm](rlm/)** - Basic live RLM run
- **[rlm_context_policy](rlm_context_policy/)** - Compare `full`, `checkpointed`, and `adaptive` replay
- **[rlm_subrlm_budgets](rlm_subrlm_budgets/)** - Deterministic sub-RLM direct/total budget demo
- **[rlm_oolong](rlm_oolong/)** - Benchmark the RLM module on OOLONG tasks
- **[rlm_oolong_gepa](rlm_oolong_gepa/)** - Optimize an adaptive RLM agent, save the optimized program, restore it, and replay it

### Modules

- **[parallel](parallel/)** - Concurrent batch processing
- **[refine](refine/)** - Iterative answer improvement
- **[xml_adapter](xml_adapter/)** - XML structured output adapters
- **[multimodal](multimodal/)** - Image + text workflows

### Tools And Integrations

- **[typesafe_decide](typesafe_decide/)** - Confidence-gated `Decide` + `ChainOfThought` program with offline replay
- **[typesafe_threshold_tuning](typesafe_threshold_tuning/)** - Calibrate a two-sided abstention band, then report a held-out split
- **[typesafe_cascade](typesafe_cascade/)** - Jev-first classification with uncertainty-triggered `Predict` escalation
- **[typesafe_vs_predict](typesafe_vs_predict/)** - W7 accuracy, latency, call, and token comparison harness
- **[smart_tool_registry](smart_tool_registry/)** - Intelligent tool selection
- **[tool_chaining](tool_chaining/)** - Sequential tool pipelines
- **[tool_composition](tool_composition/)** - Composite tools and orchestration
- **[mcp_optimizer](mcp_optimizer/)** - MCP-backed optimization workflow

### Research And Miscellaneous

- **[others/gepa](others/gepa/)** - Lower-level GEPA examples
- **[others/mipro](others/mipro/)** - MIPRO examples
- **[others/simba](others/simba/)** - SIMBA examples
- **[multi_chain_comparison](multi_chain_comparison/)** - Compare reasoning strategies

## Running Examples

From the repo root:

```bash
go run ./examples/rlm_subrlm_budgets
go run ./examples/parallel
go run ./examples/typesafe_decide -replay
go run ./examples/typesafe_threshold_tuning -replay
go run ./examples/typesafe_cascade -replay
go run ./examples/typesafe_vs_predict -replay
TYPESAFE_API_KEY=... GEMINI_API_KEY=... go run ./examples/typesafe_decide
GEMINI_API_KEY=... go run ./examples/rlm_context_policy -provider gemini
GOOGLE_API_KEY=... go run ./examples/rlm_oolong_gepa -provider gemini -artifact /tmp/oolong-program.json
```

Many example directories also include their own `README.md` with task-specific notes and flags.

## Notes

- Examples under `rlm`, `rlm_context_policy`, `rlm_oolong`, `rlm_oolong_gepa`, and all four `typesafe_*` directories can make live provider calls unless replay or scripted mode is selected.
- Every `typesafe_*` example supports deterministic `-replay` without API keys; `rlm_subrlm_budgets` is also deterministic.
- The `others/` subtree is intentionally more experimental and lower-level than the curated examples above.

## 🤝 Contributing

When adding new examples:

1. Create a dedicated directory for your example
2. Include a README.md with clear usage instructions
3. Add comprehensive comments explaining key concepts
4. Include both basic and advanced usage patterns
5. Ensure examples are self-contained and runnable

## 📄 License

All examples are provided under the same license as DSPy-Go. See the main repository LICENSE file for details.
