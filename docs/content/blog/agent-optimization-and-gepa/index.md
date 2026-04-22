---
title: "Trying to Reproduce Agent Self-Optimization in DSPy-Go"
description: "Inspired by Hermes-style agent self-evolution, we pushed DSPy-Go from prompt optimization toward real agent optimization with GEPA parity work, native tool-calling agents, and TBLite benchmarking."
summary: "A walkthrough of what it took to make agent optimization in DSPy-Go real: GEPA parity fixes, an agent optimization bridge, native tool-calling support, and a TBLite benchmark path that exposed the hard parts."
date: 2026-03-17T10:00:00+00:00
lastmod: 2026-03-17T10:00:00+00:00
draft: true
weight: 40
categories: ["Features", "Agents", "Optimization"]
tags: ["agents", "gepa", "benchmarks", "rlm", "tblite", "optimization"]
contributors: ["Xiao Constantine"]
pinned: false
homepage: false
seo:
  title: "Trying to Reproduce Agent Self-Optimization in DSPy-Go"
  description: "DSPy-Go now has a much more complete path from GEPA parity to real agent optimization, including native tool-calling agents and TBLite benchmarking."
  canonical: ""
  noindex: false
---

## The Goal

The motivating question was simple:

Can `dspy-go` do something closer to **agent self-optimization**, not just prompt rewriting?

That question was heavily influenced by two threads of work:

- Hermes-style self-evolution efforts, where the thing being improved is an agent workflow rather than a single prompt
- GEPA's broader "optimize anything" framing, where the optimizer can search over policies, code, configuration, and other structured artifacts

In practice, that meant trying to reproduce the shape of that workflow in `dspy-go`:

1. take a real agent
2. expose a stable set of mutable artifacts
3. evaluate it on real tasks
4. let GEPA search over those artifacts
5. compare baseline vs tuned behavior on held-out tasks

That sounds straightforward. It was not.

## What Was Missing

At the start, the gap was not one single missing optimizer feature. It was the whole path between:

- the **core GEPA engine**
- the **agent adapter layer**
- the **benchmark/evaluator loop**

We needed all three to be trustworthy at the same time.

Without that, "agent optimization" is mostly theater:

- maybe the optimizer picks the wrong candidate
- maybe validation behavior is not preserved
- maybe traces are too weak to support grounded reflection
- maybe the benchmark is really measuring Docker path mismatches or provider quirks

So the work split into two tracks:

- make core GEPA behavior line up with upstream DSPy where it should
- make agent optimization and benchmarking honest enough to trust

## First: GEPA Parity Had To Be Real

Before talking about agents, we needed confidence that the optimizer itself was behaving correctly.

That led to a deterministic GEPA compatibility harness built around upstream Python `dspy` and `dspy-go`, instead of vague "both seem to work" comparisons.

The parity harness now covers:

- component selection
- validation frontier behavior
- ancestor-aware merge proposals
- feedback-guided rewrites
- format-failure-as-feedback
- minibatch accept/reject behavior
- checkpoint resume
- custom stopper and metric-budget early stopping

This matters because agent optimization depends on exactly those behaviors:

- did validation actually pick the right winner?
- did resume preserve the same outcome as a fresh run?
- did minibatch acceptance admit or reject candidates for the right reason?
- did stoppers end a run early without changing the final semantics?

That fixture harness is the current parity source of truth:

```bash
./compatibility_test/run_gepa_fixture.sh
```

One important lesson from this phase: not every implementation detail deserves to be called "parity." Some things are clearly cross-language GEPA semantics, and some things are just `dspy-go` implementation details. We kept the parity harness focused on the former.

## Then: The Agent Adapter Needed To Stop Being Clever

Once core GEPA was in much better shape, the next gap was the agent bridge.

`dspy-go` now has a real agent optimization surface through `OptimizableAgent`, which exposes:

- `GetArtifacts()`
- `SetArtifacts(...)`
- `Clone()`

Some agents also expose `LastExecutionTrace()`, which lets evaluators attach richer side information, but that is intentionally optional rather than part of the base `OptimizableAgent` contract.

The GEPA bridge around that interface now lives in the agent optimizer layer. The important shift is that the user-facing story is no longer just `Optimize(...)` followed by an in-memory `SetArtifacts(...)`. It is now a real workflow:

- baseline
- optimize
- save optimized program
- restore it onto a fresh agent
- replay on held-out tasks

That looks like this:

```go
workflow, err := optimize.RunGEPAWorkflow(ctx, baseAgent, optimize.GEPAWorkflowRequest{
    Evaluator:          evaluator,
    TrainingExamples:   trainExamples,
    ValidationExamples: validationExamples,
    ReplayExamples:     heldOutExamples,
    PassThreshold:      0.9,
    ApplyBest:          true,
    ArtifactPath:       "optimized_program.json",
    Config: optimize.GEPAAdapterConfig{
        PopulationSize:             4,
        MaxGenerations:             2,
        ReflectionFreq:             1,
        ValidationFrequency:        1,
        MaxMetricCalls:             50,
        AddFormatFailureAsFeedback: true,
        PrimaryArtifact:            optimize.ArtifactRLMIterationPrompt,
    },
})
if err != nil {
    panic(err)
}

restored, err := optimize.ReadOptimizedAgentProgram("optimized_program.json")
if err != nil {
    panic(err)
}

replayAgent, _ := baseAgent.Clone()
_ = optimize.ApplyOptimizedAgentProgram(replayAgent, restored)

_ = workflow
```

The important fixes here were not flashy. They were semantic:

- prefer the GEPA **validation winner**, not just the best training candidate
- preserve explicit validation sets instead of reshuffling them back into a split ratio
- support multi-artifact candidates cleanly, including bounded integer knobs like `max_turns`
- persist optimized agent state in a shared versioned envelope instead of leaving it only in memory

Those changes are what made the agent path line up with the main GEPA engine instead of quietly drifting from it.

## RLM Became a Real Optimization Target

In parallel, `RLM` stopped being "just a module with an internal loop" and became something you can actually optimize as an agent.

That work added:

- richer execution traces
- request-scoped token accounting
- configurable outer and iteration prompt artifacts
- an `RLM`-backed `OptimizableAgent`

That changed the optimization target from "one prompt string" into something closer to a real reasoning loop. It also made the reflection/evaluation story more grounded, because the evaluator can now talk about what happened inside the run:

- the loop exited too early
- it overused subcalls
- it spent budget in the wrong place
- it failed to converge before the max-iteration limit

That is what the OOLONG example is meant to demonstrate:

```bash
GOOGLE_API_KEY=... go run ./examples/rlm_oolong_gepa \
  -provider gemini \
  -hf \
  -task-offset 20 \
  -tasks 10 \
  -output /tmp/oolong-report.json
```

This is closer to the kind of agent-optimization loop people actually care about: optimize an adaptive reasoning system against real task outcomes, not just prompt prettiness.

## Native Tool Calling Was The Other Missing Half

The other half of the story was tool-using agents.

`dspy-go` already had strong ReAct support, but for benchmarking and optimization we wanted a cleaner native tool-calling path:

- fewer prompting layers
- clearer traces
- fewer ambiguities about whether a failure came from the protocol or the model

That led to a shared native tool-calling harness, plus a TBLite-specific adapter on top of it.

This was important for two reasons:

1. it gave GEPA a stable artifact surface for tool-using agents
2. it gave the benchmark path a simpler and more inspectable execution model

## TBLite Is Where The Nice Story Broke

Once the native tool-calling agent and the benchmark runner existed, the obvious next step was to try the whole loop on TBLite:

- materialize real tasks
- run the agent in a task workspace or task container
- execute the verifier
- compare baseline vs tuned behavior

From the CLI, that now looks like:

```bash
cd cmd/dspy-cli

go run -mod=mod . benchmark tblite \
  --provider openai \
  --model gpt-5.2 \
  --offset 0 \
  --limit 20 \
  --root-dir /tmp/tblite-baseline \
  --output /tmp/tblite-baseline.json
```

And the GEPA comparison path looks like:

```bash
cd cmd/dspy-cli

go run -mod=mod . benchmark tblite \
  --provider openai \
  --model gpt-5.2 \
  --offset 0 \
  --limit 20 \
  --root-dir /tmp/tblite-gepa \
  --output /tmp/tblite-gepa.json \
  --gepa \
  --population 4 \
  --generations 2 \
  --reflection-freq 1 \
  --validation-split 0.2 \
  --test-split 0.2
```

This is where the work got much less theoretical.

TBLite immediately surfaced issues that have very little to do with "is GEPA smart enough":

- environment root mismatches like `/app` vs mounted task paths
- verifier reward parsing bugs
- task-level timeout handling
- task-name and path traversal hardening
- native tool-calling provider quirks
- agents that solved enough for the verifier but forgot to call `Finish`

That was frustrating, but useful. It forced the benchmark path to grow up.

## What The Benchmark Work Actually Taught Us

The mixed 20-task TBLite runs were valuable, but not in the way we first hoped.

They were excellent at revealing:

- runtime bugs
- provider failures
- path-contract mismatches
- bad termination behavior
- places where the benchmark harness itself was lying

They were much worse as an **inner-loop optimization signal**.

That led to a more realistic view of the workflow:

- use deterministic GEPA fixtures for parity
- use narrower, curated task slices for optimization experiments
- use broader TBLite slices as regression gates

That is why `dspy-cli` now supports curated manifests through `--tasks-file`, and why we added focused slices like:

- `pkg/benchmarks/tblite/focused_coding_repair_slice.json`

The right mental model is:

- **inner loop:** cheaper, denser, more coherent task slices
- **outer loop:** broader benchmark gates like TBLite

That is much closer to how the Hermes-style self-evolution story actually needs to work in practice.

## So What Is True Today?

The honest answer is not "we fully reproduced self-optimizing agents."

The honest answer is:

- `dspy-go` now has a much stronger **core GEPA parity story**
- it has a real **agent optimization bridge**
- it has native **tool-calling agent support**
- it has a real **TBLite benchmark path**
- and it now supports **baseline vs tuned** evaluation over held-out task splits

That is a real foundation.

It is also a very different claim from:

"we already matched the strongest public self-evolving coding-agent results."

We have the infrastructure and the experimental loop. The next stage is producing stronger benchmark evidence on better-shaped datasets.

## What Comes Next

The next work is less about inventing new optimizer terms and more about tightening the loop:

- use focused task slices as the primary GEPA signal
- keep outer benchmarks as gates, not as the only source of learning
- evolve richer multi-artifact agent surfaces
- continue reducing provider/runtime noise
- turn more agent families into clean `OptimizableAgent` targets

That is the real outcome of this phase.

The original motivation was to see whether `dspy-go` could support something closer to Hermes-style agent self-optimization.

The answer now is:

Not as a one-click demo. But as a serious system with parity-tested optimizer behavior, a real agent interface, and an honest benchmark loop? Yes, we are much closer than we were.

## Where To Start

If you want to follow the same path in the repo, the best entry points are:

- `./compatibility_test/run_gepa_fixture.sh`
- `./examples/rlm_oolong_gepa`
- `./examples/others/gepa`
- `cd cmd/dspy-cli && go run -mod=mod . benchmark tblite ...`

That combination tells the whole story:

- parity first
- agent artifacts second
- real benchmarks third

That was the path from "prompt optimization" to something that actually starts to look like agent optimization.
