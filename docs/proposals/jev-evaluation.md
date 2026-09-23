# Proposal: Jev for Evaluation and Experimental Decisions in dspy-go

- **Status:** Draft, revised after architecture review
- **Date:** 2026-09-21
- **Author:** Xiao Cui

## Summary

We propose using TypeSafe's Jev as a fast judge inside dspy-go's evaluation loop, and adding evaluation tools for programs that return probabilities. Today optimizers must choose between cheap word-matching metrics and slower LLM judges. Jev may fill the gap between them.

The work has seven workstreams. The provider-neutral evaluator contract comes before any network-backed judge so cancellation and failures are represented correctly from the first experiment. A human-labeled pilot then decides whether the Jev-judge work continues. Probability reporting is a separate provider-neutral track because calibration and coverage are dataset-level properties, not ordinary per-example metrics.

A bounded experimental track also adds a dspy-go `Decide` module analogous to Stanford DSPy's experiment. It is a Predict-like module at the user level, but a sibling rather than the concrete `modules.Predict` type because it uses a System One client, has no demonstrations, and owns numeric decision parameters instead of generative prompt state.

The proposal does not change `core.Metric` or pass provider data through hidden trace state. Existing optimizer APIs remain available, while richer evaluators and decision artifacts use additive interfaces.

## Implementation status

An initial fixture-driven vertical slice now exists under `pkg/experimental`:

- `typesafe` implements the verified `POST /v1/systemone` request/response shape, environment and explicit configuration, context cancellation, bounded retries, request IDs, typed errors, and Noul/Choice/Score decoding. Its compatibility references are TypeSafe's official Python SDK v0.7.1, official JavaScript SDK v0.6.0, and the OpenAPI-generated models committed in the official Python SDK; there is no official Go SDK at the time of implementation.
- `decide` implements a sibling `core.Module` with native `Process` outputs, typed `ProcessDecision` evidence, `Get[T]` evidence lookup, local Noul thresholds, Score anchors, Choice multipliers, cloning, and versioned tuned-parameter persistence. `Reinterpret` reapplies current local parameters to compatible, provenance-bound evidence without a provider call, and `ProcessDecision` records provider provenance and one `token_usage` annotation in a context-bound tracing span. System One usage stays separate from the overwrite-only program LLM counter. `SetLLM` is explicitly a no-op and the module implements no demonstration interfaces.
- An import guard enforces that stable packages elsewhere under `pkg` do not depend on experimental packages.
- Normal tests use recorded JSON fixtures and local transports only. Four runnable replay modes exercise a `Decide` + `ChainOfThought` program, two-sided threshold calibration with a held-out split, a Jev-first `Predict` cascade, and a labeled `Decide`-versus-`Predict` harness without credentials or live calls.

This is an M0/W7 implementation slice, not completion of either milestone. It intentionally omits response caching, request coalescing, model listing, judge integration, probability reports, and live quality/calibration evidence.

## Revision notes

This revision incorporates architecture review findings:

- W2 now precedes W1. A remote judge is never exposed as `core.Metric`, whose signature cannot carry a context or error.
- Optimizers gain an additive `ExampleMetricCompiler` interface instead of attempting runtime detection through the concrete `core.Metric` function type.
- `AgentEvaluator`, GEPA feedback, and Refine are integrations around the richer metric; they are not claimed to be the same abstraction.
- Evaluator failures, candidate execution failures, cancellation, aggregation, and retry behavior are specified separately.
- Cache identity includes the evaluator, rubric, model, and request schema; transient failures are not cached.
- ECE, macro-F1, coverage curves, and threshold tuning move to a dataset-level accumulator/report API.
- Probabilities are represented by typed records. The proposal rejects an output-map sentinel and the current execution trace as the probability transport.
- The M2 validation design now requires disjoint development, calibration, and test data, a fixed LLM baseline, human annotation procedures, confidence intervals, and a pre-registered non-inferiority margin.
- Escalation depends on measured selective risk, not raw provider confidence alone.
- Review of Stanford DSPy PR [#10463](https://github.com/stanfordnlp/dspy/pull/10463) adds explicit distinctions between provider probabilities, provider confidence, local decision confidence, and generative self-reported confidence.
- The upstream PR also motivates caching raw distributions before local threshold/utility decisions, rejecting answer-space mutations, preserving typed Choice values separately from provider string labels, and keeping the new surface experimental.
- W7 now includes a comparable dspy-go `Decide` experiment as a `core.Module` sibling of `modules.Predict`, with explicit decision artifacts and tuned-parameter persistence.
- A review of Go, gRPC-Go, OpenTelemetry-Go, Prometheus client_golang, Zap, Google Cloud Go, etcd, SPIFFE, and Kubernetes now grounds the experimental package and graduation policy.
- Jev-specific client, judge, and `Decide` APIs begin under `pkg/experimental`; stable packages may not depend on them.
- Stabilizing the TypeSafe client remains blocked on provider limits, versioning, and support commitments beyond the request/response contract verified from TypeSafe's official SDK repositories.

## Motivation

An optimizer can only improve what its evaluator can see. dspy-go's current metric contract leaves a gap, and the contract blocks the obvious fix.

### The gap between cheap and smart metrics

- **Word-matching metrics** (`ExactMatch`, `F1Score`, `AnyMatch`) are cheap but brittle. "Paris, France" scores low against "Paris", open-ended outputs cannot be judged, and every example needs a gold answer.
- **An LLM judge** understands meaning but is slow and costly per call. Optimizers call the metric once per candidate, per example, per round.
- **The volume adds up fast.** Twenty candidates on a 200-example validation set means 4,000 judge calls in one round. LLM judges therefore get rationed or skipped.

### The current metric contract blocks network-backed judges

- **No inputs.** Optimizers call `metric(example.Outputs, prediction)`. A judge asked "is this grounded in the context?" never sees the context.
- **No context.** A remote judge cannot inherit cancellation, deadlines, or tracing from an optimizer run.
- **No errors.** `core.Metric` returns only a float. A failed judge call must otherwise become a score, commonly 0, so an API outage looks like a bad candidate.

### Probabilistic programs need aggregate evaluation

When Jev or another probabilistic model is under test, accuracy alone throws away its primary output. dspy-go also lacks calibration reports and accuracy-versus-coverage curves. Some of these statistics, including ECE and macro-F1, require a whole dataset and cannot be implemented by averaging an ordinary per-example metric.

## Background: Jev and prior art

Jev is TypeSafe AI's "System One" model: it answers typed questions about a given state and returns probabilities instead of text. TypeSafe describes it as sub-second; the pilot will measure that claim.

A request sends a `state` plus named questions of three types:

| Type | Asks | Returns |
|---|---|---|
| Noul | One yes/no judgment | P(yes); no confidence field |
| Choice | Pick one of 1–255 labels | Chosen label, per-label probabilities, confidence |
| Score | Place on 2–10 ordered levels | Fractional level from 0, distribution, legend, confidence |

Each question carries instructions and criteria: a description per label or level. Ax indicates that examples may also be supported, but Stanford DSPy PR #10463 deliberately sends no demonstrations, so M0 must verify this capability rather than treating it as part of the contract. Responses report the model and token usage.

**Prior art.** Ax, the TypeScript DSPy port, ships Jev in two forms from one package. A provider adapter maps boolean and class outputs to Noul and Choice, so ordinary programs run unchanged. A separate native client exposes all three question types, raw probabilities, and model listing.

Stanford DSPy PR [#10463](https://github.com/stanfordnlp/dspy/pull/10463) is a closer design precedent, although it is open and explicitly experimental. It introduces rich `Noul`, `Score`, and `Choice` values, native shorthand forms, an optional TypeSafe client backed by TypeSafe's Python SDK, and a `Decide` module that is a sibling of `Predict`. Its important distinctions are:

- rich values preserve provider evidence, while shorthand values return only the native value;
- generative `Predict` confidence is self-reported and does not imply a provider distribution;
- Noul exposes P(true), while its local confidence is distance from the configured threshold rather than P(true);
- Score values are expected values over explicit local numeric anchors, which need not be evenly spaced;
- Choice keeps typed values separate from provider string labels and rejects labels that collide after string conversion;
- local thresholds, Score anchors, and Choice multipliers reinterpret cached raw distributions without changing the provider request;
- provider confidence remains attached to the provider's original decision and does not become confidence in a locally reweighted selection;
- answer-space-changing signature overrides are rejected before making a provider request;
- API keys are excluded from persisted client state and request history.

These are implementation insights, not a substitute for TypeSafe's official API contract.

### How major Go projects incubate APIs

There is no language-level experimental annotation in Go. Established projects make instability visible through import paths, module versions, documentation, dependency direction, and release policy.

| Project | Mechanism | Relevant lesson |
|---|---|---|
| Go | Separate `golang.org/x/exp` module | Explicitly outside the Go 1 compatibility promise; packages may change arbitrarily or disappear |
| gRPC-Go | `google.golang.org/grpc/experimental/...` in the main module | Import path and package docs mark risk; its versioning policy permits minor-release changes only for APIs marked experimental at introduction |
| Google Cloud Go Storage | `cloud.google.com/go/storage/experimental` in the Storage module | Public experimental options are isolated in one package and often forward to internal implementation hooks |
| etcd and SPIFFE | `client/v3/experimental/...` and `exp/...` inside existing modules | Small, domain-scoped experiments can remain in the main module when dependency and release isolation are unnecessary |
| OpenTelemetry-Go | Separate nested modules such as `metric/x`, released at `v0` | Stable modules must not depend on experimental modules; independent versioning isolates churn |
| Prometheus client_golang | Separate `github.com/prometheus/client_golang/exp` module | Calls the code production quality while explicitly declaring the API unstable and intended for possible graduation |
| Uber Zap | Separate `go.uber.org/zap/exp` module with `exp/v0.x` tags and its own changelog | Breaking changes ship as experimental minor releases and are still documented explicitly |
| Kubernetes | `v1alphaN` APIs plus feature gates disabled by default and formal graduation criteria | Runtime gates suit server behavior and rollback; they are not a substitute for an experimental Go import path |

The common rules are stronger than the spelling of `exp` versus `experimental`:

1. The import path makes instability obvious.
2. Stable code does not depend on experimental code.
3. Experimental means compatibility may change, not that testing or error handling is optional.
4. Adoption is explicit rather than activated by global registration.
5. Graduation and removal have written criteria.
6. Breaking changes are still documented.

For dspy-go, the first experiment stays in the existing module under `pkg/experimental`. The root module is already `v0`, the experiment is tightly coupled to `core.Module`, and no verified heavy Go SDK currently requires dependency isolation. A nested `dspy-go/exp` module should be reconsidered if dspy-go reaches `v1`, TypeSafe introduces volatile dependencies, or the experiment needs an independent release cadence.

### What Jev may bring to evaluation

| Jev property | Potential benefit for evaluation |
|---|---|
| Sub-second answers (TypeSafe's figure) | The judge may be fast enough for optimizer loops |
| Several questions in one request | A multi-criterion rubric costs one request per example |
| Probabilities, not bare labels | Continuous scores can rank near-misses |
| Criteria are data | Rubrics are explicit, reviewable, and versioned in code |
| Confidence on Choice and Score | A possible signal for selective escalation after calibration |
| Model and token usage in every response | Evaluation runs can record provenance and usage |

These are hypotheses until the pilot measures quality, latency, reliability, and selective risk.

## Goals and non-goals

### Goals

- A meaning-aware judge cheap enough to consider on every optimizer evaluation.
- Evaluation of datasets that have no gold answers.
- Optimizers that distinguish candidate execution failures from evaluator failures.
- Calibration and coverage reports for any program that exposes typed probabilities.
- Evidence that the judge agrees with humans before scaling its use.
- An experimental closed-set `Decide` module that composes with dspy-go programs without pretending Jev is a text-generating `core.LLM`.
- Additive APIs that preserve existing `core.Metric` callers.

### Non-goals

- Replacing gold metrics where exact answers exist, such as GSM8K.
- Explanations from the judge; Jev returns scores, not reasons.
- Generating rubrics automatically.
- Treating provider confidence as correctness.
- Hiding probability data in `context.Context`, execution traces, or reserved output keys.
- Assuming token prices; runs report usage and, where available, provider-reported cost.
- Making `Decide` a stable API in this proposal; W7 is explicitly experimental.
- Building the numeric optimizer for `Decide` parameters; W7 only exposes and persists those parameters.

## Design decisions

Six decisions constrain all workstreams:

1. **Remote judges use the richer evaluator contract only.** `FromMetric` adapts an infallible legacy metric into the richer contract, never the other way around.
2. **The optimizer API is additive.** Existing `Compile` and `CompileExamples` methods remain unchanged. Optimizers opt into a new interface with a differently named method.
3. **An evaluator error is not a score.** Exhausted infrastructure errors fail evaluation by default; an explicit skip policy must enforce and report minimum coverage.
4. **Evaluator identity is part of reproducibility.** Model, rubric digest, evaluator version, and schema version participate in cache keys and run reports.
5. **Probability reports consume typed records.** A future provider adapter must expose an explicit result artifact; the current trace is not a data transport.
6. **The new surface remains experimental initially.** Rich decision values, provider adapters, and local probability reinterpretation are not exported as stable root APIs until persistence, copying, cache identity, and calibration semantics have production evidence.

## Proposal

We propose seven workstreams. W2 is the evaluation foundation. W1 and W6 test Jev as a judge. W3 and W4 build on a successful judge pilot. W5 is provider-neutral and may proceed independently after its typed record format is reviewed. W7 is a separate bounded experiment using Jev as the program's decision backend rather than as its evaluator.

### W1. A Jev judge that compares against the gold answer

- **What:** `jevjudge.New`, initially under `pkg/experimental/jevjudge`, which asks Jev how well a prediction matches the gold answer and implements `core.ExampleMetric`.
- **Depends on:** W2 and the M0 client verification.
- **Why:** every migrated optimizer can replace word-overlap F1 with a meaning-aware score while preserving cancellation and errors.
- **Benefit:** it produces the first real data on Jev as a judge without weakening error handling.

```go
judge := jevjudge.New(client,
    jevjudge.WithModel("jev-pinned-model"),
    jevjudge.Score(
        "matches_gold",
        "How closely does the answer match the reference?",
        "Contradicts or misses it",
        "Partly matches",
        "Same meaning",
    ),
)

optimized, err := compiler.CompileExamplesWithMetric(ctx, program, examples, judge)
```

The implementation uses these scoring rules:

- Noul contributes `P(yes)`.
- Choice requires an explicit utility in `[0,1]` for each label and contributes `sum(P(label) * utility(label))`.
- Score uses explicit local utilities for its ordered levels and contributes `sum(P(level=i) * utility(i)) / sum(P(level=i))`. Utilities default visibly to evenly spaced values in `[0,1]`, but callers may declare non-uniform anchors such as `0`, `0.2`, and `1`. The judge does not assume that a provider's fractional `score` has the desired evaluation utility.
- Label utilities, Score level utilities, question aggregation weights, and any future Choice selection multipliers are distinct concepts and use distinct API names.
- Local utilities do not enter the provider request. They reinterpret the returned distribution, allowing score tuning without another Jev call.
- Question aggregation weights must be finite and non-negative. The constructor normalizes them to sum to 1 and rejects an all-zero rubric.
- Missing labels, malformed distributions, non-finite values, or probabilities outside their allowed tolerance return an error rather than a score.
- The final score and every subscore are finite values in `[0,1]`.

The state encoding keeps reference, prediction, and any inputs in separately named fields. By default, it sends the prediction's task value, not the program's confidence or probability evidence, so the judge cannot copy the program's own certainty. A rubric that intentionally evaluates calibration must opt into that evidence explicitly. Rubric text and serialization order are deterministic so requests can be reproduced and cached safely.

### W2. A metric that sees inputs and reports errors

- **What:** add `ExampleMetric` and `ExampleMetricCompiler` to `pkg/core`, plus an adapter for existing `core.Metric` functions.
- **Why:** this supplies inputs, cancellation, structured results, and errors without changing the current `Optimizer` and `ExamplesCompiler` method signatures.
- **Benefits:**
  - Evaluation without gold answers.
  - Evaluator failures no longer masquerade as bad candidates.
  - Shared scoring can be composed into GEPA, Refine, and agent evaluation while preserving their distinct execution responsibilities.
  - Existing users and third-party optimizer implementations continue to compile.

The intended API shape is:

```go
type EvaluationMetadata struct {
    EvaluatorID      string
    EvaluatorVersion string
    Model            string
    Usage            map[string]int64
    Latency           time.Duration
    CacheHit          bool
}

type MetricResult struct {
    Score          float64
    Subscores      map[string]float64
    Feedback       string
    FeedbackTarget string
    Metadata       map[string]any
    Evaluation     EvaluationMetadata
}

type ExampleMetric interface {
    Evaluate(ctx context.Context, ex Example, pred map[string]any) (MetricResult, error)
}

type ExampleMetricFunc func(context.Context, Example, map[string]any) (MetricResult, error)

func FromMetric(m Metric) ExampleMetric

type ExampleMetricCompiler interface {
    CompileExamplesWithMetric(
        ctx context.Context,
        program Program,
        examples []Example,
        metric ExampleMetric,
    ) (Program, error)
}
```

The exact metadata fields may be refined during API review, but evaluator identity, model provenance, usage, latency, and cache status must not exist only as undocumented map keys.

#### Compatibility and migration

Legacy methods remain authoritative compatibility entry points. A migrated optimizer implements them by wrapping `core.Metric` with `FromMetric` and forwarding to `CompileExamplesWithMetric`. A remote evaluator cannot be converted back to `core.Metric`.

| Consumer | Migration |
|---|---|
| GEPA | Implement `ExampleMetricCompiler`; map subscores and feedback into existing evaluation cases |
| COPRO | Implement `ExampleMetricCompiler`; preserve bounded parallel execution |
| SIMBA | Implement `ExampleMetricCompiler`; retain trajectories and record evaluation status |
| MIPRO | Replace its context-aware internal metric with `ExampleMetric` |
| BootstrapFewShot | Implement the same additive interface or explicitly document why it cannot consume remote metrics |
| MCPOptimizer | Implement the same additive interface or explicitly document exclusion |
| Refine | Add a context/error-aware evaluator field; keep `RewardFunction` for local legacy rewards |
| Agent evaluation | Run the agent through `AgentEvaluator`, then compose an `ExampleMetric` as its output scorer; do not replace `AgentEvaluator` |

#### Failure and aggregation semantics

- A context cancellation or deadline always propagates immediately.
- A candidate program execution failure is distinct from an evaluator failure. The evaluator is not called when no prediction exists. Existing candidate-failure policy is preserved initially but must be reported explicitly.
- An evaluator error causes its `MetricResult` to be ignored. Retryable errors may be retried under the configured retry budget; exhausted errors fail the candidate evaluation by default.
- An optional skip policy must specify a minimum evaluated count or coverage ratio. Candidate rankings report attempted, scored, candidate-failed, evaluator-failed, and skipped counts. Candidates below minimum coverage are ineligible.
- Optimizers reject non-finite scores. The new interface defaults to `[0,1]`; an optimizer supporting another range must declare it.
- No optimizer converts an evaluator error into 0, `-Inf`, or another numeric score.

This work also reconciles current optimizer differences instead of accidentally preserving incompatible failure behavior behind a new interface.

### W3. A rubric judge without gold answers, plus integrations

- **What:** a Jev rubric judge that sees example inputs and predictions, plus explicit integrations for GEPA feedback, Refine rewards, and agent evaluation.
- **Depends on:** a successful Jev pilot and W2.
- **Why:** questions such as "grounded?", "relevant?", and "complete?" often have no gold answer.
- **Benefits:**
  - GEPA's reflection step receives per-criterion scores such as `grounded=0.18` and can make targeted edits.
  - Refine can use a context/error-aware evaluator for affordable best-of-N inference.
  - Agent runs can be scored after the existing `AgentEvaluator` captures execution traces and side information.

These are adapters, not interface replacements:

- GEPA retains candidate and target-component context, then merges `MetricResult` feedback and subscores.
- Refine receives a new evaluator option because its existing `RewardFunction` also lacks context and error returns and is not safe for a remote call.
- `AgentEvaluator` continues to own agent execution. A wrapper converts the resulting output and `AgentExample` into an `ExampleMetric` call and merges evaluation details into `SideInfo`.

### W4. Escalating hard cases to an LLM judge

- **What:** route selected Jev judgments to a pinned LLM judge and combine both judges on the same normalized rubric scale.
- **Depends on:** W3 and calibrated selective-risk results from W5/W6.
- **Why:** a cheap judge's mistakes may cluster on cases detectable by uncertainty signals.
- **Benefit:** the cascade may retain LLM-judge quality while reducing LLM calls. The escalation rate also identifies ambiguous rubric questions.

Raw confidence is only a candidate routing feature, and its source is part of its type:

- Choice and Score may use provider confidence, entropy, top-two margin, and rubric-specific calibration.
- Noul has no provider confidence. For a configured decision threshold `t`, its local boundary distance is `abs(P(yes)-t) / max(t, 1-t)`, not P(yes) and not a calibrated probability of correctness.
- Provider confidence for Choice or Score describes the provider's original decision. If local utilities, thresholds, or multipliers change the selected value, that confidence must not be relabeled as confidence in the local selection.
- Generative self-reported confidence is a separate signal and is never pooled with provider confidence without independent calibration.
- Thresholds are selected per rubric field and signal source on a calibration split and frozen before the test split is opened.
- The report includes risk-versus-coverage curves, LLM call reduction, end-to-end latency, evaluator failures, and quality confidence intervals.
- A maximum escalation budget and circuit-breaker behavior are required. An LLM-judge failure is an evaluator error, not an instruction to keep the Jev score silently.

### W5. Metrics and reports that use probabilities

- **What:** a provider-neutral typed probability record plus per-example scoring and dataset-level reports.
- **Why:** calibration, macro-F1, coverage, and threshold tuning cannot all be represented as ordinary `core.Metric` values.
- **Benefit:** the API answers product questions such as "what share of traffic can we decide automatically at 95% precision?" for any probabilistic program.

The initial input is an explicit record rather than trace state:

```go
type ProbabilityKind string

const (
    ProbabilityBinary  ProbabilityKind = "binary"
    ProbabilityChoice  ProbabilityKind = "choice"
    ProbabilityOrdinal ProbabilityKind = "ordinal"
)

type ProbabilityOption struct {
    Label       string  // Provider label; unique after string conversion.
    Value       any     // Typed application value, such as int(1) versus string("1").
    Probability float64
    Utility     *float64
}

type ProbabilityPrediction struct {
    Name                string
    Kind                ProbabilityKind
    Options             []ProbabilityOption
    ProviderSelection   string
    LocalSelection      string
    ProviderConfidence  *float64
    DecisionConfidence  *float64
    DecisionThreshold   *float64
}

type ProbabilityCase struct {
    Prediction ProbabilityPrediction
    GoldLabel  string
    GoldLevel  *float64
}
```

The final form may use separate typed variants, but it must validate exact label alignment, uniqueness after provider string conversion, positive finite probability mass, distribution sums, finite utilities, and ordinal level order. Raw provider evidence is immutable; thresholds and utilities produce separate derived selections or scores. A record with only generative self-reported confidence is not represented as a provider distribution.

Per-example functions provide binary and multiclass Brier contributions, clipped log loss, correctness, and ordinal expected distance. Dataset accumulators provide:

- ECE with the binning strategy, bin boundaries, counts, and per-bin statistics in the report;
- macro-F1 from an aggregate confusion matrix;
- accuracy/precision versus coverage curves with confidence intervals;
- mean absolute distance and optional ranked/weighted agreement for ordinal outputs;
- threshold selection on development or calibration data, never on the final test set.

Log loss uses a documented epsilon. Reports distinguish top-label calibration, classwise calibration, and binary positive-class calibration rather than calling all three simply "ECE".

W7 defines an experimental program execution artifact that can produce these records. It does not use a reserved output key or the current `ExecutionState`, which stores insufficient and potentially ambiguous model-call data. Stanford DSPy PR #10463 demonstrates one viable shape—rich typed values plus native shorthand and a separate `Decide` parameter—but the dspy-go experiment adapts that idea to Go's explicit interfaces and currently untyped text signature fields.

### W6. Checking the judge against human labels

- **What:** a reproducible harness comparing Jev, a fixed LLM judge, simple gold metrics, and adjudicated human labels.
- **Why:** a judge is itself a model. An unchecked judge lets optimizers confidently improve the wrong objective.
- **Benefit:** trustworthy quality estimates, early warning on poor rubric wording, and a repeatable check when TypeSafe changes a model.

#### Dataset construction

- Freeze outputs from several program checkpoints and quality levels before judging them.
- Include exact answers, valid paraphrases, partial answers, contradictions, unsupported answers, long answers, and adversarial/judge-gaming cases.
- Keep rubric-development, calibration, and final test examples disjoint.
- Start with 100–200 examples only as a pilot. Determine the final sample size from the chosen non-inferiority margin, expected class balance, and desired confidence interval.

#### Human annotation

- Publish instructions and examples for every rubric question.
- Double-label a meaningful overlap, report inter-rater agreement, and adjudicate disagreements without exposing model-judge outputs.
- Record label provenance and rubric version.

#### Comparison

- Pin the LLM judge model, prompt, decoding settings, and output parser.
- Use metrics suited to the question: balanced accuracy/AUROC/Brier for binary judgments, macro-F1/log loss for Choice, and weighted agreement/MAE/rank correlation for Score.
- Report bootstrap confidence intervals and paired differences because all judges evaluate the same examples.
- Repeat a subset of uncached Jev requests to measure numerical run-to-run variation separately from cache determinism.
- Compare rich and shorthand projections only from identical captured provider evidence; do not infer equivalence from independent live calls.
- Pre-register the primary endpoint, non-inferiority margin, latency procedure, and go/no-go rule before opening final test labels.

### W7. Experimental typed decisions and `Decide` module

- **What:** add an experimental `Decide` module, typed Noul/Score/Choice evidence, and explicit native-value projection under `pkg/experimental/decide`.
- **Depends on:** the verified M0 client boundary. Integration with W5 records depends on the W5 record review.
- **Why:** Jev is useful not only as a judge but also as a fast closed-set program backend. Stanford DSPy's experiment gives dspy-go a concrete behavior matrix to test.
- **Scope:** one request may answer several declared outputs; no demonstrations, prompt optimizer integration, or numeric optimizer is built in. Generative fallback and cascades remain explicit program composition rather than `Decide` behavior.

`Decide` is a predictor in the general sense, but it is not the concrete `*modules.Predict` type:

- `modules.Predict` calls a text-generating `core.LLM`, formats prompts and demonstrations, parses XML/JSON/text, and is discovered explicitly by several prompt optimizers.
- `Decide` calls a narrow System One client, validates a closed answer space, preserves raw distributions, and applies local thresholds/utilities.
- Making `Decide` a `*modules.Predict` would let COPRO, SIMBA, BootstrapFewShot, and similar code attach demonstrations or mutate prompt state that `Decide` does not have.
- Both remain `core.Module` values and therefore compose in the same `core.Program`.

The initial Go shape is explicit because `core.Signature` does not currently carry Boolean, enum, or ordinal output annotations:

```go
module, err := decide.New(
    client,
    signature,
    decide.Noul("urgent"),
    decide.Score("severity",
        decide.Level(0, "Minor"),
        decide.Level(2, "Disruptive"),
        decide.Level(10, "Blocking"),
    ),
    decide.Choice[string]("category",
        decide.Option("billing", "Payment issue"),
        decide.Option("technical", "Product malfunction"),
    ),
)

// Process satisfies core.Module and returns native task values for composition.
outputs, err := module.Process(ctx, inputs)

// ProcessDecision returns the same values plus typed provider evidence.
result, err := module.ProcessDecision(ctx, inputs)
urgent, found := decide.Get[decide.NoulDecision](result, "urgent")
if !found {
    return fmt.Errorf("urgent evidence is missing")
}
probability := urgent.Probability
```

The exact generic constructors may change during the experiment. The behavioral contract is more important:

- `Process` returns native values (`bool`, `float64`, or the declared Choice value) so ordinary program composition is unsurprising.
- `ProcessDecision` is an additive interface returning native outputs plus rich typed evidence. It is the source for W5 records; `Process` does not hide evidence in context or under an output sentinel.
- Noul stores P(true), the local threshold, selected value, and threshold-relative decision confidence.
- Score stores the raw index-keyed distribution, provider confidence, declared anchors, and expected local value.
- Choice stores provider string labels, typed application values, raw probabilities, provider selection/confidence, and any distinct local selection. Labels that collide after string conversion are rejected.
- Provider evidence is retained separately from local parameters. After changing a threshold, Score anchor, or Choice multiplier, `Reinterpret` preserves that evidence while deriving a new local result without a provider call. Each result carries private signature and answer-space provenance, and reinterpretation rejects incompatible evidence.
- Output names and answer spaces are fixed at construction. Incompatible `SetSignature` changes are recorded as validation failures and cause `Process` to fail before a provider call; instruction and input-description edits may remain compatible.

The module's `Clone` method deep-copies its local parameters, and the existing `core.ParameterProvider`/`core.ParameterConsumer` interfaces persist thresholds, Score anchors, and Choice multipliers. It deliberately does not implement `core.DemoProvider` or `core.DemoConsumer`. Existing prompt optimizers that discover `*modules.Predict` therefore leave it alone. A later numeric optimizer can target its tuned parameters through a dedicated capability interface.

The System One client is passed explicitly and is not serialized with credentials. Because `core.Module` currently requires `SetLLM`, the experiment must document and test that `SetLLM` does not replace the System One client; a follow-up may split provider injection into optional capability interfaces rather than silently treating Jev as a full `core.LLM`.

The experiment remains under an `experimental` import path with no stable root alias. Its exit report covers:

- Noul, non-uniform Score, and typed Choice request/response fixtures;
- multi-output request batching and context cancellation;
- native and rich projections from identical evidence;
- local reinterpretation after parameter changes without another provider call;
- cloning and program-state round trips without credentials;
- exclusion from demo/prompt optimizers and discovery through its dedicated parameter capability;
- an opt-in live comparison with ordinary `Predict` on a closed-set dataset, reporting quality, latency, calibration, and uncached variance.

[`examples/typesafe_vs_predict`](../../examples/typesafe_vs_predict/) is the runnable W7 head-to-head harness for labeled accuracy, per-call latency, calls, and separate provider token counts. Its live measurement bypasses dspy-go's transparent LLM response cache, while replay fixtures match the complete recorded System One request. Replay validates mechanics only; a pre-registered live run on adequate held-out data is still required for W7 evidence, including calibration and uncached variance. [`examples/typesafe_cascade`](../../examples/typesafe_cascade/) separately demonstrates evidence-aware escalation to `modules.Predict` without making raw provider confidence a correctness probability.

## Cross-cutting requirements

### Client contract

Before M0 stabilizes a public client, verify TypeSafe's official documentation, SDK, or a vendor-provided schema for:

- endpoint and authentication;
- model IDs and pinning behavior;
- request and response schemas;
- question, demonstration, and payload limits;
- probability, fractional Score, and confidence semantics;
- token-usage fields;
- timeout, rate-limit, and retry guidance;
- structured error responses and API-version compatibility;
- supported SDK versions and whether an official Go SDK exists.

Stanford DSPy PR #10463 uses TypeSafe's Python SDK and contains compatibility code for SDK 0.6 and 0.7 response models. That is useful evidence that an SDK boundary can isolate transport changes, but neither it nor Ax is the source of truth for a public Go API. API keys must be excluded from persisted state, cache keys, logs, traces, and request history.

### Experimental API policy

All Jev-specific public surfaces begin under:

```text
pkg/experimental/
├── typesafe/    # System One client and wire types
├── jevjudge/    # ExampleMetric implementation
└── decide/      # core.Module and typed decision evidence
```

The leaf package names remain `typesafe`, `jevjudge`, and `decide`; exported identifiers do not repeat an `Experimental` prefix.

Each package must contain a package comment equivalent to:

```go
// Package decide provides closed-set decisions backed by System One.
//
// EXPERIMENTAL: This package has no compatibility guarantee and may change or
// be removed in a v0 minor release. Persisted state and cache formats are also
// unstable unless explicitly versioned otherwise.
package decide
```

The policy is:

- Stable packages such as `pkg/core`, `pkg/metrics`, `pkg/modules`, and `pkg/optimizers` must not import `pkg/experimental`; an architecture test enforces this direction.
- Experimental packages may import stable packages and implement their public interfaces.
- There are no stable root aliases, global provider registration, or default configuration for Jev. Importing the package and constructing the client/module is the opt-in mechanism.
- Build tags and `GOEXPERIMENT` are not used as maturity markers. A runtime feature gate is added only if a shipped CLI/server later enables Jev behavior implicitly; such a gate starts disabled.
- Experimental code receives ordinary unit, race, vet, fixture, persistence, and security coverage. Live tests remain credential-gated.
- Backward-incompatible changes are allowed in a dspy-go `v0` minor release but must appear in release notes with state/cache migration impact.
- Every serialized experimental artifact and persistent cache entry carries a schema version and rejects unknown incompatible versions clearly.
- Graduation requires the milestone evidence in this proposal, an API review, official provider-contract verification, and at least two independent real use cases.
- On graduation, the stable package is introduced first. Where practical, the experimental path forwards to it and is marked `Deprecated:` for at least one release before removal.

If an official TypeSafe Go SDK introduces substantial or fast-moving dependencies, move all three packages into a nested `github.com/XiaoConstantine/dspy-go/exp` module before public release. That follows the OpenTelemetry, Prometheus, and Zap model and prevents the root module from inheriting experimental dependency churn.

### Caching

Use two cache identities rather than conflating provider evidence with a derived metric:

- **Provider-response cache:** store the exact successful Jev response, not a selected label or scalar score. Its key includes endpoint/provider, pinned model, API/schema version, rubric and question order, full state, and serialization version. Local thresholds, label utilities, Score anchors, aggregation weights, and Choice multipliers are excluded when they do not change the provider request, so the same raw distribution can be reinterpreted without another call.
- **Derived-evaluation cache:** store `MetricResult` only when useful. Its key adds evaluator ID/version and every local scoring or selection parameter to the provider-response identity.
- GEPA's existing candidate/example cache is a derived-evaluation cache and must include that full evaluator identity before storing rich results.
- A judge's answer space is immutable after construction. Changing output names, label types, provider labels, Score levels, or rubric criteria requires a new judge identity and is rejected before a provider call; changing only local parameters invalidates the derived layer, not the provider-response layer.
- Do not persist context cancellation, rate limiting, transport failures, 5xx responses, or malformed responses. Permanent request validation errors may be negatively cached only for a short, documented TTL.
- Coalesce identical in-flight requests so parallel optimizers do not stampede the service.
- Persistent caching of potentially sensitive production examples is opt-in and documented.
- A cache hit records the source response's model/usage separately from usage billed during the current run.

### Concurrency and reliability

- Use a shared client-side rate limiter across optimizer workers, not one limiter per candidate.
- Bound concurrency independently from request rate.
- Retry only errors classified as retryable, honor `Retry-After`, use jittered backoff, and respect context deadlines.
- Send one request per example carrying the whole rubric when allowed by the verified API limits; split deterministically otherwise.

### Reproducibility and testing

- Record evaluator version, rubric digest, model, API version, latency, retries, cache status, and token usage for every call and aggregate them per run.
- Test request serialization and response validation against recorded fixtures.
- Test cancellation, deadlines, retries, malformed distributions, partial responses, cache identity, cache stampede prevention, and error aggregation.
- Test native and rich projections from the same captured response, colliding Choice labels, non-uniform Score utilities, local selection changes that preserve provider confidence provenance, and rejection of answer-space-changing overrides before any request.
- Test copy/save/load behavior and prove credentials and runtime history are not persisted.
- Run live tests only when `TYPESAFE_API_KEY` is set and mark them separately from normal unit tests. Include adversarial inputs, but document that bounded live checks are not a prompt-injection or calibration guarantee.
- Include invariant tests proving that evaluator failures never enter score aggregates.

## Plan and milestones

The revised plan establishes the safe evaluator boundary before using a network judge.

| Milestone | Work | Depends on | Exit condition |
|---|---|---|---|
| M0 | Verify official TypeSafe contract; build a System One client spike in `pkg/experimental/typesafe` | Official API information | Fixtures, cancellation, validation, model pinning, and error classification pass review |
| M1 | W2: additive evaluator API, optimizer migration, failure policy, evaluator-aware cache identity | — | All in-scope metric consumers pass compatibility and failure-invariant tests |
| M2 | W1 + W6 gold-answer pilot on `examples/hotpotqa` | M0, M1 | Pre-registered human/LLM comparison and cold-cache performance report |
| M3 | W5 typed probability records and aggregate reports | API review only | Reference fixtures validate Brier, log loss, ECE, macro-F1, ordinal metrics, and coverage curves |
| M4 | W7 experimental `Decide` module and typed decision artifacts | M0; M3 for report integration | Fixture matrix, clone/state, cache reinterpretation, optimizer-isolation, and opt-in live report |
| M5 | W3 rubric judge and integrations | M2 go, M1 | Grounding/relevance pilot without gold answers |
| M6 | W4 calibrated Jev→LLM cascade | M3, M5 | Held-out selective-risk and end-to-end cost/latency report |

W7 has its own experiment report because failure as a general-purpose judge does not prove failure on a constrained decision task. It remains experimental regardless of the M2 outcome and becomes stable only through a later API decision.

## Go/no-go rules

The final thresholds are frozen in the M2 experiment plan before test labels are opened. The proposed defaults are:

1. **Judge quality:** the lower bound of the paired 95% confidence interval for Jev minus the fixed LLM judge on the primary human-agreement endpoint is at least `-0.05`.
2. **Reliability:** fewer than 1% of calls have exhausted evaluator failures under the declared retry policy; all failures remain outside score aggregates.
3. **Performance:** on the same hardware, concurrency, and cold-cache workload, the Jev evaluation round is at least 2× faster than the LLM-judge round at the quality margin above.

Outcomes:

- Proceed to W3 when standalone Jev meets all three conditions.
- If standalone Jev misses the quality margin but its uncertainty features show useful selective risk, run a bounded W4 cascade experiment before deciding against Jev.
- Stop Jev-specific W3/W4 work when it misses the quality margin and offers no held-out selective-risk separation, or when reliability/performance removes its operational advantage.
- Continue W2 and W5 regardless because neither depends on Jev.
- Judge W7 independently on constrained-task quality, calibration, latency, API fit, and maintenance cost; do not infer its result solely from M2.

## Success metrics

The proposal succeeds when the relevant milestone criteria hold:

1. **Contract correctness:** compatibility tests preserve legacy `core.Metric` behavior, and invariant tests prove no evaluator failure is converted into a score.
2. **Judge validity:** Jev meets the pre-registered M2 human-agreement non-inferiority margin, or the M6 cascade meets it while materially reducing LLM calls.
3. **Optimizer performance:** across at least five fixed optimizer seeds, Jev-optimized programs are non-inferior to F1-optimized programs on held-out official HotpotQA metrics and adjudicated correctness. The default non-inferiority margin is 0.02 on a normalized `[0,1]` scale, reported with paired confidence intervals.
4. **Operational value:** cold-cache p50/p95 latency, end-to-end round time, request counts, retries, cache hits, error rates, and token usage are reported for Jev and the fixed LLM baseline under equal concurrency limits.
5. **Cascade value, if used:** the cascade meets the quality margin while reducing LLM-judge calls by at least 50%; the report includes the full risk/coverage curve rather than only the chosen threshold.
6. **Decision experiment:** W7 demonstrates correct typed evidence, native composition, cache reinterpretation, clone/state safety, and optimizer isolation, and publishes a held-out comparison with `modules.Predict`. This is evidence for a later stabilization decision, not an automatic stable API commitment.

A final evaluation never uses the same judge and examples that drove optimization as its sole outcome measure.

## Risks and mitigations

The largest risk is an optimizer learning to please the judge instead of doing the task.

| Risk | Mitigation |
|---|---|
| The optimizer games the judge | Evaluate held-out outputs with official gold metrics and adjudicated humans or an independently specified judge; include adversarial cases in W6 |
| The judge is poorly calibrated in a new domain | Repeat calibration and validation per domain; do not reuse thresholds blindly |
| The model behind `jev-latest` changes | Pin the model for experiments and record the response model; never use `latest` for a reproducibility claim |
| Dependence on the TypeSafe API | Keep W2/W5 provider-neutral; put judges behind a narrow client interface; use fixtures and explicit failure policy |
| Long outputs or agent traces exceed context limits | Reject or summarize through an explicit, versioned preprocessing step; never truncate silently |
| Confidence is read as correctness | Type provider, decision-boundary, and self-reported confidence separately; calibrate each on held-out selective-risk data |
| Local weighting changes a selection but confidence still describes the provider choice | Preserve provider selection/confidence and local selection as separate fields; never overwrite provenance |
| Evaluator errors bias candidate ranking | Fail evaluation by default or enforce minimum coverage; report all attempted/scored/failed counts |
| Cached scores become stale after rubric/model changes | Separate provider-response and derived-result keys; make answer-space changes create a new identity |
| Independent live calls vary numerically | Compare projections from captured evidence, measure uncached repeatability, and report variance |
| Credentials leak through persistence or history | Exclude API keys from state, cache keys, logs, traces, and request history; test round trips |
| Persistent cache stores sensitive examples | Make persistent evaluator caching opt-in and document retention/deletion behavior |
| Human labels are noisy or class-imbalanced | Double-label overlap, adjudicate, report agreement, stratify sampling, and use appropriate balanced metrics |
| Prediction text manipulates the rubric judge | Keep state fields separate, test adversarial instructions, and use independent final evaluation |

## Alternatives considered

| Alternative | Why not |
|---|---|
| Keep word-matching metrics only | Cannot judge open-ended outputs or data without gold answers and penalizes valid paraphrases |
| Use an LLM judge everywhere | Too slow and costly for every optimizer call; W4 still uses one selectively |
| Return a remote Jev judge as `core.Metric` | Cannot carry context or errors, so cancellation and API failures are represented incorrectly |
| Change `core.Optimizer.Compile` to accept `any` or a union-like interface | Breaks public and third-party implementations and weakens compile-time checking |
| Detect `ExampleMetric` from the existing metric argument | Impossible because the argument's static type is the concrete `core.Metric` function type |
| Put probabilities in execution context or traces | Current state is not returned reliably to callers, stores insufficient data, and is ambiguous for multiple model calls |
| Put probabilities under a reserved output key | Pollutes user outputs and can affect existing metrics and signatures |
| Add richer metrics to GEPA only | Leaves other optimizer and inference paths unable to use context/error-aware evaluation |
| Make experimental `Decide` a subtype or wrapper of `modules.Predict` | `Predict` is coupled to `core.LLM`, demos, prompt formatting/parsing, and concrete-type optimizer discovery. A sibling `core.Module` preserves program composition without inheriting those incorrect contracts |
| Stabilize `Decide` immediately | The answer-space API, rich/native projection, `SetLLM` mismatch, persistence, and numeric optimization contract need experimental evidence first |

## Open decisions

The architecture review resolves the metric migration and trace questions. The following decisions remain:

1. **Remaining official TypeSafe contract details.** The initial wire shape is verified against TypeSafe's official Python/JavaScript SDKs and the Python SDK's generated OpenAPI models. Which published limits, model-pinning guarantees, API-version policy, and support commitment can the client rely on? Blocks M0 exit.
2. **Human-labeling ownership and budget.** Who writes the annotation guide, labels the overlap, adjudicates disagreements, and approves the power calculation? Blocks M2.
3. **Pre-registered thresholds.** Maintainers must approve or replace the proposed 5-point quality margin, 1% exhausted-failure ceiling, and 2× speed target before final labels are opened. Blocks M2 test evaluation.
4. **Experimental projection API.** W7 proposes native values from `Process` and rich evidence from `ProcessDecision`. The experiment must decide whether a per-output rich projection is also worthwhile or whether the explicit result method is sufficient. Blocks stabilization, not M4.
5. **Provider injection for non-LLM modules.** `core.Module.SetLLM` is mandatory even though `Decide` uses a System One client. M4 must document the compatibility behavior and recommend whether provider setters should become optional capabilities. Blocks stabilization, not the experiment.

The judge and W7 share narrow consumer-owned System One client interfaces. Jev-specific implementations live under `pkg/experimental/{typesafe,jevjudge,decide}` and import stable contracts, never the reverse. The client should use an official Go SDK if one exists and otherwise isolate direct HTTP behind its package boundary. W2 and W5 remain provider-neutral stable work.

## Appendix: dspy-go references

| Item | Location | Why it matters |
|---|---|---|
| `core.Metric` | `pkg/core/optimizer.go:26` | Current function type has no context, inputs, or error |
| Optimizer interfaces | `pkg/core/optimizer.go:11-24` | Public methods statically accept `core.Metric`; runtime detection is not available |
| GEPA metric calls | `pkg/optimizers/gepa_evaluation_adapter.go:170,188` | Pass only gold outputs and prediction |
| GEPA cache key | `pkg/optimizers/gepa_evaluation_adapter.go:297-310` | Currently includes candidate and example, not evaluator identity |
| COPRO metric call | `pkg/optimizers/copro.go:570` | Same legacy metric shape, under parallel evaluation |
| SIMBA metric calls | `pkg/optimizers/simba.go:687,1307` | Same legacy shape with different failure behavior |
| MIPRO metric adapter | `pkg/optimizers/mipro.go:522-532` | Has an internal context-aware function but discards context for `core.Metric` |
| `GEPAFeedbackEvaluator` | `pkg/optimizers/gepa_feedback_metric.go:31` | Receives candidate/example context and remains a GEPA integration surface |
| `AgentEvaluator` | `pkg/agents/optimize/evaluator.go:36` | Owns agent execution and returns score, side information, and error |
| Deterministic agent evaluator | `pkg/agents/optimize/deterministic_evaluator.go:87-121` | Demonstrates existing candidate/comparison error semantics that need deliberate migration |
| Refine `RewardFunction` | `pkg/modules/refine.go:18` | Receives inputs but has neither context nor error; unsafe for direct remote adaptation |
| Execution state | `pkg/core/execution_context.go:15-25` | Stores only one model ID and token-usage value, not typed probabilities |
| Program execution context | `pkg/core/program.go:50-51` | May create internal execution state that is not returned to the caller |
| Word-matching metrics | `pkg/metrics/accuracy.go` | `ExactMatch`, `F1Score`, and `AnyMatch` baselines |
| `core.Module` | `pkg/core/module.go:11-28` | Common composition surface, but currently requires `SetLLM` even for non-generative modules |
| `modules.Predict` | `pkg/modules/predict.go:17-37` | Concrete generative module with demos, `core.LLM`, and persistence capabilities that `Decide` should not inherit |
| Tuned parameter persistence | `pkg/core/state.go:32-42` | Existing `ParameterProvider`/`ParameterConsumer` seam for experimental thresholds and weights |
| Concrete Predict discovery | `pkg/optimizers/copro.go:220-228`, `pkg/optimizers/simba.go:814,885` | A sibling module is naturally excluded from current prompt-specific optimizers |

## Appendix: external sources

### Go experimental API precedents

- [Go `x/exp` policy](https://github.com/golang/exp#readme)
- [gRPC-Go experimental package](https://github.com/grpc/grpc-go/blob/master/experimental/experimental.go)
- [gRPC-Go versioning policy](https://github.com/grpc/grpc-go/blob/master/Documentation/versioning.md)
- [Google Cloud Go Storage experimental package](https://github.com/googleapis/google-cloud-go/blob/main/storage/experimental/experimental.go)
- [OpenTelemetry-Go versioning policy](https://github.com/open-telemetry/opentelemetry-go/blob/main/VERSIONING.md)
- [OpenTelemetry-Go experimental metric module](https://github.com/open-telemetry/opentelemetry-go/tree/main/metric/x)
- [Prometheus client_golang experimental module](https://github.com/prometheus/client_golang/tree/main/exp)
- [Uber Zap experimental module changelog](https://github.com/uber-go/zap/blob/master/exp/CHANGELOG.md)
- [etcd experimental client recipes](https://github.com/etcd-io/etcd/tree/main/client/v3/experimental)
- [SPIFFE experimental package](https://github.com/spiffe/go-spiffe/blob/main/exp/doc.go)
- [Kubernetes feature-gate lifecycle](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/feature-gates.md)
- [Kubernetes API maturity levels](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api_changes.md#alpha-beta-and-stable-versions)

### Jev and System One sources

- [TypeSafe official Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python/tree/v0.7.1)
- [TypeSafe official Python SDK OpenAPI-generated models](https://github.com/typesafe-ai/typesafe-sdk-python/blob/v0.7.1/src/typesafe_sdk/_schemas/models.py)
- [TypeSafe official JavaScript SDK](https://github.com/typesafe-ai/typesafe-sdk-js/tree/v0.6.0)
- [Stanford DSPy PR #10463: experimental decision types and System One `Decide`](https://github.com/stanfordnlp/dspy/pull/10463)
- [PR #10463 TypeSafe client](https://github.com/stanfordnlp/dspy/blob/96c52f70c14a972a84ad24df2590b9ce67655d44/dspy/clients/typesafe.py)
- [PR #10463 decision types](https://github.com/stanfordnlp/dspy/blob/96c52f70c14a972a84ad24df2590b9ce67655d44/dspy/adapters/types/decision.py)
- [PR #10463 `Decide` implementation](https://github.com/stanfordnlp/dspy/blob/96c52f70c14a972a84ad24df2590b9ce67655d44/dspy/predict/decide.py)
- [Ax TypeSafe adapter (`api.ts`)](https://github.com/ax-llm/ax/blob/main/src/ax/ai/typesafe/api.ts)
- [Ax native TypeSafe client (`client.ts`)](https://github.com/ax-llm/ax/blob/main/src/ax/ai/typesafe/client.ts)
- [Ax TypeSafe request and response types (`types.ts`)](https://github.com/ax-llm/ax/blob/main/src/ax/ai/typesafe/types.ts)
- [Ax TypeSafe / Jev guide](https://github.com/ax-llm/ax/blob/main/src/ax/skills/ax-typesafe.md)
- [Ax architecture notes](https://github.com/ax-llm/ax/blob/main/docs/ARCHITECTURE.md)

The initial implementation additionally checks TypeSafe's official Python SDK v0.7.1, official JavaScript SDK v0.6.0, and the OpenAPI-generated models committed in the Python SDK. Those sources establish the endpoint, authentication, default model, core wire shape, retry headers, and error conventions used by the fixture slice. They do not by themselves establish payload limits, model-pinning guarantees, API-version policy, or a long-term support commitment. The sub-second figure remains TypeSafe's claim. M0 must resolve those remaining contract details before stabilization or before the pilot's assumptions are frozen.
