#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "dspy==3.1.3",
# ]
# ///
"""Deterministic GEPA fixture runner against upstream DSPy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import dspy
import dspy.clients
from dspy.predict import Predict
from dspy.utils.dummies import DummyLM


class MultiComponentModule(dspy.Module):
    """Minimal two-component module for deterministic GEPA parity fixtures."""

    def __init__(self):
        super().__init__()
        self.classifier = Predict("input -> category")
        self.generator = Predict("category, input -> output")

    def forward(self, input: str):
        category = self.classifier(input=input).category
        output = self.generator(category=category, input=input).output
        return dspy.Prediction(category=category, output=output)


class FeedbackModule(dspy.Module):
    """Single-component module for deterministic feedback-guided GEPA fixtures."""

    def __init__(self):
        super().__init__()
        self.classifier = Predict("input -> category")

    def forward(self, input: str):
        return dspy.Prediction(category=self.classifier(input=input).category)


class MinibatchModule(dspy.Module):
    """Single-component module for deterministic minibatch acceptance fixtures."""

    def __init__(self):
        super().__init__()
        self.classifier = Predict("input -> output")
        self.classifier.signature = self.classifier.signature.with_instructions("alpha base")

    def forward(self, input: str):
        return dspy.Prediction(output=self.classifier(input=input).output)


def lm_content(prompt=None, messages=None) -> str:
    return (prompt or "") + "\n".join(str(message.get("content", "")) for message in messages or [])


class FrontierTaskLM(dspy.clients.lm.LM):
    """Task LM that encodes component state into predictor outputs deterministically."""

    def __init__(self):
        super().__init__("dummy", "chat", 0.0, 1000, True)

    def __call__(self, prompt=None, messages=None, **kwargs):
        del prompt, kwargs
        content = lm_content(messages=messages)
        classifier_state = "classifier_improved" if "Updated classifier instruction" in content else "classifier_base"
        generator_state = "generator_improved" if "Updated generator instruction" in content else "generator_base"

        if "Your output fields are:\n1. `category`" in content:
            return [{"text": json.dumps({"category": classifier_state})}]

        return [{"text": json.dumps({"output": generator_state})}]


class FeedbackTaskLM(dspy.clients.lm.LM):
    """Task LM whose outputs improve only after the feedback-guided rewrite lands."""

    def __init__(self):
        super().__init__("dummy", "chat", 0.0, 1000, True)

    def __call__(self, prompt=None, messages=None, **kwargs):
        del kwargs
        content = lm_content(prompt=prompt, messages=messages)
        category = "correct" if "feedback tuned classifier instruction" in content else "wrong"
        return [{"text": json.dumps({"category": category})}]


class FeedbackReflectionLM(dspy.clients.lm.LM):
    """Reflection LM that only emits the improved instruction when feedback is present."""

    def __init__(self):
        super().__init__("dummy", "chat", 0.0, 1000, True)

    def __call__(self, prompt=None, messages=None, **kwargs):
        del kwargs
        content = lm_content(prompt=prompt, messages=messages)
        if "Use classifier terminology exactly" in content:
            return [{"text": "```feedback tuned classifier instruction```"}]
        return [{"text": "```feedback fallback classifier instruction```"}]


class FormatTaskLM(dspy.clients.lm.LM):
    """Task LM that starts with malformed output and only succeeds after a format fix."""

    def __init__(self):
        super().__init__("dummy", "chat", 0.0, 1000, True)

    def __call__(self, prompt=None, messages=None, **kwargs):
        del kwargs
        content = lm_content(prompt=prompt, messages=messages)
        if "format tuned classifier instruction" in content:
            return [{"text": json.dumps({"category": "correct"})}]
        return [{"text": "not json at all"}]


class FormatReflectionLM(dspy.clients.lm.LM):
    """Reflection LM that reacts only to parse-failure-as-feedback evidence.

    This fixture matches on DSPy's built-in parse-failure feedback wording.
    The Go fixture uses the equivalent dspy-go feedback text instead, so this
    scenario asserts semantic rewrite parity rather than exact feedback-string
    parity across implementations.
    """

    def __init__(self):
        super().__init__("dummy", "chat", 0.0, 1000, True)

    def __call__(self, prompt=None, messages=None, **kwargs):
        del kwargs
        content = lm_content(prompt=prompt, messages=messages)
        if "Your output failed to parse." in content:
            return [{"text": "```format tuned classifier instruction```"}]
        return [{"text": "```format fallback classifier instruction```"}]


class ResumeTaskLM(dspy.clients.lm.LM):
    """Task LM that only succeeds after the deterministic resume rewrite lands."""

    def __init__(self):
        super().__init__("dummy", "chat", 0.0, 1000, True)

    def __call__(self, prompt=None, messages=None, **kwargs):
        del kwargs
        content = lm_content(prompt=prompt, messages=messages)
        if "classifier best" in content:
            category = "correct"
        elif "classifier better" in content:
            category = "almost"
        else:
            category = "wrong"
        return [{"text": json.dumps({"category": category})}]


class MinibatchTaskLM(dspy.clients.lm.LM):
    """Task LM that mirrors the current instruction into a deterministic output."""

    def __init__(self):
        super().__init__("dummy", "chat", 0.0, 1000, True)

    def __call__(self, prompt=None, messages=None, **kwargs):
        del kwargs
        content = lm_content(prompt=prompt, messages=messages)
        if "alpha tuned" in content:
            output = "alpha tuned"
        elif "alpha worse" in content:
            output = "alpha worse"
        else:
            output = "alpha base"
        return [{"text": json.dumps({"output": output})}]


class ResumeReflectionLM(dspy.clients.lm.LM):
    """Reflection LM that deterministically performs a two-step rewrite."""

    def __init__(self):
        super().__init__("dummy", "chat", 0.0, 1000, True)

    def __call__(self, prompt=None, messages=None, **kwargs):
        del kwargs
        content = lm_content(prompt=prompt, messages=messages)
        improved_instruction = "classifier best" if "classifier better" in content else "classifier better"
        return [{"text": json.dumps({"improved_instruction": improved_instruction})}]


class StopAfterIntermediateBest:
    """Stop once the partially improved candidate wins, but before the best rewrite lands."""

    def __call__(self, gepa_state) -> bool:
        scores = list(getattr(gepa_state, "program_full_scores_val_set", []) or [])
        best_score = max(scores) if scores else 0.0
        return 0.5 <= best_score < 1.0


def run_early_stop_case(stopped_max_metric_calls: int, use_custom_stopper: bool) -> dict[str, Any]:
    example = dspy.Example(input="early stop fixture input", category="correct").with_inputs("input")
    task_lm = ResumeTaskLM()
    reflection_lm = ResumeReflectionLM()

    def resume_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        if prediction.category == example.category:
            score = 1.0
        elif prediction.category == "almost":
            score = 0.5
        else:
            score = 0.0
        return dspy.Prediction(score=score, feedback="early stop fixture feedback")

    stopped_kwargs: dict[str, Any] = {}
    if use_custom_stopper:
        stopped_kwargs["gepa_kwargs"] = {"stop_callbacks": [StopAfterIntermediateBest()]}

    with dspy.context(lm=task_lm):
        stopped_optimizer = dspy.GEPA(
            metric=resume_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=stopped_max_metric_calls,
            component_selector="round_robin",
            track_stats=True,
            num_threads=1,
            seed=7,
            **stopped_kwargs,
        )
        stopped_program = stopped_optimizer.compile(FeedbackModule(), trainset=[example], valset=[example])

        fresh_optimizer = dspy.GEPA(
            metric=resume_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=12,
            component_selector="round_robin",
            track_stats=True,
            num_threads=1,
            seed=7,
        )
        fresh_program = fresh_optimizer.compile(FeedbackModule(), trainset=[example], valset=[example])

    stopped_details = getattr(stopped_program, "detailed_results", None)
    fresh_details = getattr(fresh_program, "detailed_results", None)
    if stopped_details is None or fresh_details is None:
        raise RuntimeError("DSPy GEPA did not expose detailed_results for early-stop fixture")

    return {
        "stopped_metric_calls": getattr(stopped_details, "total_metric_calls", 0),
        "fresh_metric_calls": getattr(fresh_details, "total_metric_calls", 0),
        "stopped_candidate_count": len(getattr(stopped_details, "candidates", [])),
        "fresh_candidate_count": len(getattr(fresh_details, "candidates", [])),
        "stopped_final_program_instruction": stopped_program.classifier.signature.instructions,
        "fresh_final_program_instruction": fresh_program.classifier.signature.instructions,
    }


def run_stopper_budget_parity_fixture() -> dict[str, Any]:
    return {
        "custom_stopper": run_early_stop_case(12, True),
        "metric_budget": run_early_stop_case(8, False),
    }


def component_instructions(program: dspy.Module) -> dict[str, str]:
    return {
        "classifier": program.classifier.signature.instructions,
        "generator": program.generator.signature.instructions,
    }


def updated_components(program: dspy.Module, original_instructions: dict[str, str]) -> list[str]:
    updated = [
        component
        for component, instruction in component_instructions(program).items()
        if original_instructions.get(component) != instruction
    ]
    updated.sort()
    return updated


def candidate_label(program: dspy.Module, original_instructions: dict[str, str]) -> str:
    updated = updated_components(program, original_instructions)
    if updated == ["classifier"]:
        return "classifier"
    if updated == ["generator"]:
        return "generator"
    if updated == ["classifier", "generator"]:
        return "merged"
    return "seed"


def run_selector_fixture(selector: str) -> dict[str, Any]:
    student = MultiComponentModule()
    original_instructions = component_instructions(student)

    call_count = 0

    def improving_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        nonlocal call_count
        del example, prediction, trace, pred_name, pred_trace
        call_count += 1
        score = min(0.3 + (call_count * 0.1), 1.0)
        return dspy.Prediction(score=score, feedback="Improving feedback")

    task_lm = DummyLM([{"category": "fixture_category", "output": "fixture_output"}] * 20)
    reflection_lm = DummyLM(
        [
            {"improved_instruction": "Updated classifier instruction"},
            {"improved_instruction": "Updated generator instruction"},
        ]
        * 10
    )

    example = dspy.Example(input="fixture input", output="fixture output").with_inputs("input")

    with dspy.context(lm=task_lm):
        optimizer = dspy.GEPA(
            metric=improving_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=8,
            component_selector=selector,
            track_stats=True,
            num_threads=1,
        )
        optimized_program = optimizer.compile(student, trainset=[example], valset=[example])

    # DSPy GEPA exposes intermediate candidates through detailed_results today.
    # Treat this as best-effort fixture plumbing and fall back to the final
    # optimized program if that API surface changes or is unavailable.
    candidates = list(getattr(getattr(optimized_program, "detailed_results", None), "candidates", []))
    first_candidate = candidates[1] if len(candidates) > 1 else optimized_program

    return {
        "selector": selector,
        "first_candidate_updated_components": updated_components(first_candidate, original_instructions),
        "first_candidate_instructions": component_instructions(first_candidate),
        "final_program_updated_components": updated_components(optimized_program, original_instructions),
        "final_program_instructions": component_instructions(optimized_program),
        "candidate_count": len(candidates),
    }


def run_validation_frontier_fixture() -> dict[str, Any]:
    student = MultiComponentModule()
    original_instructions = component_instructions(student)

    def frontier_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        classifier_state = prediction.category
        generator_state = prediction.output

        if example.kind == "train":
            score = 1.0 if (classifier_state == "classifier_improved") ^ (generator_state == "generator_improved") else 0.0
        elif example.kind == "alpha":
            score = 1.0 if classifier_state == "classifier_improved" and generator_state == "generator_base" else 0.5 if classifier_state == "classifier_base" and generator_state == "generator_base" else 0.0
        else:
            score = 1.0 if generator_state == "generator_improved" and classifier_state == "classifier_base" else 0.5 if classifier_state == "classifier_base" and generator_state == "generator_base" else 0.0

        return dspy.Prediction(score=score, feedback=f"frontier feedback for {example.kind}")

    task_lm = FrontierTaskLM()
    reflection_lm = DummyLM(
        [
            {"improved_instruction": "Updated classifier instruction"},
            {"improved_instruction": "Updated generator instruction"},
        ]
        * 100
    )

    trainset = [dspy.Example(kind="train", input="fixture train input", output="fixture output").with_inputs("input")]
    valset = [
        dspy.Example(kind="alpha", input="fixture alpha input", output="fixture output").with_inputs("input"),
        dspy.Example(kind="beta", input="fixture beta input", output="fixture output").with_inputs("input"),
    ]

    with dspy.context(lm=task_lm):
        optimizer = dspy.GEPA(
            metric=frontier_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=24,
            component_selector="round_robin",
            track_stats=True,
            num_threads=1,
        )
        optimized_program = optimizer.compile(student, trainset=trainset, valset=valset)

    detailed_results = getattr(optimized_program, "detailed_results", None)
    if detailed_results is None:
        raise RuntimeError("DSPy GEPA did not expose detailed_results for validation frontier fixture")

    frontier_winner_labels_by_example: list[str] = []
    frontier_coverage_labels: dict[str, int] = {}
    # This fixture currently depends on DSPy exposing
    # per_val_instance_best_candidates on detailed_results. Treat that as
    # upstream-result plumbing rather than a stable public API promise.
    per_val_best = getattr(detailed_results, "per_val_instance_best_candidates", {})
    for case_index in range(len(valset)):
        candidate_indexes = sorted(per_val_best.get(case_index, []))
        if not candidate_indexes:
            continue

        winner_label = candidate_label(detailed_results.candidates[candidate_indexes[0]], original_instructions)
        frontier_winner_labels_by_example.append(winner_label)
        frontier_coverage_labels[winner_label] = frontier_coverage_labels.get(winner_label, 0) + 1

    return {
        "frontier_winner_labels_by_example": frontier_winner_labels_by_example,
        "frontier_coverage_labels": frontier_coverage_labels,
        "candidate_count": len(getattr(detailed_results, "candidates", [])),
    }


def run_ancestor_merge_fixture() -> dict[str, Any]:
    student = MultiComponentModule()
    original_instructions = component_instructions(student)

    def merge_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        classifier_improved = prediction.category == "classifier_improved"
        generator_improved = prediction.output == "generator_improved"

        if example.kind == "train":
            score = 1.0 if classifier_improved ^ generator_improved else 0.0
        elif example.kind == "alpha":
            score = 1.0 if classifier_improved else 0.5 if (not classifier_improved and not generator_improved) else 0.0
        else:
            score = 1.0 if generator_improved else 0.5 if (not classifier_improved and not generator_improved) else 0.0

        return dspy.Prediction(score=score, feedback=f"merge feedback for {example.kind}")

    task_lm = FrontierTaskLM()
    reflection_lm = DummyLM(
        [
            {"improved_instruction": "Updated classifier instruction"},
            {"improved_instruction": "Updated generator instruction"},
        ]
        * 100
    )

    trainset = [dspy.Example(kind="train", input="merge train input", output="fixture output").with_inputs("input")]
    valset = [
        dspy.Example(kind="alpha", input="merge alpha-1 input", output="fixture output").with_inputs("input"),
        dspy.Example(kind="alpha", input="merge alpha-2 input", output="fixture output").with_inputs("input"),
        dspy.Example(kind="beta", input="merge beta-1 input", output="fixture output").with_inputs("input"),
        dspy.Example(kind="beta", input="merge beta-2 input", output="fixture output").with_inputs("input"),
    ]

    with dspy.context(lm=task_lm):
        optimizer = dspy.GEPA(
            metric=merge_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=40,
            reflection_minibatch_size=4,
            component_selector="round_robin",
            track_stats=True,
            num_threads=1,
            seed=7,
            use_merge=True,
            max_merge_invocations=2,
            gepa_kwargs={"merge_val_overlap_floor": 1},
        )
        optimized_program = optimizer.compile(student, trainset=trainset, valset=valset)

    detailed_results = getattr(optimized_program, "detailed_results", None)
    if detailed_results is None:
        raise RuntimeError("DSPy GEPA did not expose detailed_results for ancestor merge fixture")

    # This fixture currently depends on DSPy exposing parent indexes on
    # detailed_results. Treat that as upstream-result plumbing rather than a
    # stable public API promise.
    parent_lists = list(getattr(detailed_results, "parents", []))
    merged_index = next(
        (
            index
            for index, parents in enumerate(parent_lists)
            if len([parent for parent in parents if parent is not None]) == 2
        ),
        None,
    )
    if merged_index is None:
        return {
            "merged_candidate_present": False,
            "merged_candidate_updated_components": [],
            "merged_candidate_parent_labels": [],
            "merged_candidate_parent_count": 0,
        }

    merged_candidate = detailed_results.candidates[merged_index]
    merged_parent_labels = sorted(
        candidate_label(detailed_results.candidates[parent], original_instructions)
        for parent in parent_lists[merged_index]
        if parent is not None
    )
    return {
        "merged_candidate_present": True,
        "merged_candidate_updated_components": updated_components(merged_candidate, original_instructions),
        "merged_candidate_parent_labels": merged_parent_labels,
        "merged_candidate_parent_count": len(merged_parent_labels),
    }


def run_feedback_guided_fixture() -> dict[str, Any]:
    student = FeedbackModule()
    original_instruction = student.classifier.signature.instructions

    def feedback_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        score = 1.0 if prediction.category == example.category else 0.0
        return dspy.Prediction(score=score, feedback="Use classifier terminology exactly")

    task_lm = FeedbackTaskLM()
    reflection_lm = FeedbackReflectionLM()
    example = dspy.Example(input="feedback fixture input", category="correct").with_inputs("input")

    with dspy.context(lm=task_lm):
        optimizer = dspy.GEPA(
            metric=feedback_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=4,
            component_selector="round_robin",
            track_stats=True,
            num_threads=1,
            seed=7,
        )
        optimized_program = optimizer.compile(student, trainset=[example], valset=[example])

    candidates = list(getattr(getattr(optimized_program, "detailed_results", None), "candidates", []))
    candidate_instruction = candidates[1].classifier.signature.instructions if len(candidates) > 1 else optimized_program.classifier.signature.instructions
    final_instruction = optimized_program.classifier.signature.instructions
    return {
        "original_instruction": original_instruction,
        "candidate_instruction": candidate_instruction,
        "final_program_instruction": final_instruction,
        "candidate_count": len(candidates),
    }


def run_format_failure_feedback_fixture() -> dict[str, Any]:
    student = FeedbackModule()
    original_instruction = student.classifier.signature.instructions

    def format_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        score = 1.0 if prediction.category == example.category else 0.0
        return dspy.Prediction(score=score, feedback="metric feedback should not be needed")

    task_lm = FormatTaskLM()
    reflection_lm = FormatReflectionLM()
    example = dspy.Example(input="format fixture input", category="correct").with_inputs("input")

    with dspy.context(lm=task_lm):
        optimizer = dspy.GEPA(
            metric=format_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=4,
            component_selector="round_robin",
            track_stats=True,
            num_threads=1,
            seed=7,
            add_format_failure_as_feedback=True,
        )
        optimized_program = optimizer.compile(student, trainset=[example], valset=[example])

    candidates = list(getattr(getattr(optimized_program, "detailed_results", None), "candidates", []))
    candidate_instruction = candidates[1].classifier.signature.instructions if len(candidates) > 1 else optimized_program.classifier.signature.instructions
    final_instruction = optimized_program.classifier.signature.instructions
    return {
        "original_instruction": original_instruction,
        "candidate_instruction": candidate_instruction,
        "final_program_instruction": final_instruction,
        "candidate_count": len(candidates),
    }


def run_minibatch_case(proposed_instruction: str, expected_output: str) -> dict[str, Any]:
    student = MinibatchModule()

    def minibatch_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        score = 1.0 if prediction.output == example.output else 0.0
        return dspy.Prediction(score=score, feedback="minibatch fixture feedback")

    task_lm = MinibatchTaskLM()
    reflection_lm = DummyLM([{"improved_instruction": proposed_instruction}] * 20)
    trainset = [
        dspy.Example(input="mini-1", output=expected_output).with_inputs("input"),
        dspy.Example(input="mini-2", output=expected_output).with_inputs("input"),
        dspy.Example(input="mini-3", output=expected_output).with_inputs("input"),
    ]
    valset = [dspy.Example(input="mini-val", output=expected_output).with_inputs("input")]

    with dspy.context(lm=task_lm):
        optimizer = dspy.GEPA(
            metric=minibatch_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=8,
            reflection_minibatch_size=2,
            component_selector="round_robin",
            track_stats=True,
            num_threads=1,
            seed=7,
        )
        optimized_program = optimizer.compile(student, trainset=trainset, valset=valset)

    details = getattr(optimized_program, "detailed_results", None)
    candidates = list(getattr(details, "candidates", []))
    candidate_count = len(candidates)
    winning_instruction = optimized_program.classifier.signature.instructions
    if candidate_count > 0 and getattr(details, "best_idx", None) is not None:
        best_idx = details.best_idx
        if 0 <= best_idx < candidate_count:
            winning_instruction = candidates[best_idx].classifier.signature.instructions

    return {
        "candidate_count": candidate_count,
        "candidate_added": candidate_count > 1,
        "final_program_instruction": optimized_program.classifier.signature.instructions,
        "winning_candidate_instruction": winning_instruction,
    }


def run_minibatch_acceptance_fixture() -> dict[str, Any]:
    return {
        "accepted_case": run_minibatch_case("alpha tuned", "alpha tuned"),
        "rejected_case": run_minibatch_case("alpha worse", "alpha base"),
    }


def run_resume_parity_fixture() -> dict[str, Any]:
    example = dspy.Example(input="resume fixture input", category="correct").with_inputs("input")
    task_lm = ResumeTaskLM()
    reflection_lm = ResumeReflectionLM()

    def resume_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        del trace, pred_name, pred_trace
        if prediction.category == example.category:
            score = 1.0
        elif prediction.category == "almost":
            score = 0.5
        else:
            score = 0.0
        return dspy.Prediction(score=score, feedback="resume fixture feedback")

    with tempfile.TemporaryDirectory(prefix="dspy-gepa-resume-stop-") as stopped_run_dir:
        with dspy.context(lm=task_lm):
            stopped_optimizer = dspy.GEPA(
                metric=resume_metric,
                reflection_lm=reflection_lm,
                max_metric_calls=12,
                component_selector="round_robin",
                track_stats=True,
                num_threads=1,
                seed=7,
                log_dir=stopped_run_dir,
                gepa_kwargs={"stop_callbacks": [StopAfterIntermediateBest()]},
            )
            first_program = stopped_optimizer.compile(FeedbackModule(), trainset=[example], valset=[example])

            resumed_optimizer = dspy.GEPA(
                metric=resume_metric,
                reflection_lm=reflection_lm,
                max_metric_calls=12,
                component_selector="round_robin",
                track_stats=True,
                num_threads=1,
                seed=7,
                log_dir=stopped_run_dir,
            )
            resumed_program = resumed_optimizer.compile(FeedbackModule(), trainset=[example], valset=[example])

        checkpoint_written = Path(stopped_run_dir, "gepa_state.bin").exists()

        first_details = getattr(first_program, "detailed_results", None)
        resumed_details = getattr(resumed_program, "detailed_results", None)
        if first_details is None or resumed_details is None:
            raise RuntimeError("DSPy GEPA did not expose detailed_results for resume parity fixture")

    with tempfile.TemporaryDirectory(prefix="dspy-gepa-resume-fresh-") as fresh_run_dir:
        with dspy.context(lm=task_lm):
            fresh_optimizer = dspy.GEPA(
                metric=resume_metric,
                reflection_lm=reflection_lm,
                max_metric_calls=12,
                component_selector="round_robin",
                track_stats=True,
                num_threads=1,
                seed=7,
                log_dir=fresh_run_dir,
            )
            fresh_program = fresh_optimizer.compile(FeedbackModule(), trainset=[example], valset=[example])

        fresh_details = getattr(fresh_program, "detailed_results", None)
        if fresh_details is None:
            raise RuntimeError("DSPy GEPA did not expose detailed_results for fresh resume fixture")

    return {
        "checkpoint_written": checkpoint_written,
        "stopped_metric_calls": getattr(first_details, "total_metric_calls", 0),
        "resumed_metric_calls": getattr(resumed_details, "total_metric_calls", 0),
        "fresh_metric_calls": getattr(fresh_details, "total_metric_calls", 0),
        "resumed_candidate_count": len(getattr(resumed_details, "candidates", [])),
        "fresh_candidate_count": len(getattr(fresh_details, "candidates", [])),
        "resumed_final_program_instruction": resumed_program.classifier.signature.instructions,
        "fresh_final_program_instruction": fresh_program.classifier.signature.instructions,
    }


def build_fixture_report() -> dict[str, Any]:
    return {
        "runner": "python_dspy",
        "dspy_version": getattr(dspy, "__version__", "unknown"),
        "fixtures": {
            "component_selection": {
                "scenarios": {
                    "round_robin": run_selector_fixture("round_robin"),
                    "all": run_selector_fixture("all"),
                },
            },
            "validation_frontier": run_validation_frontier_fixture(),
            "ancestor_merge": run_ancestor_merge_fixture(),
            "feedback_guided": run_feedback_guided_fixture(),
            "format_failure_feedback": run_format_failure_feedback_fixture(),
            "minibatch_acceptance": run_minibatch_acceptance_fixture(),
            "stopper_budget_parity": run_stopper_budget_parity_fixture(),
            "resume_parity": run_resume_parity_fixture(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic DSPy GEPA compatibility fixtures.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write JSON results.")
    args = parser.parse_args()

    report = build_fixture_report()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
