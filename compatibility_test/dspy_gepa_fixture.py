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


class FrontierTaskLM(dspy.clients.lm.LM):
    """Task LM that encodes component state into predictor outputs deterministically."""

    def __init__(self):
        super().__init__("dummy", "chat", 0.0, 1000, True)

    def __call__(self, prompt=None, messages=None, **kwargs):
        del prompt, kwargs
        content = "\n".join(str(message.get("content", "")) for message in messages or [])
        classifier_state = "classifier_improved" if "Updated classifier instruction" in content else "classifier_base"
        generator_state = "generator_improved" if "Updated generator instruction" in content else "generator_base"

        if "Your output fields are:\n1. `category`" in content:
            return [{"text": json.dumps({"category": classifier_state})}]

        return [{"text": json.dumps({"output": generator_state})}]


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
