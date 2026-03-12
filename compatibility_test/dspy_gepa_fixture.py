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


def component_instructions(program: MultiComponentModule) -> dict[str, str]:
    return {
        "classifier": program.classifier.signature.instructions,
        "generator": program.generator.signature.instructions,
    }


def updated_components(program: MultiComponentModule, original_instructions: dict[str, str]) -> list[str]:
    updated = [
        component
        for component, instruction in component_instructions(program).items()
        if original_instructions.get(component) != instruction
    ]
    updated.sort()
    return updated


def run_selector_fixture(selector: str) -> dict[str, Any]:
    student = MultiComponentModule()
    original_instructions = component_instructions(student)

    call_count = 0

    def improving_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        nonlocal call_count
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


def build_fixture_report() -> dict[str, Any]:
    return {
        "runner": "python_dspy",
        "dspy_version": getattr(dspy, "__version__", "unknown"),
        "fixture": "gepa_component_selection",
        "scenarios": {
            "round_robin": run_selector_fixture("round_robin"),
            "all": run_selector_fixture("all"),
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
