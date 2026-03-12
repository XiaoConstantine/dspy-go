#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# ///
"""Compare deterministic GEPA fixture outputs between Python DSPy and Go dspy-go."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_COMPONENTS = {
    "round_robin": ["classifier"],
    "all": ["classifier", "generator"],
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_scenario(name: str, python_result: dict[str, Any], go_result: dict[str, Any]) -> dict[str, Any]:
    expected = EXPECTED_COMPONENTS[name]

    python_first = sorted(python_result.get("first_candidate_updated_components", []))
    go_first = sorted(go_result.get("first_candidate_updated_components", []))
    python_final = sorted(python_result.get("final_program_updated_components", []))
    go_final = sorted(go_result.get("final_program_updated_components", []))

    return {
        "expected_updated_components": expected,
        "python_first_candidate": python_first,
        "go_first_candidate": go_first,
        "python_final_program": python_final,
        "go_final_program": go_final,
        "first_candidate_match": python_first == go_first == expected,
        "final_program_match": python_final == go_final == expected,
    }


def build_report(python_results: dict[str, Any], go_results: dict[str, Any]) -> dict[str, Any]:
    scenarios = {}
    compatible = True

    for scenario_name in sorted(EXPECTED_COMPONENTS):
        scenario_report = compare_scenario(
            scenario_name,
            python_results["scenarios"][scenario_name],
            go_results["scenarios"][scenario_name],
        )
        scenarios[scenario_name] = scenario_report
        compatible = compatible and scenario_report["first_candidate_match"] and scenario_report["final_program_match"]

    return {
        "fixture": "gepa_component_selection",
        "python_runner": python_results.get("runner"),
        "go_runner": go_results.get("runner"),
        "compatible": compatible,
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare deterministic GEPA fixture results.")
    parser.add_argument("--python-results", type=Path, required=True, help="Path to Python DSPy fixture JSON.")
    parser.add_argument("--go-results", type=Path, required=True, help="Path to Go dspy-go fixture JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write report JSON.")
    args = parser.parse_args()

    report = build_report(load_json(args.python_results), load_json(args.go_results))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
