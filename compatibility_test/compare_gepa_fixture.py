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
EXPECTED_FRONTIER_WINNERS = ["classifier", "generator"]
EXPECTED_FRONTIER_COVERAGE = {"classifier": 1, "generator": 1}
EXPECTED_MERGE_COMPONENTS = ["classifier", "generator"]
EXPECTED_MERGE_PARENT_LABELS = ["classifier", "generator"]
EXPECTED_FEEDBACK_INSTRUCTION = "feedback tuned classifier instruction"
EXPECTED_FORMAT_FAILURE_INSTRUCTION = "format tuned classifier instruction"
EXPECTED_MINIBATCH_ACCEPTED_INSTRUCTION = "alpha tuned"
EXPECTED_MINIBATCH_REJECTED_INSTRUCTION = "alpha base"
EXPECTED_RESUME_INSTRUCTION = "classifier best"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_instruction(value: str) -> str:
    normalized = value.replace("[[ ## improved_instruction ## ]]", "").strip()
    if normalized.startswith("{") and normalized.endswith("}"):
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            return normalized
        if isinstance(payload, dict):
            improved_instruction = payload.get("improved_instruction")
            if isinstance(improved_instruction, str):
                return improved_instruction.strip()
    return normalized


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


def compare_validation_frontier(python_result: dict[str, Any], go_result: dict[str, Any]) -> dict[str, Any]:
    python_winners = python_result.get("frontier_winner_labels_by_example", [])
    go_winners = go_result.get("frontier_winner_labels_by_example", [])
    python_coverage = python_result.get("frontier_coverage_labels", {})
    go_coverage = go_result.get("frontier_coverage_labels", {})

    return {
        "expected_frontier_winners": EXPECTED_FRONTIER_WINNERS,
        "expected_frontier_coverage": EXPECTED_FRONTIER_COVERAGE,
        "python_frontier_winners": python_winners,
        "go_frontier_winners": go_winners,
        "python_frontier_coverage": python_coverage,
        "go_frontier_coverage": go_coverage,
        "frontier_winners_match": python_winners == go_winners == EXPECTED_FRONTIER_WINNERS,
        "frontier_coverage_match": python_coverage == go_coverage == EXPECTED_FRONTIER_COVERAGE,
    }


def compare_ancestor_merge(python_result: dict[str, Any], go_result: dict[str, Any]) -> dict[str, Any]:
    python_components = sorted(python_result.get("merged_candidate_updated_components", []))
    go_components = sorted(go_result.get("merged_candidate_updated_components", []))
    python_parent_labels = sorted(python_result.get("merged_candidate_parent_labels", []))
    go_parent_labels = sorted(go_result.get("merged_candidate_parent_labels", []))
    python_parent_count = python_result.get("merged_candidate_parent_count", 0)
    go_parent_count = go_result.get("merged_candidate_parent_count", 0)
    python_present = bool(python_result.get("merged_candidate_present"))
    go_present = bool(go_result.get("merged_candidate_present"))

    return {
        "expected_merged_candidate_components": EXPECTED_MERGE_COMPONENTS,
        "expected_merged_candidate_parent_labels": EXPECTED_MERGE_PARENT_LABELS,
        "python_merged_candidate_present": python_present,
        "go_merged_candidate_present": go_present,
        "python_merged_candidate_components": python_components,
        "go_merged_candidate_components": go_components,
        "python_merged_candidate_parent_labels": python_parent_labels,
        "go_merged_candidate_parent_labels": go_parent_labels,
        "python_merged_candidate_parent_count": python_parent_count,
        "go_merged_candidate_parent_count": go_parent_count,
        "merged_candidate_match": (
            python_present
            and go_present
            and python_components == go_components == EXPECTED_MERGE_COMPONENTS
            and python_parent_labels == go_parent_labels == EXPECTED_MERGE_PARENT_LABELS
            and python_parent_count == go_parent_count == 2
        ),
    }


def compare_feedback_guided(python_result: dict[str, Any], go_result: dict[str, Any]) -> dict[str, Any]:
    python_candidate = python_result.get("candidate_instruction", "")
    go_candidate = go_result.get("candidate_instruction", "")
    python_final = python_result.get("final_program_instruction", "")
    go_final = go_result.get("final_program_instruction", "")

    return {
        "expected_feedback_instruction": EXPECTED_FEEDBACK_INSTRUCTION,
        "python_candidate_instruction": python_candidate,
        "go_candidate_instruction": go_candidate,
        "python_final_program_instruction": python_final,
        "go_final_program_instruction": go_final,
        "candidate_instruction_match": python_candidate == go_candidate == EXPECTED_FEEDBACK_INSTRUCTION,
        "final_program_instruction_match": python_final == go_final == EXPECTED_FEEDBACK_INSTRUCTION,
    }


def compare_format_failure_feedback(python_result: dict[str, Any], go_result: dict[str, Any]) -> dict[str, Any]:
    python_candidate = python_result.get("candidate_instruction", "")
    go_candidate = go_result.get("candidate_instruction", "")
    python_final = python_result.get("final_program_instruction", "")
    go_final = go_result.get("final_program_instruction", "")

    return {
        "expected_format_failure_instruction": EXPECTED_FORMAT_FAILURE_INSTRUCTION,
        "python_candidate_instruction": python_candidate,
        "go_candidate_instruction": go_candidate,
        "python_final_program_instruction": python_final,
        "go_final_program_instruction": go_final,
        "candidate_instruction_match": python_candidate == go_candidate == EXPECTED_FORMAT_FAILURE_INSTRUCTION,
        "final_program_instruction_match": python_final == go_final == EXPECTED_FORMAT_FAILURE_INSTRUCTION,
    }


def compare_minibatch_acceptance(python_result: dict[str, Any], go_result: dict[str, Any]) -> dict[str, Any]:
    python_accepted = python_result.get("accepted_case", {})
    go_accepted = go_result.get("accepted_case", {})
    python_rejected = python_result.get("rejected_case", {})
    go_rejected = go_result.get("rejected_case", {})

    python_accepted_instruction = normalize_instruction(python_accepted.get("final_program_instruction", ""))
    go_accepted_instruction = normalize_instruction(go_accepted.get("final_program_instruction", ""))
    python_rejected_instruction = normalize_instruction(python_rejected.get("final_program_instruction", ""))
    go_rejected_instruction = normalize_instruction(go_rejected.get("final_program_instruction", ""))

    accepted_case_match = (
        bool(python_accepted.get("candidate_added"))
        and bool(go_accepted.get("candidate_added"))
        and python_accepted_instruction == go_accepted_instruction == EXPECTED_MINIBATCH_ACCEPTED_INSTRUCTION
    )
    rejected_case_match = (
        not bool(python_rejected.get("candidate_added"))
        and not bool(go_rejected.get("candidate_added"))
        and python_rejected_instruction == go_rejected_instruction == EXPECTED_MINIBATCH_REJECTED_INSTRUCTION
    )

    return {
        "expected_accepted_instruction": EXPECTED_MINIBATCH_ACCEPTED_INSTRUCTION,
        "expected_rejected_instruction": EXPECTED_MINIBATCH_REJECTED_INSTRUCTION,
        "python_accepted_case": python_accepted,
        "go_accepted_case": go_accepted,
        "python_rejected_case": python_rejected,
        "go_rejected_case": go_rejected,
        "accepted_case_match": accepted_case_match,
        "rejected_case_match": rejected_case_match,
    }


def compare_resume_parity(python_result: dict[str, Any], go_result: dict[str, Any]) -> dict[str, Any]:
    python_resumed = normalize_instruction(python_result.get("resumed_final_program_instruction", ""))
    python_fresh = normalize_instruction(python_result.get("fresh_final_program_instruction", ""))
    go_resumed = normalize_instruction(go_result.get("resumed_final_program_instruction", ""))
    go_fresh = normalize_instruction(go_result.get("fresh_final_program_instruction", ""))

    python_resume_consistent = (
        bool(python_result.get("checkpoint_written"))
        and python_result.get("stopped_metric_calls", 0) < python_result.get("resumed_metric_calls", 0)
        and python_result.get("resumed_candidate_count", 0) == python_result.get("fresh_candidate_count", -1)
        and python_resumed == python_fresh == EXPECTED_RESUME_INSTRUCTION
    )
    go_resume_consistent = (
        bool(go_result.get("checkpoint_written"))
        and go_result.get("stopped_metric_calls", 0) < go_result.get("resumed_metric_calls", 0)
        and go_result.get("resumed_candidate_count", 0) == go_result.get("fresh_candidate_count", -1)
        and go_resumed == go_fresh == EXPECTED_RESUME_INSTRUCTION
    )

    return {
        "expected_resume_instruction": EXPECTED_RESUME_INSTRUCTION,
        "python_result": python_result,
        "go_result": go_result,
        "python_resume_consistent": python_resume_consistent,
        "go_resume_consistent": go_resume_consistent,
        "cross_language_instruction_match": python_resumed == python_fresh == go_resumed == go_fresh == EXPECTED_RESUME_INSTRUCTION,
    }


def build_report(python_results: dict[str, Any], go_results: dict[str, Any]) -> dict[str, Any]:
    fixtures = {"component_selection": {"scenarios": {}}}
    compatible = True

    for scenario_name in sorted(EXPECTED_COMPONENTS):
        scenario_report = compare_scenario(
            scenario_name,
            python_results["fixtures"]["component_selection"]["scenarios"][scenario_name],
            go_results["fixtures"]["component_selection"]["scenarios"][scenario_name],
        )
        fixtures["component_selection"]["scenarios"][scenario_name] = scenario_report
        compatible = compatible and scenario_report["first_candidate_match"] and scenario_report["final_program_match"]

    frontier_report = compare_validation_frontier(
        python_results["fixtures"]["validation_frontier"],
        go_results["fixtures"]["validation_frontier"],
    )
    fixtures["validation_frontier"] = frontier_report
    compatible = compatible and frontier_report["frontier_winners_match"] and frontier_report["frontier_coverage_match"]

    merge_report = compare_ancestor_merge(
        python_results["fixtures"]["ancestor_merge"],
        go_results["fixtures"]["ancestor_merge"],
    )
    fixtures["ancestor_merge"] = merge_report
    compatible = compatible and merge_report["merged_candidate_match"]

    feedback_report = compare_feedback_guided(
        python_results["fixtures"]["feedback_guided"],
        go_results["fixtures"]["feedback_guided"],
    )
    fixtures["feedback_guided"] = feedback_report
    compatible = compatible and feedback_report["candidate_instruction_match"] and feedback_report["final_program_instruction_match"]

    format_failure_report = compare_format_failure_feedback(
        python_results["fixtures"]["format_failure_feedback"],
        go_results["fixtures"]["format_failure_feedback"],
    )
    fixtures["format_failure_feedback"] = format_failure_report
    compatible = compatible and format_failure_report["candidate_instruction_match"] and format_failure_report["final_program_instruction_match"]

    minibatch_report = compare_minibatch_acceptance(
        python_results["fixtures"]["minibatch_acceptance"],
        go_results["fixtures"]["minibatch_acceptance"],
    )
    fixtures["minibatch_acceptance"] = minibatch_report
    compatible = compatible and minibatch_report["accepted_case_match"] and minibatch_report["rejected_case_match"]

    resume_report = compare_resume_parity(
        python_results["fixtures"]["resume_parity"],
        go_results["fixtures"]["resume_parity"],
    )
    fixtures["resume_parity"] = resume_report
    compatible = (
        compatible
        and resume_report["python_resume_consistent"]
        and resume_report["go_resume_consistent"]
        and resume_report["cross_language_instruction_match"]
    )

    return {
        "python_runner": python_results.get("runner"),
        "go_runner": go_results.get("runner"),
        "compatible": compatible,
        "fixtures": fixtures,
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
