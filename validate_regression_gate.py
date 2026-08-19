#!/usr/bin/env python3
"""Fail closed when an experimental report regresses from the locked baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROTECTED: set[str] = set()


def by_id(report: dict) -> dict[str, dict]:
    return {row["question_id"]: row for row in report.get("items", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ACL and non-regression gates to an experiment report.")
    parser.add_argument("baseline", type=Path); parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, default=Path("regression_gate.json"))
    args = parser.parse_args(); baseline = json.loads(args.baseline.read_text()); experiment = json.loads(args.experiment.read_text())
    baseline_items, experiment_items = by_id(baseline), by_id(experiment)
    failures = []
    if experiment.get("acl_preflight_failures"): failures.append("acl_preflight_failures")
    if experiment.get("acl_unauthorized_return_count", 0): failures.append("acl_unauthorized_return_count")
    for question_id in sorted(PROTECTED):
        if baseline_items.get(question_id, {}).get("hit") and not experiment_items.get(question_id, {}).get("hit"):
            failures.append(f"protected_regression:{question_id}")
    for group, score in baseline.get("by_type", {}).items():
        if score == 100 and experiment.get("by_type", {}).get(group, 0) < 100:
            failures.append(f"perfect_type_regression:{group}")
    if experiment.get("hit_rate", 0) < 95: failures.append("hit_rate_below_95")
    payload = {"passed": not failures, "failures": failures, "baseline_hit_rate": baseline.get("hit_rate"),
               "experiment_hit_rate": experiment.get("hit_rate")}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
