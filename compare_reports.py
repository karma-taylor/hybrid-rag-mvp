#!/usr/bin/env python3
"""Forensic, deterministic question-level diff for Hybrid RAG reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def item_map(report: dict) -> dict[str, dict]:
    return {item["question_id"]: item for item in report.get("items", [])}


def routes(item: dict) -> list[str]:
    return [trace.get("route_type", "not_recorded") for trace in item.get("fusion_trace", [])]


def first_death(item: dict) -> list[dict]:
    return [{"canonical_chunk_id": row["canonical_chunk_id"], "death_stage": row["death_stage"],
             "next_action": row["next_action"]} for row in item.get("diagnostics", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a baseline and experiment report without running models.")
    parser.add_argument("baseline", type=Path); parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, default=Path("report_diff.json"))
    args = parser.parse_args()
    base, experiment = json.loads(args.baseline.read_text()), json.loads(args.experiment.read_text())
    before, after = item_map(base), item_map(experiment)
    rows = []
    for question_id in sorted(set(before) | set(after)):
        left, right = before.get(question_id, {}), after.get(question_id, {})
        if bool(left.get("hit")) == bool(right.get("hit")) and routes(left) == routes(right):
            continue
        rows.append({"question_id": question_id, "baseline_hit": left.get("hit"), "experiment_hit": right.get("hit"),
                     "baseline_routes": routes(left), "experiment_routes": routes(right),
                     "experiment_deaths": first_death(right)})
    payload = {
        "baseline": {"file": str(args.baseline), "hit_rate": base.get("hit_rate"), "profile": base.get("profile")},
        "experiment": {"file": str(args.experiment), "hit_rate": experiment.get("hit_rate"), "profile": experiment.get("profile")},
        "regressed": [row for row in rows if row["baseline_hit"] and not row["experiment_hit"]],
        "improved": [row for row in rows if not row["baseline_hit"] and row["experiment_hit"]],
        "other_changed": [row for row in rows if row["baseline_hit"] == row["experiment_hit"]],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: len(value) for key, value in payload.items() if isinstance(value, list)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
