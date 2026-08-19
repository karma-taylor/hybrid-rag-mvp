#!/usr/bin/env python3
"""Deterministic five-fold selection for Hybrid RAG fusion parameters.

This intentionally has no answer generator, no network client and no LLM judge.
It is safe to use for fast, reproducible parameter selection.  Only the emitted
top-three configurations should be passed to a separate answer-generation/judge run.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from hybrid_retrieval import GOLD_SET, PROFILES, HybridRetriever, evaluate, load_chunks


# Compact, interpretable grid.  Every weight vector is already normalized, which
# makes experiment comparisons meaningful and avoids arbitrary score magnitudes.
GRID = (
    {"name": "baseline", "weights": (.75, .15, .10), "rrf_k": 60, "rerank_k": 50},
    {"name": "more_coverage", "weights": (.70, .20, .10), "rrf_k": 60, "rerank_k": 50},
    {"name": "more_rerank", "weights": (.80, .10, .10), "rrf_k": 60, "rerank_k": 50},
    {"name": "more_rrf", "weights": (.70, .15, .15), "rrf_k": 60, "rerank_k": 50},
    {"name": "rrf40", "weights": (.75, .15, .10), "rrf_k": 40, "rerank_k": 50},
    {"name": "rrf80", "weights": (.75, .15, .10), "rrf_k": 80, "rerank_k": 50},
)
PROTECTED: set[str] = set()


def group_for(item: dict[str, Any]) -> str:
    return str(item.get("category") or "single_document")


def stratified_folds(gold: list[dict[str, Any]], count: int = 5) -> list[list[dict[str, Any]]]:
    """Stable round-robin distribution within each question category."""
    folds = [[] for _ in range(count)]
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in gold:
        if item.get("expected_chunks"):
            grouped[group_for(item)].append(item)
    for group in sorted(grouped):
        for offset, item in enumerate(sorted(grouped[group], key=lambda row: row["question_id"])):
            folds[offset % count].append(item)
    return folds


def weighted_mean(reports: list[dict[str, Any]], key: str) -> float:
    total = sum(report["questions"] for report in reports)
    return round(sum(report["metrics"].get(key, 0.0) * report["questions"] for report in reports) / max(total, 1), 4)


def protected_hits(reports: list[dict[str, Any]]) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for report in reports:
        for item in report["items"]:
            if item["question_id"] in PROTECTED:
                status[item["question_id"]] = bool(item["hit"])
    return status


def run_configuration(retriever: HybridRetriever, folds: list[list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    retriever.fusion_weights = tuple(config["weights"])
    retriever.rrf_k, retriever.rerank_k = config["rrf_k"], config["rerank_k"]
    reports = [evaluate(retriever, fold, acl=True) for fold in folds]
    metrics = {key: weighted_mean(reports, key) for key in (
        "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "mrr_at_10", "ndcg_at_10", "rrf_expected_recall_at_50", "required_evidence_recall_at_8",
    )}
    protected = protected_hits(reports)
    acl_preflight = sum(len(report["acl_preflight_failures"]) for report in reports)
    acl_leaks = sum(report["acl_unauthorized_return_count"] for report in reports)
    gate_failures = []
    if acl_preflight: gate_failures.append("acl_preflight_failures")
    if acl_leaks: gate_failures.append("acl_unauthorized_return_count")
    if PROTECTED and (set(protected) != PROTECTED or not all(protected.values())):
        gate_failures.append("protected_question_regression")
    deterministic_score = round(
        .50 * metrics["required_evidence_recall_at_8"]
        + .30 * metrics["hit_at_5"]
        + .20 * metrics["mrr_at_10"], 4)
    all_ids = [item["question_id"] for fold in folds for item in fold]
    return {
        "config": config, "folds": [{"fold": index + 1, "validation_ids": [x["question_id"] for x in fold],
                                        "training_ids": [qid for qid in all_ids if qid not in {x["question_id"] for x in fold}],
                                        "hit_rate": report["hit_rate"], "metrics": report["metrics"]}
                                       for index, (fold, report) in enumerate(zip(folds, reports, strict=True))],
        "metrics": metrics, "protected_hits": protected,
        "acl_preflight_failures": acl_preflight, "acl_unauthorized_return_count": acl_leaks,
        "gate_passed": not gate_failures, "gate_failures": gate_failures,
        "deterministic_score": deterministic_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-fold deterministic Hybrid RAG fusion tuner; never calls an LLM.")
    parser.add_argument("--dense", action="store_true", help="Enable BGE-M3 dense retrieval.")
    parser.add_argument("--reranker", action="store_true", help="Enable BGE reranking.")
    parser.add_argument("--table-children", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--reranker-device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="experimental")
    parser.add_argument("--max-configs", type=int, default=len(GRID), help="Limit the grid for a smoke run.")
    parser.add_argument("--report", type=Path, default=Path("fusion_cv_report.json"))
    args = parser.parse_args()
    gold = json.loads(GOLD_SET.read_text(encoding="utf-8"))
    folds = stratified_folds(gold)
    retriever = HybridRetriever(load_chunks(include_table_children=args.table_children), args.dense, args.reranker,
                                args.batch_size, args.device, args.reranker_device, profile=args.profile)
    experiments = [run_configuration(retriever, folds, config) for config in GRID[:args.max_configs]]
    finalists = sorted((row for row in experiments if row["gate_passed"]),
                       key=lambda row: row["deterministic_score"], reverse=True)[:3]
    report = {
        "purpose": "deterministic_parameter_selection_only_no_llm_calls",
        "fold_count": 5,
        "questions": sum(len(fold) for fold in folds),
        "selection_formula": "0.50*required_evidence_recall_at_8 + 0.30*hit_at_5 + 0.20*mrr_at_10",
        "experiments": experiments,
        "top_3": finalists,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "top_3": [x["config"] for x in finalists]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
