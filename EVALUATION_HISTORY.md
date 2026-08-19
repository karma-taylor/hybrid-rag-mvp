# Evaluation history (sanitized)

## Current candidate

The reproducible candidate reported in this repository achieved **97.5% Hit@5 (39/40)** on a fixed, private evaluation suite. This document intentionally records aggregate progress only; source documents, questions, evidence IDs, and question-level traces are not published.

| Milestone | Change | Aggregate Hit@5 |
| --- | --- | ---: |
| Dense baseline | BGE-M3 semantic retrieval, Top-5 | 67.5% |
| Hybrid retrieval | ACL-first BM25 + dense candidate generation and RRF | 95.0% |
| Candidate baseline | Reranking, table handling, and regression gates | 97.5% |

## Measurement protocol

- Private evaluation questions with expected evidence identifiers.
- A retrieval hit requires expected evidence in the final Top-5.
- ACL checks are performed before candidate generation.
- Aggregate ACL results: zero preflight anomalies, unauthorized candidates, and unauthorized final results.

## Guardrails

- Parameter selection uses deterministic retrieval metrics only (Hit/Recall, MRR, nDCG, and required-evidence recall).
- Answer-quality judging, where enabled in a private environment, is separated from parameter search.
- Any configuration that regresses protected private checks or ACL behavior is rejected.

See [evaluation_summary.json](evaluation_summary.json) for the sanitized metrics snapshot.
