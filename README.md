# Hybrid RAG MVP

An ACL-first Hybrid RAG retrieval evaluation project for enterprise-style, mixed-language documents. It combines lexical precision with semantic retrieval, reranking, structured-table handling, and reproducible offline metrics.

## Results at a glance

The current candidate configuration reaches **97.5% Hit@5 (39/40)** on a private, fixed evaluation set. ACL preflight exceptions, unauthorized retrieval candidates, and unauthorized final results were all **0**.

The public repository contains code, tests, environment pins, and a sanitized aggregate report only. It deliberately contains no source documents, chunks, golden questions, question-level reports, or model cache.

## Retrieval architecture

```text
Query + user context
  → ACL allow-list
  → BM25 Top-100 + BGE-M3 dense Top-100
  → RRF (k=60), Top-50
  → bge-reranker-v2-m3
  → Top-5 evidence / bounded evidence package
```

The implementation also includes engineering-domain normalization, table parent/row relationships, deterministic evaluation traces, regression gates, and optional composite-query handling.

## Models and runtime

| Component | Configuration |
| --- | --- |
| Dense retriever | `BAAI/bge-m3` on Apple MPS |
| Reranker | `BAAI/bge-reranker-v2-m3` on CPU |
| Python | 3.12.14 |
| PyTorch | 2.13.0 |
| Sentence Transformers | 5.7.0 |
| Transformers | 5.15.0 |
| Candidate pool | Dense Top-100 + BM25 Top-100 |
| Fusion | Reciprocal Rank Fusion, `k=60` |
| Reranking | RRF Top-50 to final Top-5 |
| Evidence budget | at most 8 evidence items / 8,000 tokens |

Models are downloaded by `sentence-transformers`/Hugging Face on first use. Apple Silicon with MPS is the evaluated hardware path; CPU fallback is supported by the scripts.

## Reproduce the code-level checks

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

The end-to-end evaluation requires a separately governed private corpus mounted locally as `chunked_docs/` and a private evaluation set at `gold_set/golden_set.json`. Neither is needed for the included unit tests, and neither is distributed here.

Example private-data evaluation command:

```bash
.venv/bin/python hybrid_retrieval.py \
  --dense --reranker --device mps --reranker-device cpu \
  --batch-size 8 --report local_evaluation_report.json
```

## Repository boundaries

Excluded by design:

- Original Markdown documents and contractual, insurance, or MOC material
- `chunked_docs/`, `gold_set/`, and all question text
- Per-question evaluation traces, local reports, logs, and model caches
- Local virtual environments and operating-system metadata

`evaluation_summary.json` gives only anonymized aggregate metrics and data fingerprints for reproducibility checks.

## Limitations

The published metric is retrieval-oriented: a question is considered hit when required evidence appears in the final retrieval set. It is not a claim of production answer quality for every domain. Full answer-generation evaluation, private-corpus governance, and deployment controls remain environment-specific.

## License

Released under the [MIT License](LICENSE).
