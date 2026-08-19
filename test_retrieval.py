#!/usr/bin/env python3
"""Evaluate dense retrieval against the project's golden set (Hit Rate @ K)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHUNKS_ROOT = PROJECT_ROOT / "chunked_docs"
DEFAULT_GOLD_SET = PROJECT_ROOT / "gold_set" / "golden_set.json"
MODEL_NAME = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval with BGE-M3.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieved chunks per query.")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    return parser.parse_args()


def load_chunks(chunks_root: Path) -> list[dict]:
    chunks: list[dict] = []
    for json_path in sorted(chunks_root.rglob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        chunks.extend(payload.get("chunks", []))
    return chunks


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if not DEFAULT_CHUNKS_ROOT.is_dir():
        raise SystemExit(f"Chunk directory does not exist: {DEFAULT_CHUNKS_ROOT}")
    if not DEFAULT_GOLD_SET.is_file():
        raise SystemExit(f"Golden set does not exist: {DEFAULT_GOLD_SET}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    chunks = load_chunks(DEFAULT_CHUNKS_ROOT)
    if not chunks:
        raise SystemExit("No chunks were found under chunked_docs.")
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    chunk_texts = [chunk["content"] for chunk in chunks]
    available_ids = set(chunk_ids)

    golden_set = json.loads(DEFAULT_GOLD_SET.read_text(encoding="utf-8"))
    valid_questions = [item for item in golden_set if item.get("expected_chunks")]
    missing_ids = {
        expected_id
        for item in valid_questions
        for expected_id in item["expected_chunks"]
        if expected_id not in available_ids
    }
    if missing_ids:
        raise SystemExit(
            f"Golden set refers to {len(missing_ids)} unknown chunk IDs; "
            "rebuild chunks before evaluating."
        )

    print(f"Embedding {len(chunk_texts)} chunks ...")
    corpus_embeddings = model.encode(
        chunk_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    queries = [item["query"] for item in valid_questions]
    print(f"Evaluating {len(queries)} valid questions (Hit Rate @ {args.top_k}) ...")
    query_embeddings = model.encode(
        queries,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    hit_count = 0
    for item, query_embedding in zip(valid_questions, query_embeddings, strict=True):
        scores = corpus_embeddings @ query_embedding
        top_indices = np.argsort(scores)[-args.top_k:][::-1]
        retrieved_ids = [chunk_ids[index] for index in top_indices]
        expected_ids = item["expected_chunks"]
        is_hit = any(expected_id in retrieved_ids for expected_id in expected_ids)

        if is_hit:
            hit_count += 1
            print(f"HIT  {item['question_id']}: {item['query'][:40]}")
        else:
            print(f"MISS {item['question_id']}: {item['query'][:40]}")
            print(f"  expected: {expected_ids}")
            print(f"  retrieved: {retrieved_ids}")

    hit_rate = hit_count / len(valid_questions) * 100 if valid_questions else 0.0
    print("\n" + "=" * 48)
    print(f"Retrieval evaluation: Dense Hit Rate @ {args.top_k}")
    print(f"All golden-set questions: {len(golden_set)}")
    print(f"Questions evaluated: {len(valid_questions)}")
    print(f"Questions skipped (no expected chunks): {len(golden_set) - len(valid_questions)}")
    print(f"Hits: {hit_count}")
    print(f"Score: {hit_rate:.2f}%")
    print("=" * 48)


if __name__ == "__main__":
    main()
