#!/usr/bin/env python3
"""Write a reproducibility manifest for a fixed evaluation report."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from hybrid_retrieval import CHUNKS_ROOT, DICTIONARY_VERSION, GOLD_SET, PROFILES


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.is_file():
        hasher.update(path.read_bytes())
    else:
        for child in sorted(path.rglob("*")):
            if child.is_file():
                hasher.update(str(child.relative_to(path)).encode()); hasher.update(child.read_bytes())
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze report, data and model-runtime provenance.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", type=Path, default=Path("baseline_95_manifest.json"))
    args = parser.parse_args(); report = json.loads(args.report.read_text(encoding="utf-8"))
    payload = {
        "report": str(args.report), "report_sha256": digest(args.report), "gold_set_sha256": digest(GOLD_SET),
        "chunk_corpus_sha256": digest(CHUNKS_ROOT), "dictionary_version": DICTIONARY_VERSION,
        "profile": args.profile, "profile_spec": PROFILES[args.profile].__dict__, "command": args.command,
        "models": {"dense": "BAAI/bge-m3", "reranker": "BAAI/bge-reranker-v2-m3"},
        "runtime": {"python": platform.python_version(), "sentence-transformers": importlib.metadata.version("sentence-transformers"),
                    "transformers": importlib.metadata.version("transformers"), "torch": importlib.metadata.version("torch")},
        "acceptance_oracle": {"hit_rate": report.get("hit_rate"), "hits": report.get("hits"),
                              "misses": [item["question_id"] for item in report.get("items", []) if not item.get("hit")]},
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
