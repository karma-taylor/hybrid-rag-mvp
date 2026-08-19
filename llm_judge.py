#!/usr/bin/env python3
"""Optional, post-selection LLM judge for the three finalist retrieval configurations.

This program is intentionally separate from tuning: it never changes retrieval weights.
Set JUDGE_API_BASE, JUDGE_API_KEY and JUDGE_MODEL to enable it.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


def judge(prompt: str) -> dict:
    base, key, model = (os.environ.get(name) for name in ("JUDGE_API_BASE", "JUDGE_API_KEY", "JUDGE_MODEL"))
    if not all((base, key, model)):
        raise SystemExit("Set JUDGE_API_BASE, JUDGE_API_KEY and JUDGE_MODEL before running the LLM judge.")
    schema = {"type": "object", "properties": {name: {"type": "integer", "minimum": 1, "maximum": 5}
              for name in ("factual_completeness", "faithfulness", "cross_evidence_integration", "clarity")}
              | {"missing_fields": {"type": "array", "items": {"type": "string"}},
                 "unsupported_claims": {"type": "array", "items": {"type": "string"}}}, "required": ["factual_completeness", "faithfulness", "cross_evidence_integration", "clarity", "missing_fields", "unsupported_claims"], "additionalProperties": False}
    body = json.dumps({"model": model, "temperature": 0, "messages": [{"role": "system", "content": "Judge only from the supplied evidence. Return JSON."}, {"role": "user", "content": prompt}],
                       "response_format": {"type": "json_schema", "json_schema": {"name": "rag_judgement", "strict": True, "schema": schema}}}).encode()
    request = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(json.loads(response.read())["choices"][0]["message"]["content"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge finalist answers only; never use during tuning.")
    parser.add_argument("answers", type=Path, help="JSON list containing question_id, reference_answer, answer and evidence.")
    parser.add_argument("--output", type=Path, default=Path("llm_judgements.json"))
    args = parser.parse_args(); answers = json.loads(args.answers.read_text())
    results = []
    for item in answers:
        prompt = "\n\n".join(f"{key}: {item.get(key, '')}" for key in ("question_id", "reference_answer", "answer", "evidence"))
        results.append({"question_id": item["question_id"], "judgement": judge(prompt)})
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
