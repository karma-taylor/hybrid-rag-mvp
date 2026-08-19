#!/usr/bin/env python3
"""ACL-first hybrid retrieval with auditable BM25, RRF and optional ML backends.

This module deliberately has no mandatory dependency beyond the standard library.
SentenceTransformers is used only when the local runtime makes it available.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
CHUNKS_ROOT = ROOT / "chunked_docs"
GOLD_SET = ROOT / "gold_set" / "golden_set.json"
DICTIONARY_VERSION = "engineering-normalization-v1"
ROLE_PREFIXES = {
    "engineering": ("engineering/",), "finance": ("finance/",),
    "operations": ("operations/",), "executive": ("engineering/", "finance/", "operations/"),
}

def normalize(text: str) -> str:
    text = text.lower().replace("<sup>", "").replace("</sup>", "")
    text = text.replace("毫米", "mm").replace("芯", "芯")
    text = re.sub(r"(?<=\d)\s+(?=(?:芯|mm|m²|m2|usd|iqd|元|米|core))", "", text)
    text = re.sub(r"(?<=\d)[,，](?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"\s*([/\-])\s*", r"\1", text)
    return text

def tokens(text: str) -> list[str]:
    normalized = normalize(text)
    atoms = re.findall(r"[a-z]+(?:[/-][a-z0-9]+)*|\d+(?:[./-]\d+)*|[\u4e00-\u9fff]+", normalized)
    # Keep compact engineering terms as tokens in addition to their component terms.
    output: list[str] = re.findall(r"\d+(?:芯|mm|m2|m²|米|元|usd|iqd|core)", normalized)
    for atom in atoms:
        output.append(atom)
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", atom):
            output.extend(atom[i:i + 2] for i in range(len(atom) - 1))
    return output

def extract_identifiers(text: str) -> list[str]:
    """Extract generic identifiers such as DOC-42 or REF/2026/01."""
    return re.findall(r"\b[a-z]{2,}(?:[./-][a-z0-9]+)+\b|\b\d+[./-]\w+[./-]\d+\b", normalize(text), re.I)

@dataclass
class Chunk:
    chunk_id: str
    content: str
    source_path: str
    department: str
    chunk_type: str = "paragraph"
    parent_id: str | None = None
    table_id: str | None = None
    canonical_chunk_id: str | None = None
    table_header: str = ""
    table_row: str = ""
    is_total_row: bool = False
    chunk_index: int = 0
    title: str = ""
    identifiers: list[str] = field(default_factory=list)
    lexical_text: str = ""

    def __post_init__(self) -> None:
        self.canonical_chunk_id = self.canonical_chunk_id or self.chunk_id
        self.identifiers = self.identifiers or extract_identifiers(self.content + " " + self.source_path)
        self.lexical_text = self.lexical_text or normalize(" ".join([self.source_path, self.title, *self.identifiers, self.content]))

@dataclass(frozen=True)
class RetrievalProfile:
    """Named, auditable ranking behaviour.  Baseline never inherits experiments."""
    name: str
    use_router: bool
    use_fusion: bool
    compact_reranker_payload: bool
    preserve_same_contract_query: bool

PROFILES = {
    # Compatibility profile reconstructed from the 95% report: raw CrossEncoder
    # ordering over the full legacy paragraph is deliberately isolated here.
    "baseline_95": RetrievalProfile("baseline_95", False, False, False, False),
    "experimental": RetrievalProfile("experimental", True, True, True, True),
}

def title_for(content: str) -> str:
    for line in content.splitlines():
        if line.lstrip().startswith("#"):
            return line.lstrip("# ").strip()
    return ""

def load_chunks(root: Path = CHUNKS_ROOT, include_table_children: bool = False) -> list[Chunk]:
    """Load canonical paragraphs by default; opt in to parent/child table exploration.

    The default preserves legacy chunk IDs used by the existing golden set.  Production
    callers that answer table-wide questions set ``include_table_children=True``.
    """
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("chunks", []):
            chunks.append(Chunk(
                chunk_id=item["chunk_id"], content=item["content"], source_path=item["source_path"],
                department=item["department"], chunk_index=item.get("chunk_index", 0),
                chunk_type=item.get("chunk_type", "paragraph"), parent_id=item.get("parent_id"),
                table_id=item.get("table_id"), title=item.get("title", title_for(item["content"])),
                canonical_chunk_id=item.get("canonical_chunk_id"),
                table_header=item.get("table_header", ""), table_row=item.get("table_row", ""),
                is_total_row=item.get("is_total_row", False),
                identifiers=item.get("identifiers", []), lexical_text=item.get("lexical_text", ""),
            ))
    # Derived table chunks are scored as evidence, but always resolve to the legacy
    # paragraph ID that contains their source row so golden-set scoring remains stable.
    paragraphs = [c for c in chunks if c.chunk_type == "paragraph"]
    for chunk in chunks:
        if re.search(r"合同总价|total contractual price", chunk.content, re.I):
            chunk.is_total_row = True
        if chunk.chunk_type == "paragraph":
            continue
        header, _, row = chunk.content.partition("\n数据行：")
        chunk.table_header = header.removeprefix("表头：").strip()
        chunk.table_row = row.strip() or chunk.content
        chunk.is_total_row = chunk.is_total_row or bool(re.search(r"总计|合计|总价|total|amount", chunk.table_row, re.I))
        owner = next((p for p in paragraphs if p.source_path == chunk.source_path and chunk.table_row and chunk.table_row in p.content), None)
        if owner:
            chunk.canonical_chunk_id = owner.chunk_id
            # Canonical paragraphs inherit the compact row/header representation.
            if not owner.table_row or len(chunk.table_row) > len(owner.table_row):
                owner.table_header, owner.table_row = chunk.table_header, chunk.table_row
                owner.table_id, owner.is_total_row = chunk.table_id, chunk.is_total_row
    return chunks if include_table_children else paragraphs

def allowed(chunk: Chunk, roles: Iterable[str]) -> bool:
    return any(any(chunk.department == prefix.rstrip("/") or chunk.department.startswith(prefix) for prefix in ROLE_PREFIXES.get(role, ())) for role in roles)

def acl_preflight(chunks: list[Chunk], gold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find gold questions whose expected evidence is blocked before retrieval begins."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    failures = []
    for item in gold:
        expected = item.get("expected_chunks", [])
        if not expected:
            continue
        role = item["user_department_role"]
        accessible = [chunk_id for chunk_id in expected if chunk_id in by_id and allowed(by_id[chunk_id], [role])]
        if not accessible:
            failures.append({"question_id": item["question_id"], "role": role, "expected": expected,
                             "reason": "acl_or_gold_mismatch"})
    return failures

def min_max(values: dict[int, float]) -> dict[int, float]:
    """Query-local normalization; a flat signal remains neutral instead of disappearing."""
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if math.isclose(low, high):
        return {key: 0.5 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}

def has_reconciliation_intent(query: str) -> bool:
    return bool(re.search(r"核对|合计|总计|对比|加起来", query))

def intent_router(query: str) -> dict[str, Any]:
    """Route only clearly tabular questions into score fusion; IDs alone stay semantic."""
    reconciliation_keywords = ("合计", "总计", "加起来", "核对", "对比", "总额")
    attribute_keywords = ("单价", "数量", "工程量", "金额", "总价", "清单", "费率", "保费")
    exact_values = re.findall(r"\d+(?:/\d+)?|[A-Za-z]+(?:[-./][A-Za-z0-9]+)+|(?:USD|IQD|元|米|芯|mm²?|m²)", query, re.I)
    reconciliation = any(keyword in query for keyword in reconciliation_keywords)
    has_attribute = any(keyword in query for keyword in attribute_keywords)
    route = "exact_table_fusion" if reconciliation or (has_attribute and bool(exact_values)) else "semantic_default"
    return {"route_type": route, "is_reconciliation": reconciliation,
            "exact_fields_extracted": list(dict.fromkeys(exact_values)),
            "matched_rules": (["reconciliation"] if reconciliation else []) + (["attribute_with_exact_value"] if has_attribute and exact_values else [])}

def field_coverage(query: str, chunk: Chunk) -> float:
    """Score exact query fields plus complete numeric table rows, before normalization."""
    text = normalize(" ".join([chunk.table_header, chunk.table_row, chunk.content]))
    terms = set(extract_identifiers(query)) | {term for term in tokens(query) if any(char.isdigit() for char in term)}
    terms |= {term for term in tokens(query) if re.fullmatch(r"[a-z]{2,}", term)}
    score = float(sum(term in text for term in terms))
    # Uppercase-style item codes are compact, high-precision table entities. A
    # matching full row is more useful than a semantically similar paragraph.
    item_terms = {term for term in tokens(query) if re.fullmatch(r"[a-z]{3,}", term)}
    score += 2.0 * sum(term in text for term in item_terms)
    numeric_values = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text)
    needs_values = bool(re.search(r"工程量|多少|单价|金额|总价|合计|核对|总计|对比", query))
    if needs_values and len(numeric_values) >= 2 and (chunk.table_row or "|" in chunk.content):
        score += 0.10
    return score

class BM25:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks, self.docs = chunks, [tokens(c.lexical_text) for c in chunks]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        self.df: collections.Counter[str] = collections.Counter(t for doc in self.docs for t in set(doc))
        self.tf = [collections.Counter(doc) for doc in self.docs]

    def score(self, query: str, candidates: set[int]) -> list[tuple[int, float]]:
        q = tokens(query); n = len(self.docs); scores = []
        for i in candidates:
            score = 0.0; dl = len(self.docs[i])
            for term in q:
                if not self.tf[i][term]: continue
                idf = math.log(1 + (n - self.df[term] + .5) / (self.df[term] + .5))
                score += idf * self.tf[i][term] * 2.2 / (self.tf[i][term] + 1.2 * (1 - .75 + .75 * dl / self.avgdl))
            # Identifiers and numeric engineering fields are deliberately indivisible.
            exact_terms = set(extract_identifiers(query)) | {t for t in tokens(query) if any(char.isdigit() for char in t)}
            score += sum(3.0 for term in exact_terms if term in self.chunks[i].lexical_text)
            scores.append((i, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)

class HybridRetriever:
    def __init__(self, chunks: list[Chunk], use_dense: bool = False, use_reranker: bool = False,
                 batch_size: int = 32, device: str = "auto", reranker_device: str | None = None,
                 fusion_weights: tuple[float, float, float] = (.75, .15, .10),
                 rrf_k: int = 60, rerank_k: int = 50, profile: str = "experimental") -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose one of {', '.join(PROFILES)}")
        self.chunks, self.bm25 = chunks, BM25(chunks)
        self.fusion_weights = fusion_weights
        self.rrf_k, self.rerank_k = rrf_k, rerank_k
        self.profile = PROFILES[profile]
        self.table_totals: dict[str, Chunk] = {chunk.table_id: chunk for chunk in chunks if chunk.table_id and chunk.is_total_row}
        self.contract_totals: dict[str, Chunk] = {chunk.source_path: chunk for chunk in chunks if chunk.is_total_row and re.search(r"合同总价|total contractual price", chunk.content, re.I)}
        self.dense_model = self.reranker = None
        if device == "auto":
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if device not in {"cpu", "mps"}:
            raise ValueError("device must be auto, cpu, or mps")
        if reranker_device is None:
            reranker_device = device
        elif reranker_device == "auto":
            reranker_device = device
        if reranker_device not in {"cpu", "mps"}:
            raise ValueError("reranker_device must be auto, cpu, or mps")
        if use_dense:
            from sentence_transformers import SentenceTransformer
            self.dense_model = SentenceTransformer("BAAI/bge-m3", device=device)
            self.embeddings = self.dense_model.encode(
                [c.lexical_text for c in chunks], batch_size=batch_size,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
            )
        if use_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=reranker_device)

    def _dense(self, query: str, permitted: set[int]) -> list[tuple[int, float]]:
        if self.dense_model is None: return []
        embedding = self.dense_model.encode([query], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)[0]
        return sorted(((i, float(self.embeddings[i] @ embedding)) for i in permitted), key=lambda x: x[1], reverse=True)

    def _reranker_payload(self, query: str, chunk: Chunk) -> tuple[str, str]:
        """Keep the reranker view narrow; neighbouring/total rows are generation-only context."""
        contract = " ".join(chunk.identifiers)
        if not self.profile.compact_reranker_payload:
            # The reconstructed baseline retains the full legacy paragraph.  This is
            # intentionally separate from the table-specific compact payload.
            return query, f"文档名：{chunk.source_path}\n合同号：{contract}\n章节：{chunk.title}\n正文：{chunk.content}"
        header = chunk.table_header or "无"
        row = chunk.table_row or chunk.content
        query_text = query[:512]
        payload = f"文档名：{chunk.source_path}\n合同号：{contract}\n表头：{header}\n当前行：{row}"
        if not self.reranker:
            return query_text, payload
        tokenizer = self.reranker.tokenizer
        query_ids = tokenizer.encode(query_text, add_special_tokens=False)[:128]
        query_text = tokenizer.decode(query_ids, skip_special_tokens=True)
        prefix = f"文档名：{chunk.source_path}\n合同号：{contract}\n表头：{header}\n当前行："
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)[:160]
        row_ids = tokenizer.encode(row, add_special_tokens=False)
        row_budget = max(1, 384 - len(prefix_ids))
        # Preserve the end of a long row, where quantity/unit-price/amount commonly occur.
        if len(row_ids) > row_budget:
            head = row_ids[: min(64, row_budget)]
            tail = row_ids[-(row_budget - len(head)):]
            row_ids = head + tail
        return query_text, tokenizer.decode(prefix_ids + row_ids, skip_special_tokens=True)

    @staticmethod
    def _rrf(*rankings: list[tuple[int, float]], k: int = 60) -> dict[int, dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        for name, ranking in zip(("bm25", "dense"), rankings, strict=True):
            for rank, (index, score) in enumerate(ranking, 1):
                row = merged.setdefault(index, {"rrf": 0.0, "ranks": {}})
                row["ranks"][name] = rank; row["rrf"] += 1 / (k + rank)
        return merged

    def search(self, query: str, user_context: dict[str, Any], top_k: int = 5,
               candidate_k: int = 100, rrf_k: int | None = None,
               rerank_k: int | None = None) -> dict[str, Any]:
        roles = user_context.get("roles") or ([user_context["role"]] if user_context.get("role") else [])
        if not roles: raise ValueError("user_context must include role or roles")
        permitted = {i for i, chunk in enumerate(self.chunks) if allowed(chunk, roles)}
        intent = intent_router(query) if self.profile.use_router else {
            "route_type": "baseline_raw_reranker", "is_reconciliation": has_reconciliation_intent(query),
            "exact_fields_extracted": [], "matched_rules": ["baseline_profile"],
        }
        started = time.perf_counter()
        rrf_k = self.rrf_k if rrf_k is None else rrf_k
        rerank_k = self.rerank_k if rerank_k is None else rerank_k
        bm25 = self.bm25.score(query, permitted)[:candidate_k]
        dense = self._dense(query, permitted)[:candidate_k]
        # In no-model mode BM25 remains a transparent fallback rather than pretending to be dense.
        merged = self._rrf(bm25, dense, k=rrf_k) if dense else {i: {"rrf": score, "ranks": {"bm25": rank}} for rank, (i, score) in enumerate(bm25, 1)}
        candidates = sorted(merged, key=lambda i: merged[i]["rrf"], reverse=True)[:rerank_k]
        rrf_candidates = list(candidates)
        rerank_scores: dict[int, float] = {}
        if self.reranker and candidates:
            pairs = [self._reranker_payload(query, self.chunks[i]) for i in candidates]
            rerank_scores = dict(zip(candidates, map(float, self.reranker.predict(pairs)), strict=True))
        # Never add raw logits to rank-based signals: normalize all three per query first.
        raw_rerank = rerank_scores or {i: merged[i]["rrf"] for i in candidates}
        rerank_norm, coverage_norm = min_max(raw_rerank), min_max({i: field_coverage(query, self.chunks[i]) for i in candidates})
        rrf_norm = min_max({i: merged[i]["rrf"] for i in candidates})
        if self.profile.use_fusion and intent["route_type"] == "exact_table_fusion":
            rw, cw, fw = self.fusion_weights
            final_scores = {i: rw * rerank_norm[i] + cw * coverage_norm[i] + fw * rrf_norm[i] for i in candidates}
        else:
            # Semantic route deliberately preserves the CrossEncoder ordering that was
            # already successful for similar contracts and non-tabular policy questions.
            final_scores = {i: raw_rerank[i] for i in candidates}
        candidates.sort(key=lambda i: final_scores[i], reverse=True)
        results = []
        for index in candidates[:top_k]:
            chunk = self.chunks[index]; trace = merged[index]
            results.append({"chunk_id": chunk.chunk_id, "canonical_chunk_id": chunk.canonical_chunk_id, "source_path": chunk.source_path, "department": chunk.department,
                            "chunk_type": chunk.chunk_type, "parent_id": chunk.parent_id, "table_id": chunk.table_id,
                            "content": chunk.content, "rrf_score": trace["rrf"], "channel_ranks": trace["ranks"],
                            "rerank_score": rerank_scores.get(index), "rerank_norm": rerank_norm[index],
                            "coverage_raw": field_coverage(query, chunk), "coverage_norm": coverage_norm[index],
                            "rrf_norm": rrf_norm[index], "final_score": final_scores[index]})
        def stage(ranking: list[tuple[int, float]], name: str) -> list[dict[str, Any]]:
            return [{"chunk_id": self.chunks[i].chunk_id, "canonical_chunk_id": self.chunks[i].canonical_chunk_id,
                     "rank": rank, "score": score} for rank, (i, score) in enumerate(ranking, 1)]
        return {"query": query, "normalized_query": normalize(query), "dictionary_version": DICTIONARY_VERSION,
                "fusion_trace": {**intent, "profile": self.profile.name,
                                 "weights": dict(zip(("reranker", "coverage", "rrf"), self.fusion_weights)) if self.profile.use_fusion and intent["route_type"] == "exact_table_fusion" else {"reranker": 1.0}},
                "acl": {"roles": roles, "allowed_candidates": len(permitted), "blocked_candidates": len(self.chunks)-len(permitted)},
                "results": results,
                "stages": {"bm25_top_100": stage(bm25, "bm25"), "dense_top_100": stage(dense, "dense"),
                           "rrf_top_50": stage([(i, merged[i]["rrf"]) for i in rrf_candidates], "rrf"),
                           "reranker_top_50": [{"chunk_id": self.chunks[i].chunk_id, "canonical_chunk_id": self.chunks[i].canonical_chunk_id, "rank": rank,
                                                "rerank_raw": raw_rerank[i], "rerank_norm": rerank_norm[i], "coverage_raw": field_coverage(query, self.chunks[i]),
                                                "coverage_norm": coverage_norm[i], "rrf_norm": rrf_norm[i], "final_score": final_scores[i]} for rank, i in enumerate(candidates, 1)],
                           "reranker_top_5": [{"chunk_id": r["chunk_id"], "canonical_chunk_id": r["canonical_chunk_id"], "rank": rank, "score": r["final_score"]} for rank, r in enumerate(results, 1)]},
                "trace": {"bm25_top": len(bm25), "dense_top": len(dense), "rrf_top": len(candidates), "elapsed_ms": round((time.perf_counter()-started)*1000, 2)}}

    def search_composite(self, query: str, user_context: dict[str, Any], top_k: int = 5, token_budget: int = 8000) -> dict[str, Any]:
        """Run ACL-safe retrieval per rule-derived subquery, then enforce a global context budget."""
        subqueries = decompose(query, preserve_same_contract=self.profile.preserve_same_contract_query)
        searches = [self.search(subquery, user_context, top_k=top_k) for subquery in subqueries]
        reconciliation = has_reconciliation_intent(query)
        # Reserve one context slot for a document/table total before generic fill can consume it.
        evidence = build_evidence(searches, max_evidence=7 if reconciliation else 8, token_budget=token_budget)
        vip = []
        if reconciliation:
            table_ids = {result.get("table_id") for search in searches for result in search["results"] if result.get("table_id")}
            source_paths = {result["source_path"] for search in searches for result in search["results"]}
            present = {result.get("canonical_chunk_id") or result["chunk_id"] for result in evidence["evidence"]}
            totals = [(table_id, self.table_totals.get(table_id)) for table_id in table_ids]
            totals.extend(("contract_total", self.contract_totals.get(source_path)) for source_path in source_paths)
            for table_id, total in totals:
                if not total or total.canonical_chunk_id in present or len(evidence["evidence"]) >= 8:
                    continue
                result = {"chunk_id": total.chunk_id, "canonical_chunk_id": total.canonical_chunk_id,
                          "source_path": total.source_path, "department": total.department, "chunk_type": total.chunk_type,
                          "parent_id": total.parent_id, "table_id": total.table_id, "content": total.content,
                          "vip_total_evidence": True, "reason": "reconciliation_table_total"}
                # VIP evidence must be visible to the Hit@5/answer layer, not merely
                # appended after the usable context window.
                evidence["evidence"].insert(min(4, len(evidence["evidence"])), result); vip.append({"table_id": table_id, "canonical_chunk_id": total.canonical_chunk_id,
                                                                   "reason": "vip_total_evidence"})
        return {"query": query, "subqueries": subqueries, "searches": searches, "evidence_package": evidence,
                "vip_total_evidence": vip, "decomposition_mode": "rule" if len(subqueries) > 1 else "not_required"}

def decompose(query: str, preserve_same_contract: bool = False) -> list[str]:
    """Conservative rule-first splitting; return original query when no safe boundary exists."""
    # A single contract/table question may contain several requested fields but
    # should not exhaust the evidence budget through artificial subqueries.
    cross_domain_markers = ("policy", "insurance", "allowance", "finance", "document-control", "财务", "文控")
    if preserve_same_contract and extract_identifiers(query) and not any(marker in query for marker in cross_domain_markers):
        return [query]
    parts = [x.strip(" ，。；;？?") for x in re.split(r"(?:同时|以及|并且|另外|；|;|？|\?)", query) if len(x.strip()) >= 8]
    return parts if len(parts) > 1 else [query]

def classify_query(query: str) -> str:
    if len(decompose(query)) > 1: return "cross_document_or_composite"
    if re.search(r"(?:多少|单价|金额|保费|总价|合计|芯|米|元)", query): return "exact_field_or_table"
    if len(extract_identifiers(query)) >= 2: return "entity_disambiguation"
    return "single_document_semantic"

def build_evidence(searches: list[dict[str, Any]], max_evidence: int = 8, token_budget: int = 8000, quota: int = 2) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []; seen: set[str] = set(); budget = 0
    # First satisfy per-subquery quotas, then fill globally. A subquery's sole evidence
    # is never discarded because another topic has higher-scoring redundant chunks.
    ordered = [r for search in searches for r in search["results"]]
    per_query: list[dict[str, Any]] = []
    for query_index, search in enumerate(searches):
        kept = 0
        for result in search["results"]:
            result = {**result, "subquery_index": query_index}
            key = result.get("canonical_chunk_id") or result.get("table_id") or result["chunk_id"]
            estimated = max(1, len(result["content"]) // 3)
            if key not in seen and kept < quota and len(selected) < max_evidence and budget + estimated <= token_budget:
                seen.add(key); selected.append(result); per_query.append({"subquery_index": query_index, "chunk_id": result["chunk_id"], "reason": "quota"}); budget += estimated; kept += 1
    for result in ordered:
        key = result.get("canonical_chunk_id") or result.get("table_id") or result["chunk_id"]
        estimated = max(1, len(result["content"]) // 3)
        if key in seen or len(selected) >= max_evidence or budget + estimated > token_budget: continue
        seen.add(key); selected.append(result); budget += estimated
    return {"evidence": selected, "quota_kept": per_query, "estimated_tokens": budget, "truncated": len(selected) < len(ordered)}

def diagnose(expected: list[str], searches: list[dict[str, Any]], evidence: list[dict[str, Any]],
             final_top_k: int = 5) -> list[dict[str, Any]]:
    """Return one funnel death stage and prescribed action for each missing canonical ID."""
    def ranks(stage: str, target: str) -> list[int]:
        return [item["rank"] for search in searches for item in search["stages"][stage]
                if item.get("canonical_chunk_id") == target or item["chunk_id"] == target]
    evidence_ids = {item.get("canonical_chunk_id") or item["chunk_id"] for item in evidence}
    final_ids = {item.get("canonical_chunk_id") or item["chunk_id"] for item in evidence[:final_top_k]}
    output = []
    for target in expected:
        bm25, dense = ranks("bm25_top_100", target), ranks("dense_top_100", target)
        rrf, rerank = ranks("rrf_top_50", target), ranks("reranker_top_5", target)
        if not bm25 and not dense:
            stage, action = "not_in_top_100", "repair_chunking_or_shared_exact_dictionary"
        elif not rrf:
            stage, action = "rrf_top_50_truncated", "increase_rrf_pool_or_adjust_rrf_weights"
        elif not rerank:
            stage, action = "reranker_top_5", "add_document_table_prefix_or_adjust_reranker_template"
        elif target not in evidence_ids:
            stage, action = "evidence_budget", "preserve_subquery_quota_and_canonical_evidence"
        elif target not in final_ids:
            stage, action = "final_top5_truncated", "rebalance_subquery_evidence_or_promote_direct_evidence"
        else:
            stage, action = "hit", "none"
        output.append({"canonical_chunk_id": target, "bm25_ranks": bm25, "dense_ranks": dense,
                       "rrf_ranks": rrf, "reranker_ranks": rerank, "death_stage": stage, "next_action": action})
    return output

def ranking_metrics(expected: set[str], ranked: list[str], cutoffs: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    """Binary expected-chunk metrics; supports multiple valid chunks for composite questions."""
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        top = ranked[:cutoff]
        metrics[f"hit_at_{cutoff}"] = float(bool(expected & set(top)))
        metrics[f"recall_at_{cutoff}"] = len(expected & set(top)) / len(expected) if expected else 1.0
    first = next((rank for rank, chunk_id in enumerate(ranked[:10], 1) if chunk_id in expected), None)
    metrics["mrr_at_10"] = 1 / first if first else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank, chunk_id in enumerate(ranked[:10], 1) if chunk_id in expected)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(expected), 10) + 1))
    metrics["ndcg_at_10"] = dcg / ideal if ideal else 1.0
    return metrics

def evaluate(retriever: HybridRetriever, gold: list[dict[str, Any],], acl: bool) -> dict[str, Any]:
    valid = [q for q in gold if q.get("expected_chunks")]
    rows = []; hits = 0; groups: dict[str, list[bool]] = collections.defaultdict(list)
    totals: collections.defaultdict[str, float] = collections.defaultdict(float); acl_leaks = 0
    for item in valid:
        role = item["user_department_role"] if acl else "executive"
        composite = retriever.search_composite(item["query"], {"role": role})
        evidence = composite["evidence_package"]["evidence"]
        ids = [r.get("canonical_chunk_id") or r["chunk_id"] for r in evidence[:5]]
        all_evidence_ids = [r.get("canonical_chunk_id") or r["chunk_id"] for r in evidence]
        hit = bool(set(ids) & set(item["expected_chunks"]))
        hits += hit
        rank_metric = ranking_metrics(set(item["expected_chunks"]), ids)
        rank_metric["required_evidence_recall_at_8"] = len(set(item["expected_chunks"]) & set(all_evidence_ids)) / len(item["expected_chunks"])
        rrf_ids = {
            stage.get("canonical_chunk_id") or stage["chunk_id"]
            for search in composite["searches"]
            for stage in search["stages"]["rrf_top_50"]
        }
        rank_metric["rrf_expected_recall_at_50"] = len(set(item["expected_chunks"]) & rrf_ids) / len(item["expected_chunks"])
        for key, value in rank_metric.items(): totals[key] += value
        for search in composite["searches"]:
            acl_leaks += sum(not allowed(next(chunk for chunk in retriever.chunks if chunk.chunk_id == result["chunk_id"]), [role]) for result in search["results"])
        group = next((name for name, ids_ in QUESTION_GROUPS.items() if item["question_id"] in ids_), "single_document")
        diagnostics = diagnose(item["expected_chunks"], composite["searches"], evidence)
        groups[group].append(hit); rows.append({"question_id": item["question_id"], "type": group, "hit": hit,
                                                 "expected": item["expected_chunks"], "retrieved": ids,
                                                 "subqueries": composite["subqueries"], "quota": composite["evidence_package"]["quota_kept"],
                                                 "diagnostics": diagnostics,
                                                 "metrics": rank_metric, "fusion_trace": [search["fusion_trace"] for search in composite["searches"]],
                                                 "trace": [search["trace"] for search in composite["searches"]]})
    preflight = acl_preflight(retriever.chunks, gold) if acl else []
    mean_metrics = {key: round(value / len(valid) * 100, 2) for key, value in totals.items()}
    return {"profile": retriever.profile.name, "dictionary_version": DICTIONARY_VERSION,
            "acl_enabled": acl, "acl_preflight_failures": preflight, "acl_unauthorized_return_count": acl_leaks, "questions": len(valid), "hits": hits, "hit_rate": round(hits / len(valid) * 100, 2), "metrics": mean_metrics,
            "by_type": {k: round(sum(v)/len(v)*100, 2) for k,v in groups.items()}, "items": rows}

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ACL-first hybrid retrieval.")
    parser.add_argument("--dense", action="store_true"); parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--no-acl", action="store_true"); parser.add_argument("--report", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="experimental", help="Named retrieval behaviour (default: experimental).")
    parser.add_argument("--table-children", action="store_true", help="Include table parent/row chunks in the candidate index.")
    parser.add_argument("--batch-size", type=int, default=32, help="Dense embedding batch size (default: 32).")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto", help="Model device (default: auto).")
    parser.add_argument("--reranker-device", choices=("auto", "cpu", "mps"), default="cpu", help="CrossEncoder device (default: cpu; avoids MPS command-buffer stalls).")
    args = parser.parse_args(); chunks = load_chunks(include_table_children=args.table_children); retriever = HybridRetriever(chunks, args.dense, args.reranker, args.batch_size, args.device, args.reranker_device, profile=args.profile)
    report = evaluate(retriever, json.loads(GOLD_SET.read_text(encoding="utf-8")), not args.no_acl)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report: args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
