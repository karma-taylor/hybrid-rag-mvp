"""可解释的 ACL-first 混合检索器。

生产环境可将此模块替换为 Chroma/Elasticsearch；接口保持 ``search`` 不变。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")

ROLE_LABELS = {
    "guest": "访客", "engineering": "工程部", "finance": "财务部",
    "insurance": "保险与风险", "executive": "管理层",
}


def tokenize(text: str) -> list[str]:
    """无额外词典依赖的中文 token：英文词、单字及相邻二字。

    单字让“代码合并前”可匹配“生产代码合并前”，二字保留少量语义区分。
    """
    result: list[str] = []
    for item in TOKEN.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", item):
            result.extend(item)
            result.extend(item[i:i + 2] for i in range(len(item) - 1))
        else:
            result.append(item)
    return result


@dataclass(frozen=True)
class Evidence:
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class HybridRetriever:
    """先 ACL、再 BM25+dense、RRF 融合、轻量交叉词项重排。"""
    def __init__(self, documents: list[dict[str, Any]], enable_dense: bool = False, embedding_model: str = ""):
        self.documents = documents
        self.enable_dense = enable_dense
        self.embedding_model = embedding_model
        self._dense_model = None
        self._doc_tokens = [tokenize(d["text"]) for d in documents]
        self._df: Counter[str] = Counter(t for toks in self._doc_tokens for t in set(toks))

    @classmethod
    def from_json(cls, path: str, **kwargs: Any) -> "HybridRetriever":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), **kwargs)

    def _authorized(self, role: str) -> list[int]:
        # guest 不享有任何内部制度阅读权；这里必须先于任何召回或分数计算。
        if role == "guest":
            return []
        return [i for i, d in enumerate(self.documents) if role in d["allowed_roles"]]

    def _bm25(self, query: list[str], idx: int) -> float:
        terms, n, avg = self._doc_tokens[idx], len(self._doc_tokens[idx]), sum(map(len, self._doc_tokens)) / len(self.documents)
        counts = Counter(terms)
        score = 0.0
        for term in query:
            if term not in counts:
                continue
            idf = math.log(1 + (len(self.documents) - self._df[term] + .5) / (self._df[term] + .5))
            score += idf * counts[term] * 2.2 / (counts[term] + 1.2 * (1 - .75 + .75 * n / avg))
        return score

    def _dense_rank(self, query: str, candidate_ids: list[int]) -> list[int]:
        if not self.enable_dense:
            return candidate_ids
        try:
            if self._dense_model is None:
                from sentence_transformers import SentenceTransformer
                self._dense_model = SentenceTransformer(self.embedding_model)
            vectors = self._dense_model.encode([query] + [self.documents[i]["text"] for i in candidate_ids], normalize_embeddings=True)
            return [x for _, x in sorted(zip((vectors[0] @ vectors[1:].T).tolist(), candidate_ids), reverse=True)]
        except Exception:
            # 公共免费部署在模型下载失败时保留词法通道，且 API 可观察到 dense_disabled 标志。
            self.enable_dense = False
            return candidate_ids

    def search(self, query: str, user_role: str, top_k: int = 5) -> list[Evidence]:
        if user_role not in ROLE_LABELS:
            raise ValueError("未知角色")
        allowed = self._authorized(user_role)
        if not allowed:
            return []
        q = tokenize(query)
        # 仅“需要/什么/前后”等通用词不应让无关的已授权文档进入证据区。
        generic = {"需要", "什么", "如何", "哪些", "多久", "可以", "应该", "必须", "是否", "什么"}
        meaningful = {term for term in q if len(term) >= 2 and term not in generic}
        bm25 = sorted(allowed, key=lambda i: self._bm25(q, i), reverse=True)
        dense = self._dense_rank(query, allowed)
        rrf: defaultdict[int, float] = defaultdict(float)
        for ranking in (bm25, dense):
            for rank, idx in enumerate(ranking, 1):
                rrf[idx] += 1 / (60 + rank)
        # 轻量 reranker：增强问题词覆盖率，方便在 Demo 的 X-Ray 中解释最后排序。
        final = sorted(rrf, key=lambda i: (rrf[i] + .002 * len(set(q) & set(self._doc_tokens[i])), self._bm25(q, i)), reverse=True)[:top_k]
        best = rrf[final[0]] if final else 1
        return [Evidence(self.documents[i]["doc_id"], self.documents[i]["text"], round(rrf[i] / best, 4), {
            "title": self.documents[i]["title"], "department": self.documents[i]["department"],
            "allowed_roles": self.documents[i]["allowed_roles"], "retrieval": "ACL → BM25 + dense → RRF → rerank",
        }) for i in final if self._bm25(q, i) > 0 and (not meaningful or bool(meaningful & set(self._doc_tokens[i])))]
