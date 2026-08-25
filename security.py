"""Deterministic safety checks for untrusted RAG inputs and outputs."""
from __future__ import annotations

import re
from collections.abc import Iterable

INSTRUCTION_OVERRIDE_PATTERNS = (
    r"ignore (?:all |any |the )?(?:previous|prior|above|system) instructions?",
    r"disregard (?:all |any |the )?(?:previous|prior|above|system)",
    r"reveal (?:the )?(?:system prompt|secret|api key|password)",
    r"jailbreak",
    r"忽略(?:之前|上述|系统)?(?:的)?指令",
    r"无视(?:之前|上述|系统)?(?:的)?指令",
    r"泄露(?:系统提示词|密钥|密码)",
)
_OVERRIDE_RE = re.compile("|".join(INSTRUCTION_OVERRIDE_PATTERNS), re.IGNORECASE)


def contains_instruction_override(text: str) -> bool:
    """Return whether text matches a high-confidence prompt-injection marker."""
    return bool(_OVERRIDE_RE.search(text))


# These rules describe protected knowledge domains, not individual documents or
# evaluation questions.  A request can only enter retrieval when one of the
# caller's roles is authorised for every matched protected domain.
SENSITIVE_DOMAIN_POLICIES: tuple[tuple[tuple[str, ...], frozenset[str]], ...] = (
    (("津贴", "补贴", "工资条", "房租补贴", "人员福利", "子女补贴"), frozenset({"finance"})),
    (("保单", "保险", "保费", "责任限额", "免赔", "被保险人"), frozenset({"insurance"})),
)


def sensitive_query_is_authorized(query: str, roles: Iterable[str]) -> bool:
    """Check explicit sensitive-domain requests against a role allow-list."""
    role_set = {role.strip() for role in roles if role.strip()}
    for markers, permitted_roles in SENSITIVE_DOMAIN_POLICIES:
        if any(marker in query for marker in markers) and not (role_set & permitted_roles):
            return False
    return True


_STRUCTURED_ANCHOR_RE = re.compile(
    r"(?<![a-z0-9])(?:[a-z]+[a-z0-9]*(?:[-/.][a-z0-9]+)+|\d+(?:[-/.][a-z0-9]+)+|[a-z]+\d+)(?![a-z0-9])",
    re.IGNORECASE,
)
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{4,}(?!\d)")


def _anchor_normalize(text: str) -> str:
    return re.sub(r"[\s,，]", "", text).lower()


def extract_high_precision_anchors(query: str) -> tuple[str, ...]:
    """Extract identifiers, amounts, dates, and long numbers requiring evidence support."""
    compact = _anchor_normalize(query)
    anchors = list(_STRUCTURED_ANCHOR_RE.findall(compact))
    anchors.extend(_LONG_NUMBER_RE.findall(compact))
    return tuple(dict.fromkeys(anchor for anchor in anchors if len(anchor) >= 3))


def evidence_supports_query_anchors(query: str, evidence_texts: Iterable[str]) -> bool:
    """Fail closed when any precise query claim has no match in authorised evidence."""
    anchors = extract_high_precision_anchors(query)
    if not anchors:
        return True
    corpus = _anchor_normalize(" ".join(evidence_texts))
    return all(anchor in corpus for anchor in anchors)
