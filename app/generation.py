from __future__ import annotations

from openai import OpenAI
from .retrieval import Evidence

REFUSAL = "抱歉，基于当前的知识库检索结果，未找到与该问题相关的信息。"
SYSTEM = """你是专业、严谨的企业知识库问答助手。仅根据【参考证据】回答。
每一句事实陈述必须在句号前标注证据编号，例如：事实[1]。严禁使用先验知识或捏造。
信息不足时必须且只能回答：抱歉，基于当前的知识库检索结果，未找到与该问题相关的信息。
【参考证据】\n{context}"""


def prompt(query: str, evidences: list[Evidence]) -> str:
    context = "\n".join(f'<evidence id="{i}">{e.text}</evidence>' for i, e in enumerate(evidences, 1))
    return SYSTEM.format(context=context) + f"\n【查询】{query}"


def valid(answer: str, count: int) -> bool:
    import re
    cites = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    return bool(cites) and all(1 <= x <= count for x in cites) and all(re.search(r"\[\d+\][。！？]$", s.strip()) for s in re.split(r"(?<=[。！？])", answer) if s.strip())


class Generator:
    def __init__(self, api_key: str | None, base_url: str | None, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.model = model

    def answer(self, query: str, evidences: list[Evidence]) -> tuple[str, str]:
        if not evidences:
            return REFUSAL, "refusal"
        if self.client:
            try:
                output = self.client.chat.completions.create(model=self.model, temperature=0, max_tokens=500,
                    messages=[{"role": "system", "content": prompt(query, evidences)}, {"role": "user", "content": query}]).choices[0].message.content.strip()
                if output == REFUSAL or valid(output, len(evidences)):
                    return output, "llm"
            except Exception:
                pass
        # 无密钥或服务故障时的可演示降级；页面会明确标示，不冒充模型输出。
        return f"根据《{evidences[0].metadata['title']}》，{evidences[0].text}[1]。", "evidence-template"
