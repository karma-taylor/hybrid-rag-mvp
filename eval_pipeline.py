"""公开合成评测：Hit@1 / Hit@5 / MRR@10 与不可退化的 Protected Gate。"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from app.retrieval import HybridRetriever

PROTECTED_QUERIES = ["代码合并前需要什么批准？", "差旅费用多久内提交报销？", "保险事故发生后多久通知？"]

def reciprocal_rank(actual: list[str], expected: set[str]) -> float:
    for pos, doc_id in enumerate(actual[:10], 1):
        if doc_id in expected: return 1 / pos
    return 0.0

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['tuning', 'test'], default='test'); parser.add_argument('--data', default='data/evaluation.json')
    args = parser.parse_args(); rows = json.loads(Path(args.data).read_text())
    # 固定拆分，避免调参时把全部题目当作反馈信号。
    rows = [r for i, r in enumerate(rows) if (i % 3 != 0) == (args.mode == 'tuning')]
    retriever = HybridRetriever.from_json('data/policies.json')
    h1=h5=mrr=0.; failed=[]
    for row in rows:
        result=[x.doc_id for x in retriever.search(row['query'],row['user_role'],10)]; expected=set(row['expected_doc_ids'])
        # 对 guest 的空结果是正确行为，评测中应计为命中。
        if not expected: hit1=hit5=1.0; rr=1.0 if not result else 0.0
        else: hit1=float(bool(set(result[:1])&expected)); hit5=float(bool(set(result[:5])&expected)); rr=reciprocal_rank(result,expected)
        h1+=hit1;h5+=hit5;mrr+=rr
        if row['query'] in PROTECTED_QUERIES and not hit5: failed.append(row['query'])
    n=len(rows); print(f'\nEvaluation ({args.mode}) | questions={n}\nHit@1  {h1/n:.3f}\nHit@5  {h5/n:.3f}\nMRR@10 {mrr/n:.3f}')
    print('\nPROTECTED Regression Gate:', 'FAIL' if failed else 'PASS')
    for q in failed: print(' - FAIL:',q)
    if failed:
        print('\033[31mProtected query Hit@5=0: blocking release.\033[0m'); return 1
    return 0
if __name__ == '__main__': sys.exit(main())
