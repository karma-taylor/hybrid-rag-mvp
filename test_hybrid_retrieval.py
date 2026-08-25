from hybrid_retrieval import BM25, Chunk, HybridRetriever, acl_preflight, allowed, build_evidence, decompose, diagnose, extract_identifiers, field_coverage, intent_router, min_max, normalize, ranking_metrics, tokens
from tune_fusion import stratified_folds

def corpus():
    return [
        Chunk("engineering", "ITEM-24 24芯 CABLE 数量6500米", "engineering/a.md", "engineering/network"),
        Chunk("finance", "员工差旅补贴标准", "finance/a.md", "finance/compensation"),
    ]

def test_normalization_is_shared_by_bm25():
    assert "24芯" in tokens("24 芯 CABLE")
    assert BM25(corpus()).score("24芯 CABLE", {0})[0][0] == 0

def test_controlled_bilingual_expansion_matches_english_policy_fields():
    docs = [
        Chunk("policy", "Limit of Liability and Period of Insurance", "insurance/policy.md", "insurance"),
        Chunk("other", "员工差旅补贴标准", "finance/a.md", "finance/compensation"),
    ]
    assert BM25(docs).score("职业责任险责任限额和保险期限", {0, 1})[0][0] == 0

def test_contract_identifier_is_found_adjacent_to_chinese_text():
    assert extract_identifiers("井场SO-04的 OPGW 工程量") == ["so-04"]

def test_identified_contract_fields_prioritize_field_coverage_without_a_reranker():
    route = intent_router("井场SO-01最终计量的标题和合同号是什么？")
    assert route["coverage_priority"] is True

def test_acl_is_applied_before_candidates():
    result = HybridRetriever(corpus()).search("员工差旅补贴标准", {"role": "engineering"})
    assert result["acl"]["allowed_candidates"] == 1
    assert all(row["department"].startswith("engineering/") for row in result["results"])

def test_rrf_retains_cross_channel_rank_metadata():
    merged = HybridRetriever._rrf([(0, .8)], [(0, .9)])
    assert merged[0]["ranks"] == {"bm25": 1, "dense": 1}

def test_context_budget_preserves_one_result_per_query():
    searches = [{"results": [{"chunk_id": "a", "table_id": None, "content": "a" * 30}]}, {"results": [{"chunk_id": "b", "table_id": None, "content": "b" * 30}]}]
    evidence = build_evidence(searches, max_evidence=2, token_budget=100)
    assert [x["chunk_id"] for x in evidence["evidence"]] == ["a", "b"]

def test_composite_search_keeps_acl_and_one_evidence_per_subquery():
    result = HybridRetriever(corpus()).search_composite("ITEM-24 CABLE，同时财务预算审批流程", {"roles": ["engineering", "finance"]})
    assert len(result["subqueries"]) == 2
    assert len(result["evidence_package"]["evidence"]) == 2

def test_acl_path_is_an_exact_allowed_department():
    doc = Chunk("doc", "编号规则", "engineering/doc.md", "engineering/network")
    gold = [{"question_id": "sample", "user_department_role": "engineering", "expected_chunks": ["doc"]}]
    assert acl_preflight([doc], gold) == []

def test_legacy_department_aliases_are_explicitly_allow_listed():
    civil = Chunk("civil", "工程合同", "civil/halfaya/a.md", "civil/halfaya")
    moc = Chunk("moc", "MOC 项目合同", "civil/哈法亚MOC哈法亚油田分部运营中心EPCC项目/a.md", "civil/哈法亚MOC哈法亚油田分部运营中心EPCC项目")
    insurance = Chunk("insurance", "保险条款", "insurance/a.md", "insurance")
    general = Chunk("general", "导师制通知", "something/a.md", "something")

    assert allowed(civil, ["engineering"])
    assert allowed(civil, ["civil/halfaya"])
    assert allowed(moc, ["engineering"])
    assert allowed(moc, ["civil/哈法亚MOC哈法亚油田分部运营中心EPCC项目"])
    assert allowed(insurance, ["insurance"])
    assert allowed(general, ["something"])
    assert not allowed(insurance, ["engineering"])
    assert not allowed(general, ["insurance"])
    assert allowed(civil, ["executive"])
    assert allowed(insurance, ["executive"])
    assert allowed(general, ["executive"])
    assert not allowed(civil, ["operations"])

def test_cross_document_queries_use_deterministic_subqueries():
    parts = decompose("项目文件编号如何区分？同时说明差旅补贴适用范围。")
    assert len(parts) == 2
    assert "编号" in parts[0] and "补贴" in parts[1]

def test_quota_prevents_one_subquery_from_consuming_context():
    searches = [
        {"results": [{"chunk_id": "a1", "canonical_chunk_id": "a1", "table_id": None, "content": "a" * 90}, {"chunk_id": "a2", "canonical_chunk_id": "a2", "table_id": None, "content": "a" * 90}]},
        {"results": [{"chunk_id": "b1", "canonical_chunk_id": "b1", "table_id": None, "content": "b" * 90}]},
    ]
    evidence = build_evidence(searches, max_evidence=2, token_budget=100, quota=1)
    assert {item["chunk_id"] for item in evidence["evidence"]} == {"a1", "b1"}

def test_min_max_normalizes_and_keeps_flat_signal_neutral():
    assert min_max({1: -8.0, 2: 2.0}) == {1: 0.0, 2: 1.0}
    assert min_max({1: 3.0, 2: 3.0}) == {1: 0.5, 2: 0.5}

def test_intent_router_protects_semantic_contract_questions():
    assert intent_router("施工范围和材料责任如何划分？")["route_type"] == "semantic_default"
    assert intent_router("ITEM-24 CABLE MODEL-24工程量和单价是多少？")["route_type"] == "exact_table_fusion"
    assert intent_router("把两项加起来核对合同总价")["is_reconciliation"] is True

def test_same_contract_table_questions_are_not_artificially_decomposed_in_experimental_profile():
    query = "ITEM-24 的CABLE是几芯的？工程量多少米、单价多少？材料谁供、谁负责运输？"
    assert len(decompose(query)) > 1
    assert decompose(query, preserve_same_contract=True) == [query]
    assert len(decompose("ITEM-24 CABLE，同时财务预算审批流程", preserve_same_contract=True)) == 2

def test_table_material_entity_coverage_prefers_exact_row_over_similar_text():
    row = Chunk("row", "CABLE 24 CORE 数量6500 单价3.4", "engineering/item24.md", "engineering/network")
    other = Chunk("other", "网络施工说明", "engineering/item24.md", "engineering/network")
    query = "ITEM-24 CABLE工程量和单价是多少？"
    assert field_coverage(query, row) > field_coverage(query, other)

def test_baseline_profile_uses_raw_reranker_routing_and_full_paragraph_payload():
    retriever = HybridRetriever(corpus(), profile="baseline_95")
    result = retriever.search("ITEM-24 CABLE MODEL-24工程量和单价是多少？", {"role": "engineering"})
    assert result["fusion_trace"]["route_type"] == "baseline_raw_reranker"
    _, payload = retriever._reranker_payload("问题", corpus()[0])
    assert "正文：ITEM-24 24芯 CABLE" in payload

def test_diagnostics_distinguish_final_top5_truncation_from_evidence_loss():
    searches = [{"stages": {name: [{"chunk_id": "a", "canonical_chunk_id": "a", "rank": 1}]
                            for name in ("bm25_top_100", "dense_top_100", "rrf_top_50", "reranker_top_5")}}]
    evidence = [{"chunk_id": "x", "canonical_chunk_id": "x"}] * 5 + [{"chunk_id": "a", "canonical_chunk_id": "a"}]
    assert diagnose(["a"], searches, evidence)[0]["death_stage"] == "final_top5_truncated"

def test_ranking_metrics_supports_multiple_expected_evidence():
    metric = ranking_metrics({"a", "b"}, ["x", "a", "b"])
    assert metric["hit_at_3"] == 1.0
    assert metric["recall_at_3"] == 1.0
    assert metric["mrr_at_10"] == 0.5

def test_stratified_folds_are_stable_and_cover_each_valid_question_once():
    gold = [
        {"question_id": "sample-a", "category": "semantic", "expected_chunks": ["a"]},
        {"question_id": "sample-b", "category": "semantic", "expected_chunks": ["b"]},
        {"question_id": "sample-c", "category": "table", "expected_chunks": ["c"]},
        {"question_id": "sample-d", "category": "table", "expected_chunks": ["d"]},
        {"question_id": "q_skip", "expected_chunks": []},
    ]
    folds = stratified_folds(gold, count=2)
    ids = [row["question_id"] for fold in folds for row in fold]
    assert sorted(ids) == ["sample-a", "sample-b", "sample-c", "sample-d"]
