from app.retrieval import HybridRetriever
def test_acl_precedes_retrieval():
    r=HybridRetriever.from_json('data/policies.json')
    assert r.search('采购付款需要什么材料','engineering') == []
def test_executive_cross_department_access():
    r=HybridRetriever.from_json('data/policies.json')
    assert r.search('采购付款需要什么材料','executive')[0].doc_id == 'FIN-002'
