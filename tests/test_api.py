from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
def test_health(): assert client.get('/health').status_code == 200
def test_engineering_can_retrieve_own_policy():
    r=client.post('/api/v1/chat',json={'query':'代码合并前需要什么批准？','user_role':'engineering','top_k':5})
    assert r.status_code==200 and r.json()['evidences'][0]['doc_id']=='ENG-001'
def test_guest_is_fail_closed():
    r=client.post('/api/v1/chat',json={'query':'差旅费用多久内提交报销？','user_role':'guest','top_k':5})
    assert r.status_code==200 and r.json()['evidences']==[] and '未找到' in r.json()['answer']
def test_admin_is_read_only(): assert client.post('/api/admin/documents').status_code == 403
