# Hybrid RAG MVP · Public Interview Demo

可公开运行的企业知识库 RAG 作品。仓库只包含合成制度数据，不包含真实企业文档、客户信息、密钥、模型缓存或私有评测集。

## 演示能力

- **网页问答**：FastAPI 页面提供员工问答、X-Ray 证据透视与只读治理页。
- **ACL-first 检索**：角色在候选生成前过滤；BM25、可选 dense 通道、RRF 与轻量重排均保留可解释轨迹。
- **可信生成**：DeepSeek 或其他 OpenAI-compatible 服务生成带 `[n]` 证据引用的回答；无 Key 时降级为明确标注的检索演示。
- **安全 Harness**：独立的 JWT、受限证据包、提示注入隔离、引用校验、Trace 脱敏、并发上限、超时和熔断模块，可接入生产 API 边界。
- **质量门禁**：保留公开合成数据上的 Hit@K/MRR、受保护问题和回归比较工具；私有语料评测不在本仓库中。

## 30 秒本地演示

```bash
git clone https://github.com/karma-taylor/hybrid-rag-mvp.git
cd hybrid-rag-mvp
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
./start_demo.sh
```

首次执行会创建 `.env`。不设置 Key 也可演示 ACL、检索、X-Ray 和治理页面；需要真实生成时仅在本地 `.env` 填入：

```dotenv
OPENAI_API_KEY=你的DeepSeek密钥
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

浏览器打开 `http://127.0.0.1:8000/`。共享屏幕时不要显示 `.env` 或终端中的密钥。

需要本地运行 BGE-M3 密集召回与重排实验时，再安装 `requirements-core.txt`；面试 Demo 默认使用稳定、无需模型下载的词法通道。

## 项目结构

```text
app/                 面试演示 Web/API、X-Ray 与治理页
data/                公开合成制度与评测数据
hybrid_retrieval.py  可复现实验用 ACL-first 混合检索核心
harness.py           可接生产边界的安全编排与 Trace 控制层
security.py          提示注入检测规则
chunking_mac.py      文档切分与隔离清单生成
tests / test_*.py    演示、检索、Harness 与安全回归测试
```

## 安全边界

`app/` 是公开演示界面，角色切换用于可视化 ACL，不是企业身份认证实现。生产接入应使用认证网关与 JWT，再通过 `enterprise_api.py` + `harness.py` 提供受控 API。

文档入库时，`chunking_mac.py` 会隔离匹配高置信提示注入规则的 chunk；隔离记录只保存文档路径、chunk ID 和原因。Harness 的运行 Trace 只记录 request ID、盐化用户摘要、阶段耗时、容量和错误码，绝不记录 query、证据正文、prompt、JWT 或原始用户 ID。

## 验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/python eval_pipeline.py --mode test
```

测试涵盖 ACL、证据引用、提示注入拒绝、JWT fail-closed、并发/超时保护和文档隔离。公开评测数据用于演示回归机制，不代表真实企业场景的效果承诺。
