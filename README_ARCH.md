# EicAI_hfs 代码结构说明

## 整体逻辑（三部分）

1. **意图识别**：根据用户输入识别意图，映射到不同业务模块（知识库问答、签约等）。
2. **知识库问答**：RAG 模块，在 `rag/` 下，基于 Milvus + LangGraph 的 FAQ 问答。
3. **签约流程**：在 `signatory/` 下（自动签约 Token + 权限校验）以及 `business_agent.py` + `api_server_business.py`（五步签约流程 Mock）。

意图识别已收纳到 **`intent/`** 文件夹，并通过 **总入口 `api_server_total.py`** 将识别结果接入知识库问答与签约流程。

---

## 目录与职责

| 目录/文件 | 职责 |
|-----------|------|
| **intent/** | 意图识别模块：识别用户意图并映射到 `knowledge_qa` / `signatory` |
| **rag/** | 知识库问答：RAG 服务、LangGraph 图、Milvus 检索、配置 |
| **signatory/** | 自动签约前两步：获取 Token、校验用户权限 |
| **api_server_total.py** | 总入口：意图识别 + 分发到 RAG / 签约，统一 API |
| **intent/agent_total.py** | 旧版意图识别单文件参考实现（可保留作参考） |
| **business_agent.py** | 签约五步流程 Mock（步骤 1–5） |
| **api_server_business.py** | 签约流程专用 API（可单独起端口） |

---

## 意图识别（intent/）

- **intent/agent.py**：意图识别 Agent（LLM + 结构化输出），输出 `module`（`knowledge_qa` | `signatory`）、`business_name`、`reason_detail`。
- **intent/router.py**：`recognize_and_route(message)`：先识别意图，再返回路由结果（不执行下游）。
- **intent/__init__.py**：对外暴露 `recognize`、`recognize_and_route`、`MODULE_KNOWLEDGE_QA`、`MODULE_SIGNATORY`。

---

## 知识库问答（rag/）

- **rag/rag_server.py**：RAG 独立服务，提供 `/api/query` 等。
- **rag/rag_langgraph.py**：基于 LangGraph 的 RAG 图（检索 + LLM 总结）。
- **rag/config_loader.py**：RAG 配置（Milvus、LLM、FAQ 路径等）。
- **rag/faq_ingest.py**：FAQ 入库、Milvus 检索器。

总入口在启动时会初始化 RAG（`create_langgraph_rag(skip_ingest=True)`），知识库问答请求在总入口内直接调用 RAG 图，无需再请求 rag_server。

---

## 签约流程（signatory/ + business_agent）

- **signatory/run.py**：命令行入口，执行自动签约前两步（Token + 权限），输出姓名、职位。
- **signatory/agent.py**：`get_token`、`validate_user_permission`、`run_auto_sign_steps`。
- **business_agent.py**：五步签约流程 Mock（签约对象、合同编号、签约类型、签约日期、授权）。
- **api_server_business.py**：签约专用 API（`/api/sign-contract`），可单独运行在 8002 端口。

总入口集成了签约流程：当意图为 `signatory` 时，通过 `business_agent._sign_contract_agent_impl` 处理，并维护 `sign_contract_sessions`。

---

## 总入口 API（api_server_total.py）

- **POST /api/intent**：仅做意图识别，返回 `module`、`business_type`、`business_name`、`reason_detail` / `reasone_detail`（兼容前端）。
- **POST /api/message**：统一消息入口：先意图识别，再分发：
  - `knowledge_qa` → 调用 RAG，返回 `answer`；
  - `signatory` → 走签约流程，返回 `session_id`、`signatory_response`、`current_step`、`is_complete` 等。
- **POST /api/sign-contract**：签约流程专用接口（与 api_server_business 行为一致），支持按 `session_id` 续话。
- **GET /api/health**：健康检查，含 RAG / 签约是否可用。

启动方式（在 EicAI_hfs 目录下）：

```bash
python api_server_total.py
# 或
uvicorn api_server_total:app --host 0.0.0.0 --port 8001
```

前端 **total_agent.html** 使用 `/api/intent` 做意图识别展示；若需“一发即执行”，可改为调用 **POST /api/message**，并根据 `route` 与 `answer` / `signatory_response` 展示结果。
