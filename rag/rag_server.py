"""
RAG 系统 Web 服务器
提供基于 LangGraph RAG 系统的 Web API 接口。
启动方式：在 EicAI_hfs 目录下执行 python -m rag.rag_server
"""

import os
import sys
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

# 保证从 EicAI_hfs 运行时能找到 rag 包
_rag_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_rag_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from rag.config_loader import get_server, get_faq, get_rag, get_cors
import rag.faq_ingest as _ingest
from rag.rag_langgraph import create_langgraph_rag
from rag.faq_ingest import get_retriever_from_existing_milvus, get_milvus_vectorstore

# 从配置设置 FAQ 路径
_faq = get_faq()
if os.path.isdir(_faq["data_dir"]):
    _ingest.FAQ_DATA_DIR = _faq["data_dir"]
    print(f"[OK] FAQ 数据目录: {_faq['data_dir']}")
elif os.path.isfile(_faq["file_path"]):
    _ingest.FAQ_FILE_PATH = _faq["file_path"]
    print(f"[OK] FAQ 文件: {_faq['file_path']}")
else:
    print(f"[WARN] 未找到 FAQ 数据: {_faq['data_dir']} / {_faq['file_path']}")

# --- App ---
app = FastAPI(title="RAG 问答系统 API")
cors_cfg = get_cors()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_cfg["allow_origins"],
    allow_credentials=cors_cfg["allow_credentials"],
    allow_methods=cors_cfg["allow_methods"],
    allow_headers=cors_cfg["allow_headers"],
)

rag_app = None
milvus_vectorstore = None
_rag_cfg = get_rag()
CANDIDATES_K = _rag_cfg["candidates_k_default"]
SUMMARY_MAX_LEN = _rag_cfg.get("summary_max_len", 120)


def _make_summary(text: str, max_len: int = None) -> str:
    """在句末截断，不超过 max_len 字。"""
    if max_len is None:
        max_len = SUMMARY_MAX_LEN
    if not (text or "").strip():
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    chunk = text[: max_len + 1]
    for sep in ("。", "！", "？", "\n"):
        i = chunk.rfind(sep)
        if i >= 0:
            return chunk[: i + 1].strip()
    return chunk[:max_len].rstrip() + "…"


# --- Pydantic 模型 ---
class QueryRequest(BaseModel):
    query: str
    stream: Optional[bool] = False


class QueryResponse(BaseModel):
    answer: str
    tool_called: bool
    retrieved_content: Optional[str] = None
    message_count: int


class CandidateItem(BaseModel):
    index: int
    question: str
    summary: str
    answer: str
    score: Optional[float] = None


class QueryCandidatesRequest(BaseModel):
    query: str
    threshold_pct: Optional[int] = 0
    top_k: Optional[int] = 5


class QueryCandidatesResponse(BaseModel):
    query: str
    candidates: List[CandidateItem]


# --- Lifespan（替代已弃用的 on_event）---
@app.on_event("startup")
async def startup():
    global rag_app, milvus_vectorstore
    print("[RAG] 初始化...")
    try:
        rag_app = create_langgraph_rag(skip_ingest=True)
        milvus_vectorstore = get_milvus_vectorstore()
        print("[OK] RAG 初始化成功")
    except Exception as e:
        print(f"[FAIL] RAG 初始化失败: {e}")
        raise


# --- 路由 ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    path = os.path.join(_rag_dir, "rag_test.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>RAG 测试页面未找到</h1>")


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if rag_app is None:
        raise HTTPException(status_code=503, detail="RAG 未初始化")
    q = (request.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="查询不能为空")
    try:
        result = rag_app.invoke(
            {"messages": [HumanMessage(content=q)]},
            config={"recursion_limit": 50},
        )
        tool_called = False
        retrieved = None
        answer = ""
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                if getattr(msg, "tool_calls", None):
                    tool_called = True
                if msg.content:
                    answer = msg.content
            elif type(msg).__name__ == "ToolMessage" and hasattr(msg, "content"):
                tool_called = True
                retrieved = str(msg.content)
        if not answer and result["messages"]:
            last = result["messages"][-1]
            answer = getattr(last, "content", str(last))
        return QueryResponse(
            answer=answer or "未找到答案",
            tool_called=tool_called,
            retrieved_content=retrieved,
            message_count=len(result["messages"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query_candidates", response_model=QueryCandidatesResponse)
async def query_candidates(request: QueryCandidatesRequest):
    if milvus_vectorstore is None:
        raise HTTPException(status_code=503, detail="检索未初始化")
    q = (request.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="查询不能为空")
    k = max(0, min(10, int(request.top_k or CANDIDATES_K)))
    k = k if k > 0 else CANDIDATES_K
    threshold = max(0, min(100, int(request.threshold_pct or 0))) / 100.0
    try:
        doc_scores = milvus_vectorstore.similarity_search_with_score(q, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    candidates = []
    for d, score in doc_scores:
        sim = 1.0 - float(score)
        if sim < threshold:
            continue
        question = d.metadata.get("question", d.page_content or "")
        answer = d.metadata.get("answer", "") or "未找到答案"
        candidates.append(
            CandidateItem(
                index=len(candidates),
                question=question,
                summary=_make_summary(answer),
                answer=answer,
                score=round(float(score), 4),
            )
        )
    return QueryCandidatesResponse(query=q, candidates=candidates)


@app.get("/api/health")
async def health():
    return {"status": "ok", "rag_initialized": rag_app is not None}


if __name__ == "__main__":
    import uvicorn
    srv = get_server()
    port = srv["port"]
    print(f"[INFO] 本机浏览器请访问: http://127.0.0.1:{port}  (不要用 0.0.0.0)")
    uvicorn.run(app, host=srv["host"], port=port, reload=srv["reload"])
