# -*- coding: utf-8 -*-
# api_server_total.py - 总入口：意图识别 + 知识库问答(RAG) + 签约流程 统一接入
import asyncio
import json
import os
import sys
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

# 保证 EicAI_hfs 在 path 中
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# 意图识别
from intent import recognize as intent_recognize
from intent.agent import MODULE_KNOWLEDGE_QA, MODULE_SIGNATORY, MODULE_DAILY_QA
from intent.router import recognize_and_route

# RAG：按需在 startup 中初始化
rag_app = None
milvus_vectorstore = None
RAG_SUMMARY_MAX_LEN = 120

# 签约流程：复用 business_agent
try:
    from business_agent import _sign_contract_agent_impl, SignContractState
    from langchain.tools import ToolRuntime

    class MockToolRuntime(ToolRuntime):
        def __init__(self, state: dict):
            self.state = state

    _BUSINESS_AGENT_AVAILABLE = True
except ImportError:
    _BUSINESS_AGENT_AVAILABLE = False
    MockToolRuntime = None  # type: ignore

app = FastAPI(title="企业智能总入口：意图识别 + 知识库问答 + 签约")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sign_contract_sessions: Dict[str, dict] = {}


# --- 请求/响应模型 ---
class IntentRequest(BaseModel):
    message: str


class IntentResponse(BaseModel):
    module: str
    business_type: int
    business_name: str
    reason_detail: str
    reasone_detail: str
    success: bool = True
    error: Optional[str] = None


class MessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    # 日常问答等可调大模型（留空则用服务端 rag/config.json 的 llm）
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_temperature: Optional[float] = None
    show_thinking: bool = False  # 为 True 时返回原始技术轨迹（详细）
    show_chain_summary: bool = True  # 为 True 时用大模型总结「Agent 思维链」给用户看


class MessageResponse(BaseModel):
    intent: Dict[str, Any]
    route: str
    answer: Optional[str] = None
    hint: Optional[str] = None
    session_id: Optional[str] = None
    signatory_response: Optional[str] = None
    current_step: Optional[int] = None
    is_complete: Optional[bool] = None
    thinking: Optional[str] = None  # 原始技术轨迹（show_thinking 时）
    chain_summary: Optional[str] = None  # 大模型总结的整体 Agent 链路与思考过程（面向用户）


def _module_to_business_type(module: str) -> int:
    if module == MODULE_SIGNATORY:
        return 2
    if module == MODULE_DAILY_QA:
        return 3
    return 1


# 身份类问题固定回复
_IDENTITY_ANSWER = "我是启德教育智能系统小助手，可以帮你解答一些问题，但是你不能帮我做任何事情，比如签约、合同、自动签约等。"


def _effective_llm_cfg(request: MessageRequest) -> Dict[str, Any]:
    """合并请求中的 LLM 参数与 rag 配置，空字符串视为未填。"""
    from rag.config_loader import get_llm
    base = get_llm()
    m = (request.llm_model or "").strip()
    k = (request.llm_api_key or "").strip()
    u = (request.llm_base_url or "").strip()
    t = request.llm_temperature
    out = dict(base)
    if m:
        out["model"] = m
    if k:
        out["api_key"] = k
    if u:
        out["base_url"] = u
    if t is not None:
        out["temperature"] = float(t)
    return out


def _extract_thinking_from_ai_message(msg: Any) -> Optional[str]:
    """从 LangChain AIMessage 等对象中提取推理/思考文本（兼容 DashScope 等）。"""
    if msg is None:
        return None
    add = getattr(msg, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning", "thinking"):
        v = add.get(key)
        if v:
            return str(v).strip() or None
    meta = getattr(msg, "response_metadata", None) or {}
    for key in ("reasoning_content", "reasoning", "thinking"):
        v = meta.get(key)
        if v:
            return str(v).strip() or None
    return None


def _build_rag_thinking_trace(messages: Any) -> Optional[str]:
    """将 RAG LangGraph 消息流整理为可读过程。"""
    if not messages:
        return None
    parts: List[str] = []
    for i, m in enumerate(messages):
        name = type(m).__name__
        block = f"[{i + 1}] {name}"
        if getattr(m, "tool_calls", None):
            try:
                tc = m.tool_calls
                names = [t.get("name", "?") if isinstance(t, dict) else getattr(t, "name", "?") for t in (tc or [])]
                block += "\n工具调用: " + ", ".join(names) if names else str(tc)
            except Exception:
                block += "\n工具调用: " + str(m.tool_calls)
        content = getattr(m, "content", None)
        if content is not None and content != "":
            text = content if isinstance(content, str) else str(content)
            if len(text) > 1200:
                text = text[:1200] + "…"
            block += "\n" + text
        parts.append(block)
    return "\n\n".join(parts) if parts else None


CHAIN_SUMMARY_SYSTEM = """你是企业智能系统的「流程解说员」。下面给出一次用户提问在系统内部的执行轨迹（偏技术）。
请用通俗中文整理成给用户看的「Agent 思维链」，要求：
1. 使用 ①②③ 或 1. 2. 3. 标出清晰步骤（3～6 步为宜）。
2. 顺序一般为：理解用户问题 → 意图/路由判定 → 进入哪个业务模块 → 系统执行了哪些动作 → 如何得到最终回复。
3. 不要编造轨迹中未出现的内容；专业术语用一两句口语带过。
4. 总长度不超过 500 字，不要使用 Markdown 代码块。"""


def _llm_summarize_agent_chain(
    request: MessageRequest,
    route: str,
    intent: Dict[str, Any],
    raw_trace: str,
    answer_preview: str = "",
) -> Optional[str]:
    """用大模型把内部轨迹总结成用户可读的 Agent 思维链。"""
    if not getattr(request, "show_chain_summary", True):
        return None
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    try:
        llm_cfg = _effective_llm_cfg(request)
        if not llm_cfg.get("api_key"):
            return _fallback_chain_text(route, intent, raw_trace)
        llm = ChatOpenAI(
            model=llm_cfg.get("model", "qwen-turbo"),
            temperature=0.2,
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        user_block = (
            f"【路由】{route}\n"
            f"【意图识别】module={intent.get('module', '')} business_name={intent.get('business_name', '')}\n"
            f"【理由摘要】{intent.get('reason_detail', '')[:600]}\n"
            f"【内部轨迹】\n{raw_trace[:3500]}\n"
        )
        if answer_preview:
            user_block += f"【最终回复摘要】{answer_preview[:400]}\n"
        msg = llm.invoke([
            SystemMessage(content=CHAIN_SUMMARY_SYSTEM),
            HumanMessage(content=user_block),
        ])
        text = (getattr(msg, "content", None) or str(msg)).strip()
        return text or _fallback_chain_text(route, intent, raw_trace)
    except Exception:
        return _fallback_chain_text(route, intent, raw_trace)


def _fallback_chain_text(route: str, intent: Dict[str, Any], raw_trace: str) -> str:
    """LLM 不可用时返回简短分步说明。"""
    lines = [
        f"① 意图识别：将问题归类为「{route}」",
        f"② 依据：{(intent.get('reason_detail') or '')[:200] or '规则/模型判定'}",
    ]
    if raw_trace.strip():
        lines.append("③ 执行过程摘要：" + raw_trace.strip()[:400].replace("\n", " "))
    return "\n".join(lines)


def _ndjson_line(obj: Dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode("utf-8")


def _stream_chunk_text(chunk: Any) -> str:
    """从 LangChain 流式 chunk 中取出可展示的文本增量。"""
    c = getattr(chunk, "content", None)
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: List[str] = []
        for p in c:
            if isinstance(p, dict):
                if p.get("type") == "text" and p.get("text"):
                    parts.append(str(p["text"]))
                elif "text" in p:
                    parts.append(str(p.get("text") or ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(c)


async def _fake_stream_text(text: str, step: int = 16, delay: float = 0.012) -> AsyncIterator[str]:
    """非流式模型路径下按块推送，形成打字机效果。"""
    t = text or ""
    if not t:
        return
    for i in range(0, len(t), step):
        await asyncio.sleep(delay)
        yield t[i : i + step]


def _is_identity_question(msg: str) -> bool:
    q = (msg or "").strip().replace(" ", "")
    if not q:
        return False
    return (
        q in ("你是谁", "你叫什么", "你是啥", "你是谁?", "你叫什么?", "你是啥?")
        or q.startswith("你是谁")
        or q.startswith("你叫什么")
    )


@app.post("/api/intent", response_model=IntentResponse)
async def recognize_intent(request: IntentRequest):
    print("[INFO] 意图识别")
    result = intent_recognize(request.message)
    module = result.get("module") or MODULE_KNOWLEDGE_QA
    reason = result.get("reason_detail") or ""
    return IntentResponse(
        module=module,
        business_type=_module_to_business_type(module),
        business_name=result.get("business_name") or "",
        reason_detail=reason,
        reasone_detail=reason,
        success=result.get("success", False),
        error=result.get("error"),
    )


@app.post("/api/message", response_model=MessageResponse)
async def handle_message(request: MessageRequest):
    print("[INFO] 统一消息入口")
    if _is_identity_question(request.message):
        route_result = recognize_and_route(request.message)
        intent_d = route_result.get("intent") or {}
        raw_trace = (
            "【处理】命中身份类问题规则，直接返回固定自我介绍，未调用主对话大模型。\n"
            f"意图识别: module={intent_d.get('module', '')}, reason={intent_d.get('reason_detail', '')[:300]}"
        )
        thinking = raw_trace if request.show_thinking else None
        chain_summary = _llm_summarize_agent_chain(
            request, MODULE_DAILY_QA, intent_d, raw_trace, _IDENTITY_ANSWER[:200]
        )
        return MessageResponse(
            intent=intent_d,
            route=MODULE_DAILY_QA,
            answer=_IDENTITY_ANSWER,
            hint=None,
            thinking=thinking,
            chain_summary=chain_summary,
        )
    route_result = recognize_and_route(request.message)
    intent = route_result["intent"]
    route = route_result["route"]
    hint = route_result.get("hint")

    if route == MODULE_KNOWLEDGE_QA:
        if rag_app is None:
            raise HTTPException(
                status_code=503,
                detail="知识库问答服务未初始化，请检查 RAG 配置与 Milvus 连接",
            )
        try:
            from langchain_core.messages import HumanMessage
            result = rag_app.invoke(
                {"messages": [HumanMessage(content=request.message)]},
                config={"recursion_limit": 50},
            )
            msgs = result.get("messages") or []
            answer = ""
            for msg in reversed(msgs):
                if getattr(msg, "content", None) and hasattr(msg, "content"):
                    answer = msg.content
                    break
            raw_trace = _build_rag_thinking_trace(msgs) or "（RAG LangGraph 执行完成，无详细消息流）"
            thinking = raw_trace if request.show_thinking else None
            chain_summary = _llm_summarize_agent_chain(
                request,
                route,
                intent,
                raw_trace + f"\n【知识库答复摘要】{(answer or '')[:500]}",
                (answer or "")[:300],
            )
            return MessageResponse(
                intent=intent,
                route=route,
                answer=answer or "未找到答案",
                hint=None,
                thinking=thinking,
                chain_summary=chain_summary,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"知识库问答失败: {e}")

    if route == MODULE_SIGNATORY:
        if not _BUSINESS_AGENT_AVAILABLE:
            raw_trace = "签约模块未加载，无法执行五步签约流程。"
            chain_summary = _llm_summarize_agent_chain(request, route, intent, raw_trace, "")
            return MessageResponse(
                intent=intent,
                route=route,
                answer=None,
                hint="签约流程模块未加载，请使用签约专用接口 /api/sign-contract。",
                chain_summary=chain_summary,
            )
        sid = request.session_id
        if sid and sid in sign_contract_sessions:
            state = sign_contract_sessions[sid]
        else:
            sid = str(uuid.uuid4())
            state = {
                "current_step": 1,
                "step1_info": {},
                "step2_info": {},
                "step3_info": {},
                "step4_info": {},
                "step5_info": {},
                "is_complete": False,
                "error_msg": "",
            }
            sign_contract_sessions[sid] = state
        runtime = MockToolRuntime(state)
        signatory_response = _sign_contract_agent_impl(request.message, runtime)
        sign_contract_sessions[sid] = state
        raw_trace = (
            "【签约流程 Agent】\n"
            f"意图理由: {intent.get('reason_detail', '')}\n"
            f"当前步骤: {state.get('current_step')}，是否完成: {state.get('is_complete')}\n"
            f"本轮回复: {signatory_response[:800]}"
        )
        thinking = raw_trace if request.show_thinking else None
        chain_summary = _llm_summarize_agent_chain(
            request, route, intent, raw_trace, signatory_response[:300]
        )
        return MessageResponse(
            intent=intent,
            route=route,
            answer=None,
            hint=hint,
            session_id=sid,
            signatory_response=signatory_response,
            current_step=state.get("current_step"),
            is_complete=state.get("is_complete"),
            thinking=thinking,
            chain_summary=chain_summary,
        )

    if route == MODULE_DAILY_QA:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            llm_cfg = _effective_llm_cfg(request)
            if not llm_cfg.get("api_key"):
                raise ValueError("日常问答需要配置 RAG config.json 的 llm.api_key、环境变量，或在请求中填写 llm_api_key")
            model_kwargs: Dict[str, Any] = {}
            if request.show_thinking:
                model_kwargs["extra_body"] = {"enable_thinking": True}
            llm_args: Dict[str, Any] = {
                "model": llm_cfg.get("model", "qwen-turbo"),
                "temperature": float(llm_cfg.get("temperature", 0.1)),
                "api_key": llm_cfg["api_key"],
                "base_url": llm_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            }
            if model_kwargs:
                llm_args["model_kwargs"] = model_kwargs
            llm = ChatOpenAI(**llm_args)
            daily_qa_system = (
                "你是启德教育智能系统小助手。当用户问你是谁时，你应回答：我是启德教育智能系统小助手，可以帮你解答一些问题，但是你不能帮我做任何事情，比如签约、合同、自动签约等。平时你以该身份与用户进行日常对话。"
            )
            msg = llm.invoke([
                SystemMessage(content=daily_qa_system),
                HumanMessage(content=request.message),
            ])
            answer = getattr(msg, "content", None) or str(msg)
            thinking_parts: List[str] = []
            rc = _extract_thinking_from_ai_message(msg)
            raw_trace = (
                "【路由】日常问答（启德小助手人设）\n"
                f"意图识别理由: {(intent.get('reason_detail') or '')[:500]}\n"
            )
            if rc:
                raw_trace += f"【模型侧推理片段】\n{rc[:1500]}\n"
            raw_trace += f"【生成回复摘要】{answer[:400]}"
            if request.show_thinking:
                if rc:
                    thinking_parts.append("【模型推理/思考】\n" + rc)
                thinking_parts.append(
                    "【路由】日常问答\n"
                    f"意图识别理由摘要: {(intent.get('reason_detail') or '')[:400]}"
                )
            thinking = "\n\n".join(thinking_parts) if thinking_parts else None
            chain_summary = _llm_summarize_agent_chain(request, route, intent, raw_trace, answer[:300])
            return MessageResponse(
                intent=intent,
                route=route,
                answer=answer or "",
                hint=None,
                thinking=thinking,
                chain_summary=chain_summary,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"日常问答失败: {e}")

    raw_trace = f"兜底路由。intent={intent} hint={hint}"
    chain_summary = _llm_summarize_agent_chain(request, route, intent, raw_trace, "")
    return MessageResponse(
        intent=intent,
        route=route,
        answer=None,
        hint=hint,
        chain_summary=chain_summary,
    )


async def _message_stream_body(request: MessageRequest) -> AsyncIterator[bytes]:
    """NDJSON 流：每行一个 JSON。type 为 meta | delta | done | error。"""
    # print("[INFO] NDJSON 流")
    try:
        if _is_identity_question(request.message):
            route_result = recognize_and_route(request.message)
            intent_d = route_result.get("intent") or {}
            raw_trace = (
                "【处理】命中身份类问题规则，直接返回固定自我介绍，未调用主对话大模型。\n"
                f"意图识别: module={intent_d.get('module', '')}, reason={intent_d.get('reason_detail', '')[:300]}"
            )
            yield _ndjson_line({"type": "meta", "route": MODULE_DAILY_QA, "session_id": None, "intent": intent_d})
            async for piece in _fake_stream_text(_IDENTITY_ANSWER):
                if piece:
                    yield _ndjson_line({"type": "delta", "text": piece})
            thinking = raw_trace if request.show_thinking else None
            chain_summary = await asyncio.to_thread(
                _llm_summarize_agent_chain,
                request,
                MODULE_DAILY_QA,
                intent_d,
                raw_trace,
                _IDENTITY_ANSWER[:200],
            )
            yield _ndjson_line(
                {
                    "type": "done",
                    "route": MODULE_DAILY_QA,
                    "answer": _IDENTITY_ANSWER,
                    "thinking": thinking,
                    "chain_summary": chain_summary,
                    "session_id": None,
                }
            )
            return

        route_result = recognize_and_route(request.message)
        intent = route_result["intent"]
        route = route_result["route"]
        hint = route_result.get("hint")

        if route == MODULE_KNOWLEDGE_QA:
            if rag_app is None:
                yield _ndjson_line({"type": "error", "detail": "知识库问答服务未初始化，请检查 RAG 配置与 Milvus 连接"})
                return
            try:
                from langchain_core.messages import HumanMessage

                yield _ndjson_line({"type": "meta", "route": route, "session_id": None, "intent": intent})
                result = await asyncio.to_thread(
                    lambda: rag_app.invoke(
                        {"messages": [HumanMessage(content=request.message)]},
                        config={"recursion_limit": 50},
                    )
                )
                msgs = result.get("messages") or []
                answer = ""
                for msg in reversed(msgs):
                    if getattr(msg, "content", None) and hasattr(msg, "content"):
                        answer = msg.content
                        break
                out_text = answer or "未找到答案"
                async for piece in _fake_stream_text(out_text if isinstance(out_text, str) else str(out_text)):
                    if piece:
                        yield _ndjson_line({"type": "delta", "text": piece})
                raw_trace = _build_rag_thinking_trace(msgs) or "（RAG LangGraph 执行完成，无详细消息流）"
                thinking = raw_trace if request.show_thinking else None
                chain_summary = await asyncio.to_thread(
                    _llm_summarize_agent_chain,
                    request,
                    route,
                    intent,
                    raw_trace + f"\n【知识库答复摘要】{(answer or '')[:500]}",
                    (answer or "")[:300],
                )
                yield _ndjson_line(
                    {
                        "type": "done",
                        "route": route,
                        "answer": out_text,
                        "thinking": thinking,
                        "chain_summary": chain_summary,
                        "session_id": None,
                    }
                )
            except Exception as e:
                yield _ndjson_line({"type": "error", "detail": f"知识库问答失败: {e}"})
            return

        if route == MODULE_SIGNATORY:
            if not _BUSINESS_AGENT_AVAILABLE:
                raw_trace = "签约模块未加载，无法执行五步签约流程。"
                hint_txt = "签约流程模块未加载，请使用签约专用接口 /api/sign-contract。"
                yield _ndjson_line({"type": "meta", "route": route, "session_id": None, "intent": intent})
                async for piece in _fake_stream_text(hint_txt):
                    if piece:
                        yield _ndjson_line({"type": "delta", "text": piece})
                chain_summary = await asyncio.to_thread(
                    _llm_summarize_agent_chain, request, route, intent, raw_trace, ""
                )
                yield _ndjson_line(
                    {
                        "type": "done",
                        "route": route,
                        "answer": hint_txt,
                        "thinking": None,
                        "chain_summary": chain_summary,
                        "session_id": None,
                    }
                )
                return
            sid = request.session_id
            if sid and sid in sign_contract_sessions:
                state = sign_contract_sessions[sid]
            else:
                sid = str(uuid.uuid4())
                state = {
                    "current_step": 1,
                    "step1_info": {},
                    "step2_info": {},
                    "step3_info": {},
                    "step4_info": {},
                    "step5_info": {},
                    "is_complete": False,
                    "error_msg": "",
                }
                sign_contract_sessions[sid] = state
            runtime = MockToolRuntime(state)
            signatory_response = _sign_contract_agent_impl(request.message, runtime)
            sign_contract_sessions[sid] = state
            raw_trace = (
                "【签约流程 Agent】\n"
                f"意图理由: {intent.get('reason_detail', '')}\n"
                f"当前步骤: {state.get('current_step')}，是否完成: {state.get('is_complete')}\n"
                f"本轮回复: {signatory_response[:800]}"
            )
            thinking = raw_trace if request.show_thinking else None
            yield _ndjson_line({"type": "meta", "route": route, "session_id": sid, "intent": intent})
            sr = signatory_response or ""
            async for piece in _fake_stream_text(sr):
                if piece:
                    yield _ndjson_line({"type": "delta", "text": piece})
            chain_summary = await asyncio.to_thread(
                _llm_summarize_agent_chain,
                request,
                route,
                intent,
                raw_trace,
                (signatory_response or "")[:300],
            )
            yield _ndjson_line(
                {
                    "type": "done",
                    "route": route,
                    "answer": sr,
                    "thinking": thinking,
                    "chain_summary": chain_summary,
                    "session_id": sid,
                }
            )
            return

        if route == MODULE_DAILY_QA:
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage, SystemMessage

                llm_cfg = _effective_llm_cfg(request)
                if not llm_cfg.get("api_key"):
                    yield _ndjson_line(
                        {"type": "error", "detail": "日常问答需要配置 llm.api_key 或在请求中填写 llm_api_key"}
                    )
                    return
                model_kwargs: Dict[str, Any] = {}
                if request.show_thinking:
                    model_kwargs["extra_body"] = {"enable_thinking": True}
                llm_args: Dict[str, Any] = {
                    "model": llm_cfg.get("model", "qwen-turbo"),
                    "temperature": float(llm_cfg.get("temperature", 0.1)),
                    "api_key": llm_cfg["api_key"],
                    "base_url": llm_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                }
                if model_kwargs:
                    llm_args["model_kwargs"] = model_kwargs
                llm = ChatOpenAI(**llm_args)
                daily_qa_system = (
                    "你是启德教育智能系统小助手。当用户问你是谁时，你应回答：我是启德教育智能系统小助手，可以帮你解答一些问题，但是你不能帮我做任何事情，比如签约、合同、自动签约等。平时你以该身份与用户进行日常对话。"
                )
                messages = [
                    SystemMessage(content=daily_qa_system),
                    HumanMessage(content=request.message),
                ]
                yield _ndjson_line({"type": "meta", "route": route, "session_id": None, "intent": intent})
                answer_parts: List[str] = []
                rc_last: Optional[str] = None
                async for chunk in llm.astream(messages):
                    piece = _stream_chunk_text(chunk)
                    if piece:
                        answer_parts.append(piece)
                        yield _ndjson_line({"type": "delta", "text": piece})
                    rc = _extract_thinking_from_ai_message(chunk)
                    if rc:
                        rc_last = rc
                answer = "".join(answer_parts) or ""
                thinking_parts: List[str] = []
                raw_trace = (
                    "【路由】日常问答（启德小助手人设）\n"
                    f"意图识别理由: {(intent.get('reason_detail') or '')[:500]}\n"
                )
                if rc_last:
                    raw_trace += f"【模型侧推理片段】\n{rc_last[:1500]}\n"
                raw_trace += f"【生成回复摘要】{answer[:400]}"
                if request.show_thinking:
                    if rc_last:
                        thinking_parts.append("【模型推理/思考】\n" + rc_last)
                    thinking_parts.append(
                        "【路由】日常问答\n"
                        f"意图识别理由摘要: {(intent.get('reason_detail') or '')[:400]}"
                    )
                thinking = "\n\n".join(thinking_parts) if thinking_parts else None
                chain_summary = await asyncio.to_thread(
                    _llm_summarize_agent_chain, request, route, intent, raw_trace, answer[:300]
                )
                yield _ndjson_line(
                    {
                        "type": "done",
                        "route": route,
                        "answer": answer,
                        "thinking": thinking,
                        "chain_summary": chain_summary,
                        "session_id": None,
                    }
                )
            except Exception as e:
                yield _ndjson_line({"type": "error", "detail": f"日常问答失败: {e}"})
            return

        raw_trace = f"兜底路由。intent={intent} hint={hint}"
        reply = (hint or "").strip() or "暂无回复"
        yield _ndjson_line({"type": "meta", "route": route, "session_id": None, "intent": intent})
        async for piece in _fake_stream_text(reply):
            if piece:
                yield _ndjson_line({"type": "delta", "text": piece})
        chain_summary = await asyncio.to_thread(_llm_summarize_agent_chain, request, route, intent, raw_trace, "")
        yield _ndjson_line(
            {
                "type": "done",
                "route": route,
                "answer": reply,
                "thinking": None,
                "chain_summary": chain_summary,
                "session_id": None,
            }
        )
    except Exception as e:
        yield _ndjson_line({"type": "error", "detail": str(e)})


@app.post("/api/message/stream")
async def handle_message_stream(request: MessageRequest):
    print("[INFO] 消息流")
    return StreamingResponse(
        _message_stream_body(request),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- 签约专用接口 ---
class SignContractRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None


class SignContractResponse(BaseModel):
    response: str
    session_id: str
    current_step: int
    is_complete: bool
    error_msg: str
    step_info: Dict[str, Any] = {}


@app.post("/api/sign-contract", response_model=SignContractResponse)
async def sign_contract(request: SignContractRequest):
    print("[INFO] 签约专用接口")
    if not _BUSINESS_AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="签约流程模块未加载")
    sid = request.session_id
    if sid and sid in sign_contract_sessions:
        state = sign_contract_sessions[sid]
    else:
        sid = str(uuid.uuid4())
        state = {
            "current_step": 1,
            "step1_info": {},
            "step2_info": {},
            "step3_info": {},
            "step4_info": {},
            "step5_info": {},
            "is_complete": False,
            "error_msg": "",
        }
        sign_contract_sessions[sid] = state
    runtime = MockToolRuntime(state)
    signatory_response = _sign_contract_agent_impl(request.user_input, runtime)
    sign_contract_sessions[sid] = state
    return SignContractResponse(
        response=signatory_response,
        session_id=sid,
        current_step=state.get("current_step", 1),
        is_complete=state.get("is_complete", False),
        error_msg=state.get("error_msg", ""),
        step_info={},
    )


@app.get("/api/llm-defaults")
async def llm_defaults():
    """返回前端占位用的默认模型与地址（不含 api_key）。"""
    print("[INFO] 返回前端占位用的默认模型与地址（不含 api_key）")
    try:
        from rag.config_loader import get_llm
        cfg = get_llm()
        return {
            "model": cfg.get("model", "qwen-turbo"),
            "base_url": cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "temperature": float(cfg.get("temperature", 0.1)),
            "has_server_api_key": bool(cfg.get("api_key")),
        }
    except Exception:
        return {
            "model": "qwen-turbo",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "temperature": 0.1,
            "has_server_api_key": False,
        }


@app.get("/api/health")
async def health():
    print("[INFO] 健康检查")
    return {
        "status": "ok",
        "rag_initialized": rag_app is not None,
        "signatory_available": _BUSINESS_AGENT_AVAILABLE,
    }


# --- 静态页面与 RAG 初始化 ---
@app.on_event("startup")
async def startup():
    print("[INFO] 启动时初始化 RAG")
    global rag_app, milvus_vectorstore
    from rag.config_loader import get_milvus
    from rag.faq_ingest import get_milvus_vectorstore
    from rag.rag_langgraph import create_langgraph_rag

    mcfg = get_milvus()
    print(
        f"[INFO] Milvus: {mcfg['host']}:{mcfg['port']} "
        f"collection={mcfg.get('collection_name', '')}"
    )
    # Milvus 重启后 Proxy 可能数十秒～数分钟才就绪；单次连接默认约 10s 会报 not ready
    max_retries = int(os.environ.get("RAG_MILVUS_STARTUP_RETRIES", "24"))
    delay_sec = float(os.environ.get("RAG_MILVUS_STARTUP_DELAY_SEC", "5"))
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            rag_app = create_langgraph_rag(skip_ingest=True)
            milvus_vectorstore = get_milvus_vectorstore()
            print("[INFO] RAG 已初始化")
            return
        except Exception as e:
            last_err = e
            rag_app = None
            milvus_vectorstore = None
            print(f"[WARN] RAG 初始化失败 ({attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"[INFO] {delay_sec:g}s 后重试（等待 Milvus Proxy）…")
                await asyncio.sleep(delay_sec)
    print(f"[WARN] RAG 多次重试仍失败，最后错误: {last_err}")


def _read_html(path: str) -> str:
    p = os.path.join(_here, path)
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return ""


@app.get("/", response_class=HTMLResponse)
async def index():
    print("[INFO] 总入口")
    return _read_html("total_agent.html") or "<h1>总入口</h1><p>total_agent.html 未找到</p>"


@app.get("/signatory", response_class=HTMLResponse)
async def signatory_page():
    print("[INFO] 签约页面")
    return _read_html("sign_contract_business.html") or _read_html("sign_contract.html") or "<h1>签约页面未找到</h1>"


@app.get("/rag", response_class=HTMLResponse)
async def rag_page():
    print("[INFO] 知识库页面")
    return _read_html("rag/rag_test.html") or "<h1>知识库页面未找到</h1>"


if __name__ == "__main__":
    import socket
    import uvicorn

    if "API_SERVER_PORT" in os.environ:
        port = int(os.environ["API_SERVER_PORT"])
    else:
        port = 9001
        for alt in (9001, 9002, 9020):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", alt))
                    port = alt
                    break
                except OSError:
                    pass
        if port != 9001:
            print(
                f"[INFO] 默认端口 9001 已被占用（常见于 milvus-minio 映射），"
                f"总线 API 改用 {port}。如需固定端口请设置环境变量 API_SERVER_PORT。"
            )
    uvicorn.run(app, host="0.0.0.0", port=port)
