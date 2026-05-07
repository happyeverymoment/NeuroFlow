# -*- coding: utf-8 -*-
"""
意图识别 Agent：根据用户输入识别意图并映射到具体模块（知识库问答、签约、日常问答等）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel

try:
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy
    from langchain_openai import ChatOpenAI
except ImportError:
    create_agent = None
    ToolStrategy = None
    ChatOpenAI = None


MODULE_KNOWLEDGE_QA = "knowledge_qa"
MODULE_SIGNATORY = "signatory"
MODULE_DAILY_QA = "daily_qa"

SYSTEM_PROMPT = '''
你是企业智能的意图识别专家，需要准确识别用户意图，并判断应调用哪个业务模块。

可选模块（必须且仅能输出以下之一）：
1. daily_qa - 日常问答
   - 适用：打招呼、闲聊、简单问答、与业务无关的日常对话；用户问助手身份（你是谁、你叫什么、你是啥等）
   - 标识词："你好"、"在吗"、"谢谢"、"再见"、"你是谁"、"你叫什么"、"你是啥"
   - 当用户问「你是谁」「你叫什么」「你是啥」等关于助手身份的问题时，必须选 daily_qa。
2. knowledge_qa - 知识库问答
   - 适用：询问信息、位置、规则、操作步骤、怎么办、如何做、问题咨询等
   - 标识词："在哪"、"是什么"、"是多少"、"查询"、"查看"、"了解"、"告诉我"、"怎么办"、"如何"、"出错了"、"遇到问题"
   - 典型句式："XX在哪"、"XX是什么"、"查询XX"、"如何XX"、"XX怎么办"

3. signatory - 签约流程
   - 适用：与签约、合同、自动签约相关的业务办理
   - 标识词："签约"、"合同"、"发起签约"、"办理签约"、"给XX签约"、"自动签约"
   - 典型句式："帮我对李同学进行签约"、"我要发起签约"、"办理合同签约"

判断流程（按优先级，不得颠倒）：
1. **若用户输入含「签约」「合同」「办理签约」「发起签约」「自动签约」等与签约业务相关的词（包括「如何签约」「怎么签约」「签约流程」）→ 必须选 signatory，不得选 daily_qa 或 knowledge_qa**
2. 若为打招呼、闲聊、与业务无关的日常对话或问助手身份 → 选 daily_qa
3. 若用户是问信息、问步骤、查规则、如何办理（但与签约无关）→ 选 knowledge_qa
4. 若无法明确归类，偏向 knowledge_qa（业务咨询优先走知识库），仅纯闲聊选 daily_qa

约束：
- module 只能是 daily_qa、knowledge_qa 或 signatory 之一
- 必须基于用户实际输入判断，reason_detail 中引用用户原话中的关键词
'''

class IntentOutput(BaseModel):
    """意图识别输出"""
    module: str
    business_name: str
    reason_detail: str


def _get_llm():
    import os
    api_key = os.environ.get("INTENT_LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "sk-55d06e419aee4988944256ed77feef5e"
    return ChatOpenAI(
        model="qwen3-max",
        temperature=0,
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _create_intent_agent():
    if create_agent is None or ToolStrategy is None:
        raise RuntimeError("需要安装 langchain 及 langchain-openai，且使用支持 create_agent 的版本")
    model = _get_llm()
    return create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(IntentOutput),
    )


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = _create_intent_agent()
    return _agent


def _rule_based_module(message: str) -> Optional[str]:
    """
    规则兜底：签约/合同相关必须走 signatory，避免 LLM 误判为日常问答。
    """
    text = (message or "").strip()
    if not text:
        return None
    sign_keywords = (
        "签约", "合同", "办理签约", "发起签约", "自动签约", "电子签约",
        "合同编号", "QY", "办签约",
    )
    for kw in sign_keywords:
        if kw in text:
            return MODULE_SIGNATORY
    return None


def recognize(message: str) -> Dict[str, Any]:
    """
    识别用户消息的意图，返回模块及详情。

    Returns:
        dict: {
            "module": "daily_qa" | "knowledge_qa" | "signatory",
            "business_name": str,
            "reason_detail": str,
            "success": bool,
            "error": str | None
        }
    """
    result = {
        "module": MODULE_KNOWLEDGE_QA,
        "business_name": "",
        "reason_detail": "",
        "success": False,
        "error": None,
    }
    if not (message or "").strip():
        result["module"] = MODULE_DAILY_QA
        result["business_name"] = "日常问答"
        result["reason_detail"] = "输入为空，默认日常问答"
        result["success"] = True
        return result

    ruled = _rule_based_module(message)
    if ruled:
        result["module"] = ruled
        result["business_name"] = "签约流程" if ruled == MODULE_SIGNATORY else ""
        result["reason_detail"] = f"规则命中：输入含签约/合同相关业务词，路由 {ruled}"
        result["success"] = True
        return result

    try:
        agent = get_agent()
        response = agent.invoke({
            "messages": [{"role": "user", "content": message}],
        })
        if "structured_response" in response:
            out = response["structured_response"]
            module = (getattr(out, "module", None) or "").strip().lower()
            if module not in (MODULE_KNOWLEDGE_QA, MODULE_SIGNATORY, MODULE_DAILY_QA):
                module = MODULE_KNOWLEDGE_QA
            result["module"] = module
            result["business_name"] = getattr(out, "business_name", "") or ""
            result["reason_detail"] = getattr(out, "reason_detail", "") or ""
            result["success"] = True
        else:
            result["error"] = "意图识别未返回结构化结果"
            fallback = _rule_based_module(message)
            if fallback:
                result["module"] = fallback
                result["reason_detail"] = "LLM 未返回结构化结果，规则兜底为签约"
            else:
                result["module"] = MODULE_KNOWLEDGE_QA
                result["reason_detail"] = "LLM 未返回结构化结果，默认知识库问答"
            result["success"] = bool(fallback)
    except Exception as e:
        result["error"] = str(e)
        fallback = _rule_based_module(message)
        if fallback:
            result["module"] = fallback
            result["reason_detail"] = f"意图识别异常，规则兜底：{e}"
            result["success"] = True
        else:
            result["module"] = MODULE_KNOWLEDGE_QA
            result["reason_detail"] = str(e)
    return result
