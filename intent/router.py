# -*- coding: utf-8 -*-
"""
意图路由：根据意图识别结果分发到对应模块（知识库问答 / 签约）。
"""
from __future__ import annotations

from typing import Any, Dict

from .agent import MODULE_KNOWLEDGE_QA, MODULE_SIGNATORY, MODULE_DAILY_QA, recognize as intent_recognize


def recognize_and_route(message: str) -> Dict[str, Any]:
    """
    识别用户意图并返回路由结果（不执行下游，仅做识别与建议）。

    Returns:
        dict: {
            "intent": { "module", "business_name", "reason_detail", "success", "error" },
            "route": "daily_qa" | "knowledge_qa" | "signatory",
            "hint": str | None
        }
    """
    intent_result = intent_recognize(message)
    route = intent_result.get("module") or MODULE_DAILY_QA
    if route not in (MODULE_KNOWLEDGE_QA, MODULE_SIGNATORY, MODULE_DAILY_QA):
        route = MODULE_DAILY_QA

    hint = None
    if route == MODULE_SIGNATORY:
        hint = "请使用签约流程：调用 POST /api/sign-contract，按步骤提交签约信息。"

    return {
        "intent": intent_result,
        "route": route,
        "hint": hint,
    }
