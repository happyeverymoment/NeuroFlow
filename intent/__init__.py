# -*- coding: utf-8 -*-
# 意图识别模块：识别用户意图并映射到知识库问答、签约等业务模块
from .agent import (
    MODULE_KNOWLEDGE_QA,
    MODULE_SIGNATORY,
    MODULE_DAILY_QA,
    IntentOutput,
    recognize,
    get_agent,
)
from .router import recognize_and_route

__all__ = [
    "MODULE_KNOWLEDGE_QA",
    "MODULE_SIGNATORY",
    "MODULE_DAILY_QA",
    "IntentOutput",
    "recognize",
    "get_agent",
    "recognize_and_route",
]
