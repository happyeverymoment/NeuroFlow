# -*- coding: utf-8 -*-
# 自动签约 Agent 模块
from signatory.agent import (
    SignatoryAgent,
    extract_name,
    extract_position,
    get_token,
    validate_user_permission,
    run_auto_sign_steps,
)

__all__ = [
    "SignatoryAgent",
    "extract_name",
    "extract_position",
    "get_token",
    "validate_user_permission",
    "run_auto_sign_steps",
]
