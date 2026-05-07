# -*- coding: utf-8 -*-
"""
自动签约 Agent：实现获取 Token、校验用户权限等步骤。
"""
from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from signatory.config import (
    DEFAULT_CHECK_USERNAME,
    TOKEN_PARAMS,
    TOKEN_URL,
    USER_QUERY_BASE_URL,
)


@dataclass
class TokenResult:
    """Token 获取结果"""
    success: bool
    access_token: Optional[str] = None
    instance_url: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PermissionResult:
    """用户权限校验结果"""
    success: bool
    has_permission: bool
    user_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def extract_position(user_info: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    从用户信息中提取“职位”字段。
    优先取 Profile.Name，其次取 UserRole.Name。
    """
    if not user_info:
        return None
    profile = user_info.get("Profile") or {}
    role = user_info.get("UserRole") or {}
    position = profile.get("Name") or role.get("Name")
    return position


def extract_name(user_info: Optional[Dict[str, Any]]) -> Optional[str]:
    """从用户信息中提取姓名（name 字段）。"""
    if not user_info:
        return None
    # Salesforce User 返回字段通常为 Name（首字母大写）
    return user_info.get("Name") or user_info.get("name")


def get_token(
    timeout: int = 30,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> TokenResult:
    """
    步骤1：通过 POST 获取 Token。

    Token 请求地址（POST）：
    - URL: 配置中的 TOKEN_URL
    - 参数: grant_type, client_id, client_secret, username, password
    - username/password 若传入则覆盖配置中的值，否则使用 TOKEN_PARAMS。
    """
    data = dict(TOKEN_PARAMS)
    if username is not None:
        data["username"] = username
    if password is not None:
        data["password"] = password
    try:
        resp = requests.post(
            TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        access_token = data.get("access_token")
        instance_url = data.get("instance_url")
        if not access_token:
            return TokenResult(
                success=False,
                error=data.get("error_description") or data.get("error") or "未返回 access_token",
            )
        return TokenResult(
            success=True,
            access_token=access_token,
            instance_url=instance_url,
        )
    except requests.RequestException as e:
        err_msg = str(e)
        if hasattr(e, "response") and e.response is not None:
            try:
                body = e.response.json()
                err_msg = body.get("error_description") or body.get("error") or err_msg
            except Exception:
                pass
        return TokenResult(success=False, error=err_msg)
    except Exception as e:
        return TokenResult(success=False, error=str(e))


def validate_user_permission(
    token: str,
    username: Optional[str] = None,
    timeout: int = 30,
) -> PermissionResult:
    """
    步骤2：通过 GET 校验用户身份权限。

    用户权限查询（GET）：
    - URL: 配置中的 USER_QUERY_BASE_URL + SOQL 查询参数
    - Header: Authorization: <Token>
    """
    username = username or DEFAULT_CHECK_USERNAME
    # SOQL: SELECT name, isactive, Profile.Name, UserRole.Name FROM user WHERE Username = 'xxx' AND isactive = true
    soql = (
        "SELECT name,isactive,Profile.Name,UserRole.Name FROM user "
        f"WHERE Username = '{username}' AND isactive = true"
    )
    params = {"q": soql}
    url = f"{USER_QUERY_BASE_URL}?{urllib.parse.urlencode(params)}"

    try:
        # OAuth2：Authorization 携带 Token（Bearer access_token）
        auth_header = token if token.startswith("Bearer ") else f"Bearer {token}"
        resp = requests.get(
            url,
            headers={
                "Authorization": auth_header,
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        records = data.get("records") or []
        total_size = data.get("totalSize", 0)

        if total_size == 0:
            return PermissionResult(
                success=True,
                has_permission=False,
                user_info=None,
                error="未找到该用户或用户未激活",
            )

        user_info = records[0] if records else None
        is_active = user_info.get("IsActive", False) if user_info else False
        return PermissionResult(
            success=True,
            has_permission=is_active,
            user_info=user_info,
        )
    except requests.RequestException as e:
        err_msg = str(e)
        if hasattr(e, "response") and e.response is not None:
            try:
                body = e.response.json()
                err_msg = body if isinstance(body, str) else (body.get("message") or body.get("error_description") or str(body))
            except Exception:
                pass
        return PermissionResult(success=False, has_permission=False, error=err_msg)
    except Exception as e:
        return PermissionResult(success=False, has_permission=False, error=str(e))


class SignatoryAgent:
    """
    自动签约 Agent：串联获取 Token、校验权限等步骤。
    token_username / token_password 用于步骤1 获取 Token；不传则用配置默认。
    check_username 用于步骤2 权限校验。
    """

    def __init__(
        self,
        check_username: Optional[str] = None,
        token_username: Optional[str] = None,
        token_password: Optional[str] = None,
        timeout: int = 30,
    ):
        self.check_username = check_username or DEFAULT_CHECK_USERNAME
        self.token_username = token_username
        self.token_password = token_password
        self.timeout = timeout

    def run_first_two_steps(self) -> Dict[str, Any]:
        """
        执行前两步：1. 获取 Token  2. 校验用户权限。
        返回包含 token 结果、权限结果及整体是否成功的字典。
        """
        result = {
            "step1_token": None,
            "step2_permission": None,
            "success": False,
            "token": None,
            "position": None,
            "name": None,
            "error": None,
        }

        # Step 1: 获取 Token（可使用前端传入的 username/password）
        token_result = get_token(
            timeout=self.timeout,
            username=self.token_username,
            password=self.token_password,
        )
        result["step1_token"] = {
            "success": token_result.success,
            "error": token_result.error,
        }

        if not token_result.success:
            result["error"] = f"获取 Token 失败: {token_result.error}"
            return result

        result["token"] = token_result.access_token

        # Step 2: 校验用户权限
        perm_result = validate_user_permission(
            token=token_result.access_token,
            username=self.check_username,
            timeout=self.timeout,
        )
        result["step2_permission"] = {
            "success": perm_result.success,
            "has_permission": perm_result.has_permission,
            "user_info": perm_result.user_info,
            "error": perm_result.error,
        }
        result["position"] = extract_position(perm_result.user_info)
        result["name"] = extract_name(perm_result.user_info)

        if not perm_result.success:
            result["error"] = f"权限校验请求失败: {perm_result.error}"
            return result

        if not perm_result.has_permission:
            result["error"] = perm_result.error or "用户无权限或未激活"
            return result

        result["success"] = True
        return result


# 便于直接调用
def run_auto_sign_steps(
    username: Optional[str] = None,
    token_username: Optional[str] = None,
    token_password: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    便捷入口：执行自动签约前两步。
    username: 步骤2 要校验的用户名（不传用配置默认）。
    token_username / token_password: 步骤1 获取 Token 的账号密码（不传用配置默认）。
    """
    agent = SignatoryAgent(
        check_username=username,
        token_username=token_username,
        token_password=token_password,
        timeout=timeout,
    )
    return agent.run_first_two_steps()
