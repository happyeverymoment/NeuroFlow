# api_server_business.py - 基于 business_agent.py 的签约流程 API 服务器
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import os
from business_agent import _sign_contract_agent_impl, SignContractState
from langchain.tools import ToolRuntime

app = FastAPI(title="签约流程 API 系统")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 签约流程会话存储
sign_contract_sessions: Dict[str, SignContractState] = {}


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


class MockToolRuntime(ToolRuntime):
    """模拟 ToolRuntime"""
    def __init__(self, state: SignContractState):
        self.state = state


@app.post("/api/sign-contract", response_model=SignContractResponse)
async def sign_contract(request: SignContractRequest):
    """处理签约流程请求"""
    try:
        # 获取或创建会话
        if request.session_id and request.session_id in sign_contract_sessions:
            state = sign_contract_sessions[request.session_id]
        else:
            # 创建新会话
            session_id = str(uuid.uuid4())
            state: SignContractState = {
                "current_step": 1,
                "step1_info": {},
                "step2_info": {},
                "step3_info": {},
                "step4_info": {},
                "step5_info": {},
                "is_complete": False,
                "error_msg": ""
            }
            sign_contract_sessions[session_id] = state
            request.session_id = session_id
        
        # 创建 MockToolRuntime
        runtime = MockToolRuntime(state)
        
        # 调用签约流程处理函数
        response = _sign_contract_agent_impl(request.user_input, runtime)
        
        # 更新会话状态
        sign_contract_sessions[request.session_id] = state
        
        # 获取当前步骤的信息
        step_info = {}
        if state["current_step"] >= 1 and state.get("step1_info"):
            step_info["step1"] = state["step1_info"]
        if state["current_step"] >= 2 and state.get("step2_info"):
            step_info["step2"] = state["step2_info"]
        if state["current_step"] >= 3 and state.get("step3_info"):
            step_info["step3"] = state["step3_info"]
        if state["current_step"] >= 4 and state.get("step4_info"):
            step_info["step4"] = state["step4_info"]
        if state["current_step"] >= 5 and state.get("step5_info"):
            step_info["step5"] = state["step5_info"]
        
        return SignContractResponse(
            response=response,
            session_id=request.session_id,
            current_step=state["current_step"],
            is_complete=state["is_complete"],
            error_msg=state.get("error_msg", ""),
            step_info=step_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sign-contract/{session_id}")
async def clear_sign_contract_session(session_id: str):
    """清除签约流程会话"""
    if session_id in sign_contract_sessions:
        del sign_contract_sessions[session_id]
        return {"message": "Session cleared"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok", "system": "签约流程系统"}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回前端页面"""
    html_file_path = os.path.join(os.path.dirname(__file__), "sign_contract_business.html")
    if os.path.exists(html_file_path):
        return FileResponse(html_file_path)
    else:
        return HTMLResponse(content="<h1>前端文件 sign_contract_business.html 未找到</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    # 签约流程独立服务：8003（意图识别 8001，知识库问答 8002）
    uvicorn.run(app, host="0.0.0.0", port=8003)

