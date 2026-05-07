from langchain.tools import tool, ToolRuntime
from typing import Dict, Any
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict
import re
from datetime import datetime

# ====================== 1. 定义共享状态（存储分步信息+流程进度） ======================
class SignContractState(TypedDict):
    """签约流程共享状态：记录每步信息+当前进度"""
    current_step: int  # 当前步骤（1-5）
    step1_info: Dict[str, str]  # 步骤1：签约对象信息（姓名/身份证号）
    step2_info: Dict[str, str]  # 步骤2：合同编号
    step3_info: Dict[str, str]  # 步骤3：签约类型
    step4_info: Dict[str, str]  # 步骤4：签约日期
    step5_info: Dict[str, str]  # 步骤5：授权确认
    is_complete: bool  # 流程是否完成
    error_msg: str  # 错误/补充提示信息

# ====================== 2. Mock接口层（模拟真实业务接口，每步一个接口） ======================
class SignContractMockAPI:
    """签约流程Mock接口集合"""
    
    @staticmethod
    def step1_verify_person(info: Dict[str, str]) -> Dict[str, Any]:
        """Mock接口：步骤1 - 校验签约对象信息"""
        name = info.get("name", "").strip()
        id_card = info.get("id_card", "").strip()
        # 校验规则
        if not name:
            return {"success": False, "msg": "请补充签约对象姓名", "data": None}
        if not id_card or len(id_card) != 18 or not id_card.isdigit():
            return {"success": False, "msg": "请补充有效的18位身份证号", "data": None}
        # 校验通过
        return {
            "success": True,
            "msg": "签约对象信息校验通过",
            "data": {"name": name, "id_card": id_card}
        }
    
    @staticmethod
    def step2_verify_contract_no(info: Dict[str, str]) -> Dict[str, Any]:
        """Mock接口：步骤2 - 校验合同编号"""
        contract_no = info.get("contract_no", "").strip()
        # 校验规则：QY+8位数字
        pattern = r"^QY\d{8}$"
        if not contract_no:
            return {"success": False, "msg": "请补充合同编号", "data": None}
        if not re.match(pattern, contract_no):
            return {"success": False, "msg": "合同编号格式错误（正确格式：QY+8位数字，如QY20240121）", "data": None}
        # 校验通过
        return {
            "success": True,
            "msg": "合同编号校验通过",
            "data": {"contract_no": contract_no}
        }
    
    @staticmethod
    def step3_verify_contract_type(info: Dict[str, str]) -> Dict[str, Any]:
        """Mock接口：步骤3 - 校验签约类型"""
        contract_type = info.get("contract_type", "").strip()
        # 校验规则：仅个人/企业
        if not contract_type:
            return {"success": False, "msg": "请补充签约类型", "data": None}
        if contract_type not in ["个人", "企业"]:
            return {"success": False, "msg": "签约类型仅支持「个人/企业」", "data": None}
        # 校验通过
        return {
            "success": True,
            "msg": "签约类型确认成功",
            "data": {"contract_type": contract_type}
        }
    
    @staticmethod
    def step4_verify_date(info: Dict[str, str]) -> Dict[str, Any]:
        """Mock接口：步骤4 - 校验签约日期"""
        sign_date = info.get("sign_date", "").strip()
        # 校验规则：YYYY-MM-DD格式
        if not sign_date:
            return {"success": False, "msg": "请补充签约日期", "data": None}
        try:
            datetime.strptime(sign_date, "%Y-%m-%d")
        except ValueError:
            return {"success": False, "msg": "签约日期格式错误（正确格式：YYYY-MM-DD）", "data": None}
        # 校验通过
        return {
            "success": True,
            "msg": "签约日期校验通过",
            "data": {"sign_date": sign_date}
        }
    
    @staticmethod
    def step5_verify_authorization(info: Dict[str, str]) -> Dict[str, Any]:
        """Mock接口：步骤5 - 校验签约授权"""
        authorization = info.get("authorization", "").strip()
        # 校验规则：仅允许是/否，且必须为是
        if not authorization:
            return {"success": False, "msg": "请确认是否授权签约（是/否）", "data": None}
        if authorization not in ["是", "否"]:
            return {"success": False, "msg": "授权确认仅支持「是/否」", "data": None}
        if authorization == "否":
            return {"success": False, "msg": "未授权签约，流程终止", "data": None}
        # 校验通过（所有步骤完成）
        return {
            "success": True,
            "msg": "签约授权确认成功，所有流程完成！",
            "data": {"authorization": authorization, "final_result": "签约成功，合同编号已归档"}
        }

# ====================== 3. 业务Agent工具封装（LangChain v1 Tool） ======================
def _sign_contract_agent_impl(user_input: str, runtime: ToolRuntime[SignContractState]) -> str:
    """
    【业务Agent工具】签约流程处理（五步固定流程）
    :param user_input: 用户输入的补充信息
    :param runtime: 运行时状态（存储分步信息+流程进度）
    :return: 交互提示（缺信息提醒/步骤完成提示/最终结果）
    """
    # 初始化Mock接口
    mock_api = SignContractMockAPI()
    # 获取当前状态
    state = runtime.state
    current_step = state["current_step"]
    error_msg = ""

    # ====================== 步骤1：签约对象信息 ======================
    if current_step == 1:
        # 解析用户输入（简单拆分：姓名=XXX 身份证号=XXX）
        info = {}
        if "姓名=" in user_input:
            info["name"] = user_input.split("姓名=")[1].split()[0]
        if "身份证号=" in user_input:
            info["id_card"] = user_input.split("身份证号=")[1].split()[0]
        # 调用Mock接口校验
        res = mock_api.step1_verify_person(info)
        if res["success"]:
            # 步骤完成，更新状态
            state["step1_info"] = res["data"]
            state["current_step"] = 2
            state["error_msg"] = ""
            return f"【步骤1完成】{res['msg']} → 请补充步骤2信息：合同编号（格式：QY+8位数字）"
        else:
            # 缺信息，提醒补充
            state["error_msg"] = res["msg"]
            return f"【步骤1未完成】{res['msg']} → 请按格式补充：姓名=XXX 身份证号=18位数字"

    # ====================== 步骤2：合同编号 ======================
    elif current_step == 2:
        # 解析用户输入：合同编号=XXX
        info = {}
        if "合同编号=" in user_input:
            info["contract_no"] = user_input.split("合同编号=")[1].split()[0]
        # 调用Mock接口校验
        res = mock_api.step2_verify_contract_no(info)
        if res["success"]:
            state["step2_info"] = res["data"]
            state["current_step"] = 3
            state["error_msg"] = ""
            return f"【步骤2完成】{res['msg']} → 请补充步骤3信息：签约类型（个人/企业）"
        else:
            state["error_msg"] = res["msg"]
            return f"【步骤2未完成】{res['msg']} → 请按格式补充：合同编号=QY+8位数字（如QY20240121）"

    # ====================== 步骤3：签约类型 ======================
    elif current_step == 3:
        # 解析用户输入：签约类型=XXX
        info = {}
        if "签约类型=" in user_input:
            info["contract_type"] = user_input.split("签约类型=")[1].split()[0]
        # 调用Mock接口校验
        res = mock_api.step3_verify_contract_type(info)
        if res["success"]:
            state["step3_info"] = res["data"]
            state["current_step"] = 4
            state["error_msg"] = ""
            return f"【步骤3完成】{res['msg']} → 请补充步骤4信息：签约日期（格式：YYYY-MM-DD）"
        else:
            state["error_msg"] = res["msg"]
            return f"【步骤3未完成】{res['msg']} → 请按格式补充：签约类型=个人/企业"

    # ====================== 步骤4：签约日期 ======================
    elif current_step == 4:
        # 解析用户输入：签约日期=XXX
        info = {}
        if "签约日期=" in user_input:
            info["sign_date"] = user_input.split("签约日期=")[1].split()[0]
        # 调用Mock接口校验
        res = mock_api.step4_verify_date(info)
        if res["success"]:
            state["step4_info"] = res["data"]
            state["current_step"] = 5
            state["error_msg"] = ""
            return f"【步骤4完成】{res['msg']} → 请补充步骤5信息：是否授权签约（是/否）"
        else:
            state["error_msg"] = res["msg"]
            return f"【步骤4未完成】{res['msg']} → 请按格式补充：签约日期=YYYY-MM-DD（如2024-01-21）"

    # ====================== 步骤5：签约授权 ======================
    elif current_step == 5:
        # 解析用户输入：授权=XXX
        info = {}
        if "授权=" in user_input:
            info["authorization"] = user_input.split("授权=")[1].split()[0]
        # 调用Mock接口校验
        res = mock_api.step5_verify_authorization(info)
        if res["success"]:
            state["step5_info"] = res["data"]
            state["current_step"] = 0  # 流程结束
            state["is_complete"] = True
            state["error_msg"] = ""
            return f"【所有步骤完成】{res['msg']} → 最终结果：{res['data']['final_result']}\n" \
                   f"签约信息汇总：\n" \
                   f"1. 签约对象：{state['step1_info']['name']}（{state['step1_info']['id_card']}）\n" \
                   f"2. 合同编号：{state['step2_info']['contract_no']}\n" \
                   f"3. 签约类型：{state['step3_info']['contract_type']}\n" \
                   f"4. 签约日期：{state['step4_info']['sign_date']}\n" \
                   f"5. 授权状态：{state['step5_info']['authorization']}"
        else:
            state["error_msg"] = res["msg"]
            return f"【步骤5未完成】{res['msg']} → 请按格式补充：授权=是/否"

    # 流程已完成
    elif state["is_complete"]:
        return "签约流程已完成，无需补充信息！"
    # 无效步骤
    else:
        return "流程异常，请重新开始签约！"


# 创建工具对象（用于 Agent 调用）
sign_contract_agent = tool(_sign_contract_agent_impl)

# ====================== 4. 测试脚本（模拟用户分步补充信息） ======================
def test_sign_contract_agent():
    """测试业务Agent的五步交互流程（含缺信息场景）"""
    # 初始化状态
    init_state: SignContractState = {
        "current_step": 1,
        "step1_info": {},
        "step2_info": {},
        "step3_info": {},
        "step4_info": {},
        "step5_info": {},
        "is_complete": False,
        "error_msg": ""
    }

    # 模拟ToolRuntime（简化版，仅传递状态）
    class MockToolRuntime(ToolRuntime):
        def __init__(self, state: SignContractState):
            self.state = state

    runtime = MockToolRuntime(init_state)

    # 模拟用户交互过程（含缺信息→补充→完成的全流程）
    user_interactions = [
        # 步骤1：缺身份证号 → 提醒补充
        "姓名=李同学",
        # 步骤1：补充完整 → 到步骤2
        "姓名=李同学 身份证号=123456789012345678",
        # 步骤2：合同编号格式错误 → 提醒补充
        "合同编号=QY202401",
        # 步骤2：格式正确 → 到步骤3
        "合同编号=QY20240121",
        # 步骤3：签约类型错误 → 提醒补充
        "签约类型=个体户",
        # 步骤3：类型正确 → 到步骤4
        "签约类型=个人",
        # 步骤4：日期格式错误 → 提醒补充
        "签约日期=2024/01/21",
        # 步骤4：日期正确 → 到步骤5
        "签约日期=2024-01-21",
        # 步骤5：授权为否 → 提醒补充
        "授权=否",
        # 步骤5：授权为是 → 完成所有步骤
        "授权=是"
    ]

    # 执行交互
    print("===== 签约流程测试（五步交互）=====\n")
    for idx, user_input in enumerate(user_interactions):
        print(f"【用户输入{idx+1}】：{user_input}")
        # 调用业务Agent工具（使用原始实现函数，而不是工具对象）
        result = _sign_contract_agent_impl(user_input, runtime)
        print(f"【Agent回复】：{result}\n")
        # 流程完成则终止
        if runtime.state["is_complete"]:
            break

# ====================== 运行测试 ======================
if __name__ == "__main__":
    test_sign_contract_agent()