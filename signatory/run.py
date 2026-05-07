# -*- coding: utf-8 -*-
"""
运行自动签约前两步：获取 Token → 校验用户权限。
可直接执行: python -m signatory.run
"""
import json
import sys


def main():
    from signatory import extract_name, extract_position, run_auto_sign_steps

    # 默认校验用户 machong@eiceducation.com.cn.staging，可通过命令行参数覆盖
    username = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_auto_sign_steps(username=username)
    position = result.get("position") or extract_position(
        (result.get("step2_permission") or {}).get("user_info")
    )
    name = result.get("name") or extract_name(
        (result.get("step2_permission") or {}).get("user_info")
    )
    # 仅输出姓名 + 职位
    output = {
        "姓名": name or "",
        "职位": position or "",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
