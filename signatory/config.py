# -*- coding: utf-8 -*-
# 签约服务配置（可后续改为从环境变量读取）
import os

# Token 请求
TOKEN_URL = os.getenv(
    "SIGNATORY_TOKEN_URL",
    "https://eiceducation--staging.sandbox.my.sfcrmproducts.cn/services/oauth2/token",
)
TOKEN_PARAMS = {
    "grant_type": "password",
    "client_id": os.getenv(
        "SIGNATORY_CLIENT_ID",
        "3MVG9xXFQGV9F.WrUMOnv7I3i3tIVsXIqqtUZKScyLl8TAOZDMNC2u7N1jURSjKORzHnM9_wHzIpZ.OpsWfjQ",
    ),
    "client_secret": os.getenv(
        "SIGNATORY_CLIENT_SECRET",
        "08CECC12D6098AFEE5AC951160EF1529E996702AACF2A373E3DB568E6531E70F",
    ),
    "username": os.getenv(
        "SIGNATORY_USERNAME",
        "eicintegration@eiceducation.com.cn.staging",
    ),
    "password": os.getenv("SIGNATORY_PASSWORD", "eic123456"),
}

# 用户权限查询 base URL（SOQL 通过 q 参数传递）
USER_QUERY_BASE_URL = os.getenv(
    "SIGNATORY_QUERY_BASE_URL",
    "https://eiceducation--staging.sandbox.my.sfcrmproducts.cn/services/data/v60.0/query",
)

# 默认要校验的用户名（用于权限查询）
DEFAULT_CHECK_USERNAME = "machong@eiceducation.com.cn.staging"
