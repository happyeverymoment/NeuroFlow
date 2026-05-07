"""
RAG 统一配置加载
从 config.json 读取配置，支持环境变量覆盖（如 RAG_SERVER_PORT、MILVUS_HOST、RAG_EMBEDDING_API_KEY、RAG_LLM_API_KEY）。
路径类配置若为相对路径，则相对于 EicAI_hfs 目录解析。
"""

import json
import os

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CONFIG_DIR)
_config: dict = {}


def _load_config() -> dict:
    global _config
    if _config:
        return _config
    path = os.path.join(_CONFIG_DIR, "config.json")
    if not os.path.isfile(path):
        _config = {}
        return _config
    with open(path, "r", encoding="utf-8") as f:
        _config = json.load(f)
    return _config


def _resolve_path(value: str) -> str:
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.join(_PARENT_DIR, value)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _effective_api_key(env_key: str, file_val: str, fallback_env: str = None) -> str:
    """环境变量优先；若配置文件值为占位提示则视为未配置；可再回退到 fallback_env（如 DASHSCOPE_API_KEY）。"""
    v = _env(env_key)
    if v:
        return v
    raw = (file_val or "").strip()
    if raw and "请填写" not in raw:
        return raw
    if fallback_env:
        return _env(fallback_env)
    return ""


def get_server() -> dict:
    c = _load_config().get("server", {})
    return {
        "host": _env("RAG_SERVER_HOST") or c.get("host", "0.0.0.0"),
        "port": int(_env("RAG_SERVER_PORT") or str(c.get("port", 8001))),
        "reload": c.get("reload", False),
    }


def get_milvus() -> dict:
    c = _load_config().get("milvus", {})
    return {
        "host": _env("MILVUS_HOST") or c.get("host", "192.168.40.197"),
        "port": int(_env("MILVUS_PORT") or str(c.get("port", 19530))),
        "collection_name": c.get("collection_name", "langgraph_rag_knowledge_base"),
    }


def get_embedding() -> dict:
    c = _load_config().get("embedding", {})
    return {
        "api_key": _effective_api_key("RAG_EMBEDDING_API_KEY", c.get("api_key", ""), "DASHSCOPE_API_KEY"),
        "model": c.get("model", "text-embedding-v4"),
    }


def get_llm() -> dict:
    c = _load_config().get("llm", {})
    return {
        "api_key": _effective_api_key("RAG_LLM_API_KEY", c.get("api_key", ""), "DASHSCOPE_API_KEY"),
        "model": c.get("model", "qwen-turbo"),
        "base_url": c.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "temperature": c.get("temperature", 0.1),
    }


def get_faq() -> dict:
    c = _load_config().get("faq", {})
    data_dir = c.get("data_dir", "rag_faqdata")
    file_path = c.get("file_path", "常见问题FAQ（0928更新）.xlsx")
    return {
        "data_dir": _resolve_path(data_dir),
        "file_path": _resolve_path(file_path),
    }


def get_pdf() -> dict:
    """PDF 入库（MinerU）相关配置；路径相对于项目根目录解析。"""
    c = _load_config().get("pdf", {})
    data_dir = c.get("data_dir", "rag_faqdata")
    files = c.get("files", [])
    if isinstance(files, str):
        files = [files]
    work_dir = c.get("work_dir", ".mineru_output")
    sp = c.get("start_page_id", None)
    ep = c.get("end_page_id", None)
    return {
        "data_dir": _resolve_path(data_dir),
        "files": list(files),
        "parse_method": c.get("parse_method", "auto"),
        "chunk_size": int(c.get("chunk_size", 900)),
        "chunk_overlap": int(c.get("chunk_overlap", 150)),
        "work_dir": _resolve_path(work_dir),
        "magic_pdf_bin": _env("MINERU_MAGIC_PDF_BIN") or (c.get("magic_pdf_bin") or "").strip(),
        "add_batch_size": int(c.get("add_batch_size", 32)),
        "start_page_id": int(sp) if sp is not None and str(sp).strip() != "" else None,
        "end_page_id": int(ep) if ep is not None and str(ep).strip() != "" else None,
    }


def get_rag() -> dict:
    c = _load_config().get("rag", {})
    return {
        "candidates_k_default": c.get("candidates_k_default", 5),
        "summary_max_len": c.get("summary_max_len", 120),
        "similarity_threshold": float(c.get("similarity_threshold", 0.3)),
    }


def get_cors() -> dict:
    c = _load_config().get("cors", {})
    return {
        "allow_origins": c.get("allow_origins", ["*"]),
        "allow_credentials": c.get("allow_credentials", True),
        "allow_methods": c.get("allow_methods", ["*"]),
        "allow_headers": c.get("allow_headers", ["*"]),
    }
