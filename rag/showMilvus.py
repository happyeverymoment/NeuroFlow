# -*- coding: utf-8 -*-
"""
查看 Milvus 中的集合与抽样数据（与 faq_ingest 相同：config_loader + pymilvus URI）。

文件名「Mlivus」为常见笔误，等价脚本也可命名为 showMilvus.py。

运行（在项目根目录）:
  cd /path/to/HFS_9002 && python -m rag.showMlivus

环境变量可覆盖连接（与 config_loader 一致）:
  MILVUS_HOST  MILVUS_PORT
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pymilvus import Collection, DataType, connections, utility  # noqa: E402

from rag.config_loader import get_milvus  # noqa: E402


def _vector_dtypes() -> set:
    s = {DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR}
    for name in ("FLOAT16_VECTOR", "BFLOAT16_VECTOR", "SPARSE_FLOAT_VECTOR"):
        dt = getattr(DataType, name, None)
        if dt is not None:
            s.add(dt)
    return s


def _scalar_field_names(collection: Collection) -> List[str]:
    skip = _vector_dtypes()
    return [f.name for f in collection.schema.fields if f.dtype not in skip]


def _primary_field(collection: Collection) -> tuple[str, Any]:
    for f in collection.schema.fields:
        if getattr(f, "is_primary", False):
            return f.name, f.dtype
    return "pk", None


def _pick_query_expr(collection: Collection) -> str:
    name, dtype = _primary_field(collection)
    if dtype == DataType.INT64:
        return f"{name} >= 0"
    return f'{name} != ""'


def _truncate(val: Any, max_len: int = 500) -> str:
    if val is None:
        return ""
    s = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
    s = str(s).replace("\r\n", "\n")
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...(共{len(s)}字符)"


def connect() -> None:
    cfg = get_milvus()
    host = os.environ.get("MILVUS_HOST", cfg["host"]).strip()
    port = int(os.environ.get("MILVUS_PORT", str(cfg["port"])))
    uri = f"http://{host}:{port}"
    print(f"连接 Milvus: {uri}")
    print(f"配置中的业务集合名: {cfg.get('collection_name', '')}")
    connections.connect("default", uri=uri, timeout=60)


def dump_collection(name: str, sample_limit: int = 105) -> None:
    c = Collection(name)
    c.load()
    try:
        c.flush()
    except Exception:
        pass
    print(f"\n{'=' * 72}")
    print(f"集合: {name}")
    print(f"实体数(约): {c.num_entities}")
    print("--- Schema ---")
    for f in c.schema.fields:
        pk = " PK" if getattr(f, "is_primary", False) else ""
        extra = ""
        if f.dtype in _vector_dtypes():
            dim = (f.params or {}).get("dim")
            extra = f" dim={dim}" if dim else ""
        print(f"  - {f.name}: {f.dtype}{pk}{extra}")

    outs = _scalar_field_names(c)
    if not outs:
        print("  (无非向量字段可展示)")
        return

    expr = _pick_query_expr(c)
    try:
        rows = c.query(expr=expr, output_fields=outs, limit=sample_limit)
    except Exception as e:
        print(f"  抽样 query 失败 expr={expr!r}: {e}")
        return

    print(f"--- 抽样最多 {sample_limit} 条 (已省略/截断向量字段) ---")
    for i, row in enumerate(rows, 1):
        print(f"\n  [{i}]")
        for k in outs:
            v = row.get(k)
            if isinstance(v, str) and len(v) > 500:
                print(f"    {k}: {_truncate(v, 500)}")
            elif isinstance(v, list) and len(str(v)) > 200:
                print(f"    {k}: <list len={len(v)}>")
            else:
                print(f"    {k}: {_truncate(v, 500)}")


def main() -> None:
    connect()
    names = utility.list_collections()
    if not names:
        print("当前无任何 collection。")
        return
    print(f"\n全部集合 ({len(names)} 个): {sorted(names)}")
    cfg = get_milvus()
    preferred = cfg.get("collection_name")
    order = []
    if preferred and preferred in names:
        order.append(preferred)
    for n in sorted(names):
        if n not in order:
            order.append(n)
    for n in order:
        try:
            dump_collection(n, sample_limit=105)
        except Exception as e:
            print(f"\n集合 {n} 读取失败: {e}")
    connections.disconnect("default")
    print(f"\n{'=' * 72}\n完成。")


if __name__ == "__main__":
    main()
