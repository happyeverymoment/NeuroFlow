"""
PDF（MinerU）解析、分块并追加写入 Milvus。
风格与 faq_ingest 一致：metadata 须与已有集合字段兼容（含必填 row_data）。

运行：在项目根目录执行  python -m rag.pdf_ingest
若当前环境无 magic-pdf：export MINERU_MAGIC_PDF_BIN=.../magic-pdf 或配置 pdf.magic_pdf_bin。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .config_loader import get_embedding, get_milvus, get_pdf
except ImportError:
    from config_loader import get_embedding, get_milvus, get_pdf

_milvus = get_milvus()
_embedding = get_embedding()
_pdf_cfg = get_pdf()

MILVUS_HOST = _milvus["host"]
MILVUS_PORT = _milvus["port"]
COLLECTION_NAME = _milvus["collection_name"]
EMBEDDING_API_KEY = _embedding["api_key"]
EMBEDDING_MODEL = _embedding["model"]

PDF_DOC_TYPE = "pdf"


def _escape_milvus_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _try_import_mineru() -> bool:
    try:
        from magic_pdf.tools.common import do_parse  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_magic_pdf_bin(configured: str) -> Optional[str]:
    if configured and os.path.isfile(configured) and os.access(configured, os.X_OK):
        return configured
    which = shutil.which("magic-pdf")
    if which:
        return which
    return None


def _expected_md_path(output_dir: str, pdf_stem: str, parse_method: str) -> str:
    return os.path.join(output_dir, pdf_stem, parse_method, f"{pdf_stem}.md")


def _parse_pdf_mineru_inline(
    pdf_path: str,
    output_dir: str,
    parse_method: str,
    start_page_id: Optional[int],
    end_page_id: Optional[int],
) -> str:
    from magic_pdf.data.data_reader_writer import FileBasedDataReader
    from magic_pdf.tools.common import do_parse

    pdf_path = os.path.abspath(pdf_path)
    stem = Path(pdf_path).stem
    parent = os.path.dirname(pdf_path)
    disk_rw = FileBasedDataReader(parent)
    pdf_bytes = disk_rw.read(os.path.basename(pdf_path))
    os.makedirs(output_dir, exist_ok=True)
    do_parse(
        output_dir,
        stem,
        pdf_bytes,
        [],
        parse_method,
        debug_able=False,
        start_page_id=start_page_id if start_page_id is not None else 0,
        end_page_id=end_page_id,
        f_draw_span_bbox=False,
        f_draw_layout_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=False,
        f_dump_model_json=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=False,
    )
    md_path = _expected_md_path(output_dir, stem, parse_method)
    if not os.path.isfile(md_path):
        raise FileNotFoundError(f"MinerU 未生成 markdown: {md_path}")
    return md_path


def _parse_pdf_mineru_subprocess(
    pdf_path: str,
    output_dir: str,
    parse_method: str,
    magic_pdf_bin: str,
    start_page_id: Optional[int],
    end_page_id: Optional[int],
) -> str:
    pdf_path = os.path.abspath(pdf_path)
    stem = Path(pdf_path).stem
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        magic_pdf_bin,
        "-p",
        pdf_path,
        "-o",
        output_dir,
        "-m",
        parse_method,
    ]
    if start_page_id is not None:
        cmd.extend(["-s", str(start_page_id)])
    if end_page_id is not None:
        cmd.extend(["-e", str(end_page_id)])
    print(f"🧩 调用 MinerU: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    md_path = _expected_md_path(output_dir, stem, parse_method)
    if not os.path.isfile(md_path):
        raise FileNotFoundError(f"MinerU 未生成 markdown: {md_path}")
    return md_path


def parse_pdf_to_markdown(
    pdf_path: str,
    output_dir: str,
    parse_method: str = "auto",
    magic_pdf_bin: str = "",
    start_page_id: Optional[int] = None,
    end_page_id: Optional[int] = None,
) -> str:
    """
    使用 MinerU（magic-pdf）将 PDF 转为 markdown 文件路径。
    优先在当前解释器内 import magic_pdf；失败则使用 magic_pdf_bin 子进程。
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    if _try_import_mineru():
        print("📎 使用当前 Python 环境内的 magic_pdf 解析")
        return _parse_pdf_mineru_inline(
            pdf_path, output_dir, parse_method, start_page_id, end_page_id
        )

    bin_path = _resolve_magic_pdf_bin(magic_pdf_bin.strip())
    if not bin_path:
        raise RuntimeError(
            "未找到 MinerU：当前环境无法 import magic_pdf，且未配置可执行的 magic-pdf。\n"
            "请安装 MinerU（magic-pdf），或设置环境变量 MINERU_MAGIC_PDF_BIN，"
            "或在 config.json 的 pdf.magic_pdf_bin 中填写 magic-pdf 绝对路径。"
        )
    print(f"📎 使用子进程 MinerU: {bin_path}")
    return _parse_pdf_mineru_subprocess(
        pdf_path, output_dir, parse_method, bin_path, start_page_id, end_page_id
    )


def markdown_to_documents(
    md_text: str,
    source_pdf: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    将 Markdown 正文切分为 Document。
    与 read_faq_from_folder 入库字段对齐：question、answer、source、row_data（JSON 字符串）；
    PDF 专有信息放在 row_data 内，避免使用集合中不存在的标量字段（如 doc_type）。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_text(md_text)
    docs: List[Document] = []
    basename = os.path.basename(source_pdf)
    abs_src = os.path.abspath(source_pdf)

    for i, ch in enumerate(chunks):
        ch = ch.strip()
        if not ch:
            continue
        first_line = ch.split("\n", 1)[0].strip()
        if first_line.startswith("#"):
            question = re.sub(r"^#+\s*", "", first_line).strip() or f"[PDF] {basename}"
        else:
            question = f"[PDF] {basename} · 片段 {i + 1}"

        page_content = f"{question}\n\n{ch}"
        row_payload = {"doc_type": PDF_DOC_TYPE, "chunk_index": i}
        meta = {
            "question": question,
            "answer": ch,
            "source": abs_src,
            "row_data": json.dumps(row_payload, ensure_ascii=False),
        }
        docs.append(Document(page_content=page_content, metadata=meta))

    return docs


def read_pdf_as_documents(
    pdf_path: str,
    *,
    work_dir: Optional[str] = None,
    parse_method: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    magic_pdf_bin: Optional[str] = None,
    start_page_id: Optional[int] = None,
    end_page_id: Optional[int] = None,
) -> Tuple[List[Document], str]:
    """
    解析单个 PDF 并返回 (Document 列表, 生成的 markdown 路径)。
    """
    cfg = get_pdf()
    work_dir = work_dir or cfg["work_dir"]
    parse_method = parse_method if parse_method is not None else cfg["parse_method"]
    chunk_size = chunk_size if chunk_size is not None else cfg["chunk_size"]
    chunk_overlap = chunk_overlap if chunk_overlap is not None else cfg["chunk_overlap"]
    magic_pdf_bin = magic_pdf_bin if magic_pdf_bin is not None else cfg["magic_pdf_bin"]
    if start_page_id is None:
        start_page_id = cfg.get("start_page_id")
    if end_page_id is None:
        end_page_id = cfg.get("end_page_id")

    os.makedirs(work_dir, exist_ok=True)
    stem = Path(os.path.abspath(pdf_path)).stem
    out_sub = os.path.join(work_dir, f"mineru_{stem}")
    if os.path.isdir(out_sub):
        shutil.rmtree(out_sub, ignore_errors=True)
    os.makedirs(out_sub, exist_ok=True)

    md_path = parse_pdf_to_markdown(
        pdf_path,
        out_sub,
        parse_method=parse_method,
        magic_pdf_bin=magic_pdf_bin,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
    )
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    print(f"📄 Markdown 已生成: {md_path}，长度 {len(md_text)} 字符")
    docs = markdown_to_documents(md_text, pdf_path, chunk_size, chunk_overlap)
    print(f"✂️ 分块得到 {len(docs)} 条 Document（chunk_size={chunk_size}, overlap={chunk_overlap}）")
    return docs, md_path


def delete_pdf_vectors_for_source(
    vectorstore: Milvus,
    source_abs: str,
) -> None:
    """删除与给定 PDF 路径相同的 source 的旧向量（幂等重跑前使用）。
    集合 schema 通常无 doc_type 标量字段，故不按 doc_type 过滤；FAQ 行的 source 为 xlsx 路径，不会误删。
    """
    expr = f'source == "{_escape_milvus_str(source_abs)}"'
    try:
        ok = vectorstore.delete(expr=expr)
        if ok:
            print(f"🗑️ 已按 source 删除旧 PDF 向量: {source_abs}")
        else:
            print(f"⚠️ 删除旧向量返回 False（可能集合中无匹配项）: {source_abs}")
    except Exception as e:
        print(f"⚠️ 按表达式删除旧向量失败（可忽略，首次入库）: {e}")


def append_documents_to_milvus(
    documents: List[Document],
    *,
    milvus_host: Optional[str] = None,
    milvus_port: Optional[int] = None,
    collection_name: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    replace_same_source: bool = True,
    add_batch_size: Optional[int] = None,
    source_for_replace: Optional[str] = None,
) -> Milvus:
    """
    将 Document 追加写入已有 Milvus 集合（不 drop 集合）。
    replace_same_source=True 时，先删除 source 与该 PDF 绝对路径相同的旧记录。
    """
    if not documents:
        raise ValueError("documents 为空")

    host = milvus_host or MILVUS_HOST
    port = milvus_port or MILVUS_PORT
    coll = collection_name or COLLECTION_NAME
    api_key = embedding_api_key or EMBEDDING_API_KEY
    model = embedding_model or EMBEDDING_MODEL
    cfg = get_pdf()
    batch = add_batch_size if add_batch_size is not None else cfg["add_batch_size"]

    emb_kwargs = {"model": model}
    if api_key:
        emb_kwargs["dashscope_api_key"] = api_key
    embeddings = DashScopeEmbeddings(**emb_kwargs)

    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=coll,
        connection_args={"uri": f"http://{host}:{port}"},
        drop_old=False,
    )

    src = source_for_replace
    if replace_same_source and src is None:
        src = documents[0].metadata.get("source")
    if replace_same_source and src:
        delete_pdf_vectors_for_source(vectorstore, os.path.abspath(str(src)))

    print(f"📤 追加写入 Milvus（batch={batch}），共 {len(documents)} 条…")
    for i in range(0, len(documents), batch):
        batch_docs = documents[i : i + batch]
        vectorstore.add_documents(batch_docs)
        print(f"   已写入 {min(i + batch, len(documents))}/{len(documents)}")

    try:
        vectorstore.client.flush(collection_name=coll)
        print("ℹ️ 已对集合执行 flush，实体数统计与检索可立即反映本次写入")
    except Exception as e:
        print(f"⚠️ Milvus flush 跳过: {e}")

    print("✅ Milvus 追加完成")
    return vectorstore


def ingest_pdf_file(
    pdf_path: str,
    *,
    replace_same_source: bool = True,
    **read_kw,
) -> Milvus:
    """解析单个 PDF 并追加到 Milvus。"""
    docs, _ = read_pdf_as_documents(pdf_path, **read_kw)
    return append_documents_to_milvus(
        docs, replace_same_source=replace_same_source, source_for_replace=os.path.abspath(pdf_path)
    )


def ingest_pdfs_from_config(
    *,
    replace_same_source: bool = True,
) -> None:
    """按 config.json 的 pdf.data_dir + pdf.files 批量入库。"""
    cfg = get_pdf()
    data_dir = cfg["data_dir"]
    files = cfg["files"]
    if not files:
        cand = sorted(Path(data_dir).glob("*.pdf"))
        files = [str(p.name) for p in cand]
        print(f"ℹ️ pdf.files 为空，将入库目录下全部 PDF（{len(files)} 个）")
    for name in files:
        fp = name if os.path.isabs(name) else os.path.join(data_dir, name)
        if not os.path.isfile(fp):
            print(f"⚠️ 跳过（文件不存在）: {fp}")
            continue
        print(f"\n{'='*60}\n📚 处理: {fp}\n{'='*60}")
        ingest_pdf_file(fp, replace_same_source=replace_same_source)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="PDF（MinerU）追加写入 Milvus")
    p.add_argument(
        "--file",
        "-f",
        default=None,
        help="单个 PDF 路径；默认使用 config 中 pdf.files",
    )
    p.add_argument(
        "--no-replace",
        action="store_true",
        help="不先按 source 删除旧 PDF 向量（可能重复）",
    )
    p.add_argument(
        "--parse-only",
        action="store_true",
        help="仅解析并打印分块数量，不写入 Milvus",
    )
    p.add_argument(
        "-s",
        "--start-page",
        type=int,
        default=None,
        help="起始页（0-based），覆盖 config；传给 MinerU -s",
    )
    p.add_argument(
        "-e",
        "--end-page",
        type=int,
        default=None,
        help="结束页（0-based），覆盖 config；传给 MinerU -e",
    )
    args = p.parse_args()

    if args.file:
        docs, md_path = read_pdf_as_documents(
            args.file,
            start_page_id=args.start_page,
            end_page_id=args.end_page,
        )
        print(f"Markdown 输出路径: {md_path}")
        if args.parse_only:
            print(f"parse-only: {len(docs)} chunks, 首条 question={docs[0].metadata.get('question', '')[:80]!r}")
            sys.exit(0)
        append_documents_to_milvus(
            docs,
            replace_same_source=not args.no_replace,
            source_for_replace=os.path.abspath(args.file),
        )
    else:
        if args.parse_only:
            print("parse-only 需要指定 --file")
            sys.exit(2)
        ingest_pdfs_from_config(replace_same_source=not args.no_replace)

    print("完成")
