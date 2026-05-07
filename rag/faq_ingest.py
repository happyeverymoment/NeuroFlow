"""
FAQ 文件处理与 Milvus 导入模块
负责：1）从 Excel / rag_faqdata 目录解析 FAQ；2）构建向量并写入 Milvus，返回检索器
支持多表：不同 xlsx 列名不同时，统一为「所有出现过的列」；某表缺少某列则该项留空。
"""

import json
import os
import re
import pandas as pd
from typing import List, Optional, Set, Tuple

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document

# 从统一配置加载（config.json），可被环境变量或 rag_server 传入覆盖
try:
    from .config_loader import get_milvus, get_embedding, get_faq
except ImportError:
    from config_loader import get_milvus, get_embedding, get_faq

_milvus = get_milvus()
_embedding = get_embedding()
_faq_paths = get_faq()

MILVUS_HOST = _milvus["host"]
MILVUS_PORT = _milvus["port"]
COLLECTION_NAME = _milvus["collection_name"]
EMBEDDING_API_KEY = _embedding["api_key"]
EMBEDDING_MODEL = _embedding["model"]
FAQ_DATA_DIR = _faq_paths["data_dir"]
FAQ_FILE_PATH = _faq_paths["file_path"]

# 用于选取「向量化内容」的列名关键词（优先匹配）
QUESTION_COL_KEYWORDS = ["问题", "question", "Question", "Q", "q", "扩展问题"]
# 用于选取「答案」的列名关键词（供下游 answer 字段兼容）
ANSWER_COL_KEYWORDS = ["答案", "answer", "Answer", "解决方案", "solution", "Solution"]


def read_faq_excel(file_path: str) -> List[Document]:
    """
    从 Excel 读取 FAQ，返回 LangChain Document 列表。
    仅用「问题」作为向量化内容，答案放入 metadata。
    """
    print(f"📖 正在读取 FAQ 文件: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    df = pd.read_excel(file_path)
    print(f"📊 读取到 {len(df)} 行数据")
    print(f"📋 列数: {len(df.columns)}")
    print(f"📋 列名: {list(df.columns)}")
    if len(df) > 0:
        print(f"📋 前3行数据预览:")
        print(df.head(3))

    question_keywords = ["问题", "question", "Question", "Q", "q"]
    answer_keywords = ["答案", "answer", "Answer", "解决方案", "solution", "Solution"]
    question_cols: List[int] = []
    answer_col: Optional[int] = None

    for col_idx, col_name in enumerate(df.columns):
        col_str = str(col_name).lower()
        if any(kw.lower() in col_str for kw in question_keywords):
            question_cols.append(col_idx)
            print(f"✅ 找到问题列: 索引 {col_idx}, 名称 '{col_name}'")
        if any(kw.lower() in col_str for kw in answer_keywords):
            answer_col = col_idx
            print(f"✅ 找到答案列: 索引 {col_idx}, 名称 '{col_name}'")

    if not question_cols:
        question_cols = [2]
        print(f"⚠️ 未找到问题列，使用默认索引 {question_cols[0]}")
    if answer_col is None:
        answer_col = len(df.columns) - 1
        print(f"⚠️ 未找到答案列，使用默认索引 {answer_col} (最后一列)")
    if len(question_cols) > 1:
        print(f"ℹ️ 发现多个问题列，将依次尝试: {question_cols}")

    docs: List[Document] = []
    skipped_count = 0
    for idx, row in df.iterrows():
        answer = (
            str(row.iloc[answer_col]).strip()
            if len(row) > answer_col and pd.notna(row.iloc[answer_col])
            else ""
        )
        if not answer:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"⚠️ 跳过第 {idx+1} 行: 答案为空")
            continue

        question = ""
        for q_col in question_cols:
            if len(row) > q_col and pd.notna(row.iloc[q_col]):
                q_text = str(row.iloc[q_col]).strip()
                if q_text:
                    question = q_text
                    break
        if not question:
            question = answer[:50] + "..." if len(answer) > 50 else answer

        system = (
            str(row.iloc[0]).strip()
            if len(row) > 0 and pd.notna(row.iloc[0])
            else "LITA系统"
        )
        business = (
            str(row.iloc[1]).strip()
            if len(row) > 1 and pd.notna(row.iloc[1])
            else "商机"
        )

        doc = Document(
            page_content=question,
            metadata={
                "question": question,
                "answer": answer,
                "system": system,
                "business": business,
                "source": file_path,
            },
        )
        docs.append(doc)
        
        # 调试：打印前3条数据的答案
        if len(docs) <= 3:
            print(f"   📝 第 {len(docs)} 条: 问题='{question[:50]}...', 答案长度={len(answer)} 字符")

    total = len(df)
    pct = (len(docs) / total * 100) if total else 0
    print(f"\n{'='*60}")
    print(f"📊 FAQ 数据解析统计:")
    print(f"   ✅ 成功解析: {len(docs)} 条 FAQ 数据")
    print(f"   ⚠️ 跳过空数据: {skipped_count} 行")
    print(f"   📈 解析率: {pct:.1f}%")
    print(f"{'='*60}\n")
    return docs


def _union_columns_from_dfs(
    folder_path: str,
) -> Tuple[List[str], List[Tuple[str, pd.DataFrame]]]:
    """扫描目录下所有 xlsx，返回「统一列名列表」和 (文件路径, DataFrame) 列表。列名按首次出现顺序、去重后保留，规范化为 strip 后字符串。"""
    seen: Set[str] = set()
    order: List[str] = []
    files_and_dfs: List[Tuple[str, pd.DataFrame]] = []
    for name in sorted(os.listdir(folder_path)):
        if not (name.endswith(".xlsx") or name.endswith(".xls")):
            continue
        fp = os.path.join(folder_path, name)
        if not os.path.isfile(fp):
            continue
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            print(f"⚠️ 跳过无法读取的文件: {fp}，错误: {e}")
            continue
        files_and_dfs.append((fp, df))
        for col in df.columns:
            c = str(col).strip() or str(col)
            if c not in seen:
                seen.add(c)
                order.append(c)
    return order, files_and_dfs


def _row_value_for_column(row: pd.Series, col_canonical: str, df_columns: List[str]) -> str:
    """从行中按「规范列名」取字符串：先精确匹配，再按 strip 匹配。"""
    for c in df_columns:
        if (c == col_canonical or str(c).strip() == col_canonical) and c in row.index and pd.notna(row[c]):
            return str(row[c]).strip()
    return ""


def _pick_question_text(
    row: pd.Series,
    df_columns: List[str],
    question_keywords: List[str],
) -> str:
    """从一行中按「问题列优先」取一段文本作为向量化内容。"""
    def matches(col: str) -> bool:
        return any(kw.lower() in str(col).lower() for kw in question_keywords)
    # 先找名称像「问题」的列
    for col in df_columns:
        if matches(col) and col in row.index and pd.notna(row[col]):
            s = str(row[col]).strip()
            if s:
                return s
    # 再取第一个非空字符串列
    for col in df_columns:
        if col in row.index and pd.notna(row[col]):
            s = str(row[col]).strip()
            if s:
                return s
    return ""


def _normalize_col_name(col: str) -> str:
    """规范化列名：strip 并合并内部空白，便于匹配。"""
    return " ".join(str(col).strip().split())


def _pick_answer_text(
    row: pd.Series,
    df_columns: List[str],
    answer_keywords: List[str],
    question_keywords: Optional[List[str]] = None,
) -> str:
    """从一行中按「答案列优先」取一段文本，用于兼容下游 answer 字段。
    优先精确匹配"解决方案"列，然后匹配其他答案关键词列；排除问题列，不回退到其他列。
    """
    question_kw = question_keywords or QUESTION_COL_KEYWORDS

    def is_question_col(col: str) -> bool:
        """列名是否为问题列（不能当答案列用）。"""
        c = _normalize_col_name(col).lower()
        return any(kw.lower() in c for kw in question_kw)

    # 优先匹配"解决方案"列（列名规范化后等于「解决方案」）
    for col in df_columns:
        if _normalize_col_name(col) == "解决方案" and col in row.index and pd.notna(row[col]):
            s = str(row[col]).strip()
            if s:
                return s

    # 然后匹配其他包含答案关键词的列，且排除问题列、排除 pandas 无表头列（Unnamed: N）
    def is_unnamed_col(col: str) -> bool:
        return bool(re.match(r"^Unnamed\s*:\s*\d+$", str(col).strip(), re.IGNORECASE))

    def matches_answer(col: str) -> bool:
        return any(kw.lower() in str(col).lower() for kw in answer_keywords)

    for col in df_columns:
        if is_question_col(col) or is_unnamed_col(col):
            continue
        if matches_answer(col) and col in row.index and pd.notna(row[col]):
            s = str(row[col]).strip()
            if s:
                return s
    return ""


def read_faq_from_folder(folder_path: str) -> List[Document]:
    """
    从 rag_faqdata 目录（或指定目录）下所有 xlsx 读取 FAQ，统一列名后产出 Document。
    - 列名 = 所有文件中出现过的列名的并集，与各 xlsx 表头一致；
    - 某表中没有的列，该行对应项填空字符串。
    - 向量化内容：优先取名称像「问题」的列，否则取第一个非空列。
    """
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"目录不存在: {folder_path}")

    all_columns, files_and_dfs = _union_columns_from_dfs(folder_path)
    if not all_columns or not files_and_dfs:
        print(f"⚠️ 目录下未发现可读 xlsx 或列为空: {folder_path}")
        return []

    print(f"📂 数据目录: {folder_path}")
    print(f"📋 统一列名（共 {len(all_columns)} 列）: {all_columns}")
    
    # 检查是否有答案列
    answer_cols_found = [col for col in all_columns if any(kw.lower() in str(col).lower() for kw in ANSWER_COL_KEYWORDS)]
    if answer_cols_found:
        print(f"✅ 识别到答案列: {answer_cols_found}")
    else:
        print(f"⚠️ 警告: 未找到匹配答案关键词的列！关键词: {ANSWER_COL_KEYWORDS}")
        print(f"   如果您的答案列名不匹配，请修改 ANSWER_COL_KEYWORDS 或确保列名包含这些关键词")

    # 从第一个有「解决方案」列的文件确定该列的索引，供无此列名的文件按位置回退
    solution_col_index: Optional[int] = None
    for _, df in files_and_dfs:
        cols = list(df.columns)
        for i, c in enumerate(cols):
            if _normalize_col_name(c) == "解决方案":
                solution_col_index = i
                break
        if solution_col_index is not None:
            break
    if solution_col_index is not None:
        print(f"ℹ️ 「解决方案」列在首表中的索引: {solution_col_index}（无此列名的表将按该位置取答案）")
    
    docs: List[Document] = []
    skipped = 0
    for fp, df in files_and_dfs:
        print(f"📖 正在读取: {os.path.basename(fp)}，行数 {len(df)}")
        cols_this = list(df.columns)
        # 当前文件是否有名为「解决方案」的列（规范化后）
        has_solution_col = any(_normalize_col_name(c) == "解决方案" for c in cols_this)
        for idx, row in df.iterrows():
            # 按统一列名建行数据，某表没有的列填空。Milvus 字段名只能含 [a-zA-Z0-9_]，故所有列放入 row_data（JSON），仅保留 question/answer/source 为独立字段。
            row_data: dict = {}
            for col in all_columns:
                row_data[col] = _row_value_for_column(row, col, cols_this)
            # 向量化内容：问题列优先
            page_content = _pick_question_text(row, cols_this, QUESTION_COL_KEYWORDS)
            if not page_content:
                skipped += 1
                if skipped <= 5:
                    print(f"⚠️ 跳过第 {idx+1} 行（无可用问题文本）: {os.path.basename(fp)}")
                continue
            answer_text = _pick_answer_text(row, cols_this, ANSWER_COL_KEYWORDS)
            # 若无答案且当前文件没有「解决方案」列名，按首表中「解决方案」的列索引取（表头被读成 Unnamed 时仍能取到）
            if not answer_text and solution_col_index is not None and not has_solution_col and len(cols_this) > solution_col_index:
                val = row.iloc[solution_col_index]
                if pd.notna(val):
                    answer_text = str(val).strip()
            # 如果答案为空，跳过该行（避免答案和问题相同）
            if not answer_text:
                skipped += 1
                if skipped <= 5:
                    print(f"⚠️ 跳过第 {idx+1} 行（无可用答案文本）: {os.path.basename(fp)}")
                continue
            # 检查答案是否和问题相同（可能是列匹配错误）
            if answer_text == page_content:
                skipped += 1
                if skipped <= 5:
                    print(f"⚠️ 跳过第 {idx+1} 行（答案和问题相同，可能是列匹配错误）: {os.path.basename(fp)}")
                    print(f"      问题: '{page_content[:50]}...'")
                    print(f"      答案: '{answer_text[:50]}...'")
                continue
            meta = {
                "question": page_content,
                "answer": answer_text,
                "source": fp,
                "row_data": json.dumps(row_data, ensure_ascii=False),
            }
            doc = Document(page_content=page_content, metadata=meta)
            docs.append(doc)
            
            # 调试：打印前3条数据的答案
            if len(docs) <= 3:
                print(f"   📝 第 {len(docs)} 条: 问题='{page_content[:50]}...', 答案长度={len(answer_text)} 字符, 答案预览='{answer_text[:50] if answer_text else '(空)'}...'")
    total_rows = sum(len(df) for _, df in files_and_dfs)
    pct = (len(docs) / total_rows * 100) if total_rows else 0
    print(f"\n{'='*60}")
    print(f"📊 多表汇总统计:")
    print(f"   ✅ 成功解析: {len(docs)} 条")
    print(f"   ⚠️ 跳过无问题文本: {skipped} 行")
    print(f"   📈 解析率: {pct:.1f}%")
    print(f"   📋 统一列: {all_columns}")
    print(f"{'='*60}\n")
    return docs


def create_retriever_from_documents(
    file_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    *,
    milvus_host: Optional[str] = None,
    milvus_port: Optional[int] = None,
    collection_name: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    k: int = 1,
    drop_old: bool = True,
):
    """
    从 FAQ 数据构建 Milvus 向量库并返回检索器。

    Args:
        file_path: 单个 FAQ Excel 路径；与 data_dir 二选一
        data_dir: rag_faqdata 目录路径，该目录下所有 xlsx 会合并入库，列名统一为所有表的并集，缺列填空
        milvus_host: Milvus 主机
        milvus_port: Milvus 端口
        collection_name: 集合名
        embedding_api_key: DashScope API Key
        embedding_model: Embedding 模型名
        k: 检索返回条数，默认 1
        drop_old: 是否先删库再建，默认 True

    Returns:
        Milvus 检索器（VectorStoreRetriever）
    """
    if data_dir is not None and os.path.isdir(data_dir):
        docs = read_faq_from_folder(data_dir)
    elif file_path and os.path.isfile(file_path):
        docs = read_faq_excel(file_path)
    elif os.path.isdir(FAQ_DATA_DIR):
        docs = read_faq_from_folder(FAQ_DATA_DIR)
    else:
        path = file_path or FAQ_FILE_PATH
        docs = read_faq_excel(path)
    if not docs:
        raise ValueError("没有读取到有效的 FAQ 数据")

    host = milvus_host or MILVUS_HOST
    port = milvus_port or MILVUS_PORT
    coll = collection_name or COLLECTION_NAME
    api_key = embedding_api_key or EMBEDDING_API_KEY
    model = embedding_model or EMBEDDING_MODEL

    print("ℹ️ 使用问题作为向量化内容，不进行文档分割")
    print(f"✅ 准备向量化 {len(docs)} 个问题-答案对")
    print("🗄️ 正在创建 Milvus 向量存储...")

    emb_kwargs = {"model": model}
    if api_key:
        emb_kwargs["dashscope_api_key"] = api_key
    embeddings = DashScopeEmbeddings(**emb_kwargs)
    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        connection_args={"uri": f"http://{host}:{port}"},
        collection_name=coll,
        drop_old=drop_old,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    print("✅ Milvus 向量存储创建成功")
    print(f"   检索策略: 返回最相似的 {k} 个问题")
    return retriever


def get_retriever_from_existing_milvus(
    *,
    milvus_host: Optional[str] = None,
    milvus_port: Optional[int] = None,
    collection_name: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    k: int = 1,
):
    """
    直接连接已有 Milvus 集合并返回检索器，不做读文件、向量化、入库。
    集合须已通过 python -m rag.faq_ingest 或 create_retriever_from_documents 创建。
    连接参数与 embedding 与入库逻辑保持一致（MILVUS_*、COLLECTION_NAME、EMBEDDING_*）。

    Returns:
        Milvus 检索器（VectorStoreRetriever）
    """
    host = milvus_host or MILVUS_HOST
    port = milvus_port or MILVUS_PORT
    coll = collection_name or COLLECTION_NAME
    api_key = embedding_api_key or EMBEDDING_API_KEY
    model = embedding_model or EMBEDDING_MODEL

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
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_milvus_vectorstore(
    *,
    milvus_host: Optional[str] = None,
    milvus_port: Optional[int] = None,
    collection_name: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    """
    直接连接已有 Milvus 集合并返回向量存储实例（用于 similarity_search_with_score 等带分数检索）。
    参数与 get_retriever_from_existing_milvus 一致（不含 k）。
    """
    host = milvus_host or MILVUS_HOST
    port = milvus_port or MILVUS_PORT
    coll = collection_name or COLLECTION_NAME
    api_key = embedding_api_key or EMBEDDING_API_KEY
    model = embedding_model or EMBEDDING_MODEL
    emb_kwargs = {"model": model}
    if api_key:
        emb_kwargs["dashscope_api_key"] = api_key
    embeddings = DashScopeEmbeddings(**emb_kwargs)
    return Milvus(
        embedding_function=embeddings,
        collection_name=coll,
        connection_args={"uri": f"http://{host}:{port}"},
        drop_old=False,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="FAQ 单独入库到 Milvus")
    p.add_argument("--file", "-f", default=None, help="单个 FAQ Excel 路径")
    p.add_argument("--dir", "-d", default=None, dest="data_dir", help="rag_faqdata 目录路径，该目录下所有 xlsx 会统一列名后入库（缺列填空）")
    p.add_argument("--no-drop", action="store_true", help="不先删库，即 drop_old=False")
    args = p.parse_args()
    if args.data_dir:
        create_retriever_from_documents(data_dir=args.data_dir, drop_old=not args.no_drop)
    else:
        create_retriever_from_documents(file_path=args.file, drop_old=not args.no_drop)
    print("入库完成")