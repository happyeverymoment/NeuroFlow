import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import yaml

from report_generator import save_reports


SUPPORTED_METRICS = {
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "answer_similarity",
}


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def call_rag_api(question: str, api_cfg: Dict) -> Tuple[str, List[str]]:
    payload = dict(api_cfg.get("extra_payload", {}))
    payload[api_cfg.get("question_field", "question")] = question
    headers = api_cfg.get("headers", {})

    retries = int(api_cfg.get("retries", 2))
    timeout = int(api_cfg.get("timeout_sec", 20))
    method = api_cfg.get("method", "POST").upper()
    url = api_cfg["url"]

    answer_key = api_cfg.get("response_answer_field", "answer")
    contexts_key = api_cfg.get("response_contexts_field", "contexts")
    response_format = str(api_cfg.get("response_format", "json")).lower()
    last_error = None

    for _ in range(retries + 1):
        try:
            if method == "GET":
                response = requests.get(url, params=payload, headers=headers, timeout=timeout, stream=response_format == "ndjson")
            else:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout, stream=response_format == "ndjson")
            response.raise_for_status()

            if response_format == "ndjson":
                answer_parts: List[str] = []
                contexts: List[str] = []
                final_answer = ""
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    event = json.loads(raw_line)
                    event_type = event.get("type")
                    if event_type == "delta":
                        answer_parts.append(str(event.get("text", "")))
                    elif event_type == "done":
                        final_answer = str(event.get("answer", "")).strip()
                    elif event_type == "meta":
                        intent = event.get("intent", {})
                        if isinstance(intent, dict):
                            reason = intent.get("reason_detail")
                            if reason:
                                contexts.append(f"intent_reason: {reason}")

                answer = final_answer if final_answer else "".join(answer_parts)
            else:
                data = response.json()
                answer = data.get(answer_key, "")
                contexts = data.get(contexts_key, [])

            if isinstance(contexts, str):
                contexts = [contexts]
            if not isinstance(contexts, list):
                contexts = [str(contexts)]
            return str(answer), [str(c) for c in contexts]
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
    raise RuntimeError(f"RAG 接口请求失败: {last_error}") from last_error


def mock_rag_result(question: str, ground_truth: str) -> Tuple[str, List[str]]:
    if ground_truth and str(ground_truth).strip():
        answer = str(ground_truth)
    else:
        answer = f"Mock answer for: {question}"
    contexts = [
        f"这是与问题相关的模拟上下文: {question}",
        f"这是用于验证评测流程的补充上下文。参考答案: {answer}",
    ]
    return answer, contexts


def enrich_dataset(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    mode = config["run"].get("mode", "mock")
    rows = []
    for _, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        ground_truth = str(row.get("ground_truth", "")).strip()
        if not question:
            continue

        if mode == "api":
            answer, contexts = call_rag_api(question, config["api"])
        else:
            answer, contexts = mock_rag_result(question, ground_truth)

        rows.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "category": str(row.get("category", "")),
                "difficulty": str(row.get("difficulty", "")),
                "answer": answer,
                "contexts": contexts,
            }
        )
    return pd.DataFrame(rows)


def build_ragas_judge_llm_and_embeddings(config: Dict) -> Tuple[Any, Any]:
    """为 Ragas 提供评测用 LLM 与向量模型，避免其内部默认调用无 Key 的 OpenAI()。"""
    judge = (config.get("ragas") or {}).get("judge") or {}
    provider = str(judge.get("provider", "dashscope")).lower()
    api_key = judge.get("api_key") or judge.get("openai_api_key") or judge.get("dashscope_api_key")

    if provider == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "RAGAS 评测需要 OpenAI Key：请设置环境变量 OPENAI_API_KEY，"
                "或在 config.yaml 的 ragas.judge.api_key 中配置（不建议把密钥提交到仓库）。"
            )
        llm_model = judge.get("llm_model", "gpt-4o-mini")
        emb_model = judge.get("embedding_model", "text-embedding-3-small")
        llm = ChatOpenAI(model=llm_model, temperature=0, api_key=key)
        embeddings = OpenAIEmbeddings(model=emb_model, api_key=key)
        return llm, embeddings

    if provider == "dashscope":
        from langchain_community.chat_models import ChatTongyi
        from langchain_community.embeddings import DashScopeEmbeddings

        key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError(
                "RAGAS 评测需要 DashScope Key：请设置环境变量 DASHSCOPE_API_KEY，"
                "或在 config.yaml 的 ragas.judge.api_key 中配置。"
            )
        llm_model = judge.get("llm_model", "qwen-turbo")
        emb_model = judge.get("embedding_model", "text-embedding-v1")
        llm = ChatTongyi(model=llm_model, api_key=key)
        embeddings = DashScopeEmbeddings(model=emb_model, dashscope_api_key=key)
        return llm, embeddings

    raise ValueError(f"ragas.judge.provider 不支持: {provider}，请使用 openai 或 dashscope。")


def run_ragas_eval(eval_df: pd.DataFrame, metric_names: List[str], config: Dict) -> pd.DataFrame:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            answer_similarity,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            f"无法导入 ragas/datasets（当前 Python: {sys.executable}）。"
            "请确认已执行 `conda activate hfs`，并在该环境中安装: pip install ragas datasets"
        ) from exc

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
        "answer_similarity": answer_similarity,
    }
    selected = [metric_map[m] for m in metric_names if m in metric_map]
    if not selected:
        raise ValueError("未选择有效 RAGAS 指标。")

    llm, embeddings = build_ragas_judge_llm_and_embeddings(config)

    ds = Dataset.from_dict(
        {
            "question": eval_df["question"].tolist(),
            "answer": eval_df["answer"].tolist(),
            "contexts": eval_df["contexts"].tolist(),
            "ground_truth": eval_df["ground_truth"].tolist(),
        }
    )

    result = evaluate(ds, metrics=selected, llm=llm, embeddings=embeddings)
    detail_df = result.to_pandas()
    merged = pd.concat([eval_df.reset_index(drop=True), detail_df.reset_index(drop=True)], axis=1)
    return merged


def run_baseline_eval(eval_df: pd.DataFrame) -> pd.DataFrame:
    scored = eval_df.copy()
    random.seed(42)
    scored["faithfulness"] = [random.uniform(0.75, 0.95) for _ in range(len(scored))]
    scored["answer_relevancy"] = [random.uniform(0.70, 0.98) for _ in range(len(scored))]
    scored["context_precision"] = [random.uniform(0.65, 0.90) for _ in range(len(scored))]
    scored["context_recall"] = [random.uniform(0.65, 0.92) for _ in range(len(scored))]
    scored["eval_engine"] = "baseline_random_DO_NOT_TRUST"
    return scored


def resolve_eval_profile(config: Dict, cli_profile: str | None) -> Tuple[str, Dict]:
    profiles = config.get("evaluation_profiles", {})
    if not profiles:
        # backward compatibility: old single evaluation block
        legacy = config.get("evaluation", {})
        return "legacy", legacy

    profile_name = cli_profile or config.get("run", {}).get("eval_profile", "fast_eval")
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"未知评测档位: {profile_name}。可选档位: {available}")
    return profile_name, profiles[profile_name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for RAG system.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config yaml.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Evaluation profile name, e.g. fast_eval/full_eval. Overrides run.eval_profile.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    dataset_path = (config_path.parent / config["run"]["dataset_path"]).resolve()
    output_root = (config_path.parent / config["run"]["output_dir"]).resolve()
    output_dir = output_root / datetime.now().strftime("%Y-%m-%d_%H%M%S")

    df = pd.read_csv(dataset_path)
    if "question" not in df.columns:
        raise ValueError("评测数据必须包含 question 列。")

    eval_df = enrich_dataset(df, config)
    profile_name, profile_cfg = resolve_eval_profile(config, args.profile)
    metric_names = [m for m in profile_cfg.get("metrics", []) if m in SUPPORTED_METRICS]
    print(f"[INFO] 当前评测档位: {profile_name}")
    print(f"[INFO] 当前指标: {metric_names}")

    if config["run"].get("enable_ragas", True):
        print(f"[INFO] 当前 Python 解释器: {sys.executable}")
        allow_fallback = config["run"].get("allow_baseline_fallback", True)
        try:
            details_df = run_ragas_eval(eval_df, metric_names, config)
            details_df["eval_engine"] = "ragas"
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] RAGAS 评测失败: {exc!r}")
            if not allow_fallback:
                raise
            print("[WARN] 已启用 allow_baseline_fallback，降级到随机 baseline（分数与答案质量无关，请勿采信）。")
            details_df = run_baseline_eval(eval_df)
    else:
        details_df = run_baseline_eval(eval_df)

    details_df["contexts"] = details_df["contexts"].apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else str(x)
    )
    payload = save_reports(
        output_dir=output_dir,
        details_df=details_df,
        thresholds=profile_cfg.get("thresholds", {}),
    )

    print(f"[OK] 评测完成，输出目录: {output_dir}")
    print(f"[OK] 全局指标: {json.dumps(payload['summary'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
