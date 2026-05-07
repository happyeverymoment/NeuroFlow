import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _summary_stats(df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in metric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "p50": float(series.quantile(0.5)),
            "p90": float(series.quantile(0.9)),
        }
    return stats


def _group_stats(df: pd.DataFrame, group_col: str, metric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    if group_col not in df.columns:
        return {}
    result: Dict[str, Dict[str, float]] = {}
    grouped = df.groupby(group_col, dropna=False)
    for group_name, group_df in grouped:
        values: Dict[str, float] = {}
        for metric in metric_cols:
            if metric in group_df.columns:
                values[metric] = float(group_df[metric].mean(skipna=True))
        result[str(group_name)] = values
    return result


def _threshold_result(summary: Dict[str, Dict[str, float]], thresholds: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for metric, threshold in thresholds.items():
        mean_val = summary.get(metric, {}).get("mean")
        if mean_val is None:
            result[metric] = {"threshold": float(threshold), "mean": None, "pass": False}
            continue
        result[metric] = {
            "threshold": float(threshold),
            "mean": float(mean_val),
            "pass": bool(mean_val >= threshold),
        }
    return result


def save_reports(
    output_dir: Path,
    details_df: pd.DataFrame,
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [c for c in _numeric_columns(details_df) if c not in {"row_id"}]
    summary = _summary_stats(details_df, metric_cols)
    by_category = _group_stats(details_df, "category", metric_cols)
    by_difficulty = _group_stats(details_df, "difficulty", metric_cols)
    threshold_check = _threshold_result(summary, thresholds)

    low_score = details_df.copy()
    if "faithfulness" in low_score.columns:
        low_score = low_score.sort_values(by="faithfulness", ascending=True)
    low_score_samples = low_score.head(10)

    details_path = output_dir / "details.csv"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    low_score_path = output_dir / "low_score_samples.csv"

    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    low_score_samples.to_csv(low_score_path, index=False, encoding="utf-8-sig")

    metrics_payload = {
        "summary": summary,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "threshold_check": threshold_check,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# RAGAS 评测报告",
        "",
        "## 全局指标",
    ]
    for metric, values in summary.items():
        lines.append(
            f"- `{metric}`: mean={values['mean']:.4f}, std={values['std']:.4f}, p50={values['p50']:.4f}, p90={values['p90']:.4f}"
        )

    lines += ["", "## 门槛检查"]
    for metric, check in threshold_check.items():
        mean_val = "N/A" if check["mean"] is None else f"{check['mean']:.4f}"
        status = "PASS" if check["pass"] else "FAIL"
        lines.append(f"- `{metric}`: mean={mean_val}, threshold={check['threshold']:.2f}, status={status}")

    lines += ["", "## 产物路径", f"- 明细：`{details_path.name}`", f"- 汇总：`{metrics_path.name}`", f"- 低分样本：`{low_score_path.name}`"]

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics_payload
