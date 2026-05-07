# RAGAS 评测报告

## 全局指标
- `faithfulness`: mean=0.0000, std=0.0000, p50=0.0000, p90=0.0000
- `answer_relevancy`: mean=0.3493, std=0.2054, p50=0.4413, p90=0.5394
- `context_precision`: mean=0.4000, std=0.4899, p50=0.0000, p90=1.0000
- `context_recall`: mean=0.4000, std=0.4899, p50=0.0000, p90=1.0000
- `answer_correctness`: mean=0.1225, std=0.0599, p50=0.1349, p90=0.1779
- `answer_similarity`: mean=0.4901, std=0.2394, p50=0.5396, p90=0.7115

## 门槛检查
- `faithfulness`: mean=0.0000, threshold=0.75, status=FAIL
- `answer_relevancy`: mean=0.3493, threshold=0.80, status=FAIL
- `context_precision`: mean=0.4000, threshold=0.70, status=FAIL
- `context_recall`: mean=0.4000, threshold=0.70, status=FAIL

## 产物路径
- 明细：`details.csv`
- 汇总：`metrics.json`
- 低分样本：`low_score_samples.csv`
