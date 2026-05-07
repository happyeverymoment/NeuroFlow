# RAGAS 评测报告

## 全局指标
- `answer_correctness`: mean=0.6280, std=0.3276, p50=0.7817, p90=0.9545
- `answer_relevancy`: mean=0.3955, std=0.2584, p50=0.4902, p90=0.7105
- `answer_similarity`: mean=0.6399, std=0.2561, p50=0.6675, p90=0.9476

## 门槛检查
- `answer_correctness`: mean=0.6280, threshold=0.65, status=FAIL
- `answer_relevancy`: mean=0.3955, threshold=0.75, status=FAIL
- `answer_similarity`: mean=0.6399, threshold=0.70, status=FAIL

## 产物路径
- 明细：`details.csv`
- 汇总：`metrics.json`
- 低分样本：`low_score_samples.csv`
