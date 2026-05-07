# RAGAS 评测报告

## 全局指标
- `answer_correctness`: mean=0.5870, std=0.3163, p50=0.5891, p90=0.9628
- `answer_relevancy`: mean=0.3499, std=0.2579, p50=0.4316, p90=0.6815

## 门槛检查
- `answer_correctness`: mean=0.5870, threshold=0.65, status=FAIL
- `answer_relevancy`: mean=0.3499, threshold=0.75, status=FAIL

## 产物路径
- 明细：`details.csv`
- 汇总：`metrics.json`
- 低分样本：`low_score_samples.csv`
