# RAGAS 评测报告

## 全局指标
- `faithfulness`: mean=0.8260, std=0.0534, p50=0.8050, p90=0.8895
- `answer_relevancy`: mean=0.8180, std=0.0930, p50=0.8181, p90=0.9257
- `context_precision`: mean=0.7300, std=0.0564, p50=0.7047, p90=0.7980
- `context_recall`: mean=0.7672, std=0.0769, p50=0.7971, p90=0.8448
+-## 门槛检查
- `faithfulness`: mean=0.8260, threshold=0.75, status=PASS
- `answer_relevancy`: mean=0.8180, threshold=0.80, status=PASS
- `context_precision`: mean=0.7300, threshold=0.70, status=PASS
- `context_recall`: mean=0.7672, threshold=0.70, status=PASS

## 产物路径
- 明细：`details.csv`
- 汇总：`metrics.json`
- 低分样本：`low_score_samples.csv`
