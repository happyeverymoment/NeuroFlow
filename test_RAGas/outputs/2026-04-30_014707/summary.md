# RAGAS 评测报告

## 全局指标
- `faithfulness`: mean=0.1247, std=0.2629, p50=0.0000, p90=0.1933
- `answer_relevancy`: mean=0.4121, std=0.2137, p50=0.4187, p90=0.7173
- `context_precision`: mean=0.6923, std=0.4615, p50=1.0000, p90=1.0000
- `context_recall`: mean=0.2372, std=0.3608, p50=0.0000, p90=0.9000
- `answer_correctness`: mean=0.5798, std=0.2635, p50=0.4488, p90=0.9209
- `answer_similarity`: mean=0.6308, std=0.2460, p50=0.6986, p90=0.9151

## 门槛检查
- `faithfulness`: mean=0.1247, threshold=0.75, status=FAIL
- `answer_relevancy`: mean=0.4121, threshold=0.80, status=FAIL
- `context_precision`: mean=0.6923, threshold=0.70, status=FAIL
- `context_recall`: mean=0.2372, threshold=0.70, status=FAIL

## 产物路径
- 明细：`details.csv`
- 汇总：`metrics.json`
- 低分样本：`low_score_samples.csv`
