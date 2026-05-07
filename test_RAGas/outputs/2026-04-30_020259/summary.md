# RAGAS 评测报告

## 全局指标
- `answer_correctness`: mean=0.5751, std=0.2735, p50=0.7179, p90=0.8507
- `answer_relevancy`: mean=0.3906, std=0.2117, p50=0.4306, p90=0.6859
- `answer_similarity`: mean=0.6341, std=0.2482, p50=0.6986, p90=0.9153

## 门槛检查
- `answer_correctness`: mean=0.5751, threshold=0.65, status=FAIL
- `answer_relevancy`: mean=0.3906, threshold=0.75, status=FAIL
- `answer_similarity`: mean=0.6341, threshold=0.70, status=FAIL

## 产物路径
- 明细：`details.csv`
- 汇总：`metrics.json`
- 低分样本：`low_score_samples.csv`
