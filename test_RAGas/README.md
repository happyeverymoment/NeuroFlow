# test_RAGas（第一阶段最小可用版）

这个目录用于给企业客服助手（RAG）提供一套可运行的基础评测流程，优先目标是**先跑通评测链路**：

1. 读取问题集（CSV）；
2. 调用待测系统拿到 `answer + contexts`（支持 `mock/api` 两种模式）；
3. 使用 RAGAS 计算指标（需单独配置**裁判模型** API Key；失败时可降级 baseline，见下文）；
4. 输出 `csv/json/markdown` 报告。

## 目录说明

- `config.yaml`：评测配置；
- `eval_dataset_template.csv`：样例数据集模板；
- `run_ragas_eval.py`：主执行脚本；
- `report_generator.py`：报告生成工具；
- `outputs/`：每次执行的结果输出目录（按时间戳）。

## 安装依赖（建议）

在项目环境中安装：

```bash
pip install ragas datasets pyyaml pandas requests
```

> **注意**：`baseline` 为随机占位分，**与答案对错无关**，仅用于在未配置 RAGAS 时联调流程；正式评测请关闭降级或配置好裁判模型。

## 快速开始

1. 进入目录：

```bash
cd test_RAGas
```

2. 激活环境并执行评测：

```bash
conda activate hfs
python run_ragas_eval.py --config config.yaml
```

快速/完整模式切换：

```bash
# 默认读取 config.yaml 里的 run.eval_profile（默认 fast_eval）
python run_ragas_eval.py --config config.yaml

# 一条命令临时切到完整模式
python run_ragas_eval.py --config config.yaml --profile full_eval
```

## 配置说明（核心字段）

- `run.mode`：
  - `mock`：不依赖真实接口，快速联调；
  - `api`：调用真实 RAG 接口。
- `run.enable_ragas`：是否启用 RAGAS 评分。
- `run.eval_profile`：默认评测档位（`fast_eval` 或 `full_eval`）。
- `api.*`：接口地址、字段映射、超时与重试。
- `api.response_format`：
  - `json`：普通 JSON 返回；
  - `ndjson`：流式事件返回（如 `meta/delta/done`）。
- `evaluation.metrics`：启用的指标列表。
- `evaluation.thresholds`：门槛阈值。
- `ragas.judge`：RAGAS 用来打分的**裁判 LLM / 向量模型**（与被测客服接口不是一回事）。
  - `provider: dashscope`：需环境变量 `DASHSCOPE_API_KEY`（与项目里 DashScope 用法一致）。
  - `provider: openai`：需环境变量 `OPENAI_API_KEY`。
- `run.allow_baseline_fallback`：RAGAS 失败时是否降级；正式评测建议设为 `false`，避免误读随机分。
- `evaluation_profiles`：两套评测配置：
  - `fast_eval`：仅保留 `answer_correctness` + `answer_relevancy` 两个代表性指标，速度更快，适合日常回归；
  - `full_eval`：完整指标，耗时更长，适合版本发布前评估。

## 常见问题：明明已安装 ragas，为何仍失败或降级？

1. **解释器不一致**  
   必须用 `conda activate hfs` 后再执行 `python run_ragas_eval.py`。脚本会打印 `[INFO] 当前 Python 解释器: ...`，请确认路径在 `hfs` 环境内。

2. **RAGAS 0.4 需要裁判模型 Key**  
   即使已安装 `ragas`，若未传入 `llm/embeddings`，库内部会调用 `OpenAI()`；未设置 `OPENAI_API_KEY` 会在**评测阶段**报错。  
   本仓库已在 `config.yaml` 中通过 `ragas.judge` 注入 `ChatTongyi` + `DashScopeEmbeddings`（或你改为 `openai`），请配置对应环境变量。

3. **如何确认走的是真 RAGAS 而不是 baseline**  
   查看 `details.csv` 中的 `eval_engine` 列：应为 `ragas`；若为 `baseline_random_DO_NOT_TRUST` 表示仍在用占位分。

## 流式接口校验

新增了 `validate_stream_api.py` 用于校验你当前接口协议（`POST + application/x-ndjson`）：

```bash
conda activate hfs
python validate_stream_api.py --url "http://192.168.40.197:9001/api/message/stream" --message "你好"
```

校验内容包括：
- 是否包含 `meta / delta / done` 三类事件；
- `meta` 是否包含 `route/intent` 关键字段；
- `delta` 是否有 `text`；
- `done` 是否有 `answer`；
- `delta` 拼接结果与 `done.answer` 是否一致。

## 输出结果

每次评测输出到 `outputs/YYYY-MM-DD_HHMMSS/`，包含：

- `details.csv`：样本级明细（问题、答案、上下文、各指标）；
- `metrics.json`：全局和分组统计；
- `summary.md`：人读版汇总；
- `low_score_samples.csv`：低分样本（默认按 faithfulness 排序）。

## 下一步建议（第二阶段）

- 扩充数据到 50-100 条并按业务分类；
- 增加失败样本日志（接口错误、超时重试信息）；
- 增加“历史版本对比”脚本；
- 接入 CI 周期性评测与质量门禁。
