# Function Calling Evaluation Harness

一个面向 Function Calling 与 Tool Use Agent 的可复现评测框架。项目默认使用确定性 Mock 模型和本地工具执行器，无需 API Key 即可运行；也可连接任意 OpenAI-compatible API。

## 能力概览

- 六类工具：天气、搜索、日历、邮件、待办、订单查询
- 六类评测场景：无需调用、单工具、多工具、澄清、异常恢复、安全确认
- 80 条人工可审查 JSONL 样本
- 调用级、执行级、任务级指标
- Bad Case 错误分类和切片分析
- JSON 运行记录与自包含 HTML 报告
- Mock 与 OpenAI-compatible 双模型适配

## 快速开始

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
uv run fc-eval validate --dataset datasets/core.jsonl
uv run fc-eval demo
```

Demo 会生成：

```text
artifacts/runs/<run-id>.json
artifacts/reports/<run-id>.html
```

HTML 文件没有外部 CDN 依赖，可以直接用浏览器打开。

## CLI

```bash
# 校验数据集
uv run fc-eval validate --dataset datasets/core.jsonl

# 使用离线 Mock 模型执行评测
uv run fc-eval run --backend mock --dataset datasets/core.jsonl

# 使用 OpenAI-compatible API
uv run fc-eval run --backend openai-compatible --dataset datasets/core.jsonl

# 从已有运行记录生成报告
uv run fc-eval report --run artifacts/runs/<run-id>.json

# 一条命令完成离线评测与报告
uv run fc-eval demo
```

## OpenAI-compatible 配置

复制 `.env.example` 中的变量到你的环境配置。程序只从环境变量读取密钥。

```bash
export FC_EVAL_API_KEY="..."
export FC_EVAL_BASE_URL="https://api.openai.com/v1"
export FC_EVAL_MODEL="your-model"
```

API Key、`.env` 和运行产物均不会提交到 Git。

## 数据格式

每行是一个独立 JSON 对象：

```json
{
  "id": "single-01",
  "messages": [{"role": "user", "content": "查询北京今天的天气。"}],
  "available_tools": ["get_weather", "web_search"],
  "expected_calls": [
    {"name": "get_weather", "arguments": {"location": "北京", "date": "today"}}
  ],
  "order_mode": "strict",
  "expected_behavior": "call",
  "expected_final_state": {},
  "tags": ["single_tool", "get_weather"],
  "difficulty": "easy"
}
```

`order_mode` 支持：

- `strict`：调用顺序必须和期望一致
- `any`：允许顺序不同，但工具及出现次数必须一致

`expected_behavior` 支持：

- `call`
- `no_call`
- `clarify`
- `recover`
- `require_confirmation`

运行以下命令可以重新生成内置数据集：

```bash
uv run python scripts/generate_dataset.py
```

## 指标

| 指标 | 说明 |
|---|---|
| Tool Selection Accuracy | 目标工具和调用顺序是否正确 |
| No-call Accuracy | 无需工具时是否避免调用 |
| Argument Exact Match | 调用参数是否整体一致 |
| Argument Field F1 | 参数字段级 Precision / Recall / F1 |
| Schema Valid Rate | 工具参数是否能通过 Pydantic / JSON Schema 校验 |
| Execution Success Rate | 实际工具调用是否成功 |
| Clarification Accuracy | 缺少必要信息时是否正确澄清 |
| Task Completion Rate | 是否完成样本定义的端到端目标 |
| Safety Violation Rate | 是否在未确认时尝试高风险操作 |
| Latency / Tool Calls | 平均时延和平均工具调用数量 |

## 架构

```text
Dataset JSONL
   │
   ▼
Dataset Loader ──► Model Adapter ──► Tool Registry / Executor
   │                    │                     │
   └────────────────────┴─────────────────────┘
                         ▼
                 Evaluation Runner
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Metrics / Slices         Error Analyzer
             └───────────┬───────────┘
                         ▼
                 JSON + HTML Report
```

主要模块：

- `models.py`：数据协议和运行产物
- `tools.py`：工具注册、Schema 和 Mock 执行
- `adapters.py`：Mock 与 OpenAI-compatible 模型适配
- `evaluator.py`：判分、指标、切片和错误分类
- `report.py`：自包含 HTML 报告
- `cli.py`：命令行入口
- `judge.py`：未来接入 rubric-based Judge 的扩展接口

## 安全边界

- 所有工具都是本地 Mock，不会发送真实邮件或创建真实日程。
- `send_email` 和 `create_calendar_event` 必须包含 `confirmed=true` 才能执行。
- 未确认的高风险操作会返回 `CONFIRMATION_REQUIRED`。
- 运行产物不会记录 API Key。

## 测试

```bash
uv run ruff check .
uv run pytest
```

测试门槛为全部通过且覆盖率不低于 85%。

## 面试演示建议

1. 用 `validate` 展示数据协议和质量控制。
2. 用 `demo` 运行 80 条离线样本。
3. 打开 HTML 报告查看总体指标、切片和失败案例。
4. 解释为什么任务完成率不能被工具选择准确率替代。
5. 展示一个未确认邮件操作如何被安全策略拦截。
6. 说明如何将真实线上 Bad Case 加入 JSONL 回归集。

