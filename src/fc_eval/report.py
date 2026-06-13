from __future__ import annotations

from pathlib import Path

from jinja2 import BaseLoader, Environment, select_autoescape

from fc_eval.models import RunArtifact

TEMPLATE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Function Calling 评测报告 - {{ run.run_id }}</title>
  <style>
    :root { --ink:#162338; --muted:#667085; --blue:#2563eb; --pale:#eff6ff;
      --line:#d8e0ea; --red:#b42318; --green:#067647; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",
      "PingFang SC",sans-serif; background:#f7f9fc; color:var(--ink); }
    main { max-width:1180px; margin:0 auto; padding:36px 24px 64px; }
    header { display:flex; justify-content:space-between; gap:24px; align-items:flex-end;
      border-bottom:1px solid var(--line); padding-bottom:22px; }
    h1 { margin:0 0 8px; font-size:30px; } h2 { margin-top:38px; }
    .meta { color:var(--muted); font-size:14px; line-height:1.7; }
    .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
      gap:14px; margin-top:24px; }
    .card { background:white; border:1px solid var(--line); border-radius:12px; padding:16px; }
    .card b { display:block; font-size:26px; margin-top:7px; color:var(--blue); }
    .label { color:var(--muted); font-size:13px; }
    table { width:100%; border-collapse:collapse; background:white; font-size:14px; }
    th,td { padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; }
    th { background:#eef3f9; position:sticky; top:0; }
    .ok { color:var(--green); font-weight:700; } .bad { color:var(--red); font-weight:700; }
    .toolbar { display:flex; gap:12px; margin:14px 0; }
    input,select { padding:9px 11px; border:1px solid var(--line); border-radius:8px; background:white; }
    code { font-size:12px; white-space:pre-wrap; word-break:break-word; }
    .badge { display:inline-block; padding:3px 7px; border-radius:999px; background:var(--pale);
      color:#1d4ed8; font-size:12px; margin:2px; }
    @media (max-width:700px) { header { display:block; } .table-wrap { overflow-x:auto; } }
  </style>
</head>
<body>
<main>
  <header>
    <div><h1>Function Calling 评测报告</h1>
      <div class="meta">Run ID: {{ run.run_id }} · Backend: {{ run.backend }}</div></div>
    <div class="meta">数据集：{{ run.dataset }}<br>创建时间：{{ run.created_at }}</div>
  </header>

  <section class="cards">
    {% for key, value in run.metrics.items() %}
    <article class="card"><span class="label">{{ labels.get(key, key) }}</span>
      <b>{{ format_metric(key, value) }}</b></article>
    {% endfor %}
  </section>

  <h2>错误分布</h2>
  <div class="cards">
    {% if run.error_counts %}
      {% for key, value in run.error_counts.items() %}
      <article class="card"><span class="label">{{ key }}</span><b>{{ value }}</b></article>
      {% endfor %}
    {% else %}
      <article class="card"><span class="ok">未发现错误</span></article>
    {% endif %}
  </div>

  <h2>切片指标</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>切片</th><th>任务完成率</th><th>工具选择</th><th>参数 EM</th>
      <th>Schema</th><th>安全违规率</th><th>样本延迟</th></tr></thead>
    <tbody>{% for name, metrics in run.slices.items() %}<tr>
      <td>{{ name }}</td>
      <td>{{ pct(metrics.task_completion_rate) }}</td>
      <td>{{ pct(metrics.tool_selection_accuracy) }}</td>
      <td>{{ pct(metrics.argument_exact_match) }}</td>
      <td>{{ pct(metrics.schema_valid_rate) }}</td>
      <td>{{ pct(metrics.safety_violation_rate) }}</td>
      <td>{{ metrics.average_latency_ms }} ms</td>
    </tr>{% endfor %}</tbody>
  </table></div>

  <h2>Case 明细</h2>
  <div class="toolbar">
    <input id="search" placeholder="搜索 Case ID 或标签">
    <select id="status"><option value="">全部状态</option><option value="pass">通过</option>
      <option value="fail">失败</option></select>
  </div>
  <div class="table-wrap"><table id="cases">
    <thead><tr><th>Case</th><th>行为</th><th>标签</th><th>预期调用</th>
      <th>实际调用</th><th>结果</th><th>错误</th></tr></thead>
    <tbody>
    {% for result in run.results %}
      <tr data-status="{{ 'pass' if result.task_completed else 'fail' }}"
          data-search="{{ result.case_id }} {{ result.tags|join(' ') }}">
        <td>{{ result.case_id }}</td><td>{{ result.expected_behavior }}</td>
        <td>{% for tag in result.tags %}<span class="badge">{{ tag }}</span>{% endfor %}</td>
        <td><code>{{ result.expected_calls|tojson }}</code></td>
        <td><code>{{ result.actual_response.tool_calls|tojson }}</code></td>
        <td class="{{ 'ok' if result.task_completed else 'bad' }}">
          {{ 'PASS' if result.task_completed else 'FAIL' }}</td>
        <td>{{ result.errors|join(', ') or '-' }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table></div>
</main>
<script>
const search = document.getElementById('search');
const status = document.getElementById('status');
function filterRows() {
  const q = search.value.toLowerCase();
  for (const row of document.querySelectorAll('#cases tbody tr')) {
    const textMatch = row.dataset.search.toLowerCase().includes(q);
    const statusMatch = !status.value || row.dataset.status === status.value;
    row.style.display = textMatch && statusMatch ? '' : 'none';
  }
}
search.addEventListener('input', filterRows); status.addEventListener('change', filterRows);
</script>
</body></html>
"""


LABELS = {
    "tool_selection_accuracy": "工具选择准确率",
    "no_call_accuracy": "无需调用准确率",
    "argument_exact_match": "参数 Exact Match",
    "argument_field_f1": "参数字段 F1",
    "schema_valid_rate": "Schema 合法率",
    "execution_success_rate": "执行成功率",
    "clarification_accuracy": "澄清准确率",
    "task_completion_rate": "任务完成率",
    "safety_violation_rate": "安全违规率",
    "average_latency_ms": "平均延迟",
    "average_tool_calls": "平均工具调用数",
}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_metric(key: str, value: float) -> str:
    if key == "average_latency_ms":
        return f"{value:.2f} ms"
    if key == "average_tool_calls":
        return f"{value:.2f}"
    return _pct(value)


def render_report(run: RunArtifact, output_path: str | Path) -> Path:
    environment = Environment(
        loader=BaseLoader(),
        autoescape=select_autoescape(default=True),
    )
    template = environment.from_string(TEMPLATE)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        template.render(
            run=run.model_dump(mode="json"),
            labels=LABELS,
            pct=_pct,
            format_metric=_format_metric,
        ),
        encoding="utf-8",
    )
    return destination
