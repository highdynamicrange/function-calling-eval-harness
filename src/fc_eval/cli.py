from __future__ import annotations

from pathlib import Path

import typer

from fc_eval.dataset import DatasetError, load_dataset
from fc_eval.report import render_report
from fc_eval.run import load_run, run_evaluation

app = typer.Typer(
    help="Function Calling 与 Tool Use 离线评测工具。",
    no_args_is_help=True,
)


@app.command()
def validate(
    dataset: Path = typer.Option(..., exists=True, dir_okay=False, help="JSONL 数据集路径。"),
) -> None:
    """校验数据集格式、工具名称和重复 ID。"""
    try:
        cases = load_dataset(dataset)
    except DatasetError as exc:
        typer.echo(f"校验失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"校验通过：{len(cases)} 条样本")


@app.command()
def run(
    backend: str = typer.Option("mock", help="mock 或 openai-compatible。"),
    dataset: Path = typer.Option(..., exists=True, dir_okay=False, help="JSONL 数据集路径。"),
    output_dir: Path = typer.Option(Path("artifacts/runs"), help="运行记录目录。"),
) -> None:
    """执行评测并保存 JSON 运行记录。"""
    try:
        artifact, output = run_evaluation(dataset, backend, output_dir)
    except (DatasetError, ValueError) as exc:
        typer.echo(f"运行失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"运行完成：{len(artifact.results)} 条样本")
    typer.echo(f"任务完成率：{artifact.metrics['task_completion_rate'] * 100:.1f}%")
    typer.echo(f"运行记录：{output}")


@app.command()
def report(
    run_file: Path = typer.Option(
        ...,
        "--run",
        exists=True,
        dir_okay=False,
        help="评测运行 JSON 文件。",
    ),
    output: Path | None = typer.Option(None, help="HTML 报告路径。"),
) -> None:
    """从运行记录生成自包含 HTML 报告。"""
    artifact = load_run(run_file)
    destination = output or Path("artifacts/reports") / f"{artifact.run_id}.html"
    render_report(artifact, destination)
    typer.echo(f"报告已生成：{destination}")


@app.command()
def demo(
    dataset: Path = typer.Option(Path("datasets/core.jsonl"), help="Demo 数据集路径。"),
) -> None:
    """一条命令运行离线评测并生成 HTML 报告。"""
    artifact, run_path = run_evaluation(dataset, "mock", "artifacts/runs")
    report_path = Path("artifacts/reports") / f"{artifact.run_id}.html"
    render_report(artifact, report_path)
    typer.echo(f"Demo 完成：{len(artifact.results)} 条样本")
    typer.echo(f"任务完成率：{artifact.metrics['task_completion_rate'] * 100:.1f}%")
    typer.echo(f"运行记录：{run_path}")
    typer.echo(f"HTML 报告：{report_path}")

