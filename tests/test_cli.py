import json

from typer.testing import CliRunner

from fc_eval.cli import app

runner = CliRunner()


def write_dataset(path):
    item = {
        "id": "cli-1",
        "messages": [{"role": "user", "content": "解释 JSON"}],
        "available_tools": ["web_search"],
        "expected_calls": [],
        "order_mode": "strict",
        "expected_behavior": "no_call",
        "expected_final_state": {},
        "tags": ["no_call"],
        "difficulty": "easy",
    }
    path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")


def test_validate_run_report_and_demo(tmp_path, monkeypatch):
    dataset = tmp_path / "data.jsonl"
    write_dataset(dataset)

    validated = runner.invoke(app, ["validate", "--dataset", str(dataset)])
    assert validated.exit_code == 0
    assert "1 条样本" in validated.stdout

    run_dir = tmp_path / "runs"
    executed = runner.invoke(
        app,
        ["run", "--backend", "mock", "--dataset", str(dataset), "--output-dir", str(run_dir)],
    )
    assert executed.exit_code == 0
    run_file = next(run_dir.glob("*.json"))

    report_file = tmp_path / "manual.html"
    reported = runner.invoke(
        app,
        ["report", "--run", str(run_file), "--output", str(report_file)],
    )
    assert reported.exit_code == 0
    assert report_file.exists()

    monkeypatch.chdir(tmp_path)
    demoed = runner.invoke(app, ["demo", "--dataset", str(dataset)])
    assert demoed.exit_code == 0
    assert next((tmp_path / "artifacts" / "reports").glob("*.html"))


def test_validate_and_run_errors(tmp_path):
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"bad": true}\n', encoding="utf-8")
    result = runner.invoke(app, ["validate", "--dataset", str(invalid)])
    assert result.exit_code == 1
    run_result = runner.invoke(
        app,
        ["run", "--backend", "missing", "--dataset", str(invalid)],
    )
    assert run_result.exit_code == 1


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["validate", "run", "report", "demo"]:
        assert command in result.stdout

