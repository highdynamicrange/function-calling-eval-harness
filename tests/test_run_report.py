import json
from pathlib import Path

import pytest

from fc_eval.adapters import MockModelAdapter
from fc_eval.report import render_report
from fc_eval.run import build_adapter, load_run, run_evaluation


def write_dataset(path: Path):
    item = {
        "id": "run-1",
        "messages": [{"role": "user", "content": "北京天气"}],
        "available_tools": ["get_weather"],
        "expected_calls": [
            {"name": "get_weather", "arguments": {"location": "北京", "date": "today"}}
        ],
        "order_mode": "strict",
        "expected_behavior": "call",
        "expected_final_state": {},
        "tags": ["weather"],
        "difficulty": "easy",
    }
    path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")


def test_run_artifact_round_trip_and_report(tmp_path):
    dataset = tmp_path / "data.jsonl"
    write_dataset(dataset)
    artifact, path = run_evaluation(dataset, output_dir=tmp_path / "runs")
    loaded = load_run(path)
    assert loaded.run_id == artifact.run_id
    output = render_report(loaded, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")
    assert "Function Calling 评测报告" in html
    assert "run-1" in html
    assert "https://" not in html


def test_build_adapter():
    assert isinstance(build_adapter("mock"), MockModelAdapter)
    with pytest.raises(ValueError, match="Unsupported"):
        build_adapter("missing")

