import json

import pytest

from fc_eval.dataset import DatasetError, load_dataset, write_dataset


def valid_case(case_id="case-1"):
    return {
        "id": case_id,
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


def test_load_and_write_dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    write_dataset([valid_case()], path)
    cases = load_dataset(path)
    assert cases[0].id == "case-1"


def test_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    item = json.dumps(valid_case())
    path.write_text(f"{item}\n{item}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="Duplicate"):
        load_dataset(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "empty"),
        ('{"id":"broken"}\n', "Invalid case"),
        (
            json.dumps(
                {
                    **valid_case(),
                    "available_tools": ["get_weather", "missing_tool"],
                }
            )
            + "\n",
            "Unknown tools",
        ),
    ],
)
def test_dataset_validation_errors(tmp_path, content, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(path)


def test_dataset_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "none.jsonl")


def test_behavior_expectation_mismatch(tmp_path):
    item = valid_case()
    item["expected_behavior"] = "no_call"
    path = tmp_path / "mismatch.jsonl"
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="must not contain"):
        load_dataset(path)
