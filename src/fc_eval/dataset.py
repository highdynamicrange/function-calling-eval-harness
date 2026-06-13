from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from fc_eval.models import EvalCase
from fc_eval.tools import build_registry


class DatasetError(ValueError):
    pass


def load_dataset(path: str | Path) -> list[EvalCase]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetError(f"Dataset not found: {dataset_path}")

    cases: list[EvalCase] = []
    seen: set[str] = set()
    known_tools = set(build_registry())
    for line_number, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            case = EvalCase.model_validate_json(raw_line)
        except (ValidationError, ValueError) as exc:
            raise DatasetError(f"Invalid case at line {line_number}: {exc}") from exc
        if case.id in seen:
            raise DatasetError(f"Duplicate case id at line {line_number}: {case.id}")
        unknown = set(case.available_tools) - known_tools
        if unknown:
            raise DatasetError(f"Unknown tools at line {line_number}: {sorted(unknown)}")
        seen.add(case.id)
        cases.append(case)

    if not cases:
        raise DatasetError("Dataset is empty")
    return cases


def write_dataset(cases: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            validated = EvalCase.model_validate(case)
            handle.write(json.dumps(validated.model_dump(mode="json"), ensure_ascii=False) + "\n")

