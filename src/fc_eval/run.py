from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fc_eval.adapters import MockModelAdapter, ModelAdapter, OpenAICompatibleAdapter
from fc_eval.dataset import load_dataset
from fc_eval.evaluator import aggregate_metrics, count_errors, evaluate_dataset, slice_metrics
from fc_eval.models import RunArtifact


def build_adapter(backend: str) -> ModelAdapter:
    if backend == "mock":
        return MockModelAdapter()
    if backend == "openai-compatible":
        return OpenAICompatibleAdapter()
    raise ValueError(f"Unsupported backend: {backend}")


def run_evaluation(
    dataset_path: str | Path,
    backend: str = "mock",
    output_dir: str | Path = "artifacts/runs",
    adapter: ModelAdapter | None = None,
) -> tuple[RunArtifact, Path]:
    cases = load_dataset(dataset_path)
    results = evaluate_dataset(cases, adapter or build_adapter(backend))
    now = datetime.now(UTC)
    run_id = f"{now:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    artifact = RunArtifact(
        run_id=run_id,
        created_at=now.isoformat(),
        backend=backend,
        dataset=str(dataset_path),
        results=results,
        metrics=aggregate_metrics(results),
        slices=slice_metrics(results),
        error_counts=count_errors(results),
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{run_id}.json"
    output_path.write_text(
        json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifact, output_path


def load_run(path: str | Path) -> RunArtifact:
    return RunArtifact.model_validate_json(Path(path).read_text(encoding="utf-8"))

