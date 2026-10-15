"""Durable per-method checkpoints for long validation searches."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from tcfs.final_evaluation import verify_final_readiness


def method_checkpoint_complete(path: str | Path, method: str) -> bool:
    """Return whether a checkpoint has complete, frozen classifier outputs."""
    root = Path(path)
    selected_path = root / "selected_configurations.csv"
    variants_path = root / "validation_variant_predictions.parquet"
    if not selected_path.is_file() or not variants_path.is_file():
        return False
    try:
        selected = pd.read_csv(selected_path)
        if (
            len(selected) != 2
            or set(selected["method"].astype(str)) != {method}
            or set(selected["classifier"].astype(str))
            != {"lightgbm", "logistic_regression"}
        ):
            return False
        for experiment_id in selected["experiment_id"].astype(str):
            verify_final_readiness(
                root / experiment_id,
                root / f".__checkpoint_verification__{experiment_id}",
            )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    return True


def merge_method_checkpoints(
    checkpoint_root: str | Path,
    output_root: str | Path,
    methods: list[str] | tuple[str, ...],
) -> None:
    """Merge complete method checkpoints into the canonical experiment root."""
    checkpoints = Path(checkpoint_root)
    output = Path(output_root)
    required_methods = tuple(str(method) for method in methods)
    if not required_methods or len(set(required_methods)) != len(required_methods):
        raise ValueError("methods must be a non-empty unique sequence")
    incomplete = [
        method
        for method in required_methods
        if not method_checkpoint_complete(checkpoints / method, method)
    ]
    if incomplete:
        raise ValueError(f"method checkpoints are incomplete: {incomplete}")
    canonical_paths = [
        output / "selected_configurations.csv",
        output / "validation_variant_predictions.parquet",
    ]
    if any(path.exists() for path in canonical_paths):
        raise FileExistsError("canonical model-selection outputs already exist")

    selected_frames = []
    prediction_frames = []
    experiment_sources: list[tuple[Path, Path]] = []
    for method in required_methods:
        checkpoint = checkpoints / method
        selected = pd.read_csv(checkpoint / "selected_configurations.csv")
        selected_frames.append(selected)
        prediction_frames.append(
            pd.read_parquet(checkpoint / "validation_variant_predictions.parquet")
        )
        for experiment_id in selected["experiment_id"].astype(str):
            source = checkpoint / experiment_id
            destination = output / experiment_id
            if destination.exists():
                raise FileExistsError(
                    f"canonical experiment already exists: {destination}"
                )
            experiment_sources.append((source, destination))

    output.mkdir(parents=True, exist_ok=True)
    for source, destination in experiment_sources:
        shutil.copytree(source, destination)
    selected = pd.concat(selected_frames, ignore_index=True).sort_values(
        ["method", "classifier"], kind="stable", ignore_index=True
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selected.to_csv(output / "selected_configurations.csv", index=False)
    predictions.to_parquet(
        output / "validation_variant_predictions.parquet", index=False
    )
    (output / "checkpoint_merge.json").write_text(
        json.dumps(
            {
                "methods": list(required_methods),
                "experiment_count": len(selected),
                "validation_prediction_rows": len(predictions),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
