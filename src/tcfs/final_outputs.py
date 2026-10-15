"""Assemble audited one-shot outputs for cross-method final evaluation."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from tcfs.temporal_split import PROTOCOL_VERSION, TemporalLeakageError


def assemble_final_evaluation_inputs(
    final_root: str | Path,
    experiment_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine complete final outputs, preserving classifier identity.

    Primary LightGBM methods keep their method name. Secondary logistic models
    receive a ``__logistic_regression`` suffix so both classifiers can coexist
    without violating the evaluation table's one-row-per-method invariant.
    """
    final_root = Path(final_root)
    experiment_root = Path(experiment_root)
    prediction_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    for manifest_path in sorted(final_root.glob("*/*/final_evaluation_manifest.json")):
        output_dir = manifest_path.parent
        file_type = output_dir.parent.name.upper()
        experiment_id = output_dir.name
        experiment_dir = experiment_root / file_type / experiment_id
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        configuration = json.loads(
            (experiment_dir / "configuration.json").read_text(encoding="utf-8")
        )
        _validate_manifest(manifest, output_dir, file_type)
        if configuration.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("final configuration protocol version mismatch")
        classifier = str(configuration["classifier"])
        base_method = str(configuration["method"])
        method = (
            base_method
            if classifier == "lightgbm"
            else f"{base_method}__{classifier}"
        )
        predictions = pd.read_parquet(output_dir / "test_predictions.parquet")
        _validate_predictions(predictions, file_type)
        predictions = predictions.copy()
        predictions["method"] = method
        predictions["classifier"] = classifier
        prediction_frames.append(predictions)

        metrics = pd.read_csv(output_dir / "test_metrics.csv")
        resource = metrics.set_index("metric")["estimate"]
        required_resources = {
            "feature_count",
            "model_size_bytes",
            "inference_latency_ms_per_sample",
        }
        if missing := required_resources - set(resource.index):
            raise ValueError(f"final metrics are missing resources: {sorted(missing)}")
        metadata_rows.append(
            {
                "file_type": file_type,
                "method": method,
                "classifier": classifier,
                "feature_count": int(resource["feature_count"]),
                "model_size_bytes": int(resource["model_size_bytes"]),
                "inference_latency_ms_per_sample": float(
                    resource["inference_latency_ms_per_sample"]
                ),
                "experiment_id": experiment_id,
                "completed_at_utc": manifest["completed_at_utc"],
            }
        )
    if not prediction_frames:
        raise FileNotFoundError("no complete final-evaluation outputs were found")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows)
    if metadata.duplicated(["file_type", "method"]).any():
        raise ValueError("duplicate final format/method outputs")
    expected_formats = {"APK", "ELF", "PDF"}
    if set(metadata["file_type"]) != expected_formats:
        raise ValueError("final outputs must include APK, ELF, and PDF")
    _validate_alignment(predictions)
    return predictions, metadata.sort_values(
        ["file_type", "method"], kind="stable", ignore_index=True
    )


def _validate_manifest(manifest: dict, output_dir: Path, file_type: str) -> None:
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete final evaluation: {output_dir}")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("final manifest protocol version mismatch")
    if str(manifest.get("file_type", "")).upper() != file_type:
        raise ValueError("final manifest file type disagrees with its path")
    hashes = manifest.get("output_sha256")
    required = {
        "test_predictions.parquet",
        "test_metrics.csv",
        "refitted_model.joblib",
    }
    if not isinstance(hashes, dict) or set(hashes) != required:
        raise ValueError("final output checksums are incomplete")
    for name in required:
        path = output_dir / name
        if not path.is_file() or _sha256_file(path) != hashes[name]:
            raise ValueError(f"final output checksum mismatch: {path}")


def _validate_predictions(frame: pd.DataFrame, file_type: str) -> None:
    required = {
        "row_id",
        "file_type",
        "protocol_week",
        "protocol_role",
        "y_true",
        "probability",
        "prediction",
    }
    if missing := required - set(frame):
        raise ValueError(f"final predictions are missing columns: {sorted(missing)}")
    if set(frame["file_type"].astype(str).str.upper()) != {file_type}:
        raise ValueError("prediction file type mismatch")
    if set(frame["protocol_role"].astype(str)) != {"sealed_test"}:
        raise TemporalLeakageError("final predictions must contain sealed-test rows")
    weeks = pd.to_numeric(frame["protocol_week"], errors="raise")
    if not weeks.between(53, 64).all() or set(weeks) != set(range(53, 65)):
        raise TemporalLeakageError("final predictions must cover weeks 53-64")
    if frame["row_id"].duplicated().any():
        raise ValueError("duplicate row IDs in one final prediction output")


def _validate_alignment(predictions: pd.DataFrame) -> None:
    for file_type, format_frame in predictions.groupby("file_type", sort=True):
        expected_rows = None
        expected_labels = None
        for _, group in format_frame.groupby("method", sort=True):
            ordered = group.sort_values("row_id", kind="stable")
            rows = ordered["row_id"].to_numpy()
            labels = ordered["y_true"].to_numpy()
            if expected_rows is None:
                expected_rows, expected_labels = rows, labels
            elif (
                len(rows) != len(expected_rows)
                or not (rows == expected_rows).all()
                or not (labels == expected_labels).all()
            ):
                raise ValueError(f"unaligned final methods for {file_type}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
