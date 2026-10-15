import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from tcfs.final_outputs import assemble_final_evaluation_inputs
from tcfs.temporal_split import PROTOCOL_VERSION


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_result(
    final_root: Path,
    experiment_root: Path,
    file_type: str,
    experiment_id: str,
    classifier: str,
) -> None:
    output = final_root / file_type / experiment_id
    experiment = experiment_root / file_type / experiment_id
    output.mkdir(parents=True)
    experiment.mkdir(parents=True)
    method = experiment_id.split("__")[0]
    (experiment / "configuration.json").write_text(
        json.dumps(
            {
                "method": method,
                "classifier": classifier,
                "protocol_version": PROTOCOL_VERSION,
            }
        ),
        encoding="utf-8",
    )
    predictions = pd.DataFrame(
        {
            "row_id": range(12),
            "file_type": file_type,
            "protocol_week": range(53, 65),
            "protocol_role": "sealed_test",
            "y_true": [0, 1] * 6,
            "probability": [0.1, 0.9] * 6,
            "prediction": [0, 1] * 6,
        }
    )
    predictions.to_parquet(output / "test_predictions.parquet", index=False)
    pd.DataFrame(
        {
            "metric": [
                "feature_count",
                "model_size_bytes",
                "inference_latency_ms_per_sample",
            ],
            "estimate": [10, 500, 0.2],
        }
    ).to_csv(output / "test_metrics.csv", index=False)
    (output / "refitted_model.joblib").write_bytes(b"model")
    files = (
        "test_predictions.parquet",
        "test_metrics.csv",
        "refitted_model.joblib",
    )
    manifest = {
        "status": "complete",
        "file_type": file_type,
        "protocol_version": PROTOCOL_VERSION,
        "completed_at_utc": "2026-01-01T00:00:00+00:00",
        "output_sha256": {name: _digest(output / name) for name in files},
    }
    (output / "final_evaluation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_assembles_primary_and_secondary_final_outputs(tmp_path: Path) -> None:
    final_root = tmp_path / "final"
    experiment_root = tmp_path / "experiments"
    for file_type in ("APK", "ELF", "PDF"):
        _write_result(
            final_root,
            experiment_root,
            file_type,
            "tcfs__lightgbm",
            "lightgbm",
        )
        _write_result(
            final_root,
            experiment_root,
            file_type,
            "tcfs__logistic_regression",
            "logistic_regression",
        )

    predictions, metadata = assemble_final_evaluation_inputs(
        final_root, experiment_root
    )

    assert set(predictions["method"]) == {
        "tcfs",
        "tcfs__logistic_regression",
    }
    assert len(predictions) == 72
    assert len(metadata) == 6
    assert metadata["feature_count"].eq(10).all()
