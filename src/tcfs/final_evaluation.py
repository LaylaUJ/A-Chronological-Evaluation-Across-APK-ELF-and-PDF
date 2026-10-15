"""One-shot evaluation of a cryptographically frozen development pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from tcfs.evaluation import (
    PREDICTIVE_METRICS,
    compute_predictive_metrics,
    profile_model,
)
from tcfs.modeling import FittedClassifier
from tcfs.temporal_split import (
    PROTOCOL_VERSION,
    TemporalLeakageError,
    unseal_official_test,
)


@dataclass(frozen=True)
class FrozenReadiness:
    """Verified frozen inputs, safe to inspect before loading final-test labels."""

    experiment_dir: Path
    output_dir: Path
    configuration: dict[str, Any]
    freeze_manifest: dict[str, Any]
    selected_features: tuple[str, ...]
    frozen_model: FittedClassifier


def verify_final_readiness(
    experiment_dir: str | Path,
    output_dir: str | Path,
) -> FrozenReadiness:
    """Verify every freeze invariant and that no final output already exists."""
    experiment = Path(experiment_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(
            f"final-test output already exists: {output}; refusing another run"
        )
    manifest_path = experiment / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"freeze manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("frozen") is not True
        or manifest.get("protocol_frozen") is not True
    ):
        raise ValueError("experiment and protocol must be explicitly frozen")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("frozen protocol version differs from the running protocol")
    if not manifest.get("frozen_at_utc"):
        raise ValueError("freeze manifest has no frozen_at_utc timestamp")
    required_frozen_fields = {
        "selected_features",
        "classifier_hyperparameters",
        "classification_threshold",
        "random_seed_or_ensemble_rule",
    }
    if set(manifest.get("frozen_fields", [])) != required_frozen_fields:
        raise ValueError("freeze manifest does not freeze every required field")

    recorded_hashes = manifest.get("sha256")
    required_files = {
        "configuration.json",
        "selected_features.csv",
        "model.joblib",
    }
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != required_files:
        raise ValueError("configuration and pipeline SHA-256 hashes are not recorded")
    for name in sorted(required_files):
        path = experiment / name
        if not path.is_file():
            raise FileNotFoundError(f"frozen artifact is missing: {path}")
        actual = _sha256_file(path)
        if actual != recorded_hashes[name]:
            raise ValueError(f"frozen artifact checksum mismatch: {name}")

    configuration = json.loads(
        (experiment / "configuration.json").read_text(encoding="utf-8")
    )
    if configuration.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("configuration protocol version is not frozen/current")
    if configuration.get("trained_through_week") != 52:
        raise ValueError("frozen development configuration must end at week 52")
    for field in ("parameters_json", "threshold", "final_seeds_json", "classifier"):
        if field not in configuration:
            raise ValueError(f"frozen configuration is missing {field}")
    parameters = json.loads(configuration["parameters_json"])
    seeds = tuple(int(seed) for seed in json.loads(configuration["final_seeds_json"]))
    threshold = float(configuration["threshold"])
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("frozen classifier hyperparameters are invalid")
    if not seeds:
        raise ValueError("frozen seed or ensemble rule is missing")
    if not 0.0 < threshold < 1.0:
        raise ValueError("frozen threshold must be within (0, 1)")

    selected_frame = pd.read_csv(experiment / "selected_features.csv")
    if list(selected_frame.columns) != ["selection_order", "feature_name"]:
        raise ValueError("selected-feature artifact has an unexpected schema")
    if selected_frame.empty or selected_frame["feature_name"].duplicated().any():
        raise ValueError("selected features must be non-empty and unique")
    expected_order = np.arange(1, len(selected_frame) + 1)
    if not np.array_equal(selected_frame["selection_order"], expected_order):
        raise ValueError("selected-feature order is not contiguous")
    selected_features = tuple(selected_frame["feature_name"].astype(str))

    frozen_model = joblib.load(experiment / "model.joblib")
    if not isinstance(frozen_model, FittedClassifier):
        raise TypeError("frozen model is not a TCFS FittedClassifier")
    if frozen_model.feature_names != selected_features:
        raise ValueError("frozen model and selected-feature artifact disagree")
    if frozen_model.classifier != configuration["classifier"]:
        raise ValueError("frozen model and classifier configuration disagree")
    if frozen_model.seeds != seeds:
        raise ValueError("frozen model and seed/ensemble configuration disagree")
    if not np.isclose(frozen_model.threshold, threshold):
        raise ValueError("frozen model and threshold configuration disagree")
    if len(frozen_model.estimators) != len(seeds):
        raise ValueError("frozen ensemble size and seed configuration disagree")
    return FrozenReadiness(
        experiment_dir=experiment,
        output_dir=output,
        configuration=configuration,
        freeze_manifest=manifest,
        selected_features=selected_features,
        frozen_model=frozen_model,
    )


def run_final_evaluation(
    X: Any,
    y: Any,
    assignments: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...],
    readiness: FrozenReadiness,
    *,
    file_type: str,
    unseal_test: bool,
    unseal_log: str | Path,
    n_bootstrap: int = 2_000,
    random_state: int = 42,
) -> Path:
    """Refit through week 52, evaluate weeks 53-64 once, and save evidence."""
    if readiness.output_dir.exists():
        raise FileExistsError("final-test output appeared after preflight")
    file_type = file_type.upper()
    if file_type not in {"APK", "ELF", "PDF"}:
        raise ValueError("file_type must be APK, ELF, or PDF")
    _validate_protocol_assignments(assignments)
    output = readiness.output_dir
    output.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(UTC).isoformat()
    record_path = output / "execution_record.json"
    _write_json(
        record_path,
        {
            "status": "started",
            "started_at_utc": started_at,
            "protocol_version": PROTOCOL_VERSION,
            "file_type": file_type,
        },
    )

    try:
        test_indices, authorization = unseal_official_test(
            assignments,
            unseal_test=unseal_test,
            audit_log=unseal_log,
            reason="one_shot_frozen_final_evaluation",
        )
        matrix, labels, selected_indices, development_indices = _validate_final_data(
            X,
            y,
            assignments,
            feature_names,
            readiness.selected_features,
            test_indices,
        )
        estimators = []
        for frozen_estimator in readiness.frozen_model.estimators:
            estimator = clone(frozen_estimator)
            estimator.fit(
                matrix[development_indices][:, selected_indices],
                labels[development_indices],
            )
            estimators.append(estimator)
        refitted = FittedClassifier(
            estimators=tuple(estimators),
            feature_indices=selected_indices,
            feature_names=readiness.selected_features,
            threshold=float(readiness.configuration["threshold"]),
            classifier=readiness.frozen_model.classifier,
            seeds=readiness.frozen_model.seeds,
        )
        evaluated_at = datetime.now(UTC).isoformat()
        probability = refitted.predict_proba(matrix[test_indices])[:, 1]
        prediction = (probability >= refitted.threshold).astype(np.int8)
        test_labels = labels[test_indices]
        predictions = pd.DataFrame(
            {
                "row_id": assignments.iloc[test_indices]["row_id"].to_numpy(),
                "file_type": file_type,
                "protocol_week": assignments.iloc[test_indices][
                    "protocol_week"
                ].to_numpy(),
                "protocol_role": "sealed_test",
                "y_true": test_labels,
                "probability": probability,
                "prediction": prediction,
                "threshold": refitted.threshold,
                "evaluated_at_utc": evaluated_at,
            }
        )
        metric_table = _final_metric_table(
            test_labels,
            prediction,
            probability,
            refitted,
            matrix[test_indices],
            file_type=file_type,
            evaluated_at=evaluated_at,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        predictions_path = output / "test_predictions.parquet"
        metrics_path = output / "test_metrics.csv"
        model_path = output / "refitted_model.joblib"
        predictions.to_parquet(predictions_path, index=False)
        metric_table.to_csv(metrics_path, index=False)
        joblib.dump(refitted, model_path)
        completed_at = datetime.now(UTC).isoformat()
        final_manifest = {
            "status": "complete",
            "protocol_version": PROTOCOL_VERSION,
            "file_type": file_type,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "unsealed_at_utc": authorization.unsealed_at_utc,
            "bootstrap_samples": n_bootstrap,
            "bootstrap_random_state": random_state,
            "input_freeze_manifest_sha256": _sha256_file(
                readiness.experiment_dir / "freeze_manifest.json"
            ),
            "frozen_input_sha256": readiness.freeze_manifest["sha256"],
            "output_sha256": {
                predictions_path.name: _sha256_file(predictions_path),
                metrics_path.name: _sha256_file(metrics_path),
                model_path.name: _sha256_file(model_path),
            },
        }
        _write_json(output / "final_evaluation_manifest.json", final_manifest)
        _write_json(record_path, final_manifest)
        return output
    except Exception as error:
        _write_json(
            record_path,
            {
                "status": "failed",
                "started_at_utc": started_at,
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "protocol_version": PROTOCOL_VERSION,
                "file_type": file_type,
                "error_type": type(error).__name__,
            },
        )
        raise


def _final_metric_table(
    y_true: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    model: FittedClassifier,
    X_test: np.ndarray,
    *,
    file_type: str,
    evaluated_at: str,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    point = compute_predictive_metrics(y_true, prediction, probability)
    distributions = {
        metric: np.empty(n_bootstrap, dtype=float) for metric in PREDICTIVE_METRICS
    }
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(y_true == label) for label in (0, 1)]
    for bootstrap_id in range(n_bootstrap):
        sample = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices
            ]
        )
        metrics = compute_predictive_metrics(
            y_true[sample], prediction[sample], probability[sample]
        )
        for metric, value in metrics.items():
            distributions[metric][bootstrap_id] = value
    counts = {
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }
    rows = []
    for metric in PREDICTIVE_METRICS:
        low, high = np.quantile(distributions[metric], [0.025, 0.975])
        rows.append(
            {
                "file_type": file_type,
                "metric": metric,
                "estimate": point[metric],
                "ci_lower": low,
                "ci_upper": high,
                "confidence_level": 0.95,
                "bootstrap_samples": n_bootstrap,
                "bootstrap_random_state": random_state,
                "evaluated_at_utc": evaluated_at,
                **counts,
            }
        )
    resource = profile_model(model, X_test)
    for metric, value in resource.items():
        rows.append(
            {
                "file_type": file_type,
                "metric": metric,
                "estimate": value,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "confidence_level": 0.95,
                "bootstrap_samples": 0,
                "bootstrap_random_state": random_state,
                "evaluated_at_utc": evaluated_at,
                **counts,
            }
        )
    return pd.DataFrame(rows)


def _validate_protocol_assignments(assignments: pd.DataFrame) -> None:
    required = {"row_id", "protocol_week", "protocol_role", "protocol_version"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"assignments is missing columns: {sorted(missing)}")
    if set(assignments["protocol_version"].astype(str)) != {PROTOCOL_VERSION}:
        raise TemporalLeakageError("assignment protocol version is not frozen/current")
    roles = set(assignments["protocol_role"].astype(str))
    if roles != {"discovery", "validation", "sealed_test"}:
        raise TemporalLeakageError("final assignments must contain all protocol roles")
    weeks = pd.to_numeric(assignments["protocol_week"], errors="raise")
    expected = {
        "discovery": set(range(1, 41)),
        "validation": set(range(41, 53)),
        "sealed_test": set(range(53, 65)),
    }
    for role, expected_weeks in expected.items():
        actual = set(weeks.loc[assignments["protocol_role"].eq(role)])
        if actual != expected_weeks:
            raise TemporalLeakageError(
                f"{role} rows do not cover the frozen protocol weeks"
            )


def _validate_final_data(
    X: Any,
    y: Any,
    assignments: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...],
    selected_features: tuple[str, ...],
    test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(X)
    labels = np.asarray(y)
    names = [str(name) for name in feature_names]
    if matrix.ndim != 2 or labels.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if len(matrix) != len(labels) or len(matrix) != len(assignments):
        raise ValueError("X, y, and assignments must be row-aligned")
    if matrix.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("feature names must uniquely identify every column")
    missing_features = set(selected_features) - set(names)
    if missing_features:
        raise ValueError(f"frozen selected features are missing: {missing_features}")
    selected_indices = np.asarray([names.index(name) for name in selected_features])
    development = np.flatnonzero(
        assignments["protocol_role"].isin(["discovery", "validation"])
    )
    relevant = np.concatenate([development, test_indices])
    if not np.isfinite(matrix[relevant]).all() or not np.isfinite(
        labels[relevant]
    ).all():
        raise ValueError("final-evaluation features and labels must be finite")
    labels = labels.astype(np.int8)
    if set(np.unique(labels[development])) != {0, 1}:
        raise ValueError("development refit requires both classes")
    if set(np.unique(labels[test_indices])) != {0, 1}:
        raise ValueError("final test evaluation requires both classes")
    return matrix, labels, selected_indices, development


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
