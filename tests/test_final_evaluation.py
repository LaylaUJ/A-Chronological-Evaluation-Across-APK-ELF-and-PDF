import json
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tcfs.final_evaluation import run_final_evaluation, verify_final_readiness
from tcfs.modeling import FittedClassifier, freeze_experiment_artifacts
from tcfs.temporal_split import PROTOCOL_VERSION


def _final_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(21)
    weeks = np.repeat(np.arange(1, 65), 2)
    labels = np.tile([0, 1], 64)
    X = rng.normal(size=(len(labels), 5))
    X[:, 0] += labels * 1.5
    assignments = pd.DataFrame(
        {
            "row_id": np.arange(len(labels)),
            "protocol_week": weeks,
            "protocol_role": np.select(
                [weeks <= 40, weeks <= 52],
                ["discovery", "validation"],
                default="sealed_test",
            ),
            "protocol_version": PROTOCOL_VERSION,
        }
    )
    return X, labels, assignments, [f"feature_{index}" for index in range(5)]


def _frozen_experiment(tmp_path: Path) -> tuple[Path, tuple]:
    X, labels, assignments, names = _final_data()
    experiment = tmp_path / "frozen_experiment"
    experiment.mkdir()
    selected = (names[0], names[1])
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    l1_ratio=0.0,
                    random_state=7,
                ),
            ),
        ]
    )
    development = np.flatnonzero(assignments["protocol_role"] != "sealed_test")
    estimator.fit(X[development, :2], labels[development])
    model = FittedClassifier(
        estimators=(estimator,),
        feature_indices=np.array([0, 1]),
        feature_names=selected,
        threshold=0.45,
        classifier="logistic_regression",
        seeds=(7,),
    )
    configuration = {
        "experiment_id": "tcfs__logistic_regression",
        "classifier": "logistic_regression",
        "parameters_json": json.dumps({"C": 1.0}, sort_keys=True),
        "threshold": 0.45,
        "final_seeds_json": json.dumps([7]),
        "trained_through_week": 52,
        "protocol_version": PROTOCOL_VERSION,
    }
    (experiment / "configuration.json").write_text(
        json.dumps(configuration, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        {"selection_order": [1, 2], "feature_name": selected}
    ).to_csv(experiment / "selected_features.csv", index=False)
    joblib.dump(model, experiment / "model.joblib")
    freeze_experiment_artifacts(experiment)
    return experiment, (X, labels, assignments, names)


def test_final_evaluation_refits_once_and_checksums_outputs(tmp_path: Path) -> None:
    experiment, data = _frozen_experiment(tmp_path)
    output = tmp_path / "final_output"
    readiness = verify_final_readiness(experiment, output)
    X, labels, assignments, names = data

    result_path = run_final_evaluation(
        X,
        labels,
        assignments,
        names,
        readiness,
        file_type="APK",
        unseal_test=True,
        unseal_log=tmp_path / "unseal.jsonl",
        n_bootstrap=10,
        random_state=9,
    )

    assert result_path == output
    predictions = pd.read_parquet(output / "test_predictions.parquet")
    assert set(predictions["protocol_week"]) == set(range(53, 65))
    assert set(predictions["protocol_role"]) == {"sealed_test"}
    assert predictions["evaluated_at_utc"].notna().all()
    metrics = pd.read_csv(output / "test_metrics.csv")
    assert metrics["n_samples"].eq(24).all()
    assert metrics["n_positive"].eq(12).all()
    assert metrics["n_negative"].eq(12).all()
    assert metrics["evaluated_at_utc"].notna().all()

    manifest = json.loads(
        (output / "final_evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    for name, expected_hash in manifest["output_sha256"].items():
        assert sha256((output / name).read_bytes()).hexdigest() == expected_hash
    with pytest.raises(FileExistsError, match="refusing another run"):
        verify_final_readiness(experiment, output)


def test_preflight_rejects_tampered_frozen_configuration(tmp_path: Path) -> None:
    experiment, _ = _frozen_experiment(tmp_path)
    configuration_path = experiment / "configuration.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["threshold"] = 0.99
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_final_readiness(experiment, tmp_path / "final")


def test_failed_unseal_leaves_a_run_blocking_record(tmp_path: Path) -> None:
    experiment, data = _frozen_experiment(tmp_path)
    output = tmp_path / "failed_final"
    readiness = verify_final_readiness(experiment, output)
    X, labels, assignments, names = data

    with pytest.raises(PermissionError, match="--unseal-test"):
        run_final_evaluation(
            X,
            labels,
            assignments,
            names,
            readiness,
            file_type="PDF",
            unseal_test=False,
            unseal_log=tmp_path / "unseal.jsonl",
            n_bootstrap=5,
        )
    record = json.loads(
        (output / "execution_record.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert record["error_type"] == "TestDataSealedError"
    with pytest.raises(FileExistsError):
        verify_final_readiness(experiment, output)
