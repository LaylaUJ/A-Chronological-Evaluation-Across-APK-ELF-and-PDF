import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from tcfs.temporal_split import (
    DISCOVERY_WINDOWS,
    OFFICIAL_TEST_WEEKS,
    PROTOCOL_VERSION,
    TemporalLeakageError,
    create_temporal_assignments,
    discovery_indices,
    fit_discovery_pipeline,
    fit_window_preprocessor,
    guard_operation_indices,
    selection_indices,
    unseal_official_test,
    validation_indices,
)
from tcfs.temporal_split import TestDataSealedError as SealedTestError


class RecordingTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        self.fitted_markers_ = X[:, 0].copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.fitted_markers_ = X[:, 0].copy()
        self.classes_ = np.unique(y)
        self.feature_importances_ = np.arange(X.shape[1], dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.repeat(self.classes_[0], len(X))


def _metadata() -> pd.DataFrame:
    weeks = np.arange(64)
    return pd.DataFrame(
        {
            "row_id": np.concatenate([np.arange(52), np.arange(12)]),
            "official_split": np.where(weeks < 52, "train", "test"),
            "derived_week_id": weeks,
        }
    )


def test_protocol_has_four_disjoint_discovery_windows_and_sealed_test() -> None:
    assignments = create_temporal_assignments(_metadata())

    assert DISCOVERY_WINDOWS == {
        1: tuple(range(1, 11)),
        2: tuple(range(11, 21)),
        3: tuple(range(21, 31)),
        4: tuple(range(31, 41)),
    }
    assert assignments.loc[:39, "protocol_role"].eq("discovery").all()
    assert assignments.loc[40:51, "protocol_role"].eq("validation").all()
    assert assignments.loc[52:, "protocol_role"].eq("sealed_test").all()
    assert tuple(assignments.loc[52:, "protocol_week"]) == OFFICIAL_TEST_WEEKS


def test_preprocessor_fits_only_requested_discovery_window() -> None:
    assignments = create_temporal_assignments(_metadata())
    X = np.column_stack([np.arange(64), np.ones(64)])

    fitted = fit_window_preprocessor(RecordingTransformer(), X, assignments, 2)

    assert fitted.fitted_markers_.tolist() == list(range(10, 20))


def test_importance_pipeline_never_fits_validation_or_test() -> None:
    assignments = create_temporal_assignments(_metadata())
    X = np.column_stack([np.arange(64), np.ones(64)])
    y = np.arange(64) % 2

    fitted = fit_discovery_pipeline(
        RecordingTransformer(),
        RecordingClassifier(),
        X,
        y,
        assignments,
        4,
    )

    transformer = fitted.pipeline.named_steps["preprocess"]
    estimator = fitted.pipeline.named_steps["estimator"]
    assert transformer.fitted_markers_.tolist() == list(range(30, 40))
    assert estimator.fitted_markers_.tolist() == list(range(30, 40))
    assert fitted.feature_importance.tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    "operation",
    [
        "preprocessing",
        "discovery_importance",
        "threshold_selection",
        "hyperparameter_tuning",
    ],
)
def test_all_development_operations_reject_official_test(operation: str) -> None:
    assignments = create_temporal_assignments(_metadata())
    test_indices = np.arange(52, 64)

    with pytest.raises(TemporalLeakageError):
        guard_operation_indices(assignments, test_indices, operation=operation)


def test_importance_rejects_validation_but_selection_helpers_allow_it() -> None:
    assignments = create_temporal_assignments(_metadata())
    validation = validation_indices(assignments)

    with pytest.raises(TemporalLeakageError):
        guard_operation_indices(
            assignments,
            validation,
            operation="discovery_importance",
        )
    for operation in ("threshold_selection", "hyperparameter_tuning"):
        selected = selection_indices(assignments, operation=operation)
        assert selected.max() == 51


def test_test_data_requires_explicit_unseal_and_logs_event(tmp_path: Path) -> None:
    assignments = create_temporal_assignments(_metadata())
    log = tmp_path / "unseal.jsonl"

    with pytest.raises(SealedTestError, match="--unseal-test"):
        unseal_official_test(
            assignments,
            unseal_test=False,
            audit_log=log,
        )
    assert not log.exists()

    indices, authorization = unseal_official_test(
        assignments,
        unseal_test=True,
        audit_log=log,
    )
    event = json.loads(log.read_text(encoding="utf-8"))
    assert indices.tolist() == list(range(52, 64))
    assert authorization.protocol_version == PROTOCOL_VERSION
    assert event["protocol_version"] == PROTOCOL_VERSION
    assert event["unsealed_at_utc"] == authorization.unsealed_at_utc


def test_official_split_week_disagreement_is_rejected() -> None:
    metadata = _metadata()
    metadata.loc[52, "official_split"] = "train"

    with pytest.raises(TemporalLeakageError):
        create_temporal_assignments(metadata)


def test_discovery_indices_have_no_validation_or_test_rows() -> None:
    assignments = create_temporal_assignments(_metadata())

    for window_id in DISCOVERY_WINDOWS:
        indices = discovery_indices(assignments, window_id)
        assert assignments.iloc[indices]["protocol_role"].eq("discovery").all()
