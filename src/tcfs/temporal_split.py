"""Leakage-resistant temporal protocol for EMBER2024 experiments."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

PROTOCOL_VERSION = "tcfs-temporal-v1.0"
DISCOVERY_WINDOWS = {
    1: tuple(range(1, 11)),
    2: tuple(range(11, 21)),
    3: tuple(range(21, 31)),
    4: tuple(range(31, 41)),
}
VALIDATION_WEEKS = tuple(range(41, 53))
OFFICIAL_TEST_WEEKS = tuple(range(53, 65))

Operation = Literal[
    "preprocessing",
    "discovery_importance",
    "threshold_selection",
    "hyperparameter_tuning",
    "final_test_evaluation",
]


class TemporalLeakageError(ValueError):
    """Raised when an operation attempts to use a forbidden temporal partition."""


class TestDataSealedError(PermissionError):
    """Raised when official test rows are requested without explicit unsealing."""


@dataclass(frozen=True)
class TestUnsealAuthorization:
    """Auditable authorization for one final-test access event."""

    protocol_version: str
    unsealed_at_utc: str
    reason: str
    test_rows: int


@dataclass(frozen=True)
class DiscoveryFit:
    """A fitted discovery pipeline and its provenance."""

    pipeline: Pipeline
    window_id: int
    protocol_version: str
    fitted_rows: int
    feature_importance: np.ndarray


def create_temporal_assignments(
    metadata: pd.DataFrame,
    *,
    week_column: str = "derived_week_id",
    week_is_zero_based: bool = True,
) -> pd.DataFrame:
    """Assign metadata rows to discovery, validation, or sealed-test roles."""
    required = {"row_id", "official_split", week_column}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    weeks = pd.to_numeric(metadata[week_column], errors="raise").astype("int32")
    protocol_weeks = weeks + 1 if week_is_zero_based else weeks
    if not protocol_weeks.between(1, 64).all():
        invalid = sorted(set(protocol_weeks.loc[~protocol_weeks.between(1, 64)]))
        raise ValueError(f"Weeks outside protocol range 1..64: {invalid}")

    split = metadata["official_split"].astype(str)
    invalid_train = split.eq("train") & ~protocol_weeks.between(1, 52)
    invalid_test = split.eq("test") & ~protocol_weeks.between(53, 64)
    unexpected_split = ~split.isin(["train", "test"])
    if invalid_train.any() or invalid_test.any() or unexpected_split.any():
        raise TemporalLeakageError(
            "Official split and protocol week disagree; refusing temporal assignment"
        )

    window = np.zeros(len(metadata), dtype=np.int8)
    role = np.full(len(metadata), "", dtype=object)
    for window_id, window_weeks in DISCOVERY_WINDOWS.items():
        mask = protocol_weeks.isin(window_weeks).to_numpy()
        window[mask] = window_id
        role[mask] = "discovery"
    validation_mask = protocol_weeks.isin(VALIDATION_WEEKS).to_numpy()
    test_mask = protocol_weeks.isin(OFFICIAL_TEST_WEEKS).to_numpy()
    role[validation_mask] = "validation"
    role[test_mask] = "sealed_test"
    if (role == "").any():
        raise ValueError("One or more rows were not assigned to the protocol")

    assignments = pd.DataFrame(
        {
            "row_id": metadata["row_id"].to_numpy(),
            "official_split": split.to_numpy(),
            "protocol_week": protocol_weeks.to_numpy(),
            "protocol_role": role,
            "discovery_window": pd.array(
                np.where(window == 0, pd.NA, window),
                dtype="Int8",
            ),
            "protocol_version": PROTOCOL_VERSION,
        },
        index=metadata.index,
    )
    return assignments


def discovery_indices(assignments: pd.DataFrame, window_id: int) -> np.ndarray:
    """Return positions for exactly one discovery window."""
    _validate_window_id(window_id)
    mask = assignments["protocol_role"].eq("discovery") & assignments[
        "discovery_window"
    ].eq(window_id)
    indices = np.flatnonzero(mask.to_numpy())
    if len(indices) == 0:
        raise ValueError(f"Discovery window {window_id} contains no rows")
    guard_operation_indices(
        assignments,
        indices,
        operation="discovery_importance",
    )
    return indices


def validation_indices(assignments: pd.DataFrame) -> np.ndarray:
    """Return validation positions without exposing official test rows."""
    return np.flatnonzero(assignments["protocol_role"].eq("validation").to_numpy())


def selection_indices(
    assignments: pd.DataFrame,
    *,
    operation: Literal["threshold_selection", "hyperparameter_tuning"],
) -> np.ndarray:
    """Return discovery plus validation positions for model-selection work."""
    indices = np.flatnonzero(
        assignments["protocol_role"].isin(["discovery", "validation"]).to_numpy()
    )
    guard_operation_indices(assignments, indices, operation=operation)
    return indices


def fit_window_preprocessor(
    preprocessor: BaseEstimator,
    X: np.ndarray,
    assignments: pd.DataFrame,
    window_id: int,
    y: np.ndarray | None = None,
) -> BaseEstimator:
    """Clone and fit preprocessing exclusively on one discovery window."""
    _assert_aligned(X, assignments, "features")
    if y is not None:
        _assert_aligned(y, assignments, "labels")
    indices = discovery_indices(assignments, window_id)
    fitted = clone(preprocessor)
    if y is None:
        fitted.fit(X[indices])
    else:
        fitted.fit(X[indices], y[indices])
    return fitted


def fit_discovery_pipeline(
    preprocessor: BaseEstimator | Literal["passthrough"],
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    assignments: pd.DataFrame,
    window_id: int,
) -> DiscoveryFit:
    """Fit preprocessing and feature-importance estimator on discovery only."""
    _assert_aligned(X, assignments, "features")
    _assert_aligned(y, assignments, "labels")
    indices = discovery_indices(assignments, window_id)
    pipeline = Pipeline(
        [
            (
                "preprocess",
                preprocessor if preprocessor == "passthrough" else clone(preprocessor),
            ),
            ("estimator", clone(estimator)),
        ]
    )
    pipeline.fit(X[indices], y[indices])
    importance = _extract_feature_importance(pipeline.named_steps["estimator"])
    return DiscoveryFit(
        pipeline=pipeline,
        window_id=window_id,
        protocol_version=PROTOCOL_VERSION,
        fitted_rows=len(indices),
        feature_importance=importance,
    )


def guard_operation_indices(
    assignments: pd.DataFrame,
    indices: Sequence[int] | np.ndarray,
    *,
    operation: Operation,
    authorization: TestUnsealAuthorization | None = None,
) -> None:
    """Reject validation/test leakage for a named experimental operation."""
    positions = np.asarray(indices, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if len(positions) and (positions.min() < 0 or positions.max() >= len(assignments)):
        raise IndexError("indices contain positions outside assignments")
    roles = set(assignments.iloc[positions]["protocol_role"])

    if operation in {"preprocessing", "discovery_importance"}:
        allowed = {"discovery"}
    elif operation in {"threshold_selection", "hyperparameter_tuning"}:
        allowed = {"discovery", "validation"}
    elif operation == "final_test_evaluation":
        allowed = {"sealed_test"}
        if authorization is None or authorization.protocol_version != PROTOCOL_VERSION:
            raise TestDataSealedError(
                "Official test data require a current unseal authorization"
            )
    else:  # pragma: no cover - protected by the Operation type
        raise ValueError(f"Unknown operation: {operation}")

    forbidden = roles - allowed
    if forbidden:
        raise TemporalLeakageError(
            f"{operation} cannot use protocol roles: {sorted(forbidden)}"
        )


def unseal_official_test(
    assignments: pd.DataFrame,
    *,
    unseal_test: bool,
    audit_log: str | Path,
    reason: str = "final_test_evaluation",
) -> tuple[np.ndarray, TestUnsealAuthorization]:
    """Return official-test indices only after explicit, logged authorization."""
    if not unseal_test:
        raise TestDataSealedError(
            "Official test weeks 53-64 are sealed. Pass --unseal-test only for "
            "final evaluation."
        )
    indices = np.flatnonzero(
        assignments["protocol_role"].eq("sealed_test").to_numpy()
    )
    if len(indices) == 0:
        raise ValueError("No official test rows were found")
    authorization = TestUnsealAuthorization(
        protocol_version=PROTOCOL_VERSION,
        unsealed_at_utc=datetime.now(UTC).isoformat(),
        reason=reason,
        test_rows=len(indices),
    )
    guard_operation_indices(
        assignments,
        indices,
        operation="final_test_evaluation",
        authorization=authorization,
    )
    log_path = Path(audit_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event": "official_test_unsealed",
        **authorization.__dict__,
        "test_week_start": min(OFFICIAL_TEST_WEEKS),
        "test_week_end": max(OFFICIAL_TEST_WEEKS),
    }
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    return indices, authorization


def _extract_feature_importance(estimator: BaseEstimator) -> np.ndarray:
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=np.float64)
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=np.float64)
        importance = np.abs(coefficients)
        if importance.ndim == 2:
            importance = importance.mean(axis=0)
    else:
        raise TypeError(
            "Estimator must expose feature_importances_ or coef_ after fitting"
        )
    if importance.ndim != 1:
        raise ValueError("Feature importance must be one-dimensional")
    return importance


def _validate_window_id(window_id: int) -> None:
    if window_id not in DISCOVERY_WINDOWS:
        raise ValueError(f"window_id must be one of {tuple(DISCOVERY_WINDOWS)}")


def _assert_aligned(
    values: np.ndarray,
    assignments: pd.DataFrame,
    name: str,
) -> None:
    if len(values) != len(assignments):
        raise ValueError(
            f"{name} and temporal assignments differ in row count: "
            f"{len(values)} != {len(assignments)}"
        )

