"""Validation-only TCFS ablation and rho-sensitivity predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from tcfs.consensus import TCFS_RHO_VALUES
from tcfs.modeling import FittedClassifier
from tcfs.temporal_split import PROTOCOL_VERSION, TemporalLeakageError

ABLATION_SETTINGS = (
    "full_tcfs",
    "without_frequency",
    "without_stability_penalty",
    "mean_rank_only",
)


def build_validation_ablation_predictions(
    X: Any,
    y: Any,
    assignments: pd.DataFrame,
    feature_names: Sequence[str],
    consensus_scores: pd.DataFrame,
    frozen_model: FittedClassifier,
    *,
    file_type: str,
    feature_budget: float,
    selected_rho: float,
) -> pd.DataFrame:
    """Refit frozen classifier templates on weeks 1-40 for Figure 4.

    Only feature ranking changes between settings. Classifier parameters, seeds,
    ensemble rule, budget, and threshold remain frozen at their validation-selected
    TCFS values. Official-test rows are rejected.
    """
    matrix, labels, names, discovery, validation = _validate_inputs(
        X, y, assignments, feature_names
    )
    file_type = str(file_type).upper()
    if file_type not in {"APK", "ELF", "PDF"}:
        raise ValueError("file_type must be APK, ELF, or PDF")
    budget = float(feature_budget)
    if not 0.0 < budget <= 1.0:
        raise ValueError("feature_budget must be within (0, 1]")
    rho = _canonical_rho(selected_rho)
    summary = _consensus_summary(consensus_scores, file_type, budget, names)

    settings: list[tuple[str, str, float | None, np.ndarray]] = []
    ablation_scores = {
        "full_tcfs": _score(summary, rho=rho, use_frequency=True),
        "without_frequency": _score(summary, rho=rho, use_frequency=False),
        "without_stability_penalty": _score(
            summary, rho=0.0, use_frequency=True
        ),
        "mean_rank_only": summary["mean_normalized_rank"].to_numpy(float),
    }
    for setting in ABLATION_SETTINGS:
        settings.append(
            (
                "ablation",
                setting,
                None,
                _rank_indices(summary, ablation_scores[setting]),
            )
        )
    for sensitivity_rho in TCFS_RHO_VALUES:
        settings.append(
            (
                "sensitivity",
                f"rho={sensitivity_rho:g}",
                sensitivity_rho,
                _rank_indices(
                    summary,
                    _score(summary, rho=sensitivity_rho, use_frequency=True),
                ),
            )
        )

    count = max(1, int(np.ceil(budget * len(names))))
    frames: list[pd.DataFrame] = []
    for analysis, setting, x_value, ranked_indices in settings:
        selected = ranked_indices[:count]
        probabilities = []
        for template in frozen_model.estimators:
            estimator = clone(template)
            estimator.fit(matrix[discovery][:, selected], labels[discovery])
            probabilities.append(
                estimator.predict_proba(matrix[validation][:, selected])[:, 1]
            )
        probability = np.mean(probabilities, axis=0)
        frames.append(
            pd.DataFrame(
                {
                    "file_type": file_type,
                    "row_id": assignments.iloc[validation]["row_id"].to_numpy(),
                    "protocol_week": assignments.iloc[validation][
                        "protocol_week"
                    ].to_numpy(),
                    "protocol_role": "validation",
                    "protocol_version": PROTOCOL_VERSION,
                    "analysis": analysis,
                    "setting": setting,
                    "x_value": x_value,
                    "feature_budget": budget,
                    "n_features": count,
                    "selected_rho": rho,
                    "classifier": frozen_model.classifier,
                    "threshold": frozen_model.threshold,
                    "y_true": labels[validation],
                    "probability": probability,
                    "prediction": (
                        probability >= frozen_model.threshold
                    ).astype(np.int8),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _validate_inputs(
    X: Any,
    y: Any,
    assignments: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    matrix = np.asarray(X)
    labels = np.asarray(y)
    names = [str(value) for value in feature_names]
    if matrix.ndim != 2 or labels.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if len(matrix) != len(labels) or len(matrix) != len(assignments):
        raise ValueError("X, y, and assignments must be row-aligned")
    if matrix.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("feature names must uniquely name every feature")
    required = {"row_id", "protocol_week", "protocol_role"}
    if missing := required - set(assignments):
        raise ValueError(f"assignments is missing columns: {sorted(missing)}")
    roles = set(assignments["protocol_role"].astype(str))
    if not roles <= {"discovery", "validation"} or roles != {
        "discovery",
        "validation",
    }:
        raise TemporalLeakageError(
            "ablation accepts discovery and validation rows only"
        )
    weeks = pd.to_numeric(assignments["protocol_week"], errors="raise")
    discovery = np.flatnonzero(assignments["protocol_role"].eq("discovery"))
    validation = np.flatnonzero(assignments["protocol_role"].eq("validation"))
    if not weeks.iloc[discovery].between(1, 40).all():
        raise TemporalLeakageError("discovery rows must be weeks 1-40")
    if not weeks.iloc[validation].between(41, 52).all():
        raise TemporalLeakageError("validation rows must be weeks 41-52")
    if set(weeks.iloc[validation]) != set(range(41, 53)):
        raise TemporalLeakageError("validation must cover weeks 41-52")
    if not np.isfinite(matrix).all() or not np.isfinite(labels).all():
        raise ValueError("features and labels must be finite")
    labels = labels.astype(np.int8)
    if set(np.unique(labels[discovery])) != {0, 1}:
        raise ValueError("discovery data require both classes")
    if set(np.unique(labels[validation])) != {0, 1}:
        raise ValueError("validation data require both classes")
    return matrix, labels, names, discovery, validation


def _consensus_summary(
    scores: pd.DataFrame,
    file_type: str,
    budget: float,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    required = {
        "file_type",
        "feature_name",
        "feature_budget",
        "rho",
        "mean_normalized_rank",
        "selection_frequency",
        "rank_variance",
    }
    if missing := required - set(scores):
        raise ValueError(f"consensus scores are missing columns: {sorted(missing)}")
    frame = scores.loc[
        scores["file_type"].astype(str).str.upper().eq(file_type)
        & np.isclose(pd.to_numeric(scores["feature_budget"]), budget)
        & np.isclose(pd.to_numeric(scores["rho"]), 0.0)
    ].copy()
    if len(frame) != len(feature_names) or set(frame["feature_name"]) != set(
        feature_names
    ):
        raise ValueError("consensus scores must contain every feature exactly once")
    for column in (
        "mean_normalized_rank",
        "selection_frequency",
        "rank_variance",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"consensus column {column} must be finite")
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    frame["feature_index"] = frame["feature_name"].map(name_to_index)
    return frame.reset_index(drop=True)


def _score(frame: pd.DataFrame, *, rho: float, use_frequency: bool) -> np.ndarray:
    score = frame["mean_normalized_rank"].to_numpy(float)
    if use_frequency:
        score = score * frame["selection_frequency"].to_numpy(float)
    return score / (1.0 + rho * frame["rank_variance"].to_numpy(float))


def _rank_indices(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    ranked = frame.assign(_score=score).sort_values(
        ["_score", "selection_frequency", "mean_normalized_rank", "feature_name"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    return ranked["feature_index"].to_numpy(dtype=int)


def _canonical_rho(value: float) -> float:
    for rho in TCFS_RHO_VALUES:
        if np.isclose(float(value), rho):
            return rho
    raise ValueError(f"rho must be one of {TCFS_RHO_VALUES}")
