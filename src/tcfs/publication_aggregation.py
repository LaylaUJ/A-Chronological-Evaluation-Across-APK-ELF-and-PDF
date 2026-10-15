"""Validation-only aggregations used by the publication figures.

The functions in this module consume saved row-level predictions.  They never
fit models and explicitly reject discovery or official-test rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tcfs.feature_selection import SUPPORTED_FEATURE_BUDGETS
from tcfs.temporal_split import PROTOCOL_VERSION, TemporalLeakageError

FORMATS = ("APK", "ELF", "PDF")
RHO_VALUES = (0.0, 0.25, 0.5, 1.0)
PREDICTION_COLUMNS = {
    "file_type",
    "row_id",
    "y_true",
    "prediction",
    "protocol_role",
    "protocol_week",
}


def aggregate_budget_metrics(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 2_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Aggregate validation macro-F1 by method and retained-feature budget."""
    required = PREDICTION_COLUMNS | {"method", "feature_budget"}
    frame = _validate_validation_predictions(predictions, required)
    frame["method"] = frame["method"].astype(str)
    frame["feature_budget"] = pd.to_numeric(
        frame["feature_budget"], errors="raise"
    ).astype(float)
    allowed = np.asarray((*SUPPORTED_FEATURE_BUDGETS, 1.0), dtype=float)
    invalid = frame["feature_budget"].map(
        lambda value: not np.isclose(value, allowed).any()
    )
    if invalid.any():
        values = sorted(frame.loc[invalid, "feature_budget"].unique())
        raise ValueError(f"unsupported feature budgets: {values}")
    keys = ["file_type", "method", "feature_budget", "row_id"]
    if frame.duplicated(keys).any():
        raise ValueError("budget predictions contain duplicate configuration rows")

    rows: list[dict[str, object]] = []
    seed_sequence = np.random.SeedSequence(random_state)
    groups = list(frame.groupby(["file_type", "method", "feature_budget"], sort=True))
    child_seeds = seed_sequence.spawn(len(groups))
    for ((file_type, method, budget), group), child_seed in zip(
        groups, child_seeds, strict=True
    ):
        ordered = group.sort_values("row_id", kind="stable")
        labels = ordered["y_true"].to_numpy(dtype=int)
        predicted = ordered["prediction"].to_numpy(dtype=int)
        estimates = _bootstrap_macro_f1(
            labels,
            {"value": predicted},
            n_bootstrap=n_bootstrap,
            random_state=child_seed,
        )["value"]
        rows.append(
            {
                "file_type": file_type,
                "method": method,
                "feature_budget": float(budget),
                "metric": "macro_f1",
                "estimate": float(f1_score(labels, predicted, average="macro")),
                "ci_lower": float(np.quantile(estimates, 0.025)),
                "ci_upper": float(np.quantile(estimates, 0.975)),
                **_sample_counts(labels),
                "bootstrap_samples": n_bootstrap,
                "bootstrap_random_state": random_state,
                "protocol_version": _protocol_version(group),
            }
        )
    return pd.DataFrame(rows)


def aggregate_ablation_sensitivity(
    predictions: pd.DataFrame,
    *,
    ablation_baseline: str = "full_tcfs",
    n_bootstrap: int = 2_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Aggregate paired ablation differences and absolute rho sensitivity."""
    required = PREDICTION_COLUMNS | {"analysis", "setting", "x_value"}
    frame = _validate_validation_predictions(predictions, required)
    frame["analysis"] = frame["analysis"].astype(str)
    frame["setting"] = frame["setting"].astype(str)
    invalid_analysis = ~frame["analysis"].isin(["ablation", "sensitivity"])
    if invalid_analysis.any():
        values = sorted(frame.loc[invalid_analysis, "analysis"].unique())
        raise ValueError(f"unsupported analysis values: {values}")
    keys = ["file_type", "analysis", "setting", "row_id"]
    if frame.duplicated(keys).any():
        raise ValueError("analysis predictions contain duplicate setting rows")

    rows: list[dict[str, object]] = []
    seed_sequence = np.random.SeedSequence(random_state)
    format_seeds = seed_sequence.spawn(len(FORMATS))
    for file_type, format_seed in zip(FORMATS, format_seeds, strict=True):
        format_frame = frame.loc[frame["file_type"].eq(file_type)]
        if format_frame.empty:
            raise ValueError(f"analysis predictions are missing {file_type}")
        ablation = format_frame.loc[format_frame["analysis"].eq("ablation")]
        sensitivity = format_frame.loc[format_frame["analysis"].eq("sensitivity")]
        if ablation.empty or sensitivity.empty:
            raise ValueError(f"{file_type} requires ablation and sensitivity rows")

        aligned_ablation = _align_settings(ablation)
        if ablation_baseline not in aligned_ablation:
            raise ValueError(
                f"{file_type} ablation is missing baseline {ablation_baseline!r}"
            )
        labels = aligned_ablation[ablation_baseline][0]
        prediction_map = {
            setting: values[1] for setting, values in aligned_ablation.items()
        }
        bootstrap = _bootstrap_macro_f1(
            labels,
            prediction_map,
            n_bootstrap=n_bootstrap,
            random_state=format_seed.spawn(1)[0],
        )
        baseline_estimate = f1_score(
            labels, prediction_map[ablation_baseline], average="macro"
        )
        for setting in sorted(set(prediction_map) - {ablation_baseline}):
            estimate = f1_score(labels, prediction_map[setting], average="macro")
            differences = bootstrap[setting] - bootstrap[ablation_baseline]
            rows.append(
                {
                    "file_type": file_type,
                    "analysis": "ablation",
                    "setting": setting,
                    "metric": "macro_f1",
                    "estimate": float(estimate - baseline_estimate),
                    "ci_lower": float(np.quantile(differences, 0.025)),
                    "ci_upper": float(np.quantile(differences, 0.975)),
                    "x_value": np.nan,
                    **_sample_counts(labels),
                    "bootstrap_samples": n_bootstrap,
                    "bootstrap_random_state": random_state,
                    "protocol_version": _protocol_version(ablation),
                }
            )

        aligned_sensitivity = _align_settings(sensitivity)
        sensitivity_labels = next(iter(aligned_sensitivity.values()))[0]
        sensitivity_predictions = {
            setting: values[1] for setting, values in aligned_sensitivity.items()
        }
        sensitivity_bootstrap = _bootstrap_macro_f1(
            sensitivity_labels,
            sensitivity_predictions,
            n_bootstrap=n_bootstrap,
            random_state=format_seed.spawn(1)[0],
        )
        for setting in sorted(sensitivity_predictions, key=str):
            setting_rows = sensitivity.loc[sensitivity["setting"].eq(setting)]
            x_values = pd.to_numeric(setting_rows["x_value"], errors="raise").unique()
            if len(x_values) != 1 or not np.isclose(x_values[0], RHO_VALUES).any():
                raise ValueError(
                    f"{file_type} sensitivity setting {setting!r} has invalid rho"
                )
            predicted = sensitivity_predictions[setting]
            estimates = sensitivity_bootstrap[setting]
            rows.append(
                {
                    "file_type": file_type,
                    "analysis": "sensitivity",
                    "setting": setting,
                    "metric": "macro_f1",
                    "estimate": float(
                        f1_score(sensitivity_labels, predicted, average="macro")
                    ),
                    "ci_lower": float(np.quantile(estimates, 0.025)),
                    "ci_upper": float(np.quantile(estimates, 0.975)),
                    "x_value": float(x_values[0]),
                    **_sample_counts(sensitivity_labels),
                    "bootstrap_samples": n_bootstrap,
                    "bootstrap_random_state": random_state,
                    "protocol_version": _protocol_version(sensitivity),
                }
            )
    return pd.DataFrame(rows)


def save_publication_aggregate(table: pd.DataFrame, path: str | Path) -> Path:
    """Write one validated publication aggregate without overwriting it."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"publication aggregate already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return destination


def _validate_validation_predictions(
    predictions: pd.DataFrame, required: set[str]
) -> pd.DataFrame:
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction input is missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("prediction input is empty")
    frame = predictions.copy()
    frame["file_type"] = frame["file_type"].astype(str).str.upper()
    invalid_formats = sorted(set(frame["file_type"]) - set(FORMATS))
    if invalid_formats:
        raise ValueError(f"unsupported file types: {invalid_formats}")
    roles = set(frame["protocol_role"].astype(str))
    if roles != {"validation"}:
        raise TemporalLeakageError(
            "publication aggregation requires validation rows only; "
            f"found protocol roles {sorted(roles)}"
        )
    weeks = pd.to_numeric(frame["protocol_week"], errors="raise").astype(int)
    if not weeks.between(41, 52).all():
        raise TemporalLeakageError(
            "publication aggregation is restricted to validation weeks 41-52"
        )
    if "protocol_version" in frame:
        versions = set(frame["protocol_version"].astype(str))
        if versions != {PROTOCOL_VERSION}:
            raise ValueError(f"expected protocol version {PROTOCOL_VERSION}")
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    frame["prediction"] = pd.to_numeric(
        frame["prediction"], errors="raise"
    ).astype(int)
    if not frame["y_true"].isin([0, 1]).all() or not frame["prediction"].isin(
        [0, 1]
    ).all():
        raise ValueError("y_true and prediction must be binary")
    return frame


def _align_settings(
    frame: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    aligned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expected_rows: np.ndarray | None = None
    expected_labels: np.ndarray | None = None
    for setting, group in frame.groupby("setting", sort=True):
        ordered = group.sort_values("row_id", kind="stable")
        rows = ordered["row_id"].to_numpy()
        labels = ordered["y_true"].to_numpy(dtype=int)
        if expected_rows is None:
            expected_rows = rows
            expected_labels = labels
        elif not np.array_equal(rows, expected_rows) or not np.array_equal(
            labels, expected_labels
        ):
            raise ValueError("analysis settings must have aligned rows and labels")
        aligned[str(setting)] = (
            labels,
            ordered["prediction"].to_numpy(dtype=int),
        )
    if expected_labels is None or set(expected_labels) != {0, 1}:
        raise ValueError("each analysis requires both classes")
    return aligned


def _bootstrap_macro_f1(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    n_bootstrap: int,
    random_state: int | np.random.SeedSequence,
) -> dict[str, np.ndarray]:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if set(labels) != {0, 1}:
        raise ValueError("bootstrap inputs require both classes")
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(labels == value) for value in (0, 1)]
    output = {
        name: np.empty(n_bootstrap, dtype=float) for name in predictions
    }
    for bootstrap_id in range(n_bootstrap):
        sampled = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices
            ]
        )
        for name, predicted in predictions.items():
            output[name][bootstrap_id] = f1_score(
                labels[sampled], predicted[sampled], average="macro"
            )
    return output


def _sample_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "n_samples": len(labels),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
    }


def _protocol_version(frame: pd.DataFrame) -> str:
    if "protocol_version" not in frame:
        return PROTOCOL_VERSION
    return str(frame["protocol_version"].iloc[0])
