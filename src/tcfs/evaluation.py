"""Per-format evaluation, paired bootstrap inference, and feature stability."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PREDICTIVE_METRICS = (
    "macro_f1",
    "mcc",
    "balanced_accuracy",
    "precision",
    "recall",
    "pr_auc",
    "roc_auc",
    "tpr_at_1pct_fpr",
    "tpr_at_0_1pct_fpr",
)
SECONDARY_CONFIRMATORY_METRICS = PREDICTIVE_METRICS[1:]
RESOURCE_METRICS = (
    "feature_count",
    "model_size_bytes",
    "inference_latency_ms_per_sample",
)
REQUIRED_PREDICTION_COLUMNS = {
    "file_type",
    "method",
    "row_id",
    "y_true",
    "probability",
    "prediction",
}
REQUIRED_METADATA_COLUMNS = {
    "file_type",
    "method",
    *RESOURCE_METRICS,
}


@dataclass(frozen=True)
class EvaluationResults:
    """All point estimates, inferential comparisons, and stability outputs."""

    metrics: pd.DataFrame
    method_differences: pd.DataFrame
    noninferiority: pd.DataFrame
    secondary_comparisons: pd.DataFrame
    jaccard_stability: pd.DataFrame
    spearman_stability: pd.DataFrame


def evaluate_predictions(
    predictions: pd.DataFrame,
    experiment_metadata: pd.DataFrame,
    *,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    comparisons: Sequence[tuple[str, str]] | None = None,
    tcfs_method: str = "tcfs",
    all_features_method: str = "all_features",
    noninferiority_margin: float = -0.01,
    feature_results: pd.DataFrame | None = None,
) -> EvaluationResults:
    """Evaluate aligned methods independently for APK, ELF, and PDF.

    Bootstrap resampling is stratified by class and paired across methods within
    each format. A comparison tuple is ``(method, reference_method)`` and all
    reported differences are method minus reference.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be within (0, 1)")
    frame, metadata = _validate_evaluation_inputs(predictions, experiment_metadata)
    formats = set(frame["file_type"])
    if formats != {"APK", "ELF", "PDF"}:
        raise ValueError("predictions must contain APK, ELF, and PDF results")

    rng = np.random.default_rng(random_state)
    alpha = 1.0 - confidence_level
    metric_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    noninferiority_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    sample_counts: list[dict[str, Any]] = []

    for file_type in sorted(formats):
        format_frame = frame.loc[frame["file_type"].eq(file_type)]
        aligned = _align_format_predictions(format_frame)
        methods = sorted(aligned)
        first = aligned[methods[0]]
        y_true = first["y_true"].to_numpy(dtype=np.int8)
        counts = _sample_counts(file_type, y_true)
        sample_counts.append(counts)

        point_metrics: dict[str, dict[str, float]] = {}
        bootstrap_metrics = {
            method: {
                metric: np.empty(n_bootstrap, dtype=float)
                for metric in PREDICTIVE_METRICS
            }
            for method in methods
        }
        for method, method_frame in aligned.items():
            point_metrics[method] = compute_predictive_metrics(
                y_true,
                method_frame["prediction"].to_numpy(dtype=np.int8),
                method_frame["probability"].to_numpy(dtype=float),
            )

        class_indices = [np.flatnonzero(y_true == label) for label in (0, 1)]
        for bootstrap_id in range(n_bootstrap):
            sample = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in class_indices
                ]
            )
            sampled_y = y_true[sample]
            for method, method_frame in aligned.items():
                sampled_metrics = compute_predictive_metrics(
                    sampled_y,
                    method_frame["prediction"].to_numpy(dtype=np.int8)[sample],
                    method_frame["probability"].to_numpy(dtype=float)[sample],
                )
                for metric, value in sampled_metrics.items():
                    bootstrap_metrics[method][metric][bootstrap_id] = value

        for method in methods:
            for metric in PREDICTIVE_METRICS:
                low, high = np.quantile(
                    bootstrap_metrics[method][metric],
                    [alpha / 2.0, 1.0 - alpha / 2.0],
                )
                metric_rows.append(
                    {
                        "file_type": file_type,
                        "method": method,
                        "metric": metric,
                        "estimate": point_metrics[method][metric],
                        "ci_lower": low,
                        "ci_upper": high,
                        "confidence_level": confidence_level,
                        "bootstrap_samples": n_bootstrap,
                        "bootstrap_random_state": random_state,
                        **counts,
                    }
                )
            resource = metadata.loc[
                metadata["file_type"].eq(file_type) & metadata["method"].eq(method)
            ].iloc[0]
            for metric in RESOURCE_METRICS:
                metric_rows.append(
                    {
                        "file_type": file_type,
                        "method": method,
                        "metric": metric,
                        "estimate": float(resource[metric]),
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "confidence_level": confidence_level,
                        "bootstrap_samples": 0,
                        "bootstrap_random_state": random_state,
                        **counts,
                    }
                )

        format_comparisons = (
            list(comparisons)
            if comparisons is not None
            else list(combinations(methods, 2))
        )
        for method, reference in format_comparisons:
            if method not in aligned or reference not in aligned:
                raise ValueError(
                    f"comparison {method} versus {reference} is unavailable "
                    f"for {file_type}"
                )
            for metric in PREDICTIVE_METRICS:
                observed = (
                    point_metrics[method][metric]
                    - point_metrics[reference][metric]
                )
                differences = (
                    bootstrap_metrics[method][metric]
                    - bootstrap_metrics[reference][metric]
                )
                low, high = np.quantile(
                    differences, [alpha / 2.0, 1.0 - alpha / 2.0]
                )
                difference_rows.append(
                    {
                        "file_type": file_type,
                        "method": method,
                        "reference_method": reference,
                        "metric": metric,
                        "difference": observed,
                        "ci_lower": low,
                        "ci_upper": high,
                        "p_value": _two_sided_bootstrap_p(differences),
                        "bootstrap_samples": n_bootstrap,
                        "bootstrap_random_state": random_state,
                        **counts,
                    }
                )

        if tcfs_method not in aligned or all_features_method not in aligned:
            raise ValueError(
                f"{file_type} requires {tcfs_method} and {all_features_method} "
                "for non-inferiority"
            )
        macro_differences = (
            bootstrap_metrics[tcfs_method]["macro_f1"]
            - bootstrap_metrics[all_features_method]["macro_f1"]
        )
        observed_macro_difference = (
            point_metrics[tcfs_method]["macro_f1"]
            - point_metrics[all_features_method]["macro_f1"]
        )
        macro_low, macro_high = np.quantile(
            macro_differences, [alpha / 2.0, 1.0 - alpha / 2.0]
        )
        noninferiority_p = _one_sided_noninferiority_p(
            macro_differences, noninferiority_margin
        )
        noninferiority_rows.append(
            {
                "file_type": file_type,
                "method": tcfs_method,
                "reference_method": all_features_method,
                "metric": "macro_f1",
                "difference": observed_macro_difference,
                "ci_lower": macro_low,
                "ci_upper": macro_high,
                "noninferiority_margin": noninferiority_margin,
                "p_value": noninferiority_p,
                "noninferior_at_0_05": bool(
                    noninferiority_p < 0.05
                    and observed_macro_difference > noninferiority_margin
                ),
                "bootstrap_samples": n_bootstrap,
                "bootstrap_random_state": random_state,
                **counts,
            }
        )

        format_secondary: list[dict[str, Any]] = []
        for metric in SECONDARY_CONFIRMATORY_METRICS:
            differences = (
                bootstrap_metrics[tcfs_method][metric]
                - bootstrap_metrics[all_features_method][metric]
            )
            low, high = np.quantile(
                differences, [alpha / 2.0, 1.0 - alpha / 2.0]
            )
            format_secondary.append(
                {
                    "file_type": file_type,
                    "method": tcfs_method,
                    "reference_method": all_features_method,
                    "metric": metric,
                    "difference": (
                        point_metrics[tcfs_method][metric]
                        - point_metrics[all_features_method][metric]
                    ),
                    "ci_lower": low,
                    "ci_upper": high,
                    "p_value": _two_sided_bootstrap_p(differences),
                    "bootstrap_samples": n_bootstrap,
                    "bootstrap_random_state": random_state,
                    **counts,
                }
            )
        adjusted = _holm_adjust([row["p_value"] for row in format_secondary])
        for row, adjusted_p in zip(format_secondary, adjusted, strict=True):
            row["holm_adjusted_p_value"] = adjusted_p
            row["significant_at_0_05"] = bool(adjusted_p < 0.05)
        secondary_rows.extend(format_secondary)

    count_frame = pd.DataFrame(sample_counts)
    if feature_results is None:
        jaccard = _empty_stability_table("jaccard")
        spearman = _empty_stability_table("spearman_rho")
    else:
        jaccard, spearman = evaluate_feature_stability(feature_results, count_frame)
    return EvaluationResults(
        metrics=pd.DataFrame(metric_rows),
        method_differences=pd.DataFrame(difference_rows),
        noninferiority=pd.DataFrame(noninferiority_rows),
        secondary_comparisons=pd.DataFrame(secondary_rows),
        jaccard_stability=jaccard,
        spearman_stability=spearman,
    )


def evaluate_feature_stability(
    feature_results: pd.DataFrame,
    sample_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pairwise Jaccard and Spearman stability by format and method.

    ``feature_results`` must contain one complete feature table per ``replicate``
    with boolean ``selected`` and numeric ``normalized_rank`` values.
    """
    required = {
        "file_type",
        "method",
        "replicate",
        "feature_name",
        "selected",
        "normalized_rank",
    }
    missing = required - set(feature_results.columns)
    if missing:
        raise ValueError(f"feature_results is missing columns: {sorted(missing)}")
    count_required = {"file_type", "n_samples", "n_positive", "n_negative"}
    if count_required - set(sample_counts.columns):
        raise ValueError("sample_counts is missing required class-count columns")
    jaccard_rows: list[dict[str, Any]] = []
    spearman_rows: list[dict[str, Any]] = []
    for (file_type, method), group in feature_results.groupby(
        ["file_type", "method"], sort=True
    ):
        replicates = sorted(group["replicate"].astype(str).unique())
        if len(replicates) < 2:
            raise ValueError(
                f"{file_type}/{method} requires at least two feature replicates"
            )
        count_match = sample_counts.loc[sample_counts["file_type"].eq(file_type)]
        if len(count_match) != 1:
            raise ValueError(f"sample counts are unavailable for {file_type}")
        counts = count_match.iloc[0][
            ["n_samples", "n_positive", "n_negative"]
        ].to_dict()
        by_replicate: dict[str, pd.DataFrame] = {}
        for replicate in replicates:
            replicate_frame = group.loc[
                group["replicate"].astype(str).eq(replicate)
            ].copy()
            if replicate_frame["feature_name"].duplicated().any():
                raise ValueError("feature names must be unique within each replicate")
            by_replicate[replicate] = replicate_frame.set_index("feature_name")
        expected_features = set(by_replicate[replicates[0]].index)
        if any(
            set(table.index) != expected_features
            for table in by_replicate.values()
        ):
            raise ValueError("all stability replicates must contain the same features")

        for replicate_a, replicate_b in combinations(replicates, 2):
            table_a = by_replicate[replicate_a]
            table_b = by_replicate[replicate_b].reindex(table_a.index)
            selected_a = set(table_a.index[table_a["selected"].astype(bool)])
            selected_b = set(table_b.index[table_b["selected"].astype(bool)])
            union = selected_a | selected_b
            jaccard = 1.0 if not union else len(selected_a & selected_b) / len(union)
            ranks_a = pd.to_numeric(table_a["normalized_rank"], errors="raise")
            ranks_b = pd.to_numeric(table_b["normalized_rank"], errors="raise")
            if len(ranks_a) == 1:
                correlation = 1.0
            else:
                correlation = float(spearmanr(ranks_a, ranks_b).statistic)
            common = {
                "file_type": file_type,
                "method": method,
                "replicate_a": replicate_a,
                "replicate_b": replicate_b,
                **counts,
            }
            jaccard_rows.append({**common, "jaccard": jaccard})
            spearman_rows.append({**common, "spearman_rho": correlation})
    return pd.DataFrame(jaccard_rows), pd.DataFrame(spearman_rows)


def profile_model(
    model: Any,
    X: Any,
    *,
    warmup_runs: int = 3,
    timed_runs: int = 20,
) -> dict[str, float | int]:
    """Measure serialized model size and median probability-inference latency."""
    matrix = np.asarray(X)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError("warmup_runs must be non-negative and timed_runs positive")
    for _ in range(warmup_runs):
        model.predict_proba(matrix)
    elapsed: list[float] = []
    for _ in range(timed_runs):
        started = perf_counter()
        model.predict_proba(matrix)
        elapsed.append(perf_counter() - started)
    model_size = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    feature_names = getattr(model, "feature_names", None)
    feature_count = (
        len(feature_names)
        if feature_names is not None
        else int(getattr(model, "n_features_in_", matrix.shape[1]))
    )
    return {
        "feature_count": feature_count,
        "model_size_bytes": model_size,
        "inference_latency_ms_per_sample": (
            float(np.median(elapsed)) * 1_000.0 / len(matrix)
        ),
    }


def save_evaluation_results(
    results: EvaluationResults, output_dir: str | Path
) -> None:
    """Save every result table as machine-readable CSV."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for field_name in EvaluationResults.__dataclass_fields__:
        table = getattr(results, field_name)
        table.to_csv(destination / f"{field_name}.csv", index=False)


def compute_predictive_metrics(
    y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    """Calculate the project's binary predictive metrics."""
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true, probability, drop_intermediate=False
    )
    return {
        "macro_f1": f1_score(y_true, prediction, average="macro"),
        "mcc": matthews_corrcoef(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "pr_auc": average_precision_score(y_true, probability),
        "roc_auc": roc_auc_score(y_true, probability),
        "tpr_at_1pct_fpr": _tpr_at_fpr(
            false_positive_rate, true_positive_rate, 0.01
        ),
        "tpr_at_0_1pct_fpr": _tpr_at_fpr(
            false_positive_rate, true_positive_rate, 0.001
        ),
    }


def _tpr_at_fpr(
    false_positive_rate: np.ndarray,
    true_positive_rate: np.ndarray,
    maximum_fpr: float,
) -> float:
    eligible = true_positive_rate[false_positive_rate <= maximum_fpr + 1e-15]
    return float(eligible.max(initial=0.0))


def _validate_evaluation_inputs(
    predictions: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing columns: {sorted(missing)}")
    metadata_missing = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if metadata_missing:
        raise ValueError(
            f"experiment_metadata is missing columns: {sorted(metadata_missing)}"
        )
    frame = predictions.copy()
    frame["file_type"] = frame["file_type"].astype(str).str.upper()
    frame["method"] = frame["method"].astype(str)
    if frame.duplicated(["file_type", "method", "row_id"]).any():
        raise ValueError("predictions contains duplicate format/method/row records")
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    frame["prediction"] = pd.to_numeric(
        frame["prediction"], errors="raise"
    ).astype(int)
    if not frame["y_true"].isin([0, 1]).all() or not frame["prediction"].isin(
        [0, 1]
    ).all():
        raise ValueError("y_true and prediction must be binary")
    frame["probability"] = pd.to_numeric(
        frame["probability"], errors="raise"
    ).astype(float)
    if not np.isfinite(frame["probability"]).all() or not frame[
        "probability"
    ].between(0, 1).all():
        raise ValueError("probabilities must be finite and within [0, 1]")

    metadata = metadata.copy()
    metadata["file_type"] = metadata["file_type"].astype(str).str.upper()
    metadata["method"] = metadata["method"].astype(str)
    if metadata.duplicated(["file_type", "method"]).any():
        raise ValueError("experiment_metadata must have one row per format/method")
    prediction_keys = set(zip(frame["file_type"], frame["method"], strict=True))
    metadata_keys = set(zip(metadata["file_type"], metadata["method"], strict=True))
    if prediction_keys != metadata_keys:
        raise ValueError("prediction and experiment-metadata format/method keys differ")
    for metric in RESOURCE_METRICS:
        metadata[metric] = pd.to_numeric(metadata[metric], errors="raise")
        if not np.isfinite(metadata[metric]).all() or (metadata[metric] < 0).any():
            raise ValueError(f"{metric} must contain finite non-negative values")
    return frame, metadata


def _align_format_predictions(
    format_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    aligned: dict[str, pd.DataFrame] = {}
    expected_rows: np.ndarray | None = None
    expected_labels: np.ndarray | None = None
    for method, group in format_frame.groupby("method", sort=True):
        ordered = group.sort_values("row_id", kind="stable").reset_index(drop=True)
        rows = ordered["row_id"].to_numpy()
        labels = ordered["y_true"].to_numpy()
        if expected_rows is None:
            expected_rows = rows
            expected_labels = labels
        elif not np.array_equal(rows, expected_rows) or not np.array_equal(
            labels, expected_labels
        ):
            raise ValueError("methods must have aligned row IDs and true labels")
        aligned[str(method)] = ordered
    if expected_labels is None or set(expected_labels) != {0, 1}:
        raise ValueError("each format requires both classes")
    return aligned


def _sample_counts(file_type: str, y_true: np.ndarray) -> dict[str, int | str]:
    return {
        "file_type": file_type,
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }


def _two_sided_bootstrap_p(differences: np.ndarray) -> float:
    denominator = len(differences) + 1
    lower = (np.count_nonzero(differences <= 0.0) + 1) / denominator
    upper = (np.count_nonzero(differences >= 0.0) + 1) / denominator
    return float(min(1.0, 2.0 * min(lower, upper)))


def _one_sided_noninferiority_p(
    differences: np.ndarray, margin: float
) -> float:
    return float(
        (np.count_nonzero(differences <= margin) + 1) / (len(differences) + 1)
    )


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def _empty_stability_table(metric: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "file_type",
            "method",
            "replicate_a",
            "replicate_b",
            "n_samples",
            "n_positive",
            "n_negative",
            metric,
        ]
    )
