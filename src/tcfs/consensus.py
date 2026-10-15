"""Temporal consensus aggregation for component feature selectors.

The TCFS score used by this project is::

    TCFS_rho = (M * F) / (1 + rho * V)

where ``M`` is mean normalized rank, ``F`` is top-k selection frequency, and
``V`` is population variance of normalized rank. Higher scores are better.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tcfs.temporal_split import TemporalLeakageError

TCFS_RHO_VALUES = (0.0, 0.25, 0.5, 1.0)
DISCOVERY_WINDOW_IDS = (1, 2, 3, 4)
REQUIRED_SCORE_COLUMNS = {
    "file_type",
    "discovery_window",
    "selector",
    "feature_name",
    "normalized_rank",
    "selected",
}

ValidationEvaluator = Callable[[pd.DataFrame, pd.DataFrame], float]


@dataclass(frozen=True)
class RhoEvaluation:
    """Validation-only rho search results and corresponding consensus tables."""

    best_rho: float
    evaluations: pd.DataFrame
    consensus_tables: dict[float, pd.DataFrame]


def aggregate_consensus(
    selector_scores: pd.DataFrame,
    *,
    rho: float,
    expected_selectors: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aggregate four-window, three-selector results into TCFS rankings.

    ``selector_scores`` must contain a complete observation for every
    file-type/feature/window/selector combination. Features may differ between
    file types. Shared scores combine identically named features across the
    formats where they occur.
    """
    rho = _validate_rho(rho)
    scores, selectors = _validate_selector_scores(
        selector_scores, expected_selectors=expected_selectors
    )
    expected_observations = len(DISCOVERY_WINDOW_IDS) * len(selectors)

    format_summary = (
        scores.groupby(["file_type", "feature_name"], sort=True, observed=True)
        .agg(
            mean_normalized_rank=("normalized_rank", "mean"),
            selection_frequency=("selected", "mean"),
            rank_variance=("normalized_rank", _population_variance),
            observation_count=("normalized_rank", "size"),
        )
        .reset_index()
    )
    if not format_summary["observation_count"].eq(expected_observations).all():
        raise ValueError("incomplete selector scores survived completeness checks")
    format_summary["format_tcfs_score"] = _tcfs_score(
        format_summary["mean_normalized_rank"],
        format_summary["selection_frequency"],
        format_summary["rank_variance"],
        rho,
    )

    shared_summary = (
        scores.groupby("feature_name", sort=True, observed=True)
        .agg(
            shared_mean_normalized_rank=("normalized_rank", "mean"),
            shared_selection_frequency=("selected", "mean"),
            shared_rank_variance=("normalized_rank", _population_variance),
        )
        .reset_index()
    )
    shared_summary["shared_tcfs_score"] = _tcfs_score(
        shared_summary["shared_mean_normalized_rank"],
        shared_summary["shared_selection_frequency"],
        shared_summary["shared_rank_variance"],
        rho,
    )
    shared_summary = _rank_rows(
        shared_summary,
        score_column="shared_tcfs_score",
        frequency_column="shared_selection_frequency",
        mean_rank_column="shared_mean_normalized_rank",
        rank_column="shared_rank",
    )

    result = format_summary.merge(
        shared_summary[["feature_name", "shared_tcfs_score", "shared_rank"]],
        on="feature_name",
        how="left",
        validate="many_to_one",
    )
    result = _rank_rows(
        result,
        score_column="format_tcfs_score",
        frequency_column="selection_frequency",
        mean_rank_column="mean_normalized_rank",
        rank_column="format_rank",
        group_column="file_type",
    )
    result.insert(2, "rho", rho)
    return result[
        [
            "file_type",
            "feature_name",
            "rho",
            "mean_normalized_rank",
            "selection_frequency",
            "rank_variance",
            "format_tcfs_score",
            "shared_tcfs_score",
            "format_rank",
            "shared_rank",
            "observation_count",
        ]
    ].reset_index(drop=True)


def evaluate_rho_values(
    selector_scores: pd.DataFrame,
    validation_data: pd.DataFrame,
    evaluator: ValidationEvaluator,
    *,
    expected_selectors: Iterable[str] | None = None,
    role_column: str = "protocol_role",
    higher_is_better: bool = True,
) -> RhoEvaluation:
    """Choose rho using an experiment-specific validation evaluator only.

    The evaluator receives a consensus table and a defensive copy of validation
    data, and must return one finite scalar metric. The role guard rejects mixed,
    discovery, or sealed-test rows. Metric ties choose the smaller rho.
    """
    if role_column not in validation_data.columns:
        raise ValueError(f"validation_data is missing role column: {role_column}")
    if validation_data.empty:
        raise ValueError("validation_data must contain at least one row")
    roles = set(validation_data[role_column].astype(str))
    if roles != {"validation"}:
        raise TemporalLeakageError(
            "rho evaluation requires validation rows only; "
            f"received protocol roles: {sorted(roles)}"
        )

    consensus_tables: dict[float, pd.DataFrame] = {}
    evaluations: list[dict[str, Any]] = []
    for rho in TCFS_RHO_VALUES:
        table = aggregate_consensus(
            selector_scores,
            rho=rho,
            expected_selectors=expected_selectors,
        )
        metric = float(evaluator(table.copy(), validation_data.copy()))
        if not np.isfinite(metric):
            raise ValueError(
                f"validation evaluator returned a non-finite value: {metric}"
            )
        consensus_tables[rho] = table
        evaluations.append({"rho": rho, "validation_score": metric})

    evaluation_table = pd.DataFrame(evaluations).sort_values(
        ["validation_score", "rho"],
        ascending=[not higher_is_better, True],
        kind="stable",
        ignore_index=True,
    )
    best_rho = float(evaluation_table.loc[0, "rho"])
    evaluation_table["selected"] = evaluation_table["rho"].eq(best_rho)
    return RhoEvaluation(
        best_rho=best_rho,
        evaluations=evaluation_table,
        consensus_tables=consensus_tables,
    )


def _validate_selector_scores(
    selector_scores: pd.DataFrame,
    *,
    expected_selectors: Iterable[str] | None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    missing_columns = REQUIRED_SCORE_COLUMNS - set(selector_scores.columns)
    if missing_columns:
        raise ValueError(
            f"selector_scores is missing columns: {sorted(missing_columns)}"
        )
    if selector_scores.empty:
        raise ValueError("selector_scores must contain observations")

    scores = selector_scores.loc[:, sorted(REQUIRED_SCORE_COLUMNS)].copy()
    text_columns = ["file_type", "selector", "feature_name"]
    for column in text_columns:
        if scores[column].isna().any():
            raise ValueError(f"selector_scores contains missing {column} values")
        scores[column] = scores[column].astype(str)
        if scores[column].str.strip().eq("").any():
            raise ValueError(f"selector_scores contains blank {column} values")

    scores["discovery_window"] = pd.to_numeric(
        scores["discovery_window"], errors="raise"
    ).astype(int)
    actual_windows = set(scores["discovery_window"])
    if actual_windows != set(DISCOVERY_WINDOW_IDS):
        raise ValueError(
            "selector_scores must contain discovery windows 1, 2, 3, and 4; "
            f"received {sorted(actual_windows)}"
        )

    actual_selectors = tuple(sorted(scores["selector"].unique()))
    if expected_selectors is None:
        selectors = actual_selectors
        if len(selectors) != 3:
            raise ValueError(
                "selector_scores must contain exactly three component selectors"
            )
    else:
        selectors = tuple(sorted(str(value) for value in expected_selectors))
        if len(selectors) != 3 or len(set(selectors)) != 3:
            raise ValueError("expected_selectors must contain three unique names")
        if set(actual_selectors) != set(selectors):
            raise ValueError(
                "selector names differ from expected_selectors: "
                f"received {list(actual_selectors)}"
            )

    scores["normalized_rank"] = pd.to_numeric(
        scores["normalized_rank"], errors="raise"
    ).astype(float)
    ranks = scores["normalized_rank"]
    if not np.isfinite(ranks).all() or not ranks.between(0.0, 1.0).all():
        raise ValueError("normalized_rank values must be finite and within [0, 1]")
    if scores["selected"].isna().any():
        raise ValueError("selected values must not be missing")
    if not scores["selected"].isin([True, False, 0, 1]).all():
        raise ValueError("selected values must be boolean")
    scores["selected"] = scores["selected"].astype(bool)

    key = [
        "file_type",
        "feature_name",
        "discovery_window",
        "selector",
    ]
    if scores.duplicated(key).any():
        raise ValueError("duplicate feature/window/selector score observations")

    expected_pairs = pd.MultiIndex.from_product(
        [DISCOVERY_WINDOW_IDS, selectors],
        names=["discovery_window", "selector"],
    )
    for (file_type, feature_name), group in scores.groupby(
        ["file_type", "feature_name"], sort=True, observed=True
    ):
        actual_pairs = pd.MultiIndex.from_frame(
            group[["discovery_window", "selector"]]
        )
        missing_pairs = expected_pairs.difference(actual_pairs)
        if len(missing_pairs):
            missing_text = ", ".join(
                f"window={window}/selector={selector}"
                for window, selector in missing_pairs
            )
            raise ValueError(
                "missing selector scores for "
                f"file_type={file_type}, feature={feature_name}: {missing_text}"
            )
    return scores, selectors


def _tcfs_score(
    mean_rank: pd.Series,
    frequency: pd.Series,
    variance: pd.Series,
    rho: float,
) -> pd.Series:
    return (mean_rank * frequency) / (1.0 + rho * variance)


def _population_variance(values: pd.Series) -> float:
    variance = float(values.var(ddof=0))
    return 0.0 if np.isclose(variance, 0.0) else variance


def _rank_rows(
    table: pd.DataFrame,
    *,
    score_column: str,
    frequency_column: str,
    mean_rank_column: str,
    rank_column: str,
    group_column: str | None = None,
) -> pd.DataFrame:
    sort_columns = [score_column, frequency_column, mean_rank_column, "feature_name"]
    ascending = [False, False, False, True]
    if group_column is not None:
        sort_columns.insert(0, group_column)
        ascending.insert(0, True)
    ranked = table.sort_values(
        sort_columns, ascending=ascending, kind="stable", ignore_index=True
    )
    if group_column is None:
        ranked[rank_column] = np.arange(1, len(ranked) + 1, dtype=int)
    else:
        ranked[rank_column] = ranked.groupby(group_column, sort=False).cumcount() + 1
    return ranked


def _validate_rho(rho: float) -> float:
    value = float(rho)
    for candidate in TCFS_RHO_VALUES:
        if np.isclose(value, candidate):
            return candidate
    allowed = ", ".join(str(value) for value in TCFS_RHO_VALUES)
    raise ValueError(f"rho must be one of: {allowed}")
