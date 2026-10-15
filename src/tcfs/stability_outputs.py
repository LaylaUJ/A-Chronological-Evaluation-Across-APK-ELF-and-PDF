"""Build window-level feature rankings for publication stability analyses."""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

COMPONENTS = ("mutual_information", "l1_logistic", "lightgbm_permutation")


def build_feature_results(
    selector_scores: pd.DataFrame,
    control_scores: pd.DataFrame,
    selected_configurations: pd.DataFrame,
) -> pd.DataFrame:
    """Construct complete four-window ranking tables at frozen budgets/rhos."""
    config_required = {
        "file_type",
        "method",
        "classifier",
        "feature_budget",
        "rho",
    }
    if missing := config_required - set(selected_configurations):
        raise ValueError(f"configurations are missing columns: {sorted(missing)}")
    configs = selected_configurations.loc[
        selected_configurations["classifier"].eq("lightgbm")
    ].copy()
    if configs.duplicated(["file_type", "method"]).any():
        raise ValueError("one primary configuration is required per format/method")
    frames: list[pd.DataFrame] = []
    for config in configs.itertuples(index=False):
        file_type = str(config.file_type)
        method = str(config.method)
        budget = float(config.feature_budget)
        rho = 0.0 if pd.isna(config.rho) else float(config.rho)
        for window in range(1, 5):
            if method in COMPONENTS:
                subset = selector_scores.loc[
                    selector_scores["file_type"].eq(file_type)
                    & selector_scores["discovery_window"].eq(window)
                    & selector_scores["selector"].eq(method)
                ]
                ranked = subset[["feature_name", "normalized_rank"]].copy()
            elif method == "random_matched":
                subset = control_scores.loc[
                    control_scores["file_type"].eq(file_type)
                    & control_scores["discovery_window"].eq(window)
                    & control_scores["selector"].eq("random_matched")
                ]
                ranked = subset[["feature_name", "normalized_rank"]].copy()
            elif method == "all_features":
                names = sorted(
                    selector_scores.loc[
                        selector_scores["file_type"].eq(file_type), "feature_name"
                    ].astype(str).unique()
                )
                ranked = pd.DataFrame(
                    {
                        "feature_name": names,
                        "normalized_rank": np.arange(len(names), 0, -1)
                        / len(names),
                    }
                )
            elif method in {"tcfs", "shared_tcfs"}:
                source = selector_scores.loc[
                    selector_scores["discovery_window"].eq(window)
                ].copy()
                if method == "tcfs":
                    source = source.loc[source["file_type"].eq(file_type)]
                ranked = _single_window_consensus(source, budget=budget, rho=rho)
            else:
                raise ValueError(f"unsupported stability method: {method}")
            if ranked.empty:
                raise ValueError(
                    f"missing scores for {file_type}/{method}/window {window}"
                )
            selected_count = (
                len(ranked)
                if method == "all_features"
                else max(1, ceil(budget * len(ranked)))
            )
            ordered = ranked.sort_values(
                ["normalized_rank", "feature_name"],
                ascending=[False, True],
                kind="stable",
                ignore_index=True,
            )
            ordered["selected"] = False
            ordered.loc[: selected_count - 1, "selected"] = True
            ordered.insert(0, "replicate", f"window_{window}")
            ordered.insert(0, "method", method)
            ordered.insert(0, "file_type", file_type)
            frames.append(ordered)
    result = pd.concat(frames, ignore_index=True)
    key = ["file_type", "method", "replicate", "feature_name"]
    if result.duplicated(key).any():
        raise RuntimeError("stability feature results contain duplicate rows")
    return result


def _single_window_consensus(
    scores: pd.DataFrame, *, budget: float, rho: float
) -> pd.DataFrame:
    """Aggregate three selectors within one window, optionally across formats."""
    if set(scores["selector"].astype(str)) != set(COMPONENTS):
        raise ValueError("single-window consensus requires all three selectors")
    working = scores.copy()
    working["selected_at_budget"] = False
    for _, index in working.groupby(
        ["file_type", "selector"], sort=False, observed=True
    ).groups.items():
        group = working.loc[index]
        count = max(1, ceil(budget * len(group)))
        chosen = group.sort_values(
            ["normalized_rank", "feature_name"],
            ascending=[False, True],
            kind="stable",
        ).index[:count]
        working.loc[chosen, "selected_at_budget"] = True
    summary = (
        working.groupby("feature_name", sort=True, observed=True)
        .agg(
            mean_rank=("normalized_rank", "mean"),
            frequency=("selected_at_budget", "mean"),
            variance=("normalized_rank", lambda values: values.var(ddof=0)),
        )
        .reset_index()
    )
    summary["score"] = (
        summary["mean_rank"] * summary["frequency"]
    ) / (1.0 + rho * summary["variance"])
    summary = summary.sort_values(
        ["score", "frequency", "mean_rank", "feature_name"],
        ascending=[False, False, False, True],
        kind="stable",
        ignore_index=True,
    )
    summary["normalized_rank"] = np.arange(len(summary), 0, -1) / len(summary)
    return summary[["feature_name", "normalized_rank"]]
