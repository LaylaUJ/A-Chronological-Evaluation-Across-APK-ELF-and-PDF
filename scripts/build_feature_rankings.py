"""Build budget-aware TCFS and baseline rankings from discovery-only scores."""

import argparse
from collections.abc import Sequence
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from tcfs.consensus import TCFS_RHO_VALUES, aggregate_consensus
from tcfs.feature_selection import SUPPORTED_FEATURE_BUDGETS

COMPONENTS = ("mutual_information", "l1_logistic", "lightgbm_permutation")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse discovery score inputs and ranking destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--control-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def build_rankings(
    selector_scores: pd.DataFrame,
    control_scores: pd.DataFrame,
    *,
    budgets: Sequence[float] = SUPPORTED_FEATURE_BUDGETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return model rankings and full budget-specific consensus statistics."""
    formats = set(selector_scores["file_type"].astype(str))
    if formats != set(control_scores["file_type"].astype(str)):
        raise ValueError("selector and control scores cover different file types")
    ranking_parts: list[pd.DataFrame] = []
    consensus_parts: list[pd.DataFrame] = []

    for budget in budgets:
        budget = float(budget)
        budget_scores = selector_scores.copy()
        budget_scores["selected"] = False
        group_columns = ["file_type", "discovery_window", "selector"]
        for _, index in budget_scores.groupby(
            group_columns, sort=False, observed=True
        ).groups.items():
            group = budget_scores.loc[index]
            count = max(1, ceil(budget * len(group)))
            selected_index = group.sort_values(
                ["normalized_rank", "feature_name"],
                ascending=[False, True],
                kind="stable",
            ).index[:count]
            budget_scores.loc[selected_index, "selected"] = True

        for rho in TCFS_RHO_VALUES:
            consensus = aggregate_consensus(
                budget_scores, rho=rho, expected_selectors=COMPONENTS
            )
            consensus.insert(3, "feature_budget", budget)
            consensus_parts.append(consensus)
            format_ranks = consensus[
                ["file_type", "feature_name", "format_rank"]
            ].rename(columns={"format_rank": "rank"})
            format_ranks.insert(0, "method", "tcfs")
            format_ranks["rho"] = rho
            format_ranks["ranking_budget"] = budget
            ranking_parts.append(format_ranks)

            shared = consensus.drop_duplicates("feature_name")[
                ["feature_name", "shared_rank"]
            ].rename(columns={"shared_rank": "rank"})
            shared = pd.concat(
                [shared.assign(file_type=file_type) for file_type in sorted(formats)],
                ignore_index=True,
            )
            shared.insert(0, "method", "shared_tcfs")
            shared["rho"] = rho
            shared["ranking_budget"] = budget
            ranking_parts.append(shared)

    for method, source in [
        *[(component, selector_scores) for component in COMPONENTS],
        ("random_matched", control_scores),
    ]:
        subset = source.loc[source["selector"].eq(method)]
        summary = (
            subset.groupby(["file_type", "feature_name"], observed=True)[
                "normalized_rank"
            ]
            .mean()
            .rename("mean_rank")
            .reset_index()
            .sort_values(
                ["file_type", "mean_rank", "feature_name"],
                ascending=[True, False, True],
                kind="stable",
            )
        )
        summary["rank"] = summary.groupby("file_type").cumcount() + 1
        summary.insert(0, "method", method)
        summary["rho"] = np.nan
        summary["ranking_budget"] = np.nan
        ranking_parts.append(
            summary[
                [
                    "method",
                    "file_type",
                    "feature_name",
                    "rank",
                    "rho",
                    "ranking_budget",
                ]
            ]
        )

    names = sorted(selector_scores["feature_name"].astype(str).unique())
    all_features = pd.concat(
        [
            pd.DataFrame(
                {
                    "method": "all_features",
                    "file_type": file_type,
                    "feature_name": names,
                    "rank": np.arange(1, len(names) + 1),
                    "rho": np.nan,
                    "ranking_budget": np.nan,
                }
            )
            for file_type in sorted(formats)
        ],
        ignore_index=True,
    )
    ranking_parts.append(all_features)
    rankings = pd.concat(ranking_parts, ignore_index=True)
    return rankings, pd.concat(consensus_parts, ignore_index=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Load discovery scores and persist complete rankings."""
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = args.output_dir / "rankings.parquet"
    consensus_path = args.output_dir / "consensus_scores.parquet"
    if rankings_path.exists() or consensus_path.exists():
        raise FileExistsError("ranking outputs already exist")
    selectors = pd.concat(
        [pd.read_parquet(path) for path in args.selector_scores], ignore_index=True
    )
    controls = pd.concat(
        [pd.read_parquet(path) for path in args.control_scores], ignore_index=True
    )
    rankings, consensus = build_rankings(selectors, controls)
    rankings.to_parquet(rankings_path, index=False)
    consensus.to_parquet(consensus_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
