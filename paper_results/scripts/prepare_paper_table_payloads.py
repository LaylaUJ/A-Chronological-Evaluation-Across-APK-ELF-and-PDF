"""Prepare JSON payloads consumed by the artifact-tool Excel table builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-results", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.paper_results.resolve()
    reports = args.project_root.resolve() / "outputs" / "reports"

    metrics = pd.read_csv(output / "metrics.csv")
    metrics["stored_method"] = np.where(
        metrics["classifier"].eq("lightgbm"),
        metrics["selector"],
        metrics["method"],
    )
    ci = pd.read_csv(reports / "evaluation" / "metrics.csv")
    ci = ci.loc[
        ci["metric"].eq("macro_f1"),
        ["file_type", "method", "ci_lower", "ci_upper"],
    ]
    metrics = metrics.merge(
        ci,
        left_on=["format", "stored_method"],
        right_on=["file_type", "method"],
        how="left",
        validate="many_to_one",
    )
    primary_columns = [
        "format",
        "classifier",
        "selector",
        "feature_count",
        "total_feature_count",
        "feature_fraction",
        "split",
        "n",
        "macro_f1",
        "ci_lower",
        "ci_upper",
        "mcc",
        "pr_auc",
        "roc_auc",
        "balanced_accuracy",
        "precision",
        "recall",
        "latency_ms",
        "model_size_mb",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    primary = metrics[primary_columns].sort_values(["format", "classifier", "selector"])

    noninferiority = pd.read_csv(output / "noninferiority.csv")
    secondary = pd.read_csv(output / "secondary_comparisons.csv")
    jaccard = pd.read_csv(output / "jaccard_stability.csv")
    spearman = pd.read_csv(output / "spearman_stability.csv")
    jaccard_summary = jaccard.groupby(
        ["format", "method", "feature_budget"], as_index=False
    ).agg(
        jaccard_mean=("jaccard", "mean"),
        jaccard_sd=("jaccard", "std"),
        jaccard_min=("jaccard", "min"),
        jaccard_max=("jaccard", "max"),
        window_pairs=("jaccard", "size"),
    )
    spearman_summary = spearman.groupby(["format", "method"], as_index=False).agg(
        spearman_mean=("spearman_rho", "mean"),
        spearman_sd=("spearman_rho", "std"),
        spearman_min=("spearman_rho", "min"),
        spearman_max=("spearman_rho", "max"),
        common_features=("common_feature_count", "min"),
    )
    stability = jaccard_summary.merge(
        spearman_summary, on=["format", "method"], validate="one_to_one"
    ).sort_values(["format", "method"])

    ablation = pd.read_csv(output / "ablation_sensitivity.csv")
    original = pd.read_csv(reports / "ablation_sensitivity.csv")
    original["ablation"] = (
        original["analysis"].astype(str) + ":" + original["setting"].astype(str)
    )
    ablation = ablation.merge(
        original[["file_type", "ablation", "ci_lower", "ci_upper"]],
        left_on=["format", "ablation"],
        right_on=["file_type", "ablation"],
        how="left",
        validate="one_to_one",
    ).drop(columns="file_type")

    def matrix(frame: pd.DataFrame) -> list[list[object]]:
        clean = frame.astype(object).where(pd.notna(frame), None)
        return [frame.columns.tolist(), *clean.values.tolist()]

    payload = {
        "primary_results": matrix(primary),
        "noninferiority": matrix(noninferiority),
        "secondary_comparisons": matrix(secondary),
        "stability_summary": matrix(stability),
        "jaccard_detail": matrix(jaccard),
        "spearman_detail": matrix(spearman),
        "ablation": matrix(ablation),
    }
    destination = output / "scripts" / "table_payloads.json"
    destination.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
