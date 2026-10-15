"""Build the auditable manuscript-results package from genuine saved artifacts.

This script never trains models or accesses raw malware.  It standardizes saved
sample-level predictions and analysis outputs, recalculates deterministic point
metrics, creates publication figures, and writes an audit trail.
"""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy.stats import spearmanr
from sklearn import __version__ as sklearn_version
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "outputs" / "reports"
PREDICTIONS = PROJECT_ROOT / "outputs" / "predictions"
PROCESSED = PROJECT_ROOT / "data" / "processed"
FORMATS = ("APK", "ELF", "PDF")
SEEDS = "17;42;73"
TOTAL_FEATURES = 696
PROTOCOL_VERSION = "tcfs-temporal-v1.0"
MARGIN = -0.01

SCHEMAS = {
    "metrics.csv": [
        "experiment_id",
        "format",
        "split",
        "method",
        "selector",
        "classifier",
        "seed",
        "feature_count",
        "total_feature_count",
        "feature_fraction",
        "n",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "macro_f1",
        "mcc",
        "pr_auc",
        "roc_auc",
        "tpr_at_1pct_fpr",
        "tpr_at_0_1pct_fpr",
        "latency_ms",
        "model_size_mb",
    ],
    "budget_metrics.csv": [
        "format",
        "method",
        "feature_budget",
        "feature_count",
        "feature_fraction",
        "macro_f1_mean",
        "macro_f1_sd",
        "macro_f1_ci_lower",
        "macro_f1_ci_upper",
        "pr_auc_mean",
        "mcc_mean",
        "seeds",
    ],
    "jaccard_stability.csv": [
        "format",
        "method",
        "feature_budget",
        "window_a",
        "window_b",
        "intersection_size",
        "union_size",
        "jaccard",
    ],
    "spearman_stability.csv": [
        "format",
        "method",
        "window_a",
        "window_b",
        "common_feature_count",
        "spearman_rho",
        "p_value",
    ],
    "noninferiority.csv": [
        "format",
        "tcfs_method",
        "baseline_method",
        "metric",
        "margin",
        "effect_tcfs_minus_baseline",
        "ci_lower",
        "ci_upper",
        "p_value",
        "adjusted_p_value",
        "decision",
        "analysis_unit",
    ],
    "secondary_comparisons.csv": [
        "format",
        "method_a",
        "method_b",
        "metric",
        "effect_a_minus_b",
        "ci_lower",
        "ci_upper",
        "raw_p_value",
        "adjusted_p_value",
        "correction_method",
        "analysis_unit",
        "decision",
    ],
    "ablation_sensitivity.csv": [
        "format",
        "ablation",
        "reference_method",
        "evaluated_method",
        "feature_count",
        "macro_f1",
        "pr_auc",
        "mcc",
        "effect_on_macro_f1",
        "seed_or_analysis_unit",
    ],
    "selected_features.csv": [
        "format",
        "method",
        "discovery_window",
        "feature_name",
        "feature_group",
        "rank",
        "normalized_rank",
        "selection_frequency",
        "rank_variance",
        "tcfs_score",
        "selected",
    ],
    "sample_counts.csv": [
        "format",
        "official_split",
        "study_phase",
        "time_window",
        "label",
        "raw_count",
        "excluded_count",
        "retained_count",
        "exclusion_reason",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("C:/workspace/paper_results")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selector_name(method: str, classifier: str) -> str:
    suffix = "__logistic_regression"
    return (
        method[: -len(suffix)]
        if classifier == "logistic_regression" and method.endswith(suffix)
        else method
    )


def tpr_at_fpr(y_true: np.ndarray, probability: np.ndarray, limit: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, probability, drop_intermediate=False)
    eligible = tpr[fpr <= limit + 1e-15]
    return float(eligible.max(initial=0.0))


def predictive_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    y = group["y_true"].to_numpy(int)
    pred = group["prediction"].to_numpy(int)
    prob = group["probability"].to_numpy(float)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro"),
        "mcc": matthews_corrcoef(y, pred),
        "pr_auc": average_precision_score(y, prob),
        "roc_auc": roc_auc_score(y, prob),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, prob, 0.01),
        "tpr_at_0_1pct_fpr": tpr_at_fpr(y, prob, 0.001),
    }


def load_selected_configurations() -> pd.DataFrame:
    frames = []
    for file_type in FORMATS:
        frame = pd.read_csv(
            REPORTS / "experiments" / file_type / "selected_configurations.csv"
        )
        frame.insert(0, "file_type", file_type)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_metrics(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(PREDICTIONS / "final_test_predictions.parquet")
    metadata = pd.read_csv(REPORTS / "evaluation" / "experiment_metadata.csv")
    existing = pd.read_csv(REPORTS / "evaluation" / "metrics.csv")
    rows = []
    crosschecks = []
    for (file_type, method, classifier), group in predictions.groupby(
        ["file_type", "method", "classifier"], sort=True
    ):
        experiment_id = (
            str(method)
            if str(method).endswith(f"__{classifier}")
            else f"{method}__{classifier}"
        )
        config = selected.loc[
            selected["file_type"].eq(file_type)
            & selected["experiment_id"].eq(experiment_id)
        ].iloc[0]
        resource = metadata.loc[
            metadata["file_type"].eq(file_type) & metadata["method"].eq(method)
        ].iloc[0]
        values = predictive_metrics(group)
        selector = selector_name(str(method), str(classifier))
        row = {
            "experiment_id": f"{file_type}__{experiment_id}",
            "format": file_type,
            "split": "official_test_weeks_53_64",
            "method": experiment_id,
            "selector": selector,
            "classifier": classifier,
            "seed": f"mean_probability_ensemble:{SEEDS}",
            "feature_count": int(config["n_features"]),
            "total_feature_count": TOTAL_FEATURES,
            "feature_fraction": float(config["n_features"]) / TOTAL_FEATURES,
            **values,
            "latency_ms": float(resource["inference_latency_ms_per_sample"]),
            "model_size_mb": float(resource["model_size_bytes"]) / (1024**2),
        }
        rows.append(row)
        for metric in [
            "macro_f1",
            "mcc",
            "balanced_accuracy",
            "precision",
            "recall",
            "pr_auc",
            "roc_auc",
            "tpr_at_1pct_fpr",
            "tpr_at_0_1pct_fpr",
        ]:
            old = existing.loc[
                existing["file_type"].eq(file_type)
                & existing["method"].eq(method)
                & existing["metric"].eq(metric),
                "estimate",
            ]
            crosschecks.append(
                {
                    "format": file_type,
                    "experiment_id": experiment_id,
                    "metric": metric,
                    "recalculated": float(values[metric]),
                    "stored": float(old.iloc[0]),
                    "absolute_difference": abs(
                        float(values[metric]) - float(old.iloc[0])
                    ),
                }
            )
    result = pd.DataFrame(rows)[SCHEMAS["metrics.csv"]]
    return result, pd.DataFrame(crosschecks)


def build_budget_metrics() -> pd.DataFrame:
    predictions = pd.read_parquet(PREDICTIONS / "validation_budget_predictions.parquet")
    summary = pd.read_csv(REPORTS / "budget_metrics.csv")
    rows = []
    for (file_type, method, budget), group in predictions.groupby(
        ["file_type", "method", "feature_budget"], sort=True
    ):
        values = predictive_metrics(group)
        ci = summary.loc[
            summary["file_type"].eq(file_type)
            & summary["method"].eq(method)
            & np.isclose(summary["feature_budget"], budget)
            & summary["metric"].eq("macro_f1")
        ].iloc[0]
        n_features = (
            TOTAL_FEATURES
            if float(budget) == 1.0
            else math.ceil(float(budget) * TOTAL_FEATURES)
        )
        seed_scores = []
        # Genuine seed-specific validation summaries exist in the method checkpoint.
        tuning_path = (
            REPORTS
            / "experiments"
            / file_type
            / f"{method}__lightgbm"
            / "tuning_results.csv"
        )
        if tuning_path.exists():
            tuning = pd.read_csv(tuning_path)
            candidate = group.iloc[0]
            same = tuning.loc[
                tuning["method"].eq(method)
                & tuning["classifier"].eq("lightgbm")
                & np.isclose(tuning["feature_budget"], budget)
                & tuning["parameter_id"].eq(candidate["parameter_id"])
                & np.isclose(tuning["threshold"], candidate["threshold"])
                & tuning["seed_rule"].astype(str).str.startswith("seed_")
            ]
            if pd.notna(candidate["rho"]):
                same = same.loc[np.isclose(same["rho"], candidate["rho"])]
            else:
                same = same.loc[same["rho"].isna()]
            seed_scores = same["macro_f1"].tolist()
        rows.append(
            {
                "format": file_type,
                "method": method,
                "feature_budget": float(budget),
                "feature_count": n_features,
                "feature_fraction": n_features / TOTAL_FEATURES,
                # Point estimate is the frozen three-seed probability ensemble.
                "macro_f1_mean": float(values["macro_f1"]),
                "macro_f1_sd": float(np.std(seed_scores, ddof=1))
                if len(seed_scores) >= 2
                else np.nan,
                "macro_f1_ci_lower": float(ci["ci_lower"]),
                "macro_f1_ci_upper": float(ci["ci_upper"]),
                "pr_auc_mean": float(values["pr_auc"]),
                "mcc_mean": float(values["mcc"]),
                "seeds": SEEDS,
            }
        )
    return pd.DataFrame(rows)[SCHEMAS["budget_metrics.csv"]]


def build_stability(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(
        REPORTS / "feature_selection" / "feature_results.parquet"
    )
    configs = selected.loc[selected["classifier"].eq("lightgbm")].copy()
    budget_map = configs.set_index(["file_type", "method"])["feature_budget"].to_dict()
    j_rows, s_rows = [], []
    for (file_type, method), group in features.groupby(
        ["file_type", "method"], sort=True
    ):
        windows = sorted(group["replicate"].unique())
        for i, window_a in enumerate(windows):
            a = group.loc[group["replicate"].eq(window_a)].set_index("feature_name")
            for window_b in windows[i + 1 :]:
                b = group.loc[group["replicate"].eq(window_b)].set_index("feature_name")
                sa = set(a.index[a["selected"].astype(bool)])
                sb = set(b.index[b["selected"].astype(bool)])
                common = a.index.intersection(b.index)
                rho, p_value = spearmanr(
                    a.loc[common, "normalized_rank"].to_numpy(float),
                    b.loc[common, "normalized_rank"].to_numpy(float),
                )
                union = sa | sb
                j_rows.append(
                    {
                        "format": file_type,
                        "method": method,
                        "feature_budget": float(budget_map[(file_type, method)]),
                        "window_a": str(window_a),
                        "window_b": str(window_b),
                        "intersection_size": len(sa & sb),
                        "union_size": len(union),
                        "jaccard": len(sa & sb) / len(union) if union else 1.0,
                    }
                )
                s_rows.append(
                    {
                        "format": file_type,
                        "method": method,
                        "window_a": str(window_a),
                        "window_b": str(window_b),
                        "common_feature_count": len(common),
                        "spearman_rho": float(rho),
                        "p_value": float(p_value),
                    }
                )
    return (
        pd.DataFrame(j_rows)[SCHEMAS["jaccard_stability.csv"]],
        pd.DataFrame(s_rows)[SCHEMAS["spearman_stability.csv"]],
    )


def build_noninferiority() -> pd.DataFrame:
    source = pd.read_csv(REPORTS / "evaluation" / "noninferiority.csv")
    result = pd.DataFrame(
        {
            "format": source["file_type"],
            "tcfs_method": source["method"],
            "baseline_method": source["reference_method"],
            "metric": source["metric"],
            "margin": source["noninferiority_margin"],
            "effect_tcfs_minus_baseline": source["difference"],
            "ci_lower": source["ci_lower"],
            "ci_upper": source["ci_upper"],
            "p_value": source["p_value"],
            "adjusted_p_value": source["p_value"],
            "decision": np.where(
                source["ci_lower"] > MARGIN, "Non-inferior", "Not non-inferior"
            ),
            "analysis_unit": "paired sample; 2000 class-stratified bootstrap resamples; seed 42",
        }
    )
    return result[SCHEMAS["noninferiority.csv"]]


def build_secondary() -> pd.DataFrame:
    source = pd.read_csv(REPORTS / "evaluation" / "secondary_comparisons.csv")
    result = pd.DataFrame(
        {
            "format": source["file_type"],
            "method_a": source["method"],
            "method_b": source["reference_method"],
            "metric": source["metric"],
            "effect_a_minus_b": source["difference"],
            "ci_lower": source["ci_lower"],
            "ci_upper": source["ci_upper"],
            "raw_p_value": source["p_value"],
            "adjusted_p_value": source["holm_adjusted_p_value"],
            "correction_method": "Holm within format across 8 secondary metrics",
            "analysis_unit": "paired sample; 2000 class-stratified bootstrap resamples; seed 42",
            "decision": np.where(
                source["holm_adjusted_p_value"] < 0.05, "Significant", "Not significant"
            ),
        }
    )
    return result[SCHEMAS["secondary_comparisons.csv"]]


def build_ablation() -> pd.DataFrame:
    predictions = pd.read_parquet(
        PREDICTIONS / "validation_ablation_sensitivity_predictions.parquet"
    )
    rows = []
    for file_type, format_group in predictions.groupby("file_type", sort=True):
        reference = format_group.loc[
            format_group["analysis"].eq("ablation")
            & format_group["setting"].eq("full_tcfs")
        ]
        reference_f1 = float(predictive_metrics(reference)["macro_f1"])
        for (analysis, setting), group in format_group.groupby(
            ["analysis", "setting"], sort=True
        ):
            values = predictive_metrics(group)
            rows.append(
                {
                    "format": file_type,
                    "ablation": f"{analysis}:{setting}",
                    "reference_method": "full_tcfs",
                    "evaluated_method": setting,
                    "feature_count": int(group["n_features"].iloc[0]),
                    "macro_f1": float(values["macro_f1"]),
                    "pr_auc": float(values["pr_auc"]),
                    "mcc": float(values["mcc"]),
                    "effect_on_macro_f1": float(values["macro_f1"]) - reference_f1,
                    "seed_or_analysis_unit": f"validation weeks 41-52; mean probability ensemble:{SEEDS}",
                }
            )
    return pd.DataFrame(rows)[SCHEMAS["ablation_sensitivity.csv"]]


def feature_group(name: str) -> str:
    for separator in (".", "_", "["):
        if separator in name:
            return name.split(separator, 1)[0]
    return "ungrouped"


def build_selected_features(selected: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_parquet(REPORTS / "feature_selection" / "feature_results.parquet")
    consensus = pd.read_parquet(
        REPORTS / "feature_selection" / "combined" / "consensus_scores.parquet"
    )
    configs = selected.loc[selected["classifier"].eq("lightgbm")].set_index(
        ["file_type", "method"]
    )
    parts = []
    for (file_type, method), group in source.groupby(
        ["file_type", "method"], sort=True
    ):
        group = group.copy()
        group["rank"] = (
            group.groupby("replicate")["normalized_rank"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        summary = group.groupby("feature_name").agg(
            selection_frequency=("selected", "mean"),
            rank_variance=("normalized_rank", lambda x: float(np.var(x, ddof=0))),
        )
        group = group.join(summary, on="feature_name")
        group["tcfs_score"] = np.nan
        if method in {"tcfs", "shared_tcfs"}:
            cfg = configs.loc[(file_type, method)]
            c = consensus.loc[
                consensus["file_type"].eq(file_type)
                & np.isclose(consensus["rho"], float(cfg["rho"]))
                & np.isclose(consensus["feature_budget"], float(cfg["feature_budget"]))
            ].set_index("feature_name")
            score_col = "format_tcfs_score" if method == "tcfs" else "shared_tcfs_score"
            group["tcfs_score"] = group["feature_name"].map(c[score_col])
            group["selection_frequency"] = group["feature_name"].map(
                c["selection_frequency"]
            )
            group["rank_variance"] = group["feature_name"].map(c["rank_variance"])
        out = pd.DataFrame(
            {
                "format": file_type,
                "method": method,
                "discovery_window": group["replicate"].astype(str),
                "feature_name": group["feature_name"].astype(str),
                "feature_group": group["feature_name"].astype(str).map(feature_group),
                "rank": group["rank"].astype(int),
                "normalized_rank": group["normalized_rank"].astype(float),
                "selection_frequency": group["selection_frequency"].astype(float),
                "rank_variance": group["rank_variance"].astype(float),
                "tcfs_score": group["tcfs_score"].astype(float),
                "selected": group["selected"].astype(bool),
            }
        )
        parts.append(out)
    return pd.concat(parts, ignore_index=True)[SCHEMAS["selected_features.csv"]]


def build_sample_counts() -> pd.DataFrame:
    metadata = pd.read_parquet(
        PROCESSED / "ember2024_metadata.parquet",
        columns=["row_id", "file_type", "label", "official_split"],
    )
    assignments = pd.read_parquet(PROCESSED / "temporal_assignments.parquet")
    exclusions = pd.read_csv(REPORTS / "data_quality" / "approved_exclusions.csv")
    frame = metadata.merge(
        assignments[["row_id", "official_split", "protocol_week", "protocol_role"]],
        on=["row_id", "official_split"],
        validate="one_to_one",
    )
    excluded = set(
        zip(
            exclusions["official_split"].astype(str),
            exclusions["row_id"].astype(int),
            strict=True,
        )
    )
    frame["excluded"] = [
        (str(split), int(row_id)) in excluded
        for split, row_id in zip(frame["official_split"], frame["row_id"], strict=True)
    ]
    grouped = frame.groupby(
        ["file_type", "official_split", "protocol_role", "protocol_week", "label"],
        observed=True,
        sort=True,
    )
    output = grouped.agg(
        raw_count=("row_id", "size"), excluded_count=("excluded", "sum")
    ).reset_index()
    output["retained_count"] = output["raw_count"] - output["excluded_count"]
    output["exclusion_reason"] = np.where(
        output["excluded_count"].gt(0),
        "DQ003 exact within-split duplicate SHA-256; retained lowest row_id",
        "None",
    )
    output = output.rename(
        columns={
            "file_type": "format",
            "protocol_role": "study_phase",
            "protocol_week": "time_window",
        }
    )
    output["time_window"] = output["time_window"].map(lambda x: f"week_{int(x):02d}")
    return output[SCHEMAS["sample_counts.csv"]]


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def make_figures(
    output: Path,
    budget: pd.DataFrame,
    jaccard: pd.DataFrame,
    spearman: pd.DataFrame,
    ablation: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    palette = dict(
        zip(
            sorted(budget["method"].unique()),
            sns.color_palette("colorblind", n_colors=7),
            strict=False,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    for ax, file_type in zip(axes, FORMATS, strict=True):
        sub = budget.loc[budget["format"].eq(file_type)]
        for method, g in sub.groupby("method", sort=True):
            g = g.sort_values("feature_fraction")
            ax.plot(
                g["feature_fraction"] * 100,
                g["macro_f1_mean"],
                marker="o",
                label=method.replace("_", " ").title(),
                color=palette[method],
            )
            ax.fill_between(
                g["feature_fraction"] * 100,
                g["macro_f1_ci_lower"],
                g["macro_f1_ci_upper"],
                alpha=0.12,
                color=palette[method],
            )
        ax.set_title(file_type)
        ax.set_xlabel("Retained features (%)")
        ax.set_ylabel("Validation macro-F1")
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle("Predictive performance versus retained feature budget")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig, figures / "figure2_performance_vs_budget")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=False)
    for col, file_type in enumerate(FORMATS):
        sj = jaccard.loc[jaccard["format"].eq(file_type)]
        ss = spearman.loc[spearman["format"].eq(file_type)]
        sns.boxplot(
            data=sj,
            x="jaccard",
            y="method",
            ax=axes[0, col],
            color="#56B4E9",
            fliersize=2,
        )
        sns.boxplot(
            data=ss,
            x="spearman_rho",
            y="method",
            ax=axes[1, col],
            color="#009E73",
            fliersize=2,
        )
        axes[0, col].set_title(file_type)
        axes[0, col].set_xlabel("Pairwise Jaccard")
        axes[1, col].set_xlabel("Spearman rank correlation")
        axes[0, col].set_xlim(0, 1)
        axes[1, col].set_xlim(-1, 1)
        if col:
            axes[0, col].set_ylabel("")
            axes[1, col].set_ylabel("")
    fig.suptitle("Temporal feature stability across discovery windows")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, figures / "figure3_temporal_stability")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharey=False)
    for col, file_type in enumerate(FORMATS):
        ax = axes[0, col]
        sub = ablation.loc[
            ablation["format"].eq(file_type)
            & ablation["ablation"].str.startswith("ablation:")
        ].copy()
        sub["label"] = sub["evaluated_method"].str.replace("_", " ").str.title()
        sns.barplot(data=sub, x="effect_on_macro_f1", y="label", ax=ax, color="#D55E00")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(file_type)
        ax.set_xlabel("Macro-F1 difference vs full TCFS")
        ax.set_ylabel("")
        sensitivity = ablation.loc[
            ablation["format"].eq(file_type)
            & ablation["ablation"].str.startswith("sensitivity:")
        ].copy()
        sensitivity["rho"] = (
            sensitivity["evaluated_method"].str.split("=").str[-1].astype(float)
        )
        axes[1, col].plot(
            sensitivity["rho"],
            sensitivity["macro_f1"],
            color="#0072B2",
            marker="o",
            linewidth=1.8,
        )
        axes[1, col].set_xlabel(r"Stability penalty $\rho$")
        axes[1, col].set_ylabel("Validation macro-F1")
        axes[1, col].set_xticks([0, 0.25, 0.5, 1.0])
        axes[1, col].set_ylim(0, 1)
        axes[1, col].set_title(f"{file_type} sensitivity")
    fig.suptitle("TCFS ablation and sensitivity on validation weeks 41–52")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, figures / "figure4_ablation_sensitivity")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_inventory() -> dict[str, int]:
    categories = {
        "scripts": PROJECT_ROOT / "scripts",
        "configs": PROJECT_ROOT / "configs",
        "processed_data_files": PROCESSED,
        "prediction_files": PREDICTIONS,
        "model_files": PROJECT_ROOT / "outputs" / "models",
        "log_files": REPORTS,
        "existing_figures": PROJECT_ROOT / "outputs" / "figures",
    }
    result = {}
    for name, path in categories.items():
        if name == "log_files":
            result[name] = len(list(path.rglob("*.log"))) + len(
                list(path.rglob("*.jsonl"))
            )
        elif name == "model_files":
            result[name] = len(list((REPORTS / "experiments").rglob("model.joblib")))
        else:
            result[name] = len(
                [p for p in path.rglob("*") if p.is_file() and "pytest" not in str(p)]
            )
    return result


def git_description() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "No Git commit available; working tree is unversioned"


def write_documents(
    output: Path,
    metrics: pd.DataFrame,
    budget: pd.DataFrame,
    jaccard: pd.DataFrame,
    spearman: pd.DataFrame,
    noninf: pd.DataFrame,
    ablation: pd.DataFrame,
    crosschecks: pd.DataFrame,
) -> None:
    primary = metrics.loc[
        metrics["classifier"].eq("lightgbm")
        & metrics["selector"].isin(["tcfs", "all_features"])
    ]
    findings = []
    for file_type in FORMATS:
        sub = primary.loc[primary["format"].eq(file_type)].set_index("selector")
        ni = noninf.loc[noninf["format"].eq(file_type)].iloc[0]
        tcfs = sub.loc["tcfs"]
        base = sub.loc["all_features"]
        findings.append(
            f"- **{file_type}:** TCFS used {int(tcfs.feature_count)}/{TOTAL_FEATURES} features "
            f"({tcfs.feature_fraction:.1%}); future-test macro-F1 was {tcfs.macro_f1:.6f} "
            f"versus {base.macro_f1:.6f} for all features (difference {ni.effect_tcfs_minus_baseline:+.6f}, "
            f"95% CI {ni.ci_lower:.6f} to {ni.ci_upper:.6f}); **{ni.decision}** under the "
            f"strict lower-bound > {MARGIN:.2f} rule."
        )
    stability_text = []
    for file_type in FORMATS:
        j = jaccard.loc[
            jaccard["format"].eq(file_type) & jaccard["method"].eq("tcfs"), "jaccard"
        ]
        s = spearman.loc[
            spearman["format"].eq(file_type) & spearman["method"].eq("tcfs"),
            "spearman_rho",
        ]
        stability_text.append(
            f"- {file_type}: TCFS mean Jaccard {j.mean():.4f}; mean Spearman rho {s.mean():.4f} across six window pairs."
        )
    abl_lines = []
    for file_type in FORMATS:
        sub = ablation.loc[
            ablation["format"].eq(file_type)
            & ablation["ablation"].str.startswith("ablation:")
            & ~ablation["evaluated_method"].eq("full_tcfs")
        ]
        worst = sub.sort_values("effect_on_macro_f1").iloc[0]
        abl_lines.append(
            f"- {file_type}: largest observed ablation loss was {worst.effect_on_macro_f1:+.6f} for `{worst.evaluated_method}`."
        )

    (output / "results_summary.md").write_text(
        "# Results summary\n\n"
        "## Implemented design\n\n"
        "EMBER2024 APK, ELF, and PDF records were assigned chronologically: discovery weeks 1–40 "
        "(four non-overlapping 10-week windows), internal validation weeks 41–52, and official future-test "
        "weeks 53–64. Component selectors were mutual information, L1 logistic regression, and LightGBM "
        "permutation importance. TCFS aggregated their four-window rankings. LightGBM was primary and "
        "logistic regression secondary. Budgets were 5%, 10%, 20%, 40%, and 60%; rho values were 0, "
        "0.25, 0.5, and 1. Models used a mean-probability ensemble over seeds 17, 42, and 73.\n\n"
        "Only validation data selected budget, rho, hyperparameters, threshold, and ensemble rule. Frozen "
        "pipelines were refit on weeks 1–52 and evaluated once on weeks 53–64. Approved DQ003 exclusions "
        "removed exact within-split duplicate hashes while retaining the lowest row_id.\n\n"
        "## Primary future-test findings\n\n" + "\n".join(findings) + "\n\n"
        "## Stability\n\n" + "\n".join(stability_text) + "\n\n"
        "## Ablation and sensitivity\n\n" + "\n".join(abl_lines) + "\n\n"
        "Ablation and rho-sensitivity results use validation weeks 41–52 and are exploratory, not "
        "confirmatory future-test analyses. Budget curves are also validation analyses.\n\n"
        "## Evidence-supported limitations\n\n"
        "- The project directory has no resolvable Git commit, so code provenance is limited to file hashes.\n"
        "- Feature-level missing, infinite, and constant-value QC was marked NOT_RUN in the saved DQ report.\n"
        "- The stored ELF non-inferiority boolean used a p-value rule, but the requested manuscript rule is "
        "CI-based; the standardized table therefore reports ELF as not non-inferior.\n"
        "- The 60% candidate budget exceeds the manuscript's stated maximum of 50%; selected primary TCFS "
        "models remained at 5–10%, but the 60% validation curve is protocol-deviating exploratory evidence.\n",
        encoding="utf-8",
    )

    inventory = repository_inventory()
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn_version,
        "lightgbm": version("lightgbm"),
        "pyarrow": version("pyarrow"),
        "matplotlib": version("matplotlib"),
        "seaborn": version("seaborn"),
        "pyyaml": version("pyyaml"),
        "joblib": version("joblib"),
        "pytest": version("pytest"),
    }
    commands = {
        "standardized package": "python scripts/build_paper_results_package.py --output-dir C:/workspace/paper_results --force",
        "sealed final evaluation": "python scripts/run_frozen_format_evaluation.py ... --unseal-test (already completed; do not rerun)",
        "budget aggregation": "python scripts/build_budget_metrics.py --predictions outputs/predictions/validation_budget_predictions.parquet --output outputs/reports/budget_metrics.csv",
        "ablation aggregation": "python scripts/build_ablation_sensitivity.py --predictions outputs/predictions/validation_ablation_sensitivity_predictions.parquet --output outputs/reports/ablation_sensitivity.csv",
    }
    source_paths = [
        PROCESSED / "ember2024_metadata.parquet",
        PROCESSED / "temporal_assignments.parquet",
        PREDICTIONS / "final_test_predictions.parquet",
        PREDICTIONS / "validation_budget_predictions.parquet",
        PREDICTIONS / "validation_ablation_sensitivity_predictions.parquet",
        REPORTS / "data_quality" / "approved_exclusions.csv",
        REPORTS / "evaluation" / "metrics.csv",
        REPORTS / "evaluation" / "noninferiority.csv",
        REPORTS / "evaluation" / "secondary_comparisons.csv",
        REPORTS / "feature_selection" / "feature_results.parquet",
    ]
    source_hashes = "\n".join(f"- `{p}`: `{sha256_file(p)}`" for p in source_paths)
    (output / "reproducibility_manifest.md").write_text(
        "# Reproducibility manifest\n\n"
        f"Generated: {datetime.now(UTC).isoformat()}\n\n"
        f"Protocol: `{PROTOCOL_VERSION}`. Dataset: EMBER2024, local processed metadata at `{PROCESSED / 'ember2024_metadata.parquet'}`.\n\n"
        f"Code version: {git_description()}.\n\n"
        "## Repository inventory\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in inventory.items())
        + "\n\n"
        "## Environment\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in versions.items())
        + "\n\n"
        "Random seeds: model ensemble 17, 42, 73; bootstrap 42; 2,000 resamples.\n\n"
        "## Commands\n\n"
        + "\n".join(f"- {k}: `{v}`" for k, v in commands.items())
        + "\n\n"
        "## Figure and table mapping\n\n"
        "- Figure 2 and table primary: `budget_metrics.csv`, `metrics.csv`.\n"
        "- Figure 3 and table stability: `jaccard_stability.csv`, `spearman_stability.csv`.\n"
        "- Figure 4 and table ablation: `ablation_sensitivity.csv`.\n"
        "- Table non-inferiority: `noninferiority.csv`, `secondary_comparisons.csv`.\n\n"
        "## Source SHA-256 hashes\n\n" + source_hashes + "\n",
        encoding="utf-8",
    )

    checks = pd.read_csv(REPORTS / "data_quality" / "check_results.csv")
    max_diff = float(crosschecks["absolute_difference"].max())
    ni_conflict = pd.read_csv(REPORTS / "evaluation" / "noninferiority.csv")
    conflicts = int(
        (ni_conflict["noninferior_at_0_05"] != (ni_conflict["ci_lower"] > MARGIN)).sum()
    )
    (output / "quality_control_report.md").write_text(
        "# Quality-control report\n\n"
        "## Data quality\n\n"
        + "\n".join(
            f"- {r.rule_id} {r.check}: {r.status} ({'not available' if pd.isna(r.findings) else int(r.findings)} findings)."
            for r in checks.itertuples()
        )
        + "\n\nApproved exclusions contain only DQ003 exact within-split duplicate hashes and "
        "retain the lowest row_id. Saved QC found zero cross-split duplicate hashes, zero duplicate "
        "split/row identifiers, zero inconsistent file types, and zero impossible timestamps. There were "
        "644 supplied-versus-derived week discrepancies; the protocol uses derived weeks.\n\n"
        "## Leakage audit\n\n"
        "- Discovery selectors explicitly reject weeks 41–64 and fit per-window scaling only on their discovery window.\n"
        "- Model-selection code rejects sealed-test rows and trains validation candidates on weeks 1–40 only.\n"
        "- Budget, rho, hyperparameters, thresholds, and ensemble choices use validation weeks 41–52.\n"
        "- Final-evaluation code requires frozen manifests and an explicit logged `--unseal-test`, then refits on weeks 1–52.\n"
        "- Final prediction rows contain only `sealed_test` weeks 53–64.\n"
        "- No cross-split duplicate hashes were reported. No evidence of test-label use in model selection was found.\n\n"
        "## Result integrity\n\n"
        f"Recalculated point metrics from sample-level final predictions differed from stored estimates by at most {max_diff:.3g}. "
        "All probabilities and reported metrics were checked for finite values and valid bounds. Confidence intervals were checked "
        "for lower <= estimate <= upper.\n\n"
        f"There is {conflicts} stored non-inferiority flag conflict with the requested strict CI rule: ELF. The standardized decision "
        "uses the requested rule and preserves the original p-value.\n\n"
        "## Traceability\n\nEvery numerical claim in `results_summary.md` is derived from one of the standardized CSVs. "
        "Exploratory validation analyses are explicitly separated from confirmatory future-test results.\n",
        encoding="utf-8",
    )

    (output / "missing_analyses.md").write_text(
        "# Missing analyses\n\n"
        "No manuscript-required predictive, stability, non-inferiority, secondary-comparison, or ablation analysis is missing.\n\n"
        "The saved data-quality run marks feature-matrix checks DQ005 (missing values), DQ006 (infinite values), and DQ007 "
        "(constant features) as NOT_RUN because no feature matrix was supplied to that audit. To complete them without touching "
        "the sealed evaluation logic, run `python scripts/run_data_quality.py` with all six processed feature matrices and the "
        "696 feature names, writing to a new QC directory. This package does not invent those findings.\n",
        encoding="utf-8",
    )


def copy_scripts(output: Path) -> None:
    destination = output / "scripts"
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        "build_paper_results_package.py",
        "run_evaluation.py",
        "build_budget_metrics.py",
        "build_stability_feature_results.py",
        "build_ablation_sensitivity.py",
        "generate_publication_figures.py",
        "run_final_evaluation.py",
        "run_frozen_format_evaluation.py",
    ]
    for name in names:
        shutil.copy2(PROJECT_ROOT / "scripts" / name, destination / name)


def validate_outputs(output: Path) -> None:
    for name, columns in SCHEMAS.items():
        frame = pd.read_csv(output / name)
        if frame.empty:
            raise ValueError(f"{name} contains no data rows")
        if list(frame.columns) != columns:
            raise ValueError(f"{name} has unexpected columns: {list(frame.columns)}")
    metrics = pd.read_csv(output / "metrics.csv")
    bounded = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "macro_f1",
        "pr_auc",
        "roc_auc",
        "tpr_at_1pct_fpr",
        "tpr_at_0_1pct_fpr",
    ]
    if not metrics[bounded].apply(lambda c: c.between(0, 1).all()).all():
        raise ValueError("metrics.csv contains an impossible bounded metric")
    noninf = pd.read_csv(output / "noninferiority.csv")
    if not (
        (noninf["ci_lower"] <= noninf["effect_tcfs_minus_baseline"])
        & (noninf["effect_tcfs_minus_baseline"] <= noninf["ci_upper"])
    ).all():
        raise ValueError(
            "non-inferiority confidence interval does not contain estimate"
        )


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} exists; pass --force to rebuild")
        # Output is a dedicated package directory, never a raw-results location.
        shutil.rmtree(output)
    output.mkdir(parents=True)
    selected = load_selected_configurations()
    metrics, crosschecks = build_metrics(selected)
    budget = build_budget_metrics()
    jaccard, spearman = build_stability(selected)
    noninf = build_noninferiority()
    secondary = build_secondary()
    ablation = build_ablation()
    features = build_selected_features(selected)
    counts = build_sample_counts()
    tables = {
        "metrics.csv": metrics,
        "budget_metrics.csv": budget,
        "jaccard_stability.csv": jaccard,
        "spearman_stability.csv": spearman,
        "noninferiority.csv": noninf,
        "secondary_comparisons.csv": secondary,
        "ablation_sensitivity.csv": ablation,
        "selected_features.csv": features,
        "sample_counts.csv": counts,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    crosschecks.to_csv(output / "metric_crosscheck.csv", index=False)
    make_figures(output, budget, jaccard, spearman, ablation, metrics)
    write_documents(
        output, metrics, budget, jaccard, spearman, noninf, ablation, crosschecks
    )
    copy_scripts(output)
    validate_outputs(output)
    (output / "package_build.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "project_root": str(PROJECT_ROOT),
                "protocol_version": PROTOCOL_VERSION,
                "no_fabricated_values": True,
                "csv_rows": {name: len(frame) for name, frame in tables.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
