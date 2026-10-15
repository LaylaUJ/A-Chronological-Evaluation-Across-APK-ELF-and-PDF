"""Reproducible publication figures and tables from saved artifacts only."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tcfs-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tcfs.evaluation import PREDICTIVE_METRICS, RESOURCE_METRICS
from tcfs.temporal_split import PROTOCOL_VERSION

FORMATS = ("APK", "ELF", "PDF")
METHOD_LABELS = {
    "tcfs": "TCFS",
    "all_features": "All features",
    "mutual_information": "Mutual information",
    "l1_logistic": "L1 logistic",
    "lightgbm_permutation": "LightGBM permutation",
    "random_matched": "Random matched",
    "shared_tcfs": "Shared TCFS",
}
METHOD_ORDER = (
    "tcfs",
    "all_features",
    "mutual_information",
    "l1_logistic",
    "lightgbm_permutation",
    "random_matched",
    "shared_tcfs",
)
ABLATION_LABELS = {
    "mean_rank_only": "Mean rank only",
    "without_frequency": "Without selection frequency",
    "without_stability_penalty": "Without stability penalty",
}
METRIC_LABELS = {
    "macro_f1": "Macro-F1",
    "mcc": "MCC",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "tpr_at_1pct_fpr": "TPR at 1% FPR",
    "tpr_at_0_1pct_fpr": "TPR at 0.1% FPR",
    "feature_count": "Features",
    "model_size_bytes": "Model size (bytes)",
    "inference_latency_ms_per_sample": "Latency (ms/sample)",
}

# Okabe-Ito: color-vision-deficiency-safe, paired with marker/line encodings.
PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)
MARKERS = ("o", "s", "^", "D", "P", "X", "v")
LINESTYLES = ("-", "--", "-.", ":")


def configure_publication_style() -> None:
    """Set deterministic, journal-friendly Matplotlib defaults."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "tcfs-malware-publication",
        }
    )


def generate_figure_1(assignments_path: str | Path, output_dir: str | Path) -> None:
    """Create the chronological study-design figure."""
    configure_publication_style()
    assignments = read_table(assignments_path)
    required = {"protocol_week", "protocol_role", "protocol_version"}
    _require_columns(assignments, required, "temporal assignments")
    if set(assignments["protocol_version"].astype(str)) != {PROTOCOL_VERSION}:
        raise ValueError("Figure 1 requires the frozen protocol version")
    weeks = pd.to_numeric(assignments["protocol_week"], errors="raise")
    if set(weeks) != set(range(1, 65)):
        raise ValueError("Figure 1 assignments must cover protocol weeks 1-64")

    fig, ax = plt.subplots(figsize=(10.0, 3.2), constrained_layout=True)
    lanes = [
        ("Discovery / feature selection", 1, 40, PALETTE[0]),
        ("Validation / model selection", 41, 52, PALETTE[2]),
        ("Sealed final test", 53, 64, PALETTE[1]),
    ]
    for lane_index, (_label, start, end, color) in enumerate(lanes):
        width = end - start + 1
        ax.barh(
            lane_index,
            width,
            left=start - 0.5,
            height=0.55,
            color=color,
            edgecolor="#222222",
            linewidth=0.7,
            hatch="///" if lane_index == 2 else None,
        )
        count = int(assignments.loc[weeks.between(start, end)].shape[0])
        ax.text(
            (start + end) / 2,
            lane_index,
            f"Weeks {start}-{end}\n{count:,} samples",
            ha="center",
            va="center",
            color="white" if lane_index != 1 else "black",
            fontsize=8,
            fontweight="bold",
        )
    for window, start in enumerate((1, 11, 21, 31), start=1):
        ax.axvline(start - 0.5, color="#444444", linewidth=0.6)
        ax.text(start + 4.5, 0.39, f"W{window}", ha="center", va="top")
    ax.axvline(40.5, color="#222222", linewidth=1.0)
    ax.axvline(52.5, color="#222222", linewidth=1.0)
    ax.set_xlim(0.5, 64.5)
    ax.set_xticks([1, 10, 20, 30, 40, 52, 64])
    ax.set_xlabel("Dataset week")
    ax.set_yticks(range(len(lanes)), [lane[0] for lane in lanes])
    ax.invert_yaxis()
    ax.set_title("Chronological study design")
    ax.grid(axis="y", visible=False)
    ax.text(
        58.5,
        1.68,
        "Opened once after all choices were frozen",
        ha="center",
        va="center",
        fontsize=8,
    )
    export_figure(fig, Path(output_dir), "figure_1_chronological_study_design")


def generate_figure_2(
    jaccard_path: str | Path,
    spearman_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Create pairwise feature-set stability small multiples."""
    configure_publication_style()
    jaccard = read_table(jaccard_path)
    spearman = read_table(spearman_path)
    common = {"file_type", "method", "replicate_a", "replicate_b"}
    _require_columns(jaccard, common | {"jaccard"}, "Jaccard stability")
    _require_columns(spearman, common | {"spearman_rho"}, "Spearman stability")
    methods = _ordered_methods(set(jaccard["method"]) | set(spearman["method"]))
    styles = _method_styles(methods)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 6.0),
        sharey="row",
        constrained_layout=True,
    )
    for column, file_type in enumerate(FORMATS):
        _stability_panel(
            axes[0, column],
            jaccard.loc[jaccard["file_type"].eq(file_type)],
            "jaccard",
            methods,
            styles,
        )
        _stability_panel(
            axes[1, column],
            spearman.loc[spearman["file_type"].eq(file_type)],
            "spearman_rho",
            methods,
            styles,
        )
        axes[0, column].set_title(file_type)
        axes[1, column].set_xlabel("Feature-selection method")
    axes[0, 0].set_ylabel("Pairwise Jaccard similarity")
    axes[1, 0].set_ylabel("Pairwise Spearman correlation")
    fig.suptitle("Feature-set stability across discovery windows", fontsize=11)
    export_figure(fig, Path(output_dir), "figure_2_feature_set_stability")


def generate_figure_3(
    budget_metrics_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Plot future macro-F1 against retained feature percentage."""
    configure_publication_style()
    metrics = read_table(budget_metrics_path)
    required = {
        "file_type",
        "method",
        "feature_budget",
        "metric",
        "estimate",
        "ci_lower",
        "ci_upper",
    }
    _require_columns(metrics, required, "budget metrics")
    metrics = metrics.loc[metrics["metric"].eq("macro_f1")].copy()
    if metrics.empty:
        raise ValueError("budget metrics contains no macro_f1 rows")
    metrics["feature_budget"] = pd.to_numeric(
        metrics["feature_budget"], errors="raise"
    )
    methods = _ordered_methods(metrics["method"].astype(str).unique())
    styles = _method_styles(methods)
    fig, axes = plt.subplots(
        1, 3, figsize=(10.5, 3.5), sharey=True, constrained_layout=True
    )
    for ax, file_type in zip(axes, FORMATS, strict=True):
        subset = metrics.loc[metrics["file_type"].eq(file_type)]
        if subset.empty:
            raise ValueError(f"budget metrics is missing {file_type}")
        for method in methods:
            series = subset.loc[subset["method"].eq(method)].sort_values(
                "feature_budget"
            )
            if series.empty:
                continue
            x = series["feature_budget"].to_numpy(dtype=float) * 100
            style = styles[method]
            ax.plot(
                x,
                series["estimate"],
                label=_method_label(method),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
            )
            ax.fill_between(
                x,
                series["ci_lower"].to_numpy(dtype=float),
                series["ci_upper"].to_numpy(dtype=float),
                color=style["color"],
                alpha=0.14,
                linewidth=0,
            )
        ax.set_title(file_type)
        ax.set_xlabel("Retained features (%)")
        ax.set_xticks(sorted(set(subset["feature_budget"] * 100)))
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Future macro-F1 (95% CI)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(3, len(labels)))
    fig.suptitle("Future performance versus retained feature percentage", fontsize=11)
    export_figure(fig, Path(output_dir), "figure_3_future_performance")


def generate_figure_4(
    ablation_sensitivity_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Create ablation forest plot and rho sensitivity curves."""
    configure_publication_style()
    results = read_table(ablation_sensitivity_path)
    required = {
        "file_type",
        "analysis",
        "setting",
        "metric",
        "estimate",
        "ci_lower",
        "ci_upper",
        "x_value",
    }
    _require_columns(results, required, "ablation and sensitivity results")
    results = results.loc[results["metric"].eq("macro_f1")].copy()
    ablation = results.loc[results["analysis"].eq("ablation")]
    sensitivity = results.loc[results["analysis"].eq("sensitivity")]
    if ablation.empty or sensitivity.empty:
        raise ValueError("Figure 4 requires ablation and sensitivity rows")

    fig, (ax_ablation, ax_sensitivity) = plt.subplots(
        1, 2, figsize=(10.5, 4.5), constrained_layout=True
    )
    settings = list(dict.fromkeys(ablation["setting"].astype(str)))
    y_positions = np.arange(len(settings))
    offsets = np.linspace(-0.22, 0.22, len(FORMATS))
    for format_index, file_type in enumerate(FORMATS):
        subset = ablation.loc[ablation["file_type"].eq(file_type)].set_index(
            "setting"
        )
        subset = subset.reindex(settings)
        if subset["estimate"].isna().any():
            raise ValueError(f"ablation results are incomplete for {file_type}")
        estimate = subset["estimate"].to_numpy(dtype=float)
        errors = np.vstack(
            [
                estimate - subset["ci_lower"].to_numpy(dtype=float),
                subset["ci_upper"].to_numpy(dtype=float) - estimate,
            ]
        )
        ax_ablation.errorbar(
            estimate,
            y_positions + offsets[format_index],
            xerr=errors,
            fmt=MARKERS[format_index],
            color=PALETTE[format_index],
            capsize=2,
            label=file_type,
        )
    ax_ablation.axvline(0, color="#333333", linewidth=0.8)
    ax_ablation.set_yticks(
        y_positions,
        [
            ABLATION_LABELS.get(setting, setting.replace("_", " "))
            for setting in settings
        ],
    )
    ax_ablation.invert_yaxis()
    ax_ablation.set_xlabel("Macro-F1 difference from full TCFS")
    ax_ablation.set_title("A  Component ablation")
    ax_ablation.legend(frameon=False)

    for format_index, file_type in enumerate(FORMATS):
        subset = sensitivity.loc[sensitivity["file_type"].eq(file_type)].sort_values(
            "x_value"
        )
        if subset.empty:
            raise ValueError(f"sensitivity results are missing {file_type}")
        x = pd.to_numeric(subset["x_value"], errors="raise").to_numpy()
        ax_sensitivity.plot(
            x,
            subset["estimate"],
            color=PALETTE[format_index],
            marker=MARKERS[format_index],
            label=file_type,
        )
        ax_sensitivity.fill_between(
            x,
            subset["ci_lower"].to_numpy(dtype=float),
            subset["ci_upper"].to_numpy(dtype=float),
            color=PALETTE[format_index],
            alpha=0.14,
            linewidth=0,
        )
    ax_sensitivity.set_xlabel(r"Stability penalty $\rho$")
    ax_sensitivity.set_ylabel("Validation macro-F1 (95% CI)")
    ax_sensitivity.set_ylim(0, 1.02)
    ax_sensitivity.set_title("B  Sensitivity to stability penalty")
    ax_sensitivity.legend(frameon=False)
    fig.suptitle("Ablation and sensitivity analysis", fontsize=11)
    export_figure(fig, Path(output_dir), "figure_4_ablation_sensitivity")


def generate_table_1(
    record_counts_path: str | Path,
    check_results_path: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Build dataset composition and quality-control Table 1."""
    counts = read_table(record_counts_path)
    checks = read_table(check_results_path)
    _require_columns(
        counts,
        {"official_split", "file_type", "label", "week_id", "record_count"},
        "record counts",
    )
    _require_columns(checks, {"rule_id", "status", "findings"}, "QC checks")
    rows: list[dict[str, Any]] = []
    for file_type in (*FORMATS, "Overall"):
        subset = counts if file_type == "Overall" else counts.loc[
            counts["file_type"].eq(file_type)
        ]
        rows.append(
            {
                "Format": file_type,
                "Total": int(subset["record_count"].sum()),
                "Train": int(
                    subset.loc[
                        subset["official_split"].eq("train"), "record_count"
                    ].sum()
                ),
                "Test": int(
                    subset.loc[
                        subset["official_split"].eq("test"), "record_count"
                    ].sum()
                ),
                "Benign": int(
                    subset.loc[subset["label"].eq(0), "record_count"].sum()
                ),
                "Malicious": int(
                    subset.loc[subset["label"].eq(1), "record_count"].sum()
                ),
                "Weeks": _week_range(subset["week_id"]),
                "QC findings": (
                    "—" if file_type != "Overall" else _qc_findings_text(checks)
                ),
            }
        )
    table = pd.DataFrame(rows)
    export_table(
        table,
        Path(output_dir),
        "table_1_dataset_composition_quality_control",
        "Table 1. Dataset composition and quality control",
    )
    return table


def generate_table_2(
    metrics_path: str | Path,
    noninferiority_path: str | Path,
    secondary_path: str | Path,
    output_dir: str | Path,
    *,
    tcfs_method: str = "tcfs",
    all_features_method: str = "all_features",
) -> pd.DataFrame:
    """Build confirmatory performance and efficiency Table 2."""
    metrics = read_table(metrics_path)
    noninferiority = read_table(noninferiority_path)
    secondary = read_table(secondary_path)
    _require_columns(
        metrics,
        {"file_type", "method", "metric", "estimate", "ci_lower", "ci_upper"},
        "evaluation metrics",
    )
    _require_columns(
        noninferiority,
        {"file_type", "metric", "difference", "ci_lower", "ci_upper", "p_value"},
        "non-inferiority results",
    )
    _require_columns(
        secondary,
        {
            "file_type",
            "metric",
            "difference",
            "ci_lower",
            "ci_upper",
            "holm_adjusted_p_value",
        },
        "secondary comparisons",
    )
    ordered_metrics = (*PREDICTIVE_METRICS, *RESOURCE_METRICS)
    rows: list[dict[str, Any]] = []
    for file_type in FORMATS:
        for metric in ordered_metrics:
            tcfs = _single_metric(metrics, file_type, tcfs_method, metric)
            all_features = _single_metric(
                metrics, file_type, all_features_method, metric
            )
            difference = float(tcfs["estimate"] - all_features["estimate"])
            ci_text = "—"
            p_text = "—"
            test = "Descriptive"
            if metric == "macro_f1":
                comparison = _single_comparison(noninferiority, file_type, metric)
                difference = float(comparison["difference"])
                ci_text = _format_interval(
                    comparison["ci_lower"], comparison["ci_upper"]
                )
                p_text = _format_p(comparison["p_value"])
                test = "Non-inferiority"
            elif metric in PREDICTIVE_METRICS[1:]:
                comparison = _single_comparison(secondary, file_type, metric)
                difference = float(comparison["difference"])
                ci_text = _format_interval(
                    comparison["ci_lower"], comparison["ci_upper"]
                )
                p_text = _format_p(comparison["holm_adjusted_p_value"])
                test = "Holm-adjusted"
            rows.append(
                {
                    "Format": file_type,
                    "Metric": METRIC_LABELS[metric],
                    "TCFS": _format_estimate(tcfs),
                    "All features": _format_estimate(all_features),
                    "Difference": _format_number(difference, metric),
                    "95% CI of difference": ci_text,
                    "Adjusted P": p_text,
                    "Comparison": test,
                    "N": int(tcfs["n_samples"]) if "n_samples" in tcfs else "—",
                    "Malicious": (
                        int(tcfs["n_positive"]) if "n_positive" in tcfs else "—"
                    ),
                    "Benign": (
                        int(tcfs["n_negative"]) if "n_negative" in tcfs else "—"
                    ),
                }
            )
    table = pd.DataFrame(rows)
    export_table(
        table,
        Path(output_dir),
        "table_2_confirmatory_performance_efficiency",
        "Table 2. Confirmatory performance and efficiency comparisons",
        font_size=6.5,
        row_height=0.25,
    )
    return table


def export_figure(fig: Figure, output_dir: Path, stem: str) -> None:
    """Export one figure as vector PDF/SVG and 300-DPI PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.pdf",
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": stem,
            "Creator": "tcfs-malware",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        output_dir / f"{stem}.svg",
        format="svg",
        bbox_inches="tight",
        metadata={"Title": stem, "Creator": "tcfs-malware", "Date": None},
    )
    fig.savefig(
        output_dir / f"{stem}.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Title": stem, "Software": "tcfs-malware"},
    )
    plt.close(fig)


def export_table(
    table: pd.DataFrame,
    output_dir: Path,
    stem: str,
    title: str,
    *,
    font_size: float = 7.5,
    row_height: float = 0.36,
) -> None:
    """Export source CSV/LaTeX plus publication-rendered table formats."""
    configure_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.tex").write_text(
        _table_to_latex(table), encoding="utf-8"
    )
    width = max(10.0, min(18.0, 1.1 * len(table.columns)))
    height = max(2.4, 1.0 + row_height * (len(table) + 2))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10, pad=8)
    rendered = ax.table(
        cellText=table.astype(str).to_numpy(),
        colLabels=table.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(font_size)
    rendered.auto_set_column_width(col=list(range(len(table.columns))))
    for (row, _), cell in rendered.get_celld().items():
        cell.set_edgecolor("#B7B7B7")
        cell.set_linewidth(0.45)
        cell.set_height(row_height / height)
        if row == 0:
            cell.set_facecolor("#DCEAF7")
            cell.set_text_props(weight="bold", color="#111111")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F4F4F4")
            cell.set_text_props(color="#111111")
    export_figure(fig, output_dir, stem)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a saved CSV or Parquet artifact."""
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported table extension: {source.suffix}")


def _stability_panel(
    ax: Axes,
    data: pd.DataFrame,
    value_column: str,
    methods: Sequence[str],
    styles: dict[str, dict[str, Any]],
) -> None:
    for position, method in enumerate(methods):
        values = data.loc[data["method"].eq(method), value_column].to_numpy(
            dtype=float
        )
        if not len(values):
            continue
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else [0]
        style = styles[method]
        ax.scatter(
            position + np.asarray(jitter),
            values,
            color=style["color"],
            marker=style["marker"],
            alpha=0.38,
            s=18,
            linewidths=0,
        )
        ax.errorbar(
            position,
            np.mean(values),
            yerr=np.std(values, ddof=0),
            fmt=style["marker"],
            color=style["color"],
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            capsize=2,
        )
    ax.set_xticks(range(len(methods)), [_method_label(method) for method in methods])
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(-1.02 if value_column == "spearman_rho" else -0.02, 1.02)
    ax.axhline(0, color="#777777", linewidth=0.5)


def _method_styles(methods: Sequence[str]) -> dict[str, dict[str, Any]]:
    unknown = sorted(set(methods) - set(METHOD_ORDER))
    style_order = [*METHOD_ORDER, *unknown]
    return {
        method: {
            "color": PALETTE[style_order.index(method) % len(PALETTE)],
            "marker": MARKERS[style_order.index(method) % len(MARKERS)],
            "linestyle": LINESTYLES[style_order.index(method) % len(LINESTYLES)],
        }
        for method in methods
    }


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def _ordered_methods(methods: Sequence[str] | set[str]) -> list[str]:
    values = {str(method) for method in methods}
    return [method for method in METHOD_ORDER if method in values] + sorted(
        values - set(METHOD_ORDER)
    )


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _week_range(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "—"
    return f"{int(numeric.min()) + 1}-{int(numeric.max()) + 1}"


def _qc_findings_text(checks: pd.DataFrame) -> str:
    findings = pd.to_numeric(checks["findings"], errors="coerce").fillna(0)
    failed = checks.loc[checks["status"].ne("PASS") & findings.gt(0)]
    if failed.empty:
        return "None"
    return "; ".join(
        f"{row.rule_id}: {int(row.findings)}" for row in failed.itertuples()
    )


def _single_metric(
    metrics: pd.DataFrame, file_type: str, method: str, metric: str
) -> pd.Series:
    match = metrics.loc[
        metrics["file_type"].eq(file_type)
        & metrics["method"].eq(method)
        & metrics["metric"].eq(metric)
    ]
    if len(match) != 1:
        raise ValueError(
            f"expected one metric row for {file_type}/{method}/{metric}; "
            f"got {len(match)}"
        )
    return match.iloc[0]


def _single_comparison(
    comparisons_frame: pd.DataFrame, file_type: str, metric: str
) -> pd.Series:
    match = comparisons_frame.loc[
        comparisons_frame["file_type"].eq(file_type)
        & comparisons_frame["metric"].eq(metric)
    ]
    if len(match) != 1:
        raise ValueError(
            f"expected one comparison for {file_type}/{metric}; got {len(match)}"
        )
    return match.iloc[0]


def _format_estimate(row: pd.Series) -> str:
    metric = str(row["metric"])
    estimate = _format_number(float(row["estimate"]), metric)
    if pd.notna(row.get("ci_lower")) and pd.notna(row.get("ci_upper")):
        return f"{estimate} {_format_interval(row['ci_lower'], row['ci_upper'])}"
    return estimate


def _format_number(value: float, metric: str) -> str:
    if metric in {"feature_count", "model_size_bytes"}:
        return f"{value:,.0f}"
    if metric == "inference_latency_ms_per_sample":
        return f"{value:.4f}"
    return f"{value:.3f}"


def _format_interval(low: float, high: float) -> str:
    return f"[{float(low):.3f}, {float(high):.3f}]"


def _format_p(value: float) -> str:
    value = float(value)
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _table_to_latex(table: pd.DataFrame) -> str:
    alignment = "l" * len(table.columns)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(_latex_escape(column) for column in table.columns) + r" \\")
    lines.append("\\midrule")
    for row in table.astype(str).itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)
