from pathlib import Path

import numpy as np
import pandas as pd

from tcfs.evaluation import PREDICTIVE_METRICS, RESOURCE_METRICS
from tcfs.reporting import (
    generate_figure_1,
    generate_figure_2,
    generate_figure_3,
    generate_figure_4,
    generate_table_1,
    generate_table_2,
)
from tcfs.temporal_split import PROTOCOL_VERSION


def _write_inputs(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    weeks = np.arange(1, 65)
    assignments = pd.DataFrame(
        {
            "protocol_week": weeks,
            "protocol_role": np.select(
                [weeks <= 40, weeks <= 52],
                ["discovery", "validation"],
                default="sealed_test",
            ),
            "protocol_version": PROTOCOL_VERSION,
        }
    )
    paths["assignments"] = root / "assignments.parquet"
    assignments.to_parquet(paths["assignments"], index=False)

    stability_rows = []
    for file_type in ("APK", "ELF", "PDF"):
        for method in ("tcfs", "all_features"):
            for pair, value in enumerate((0.70, 0.80, 0.90), start=1):
                stability_rows.append(
                    {
                        "file_type": file_type,
                        "method": method,
                        "replicate_a": f"window_{pair}",
                        "replicate_b": f"window_{pair + 1}",
                        "jaccard": value,
                        "spearman_rho": value - 0.1,
                    }
                )
    stability = pd.DataFrame(stability_rows)
    paths["jaccard"] = root / "jaccard.csv"
    paths["spearman"] = root / "spearman.csv"
    stability.drop(columns="spearman_rho").to_csv(paths["jaccard"], index=False)
    stability.drop(columns="jaccard").to_csv(paths["spearman"], index=False)

    budget_rows = []
    for file_type in ("APK", "ELF", "PDF"):
        for method_index, method in enumerate(("tcfs", "all_features")):
            for budget in (0.05, 0.10, 0.20, 0.40, 0.60):
                estimate = 0.72 + 0.12 * budget - 0.01 * method_index
                budget_rows.append(
                    {
                        "file_type": file_type,
                        "method": method,
                        "feature_budget": budget,
                        "metric": "macro_f1",
                        "estimate": estimate,
                        "ci_lower": estimate - 0.02,
                        "ci_upper": estimate + 0.02,
                    }
                )
    paths["budget"] = root / "budget_metrics.csv"
    pd.DataFrame(budget_rows).to_csv(paths["budget"], index=False)

    analysis_rows = []
    for file_type in ("APK", "ELF", "PDF"):
        for setting, estimate in (("No frequency", -0.02), ("No variance", -0.01)):
            analysis_rows.append(
                {
                    "file_type": file_type,
                    "analysis": "ablation",
                    "setting": setting,
                    "metric": "macro_f1",
                    "estimate": estimate,
                    "ci_lower": estimate - 0.01,
                    "ci_upper": estimate + 0.01,
                    "x_value": np.nan,
                }
            )
        for rho in (0.0, 0.25, 0.5, 1.0):
            analysis_rows.append(
                {
                    "file_type": file_type,
                    "analysis": "sensitivity",
                    "setting": f"rho={rho}",
                    "metric": "macro_f1",
                    "estimate": 0.78 - 0.01 * rho,
                    "ci_lower": 0.76 - 0.01 * rho,
                    "ci_upper": 0.80 - 0.01 * rho,
                    "x_value": rho,
                }
            )
    paths["ablation"] = root / "ablation.csv"
    pd.DataFrame(analysis_rows).to_csv(paths["ablation"], index=False)

    count_rows = []
    for file_type in ("APK", "ELF", "PDF"):
        for split, week in (("train", 0), ("test", 52)):
            for label in (0, 1):
                count_rows.append(
                    {
                        "official_split": split,
                        "file_type": file_type,
                        "label": label,
                        "week_id": week,
                        "record_count": 10,
                    }
                )
    paths["counts"] = root / "record_counts.csv"
    pd.DataFrame(count_rows).to_csv(paths["counts"], index=False)
    paths["checks"] = root / "check_results.csv"
    pd.DataFrame(
        {
            "rule_id": ["DQ001", "DQ004"],
            "status": ["PASS", "FAIL"],
            "findings": [0, 2],
        }
    ).to_csv(paths["checks"], index=False)

    metric_rows = []
    secondary_rows = []
    noninferiority_rows = []
    for file_type in ("APK", "ELF", "PDF"):
        for method_index, method in enumerate(("tcfs", "all_features")):
            for metric in (*PREDICTIVE_METRICS, *RESOURCE_METRICS):
                if metric == "feature_count":
                    estimate = 10 if method == "tcfs" else 100
                elif metric == "model_size_bytes":
                    estimate = 1_000 if method == "tcfs" else 2_000
                elif metric == "inference_latency_ms_per_sample":
                    estimate = 0.1 if method == "tcfs" else 0.2
                else:
                    estimate = 0.8 - method_index * 0.005
                metric_rows.append(
                    {
                        "file_type": file_type,
                        "method": method,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower": (
                            estimate - 0.02 if metric in PREDICTIVE_METRICS else np.nan
                        ),
                        "ci_upper": (
                            estimate + 0.02 if metric in PREDICTIVE_METRICS else np.nan
                        ),
                        "n_samples": 100,
                        "n_positive": 40,
                        "n_negative": 60,
                    }
                )
        noninferiority_rows.append(
            {
                "file_type": file_type,
                "metric": "macro_f1",
                "difference": 0.005,
                "ci_lower": -0.003,
                "ci_upper": 0.013,
                "p_value": 0.01,
            }
        )
        for metric in PREDICTIVE_METRICS[1:]:
            secondary_rows.append(
                {
                    "file_type": file_type,
                    "metric": metric,
                    "difference": 0.005,
                    "ci_lower": -0.005,
                    "ci_upper": 0.015,
                    "holm_adjusted_p_value": 0.20,
                }
            )
    paths["metrics"] = root / "metrics.csv"
    paths["noninferiority"] = root / "noninferiority.csv"
    paths["secondary"] = root / "secondary.csv"
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(noninferiority_rows).to_csv(paths["noninferiority"], index=False)
    pd.DataFrame(secondary_rows).to_csv(paths["secondary"], index=False)
    return paths


def test_generates_all_publication_outputs_from_saved_artifacts(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "publication"
    generate_figure_1(inputs["assignments"], output)
    generate_figure_2(inputs["jaccard"], inputs["spearman"], output)
    generate_figure_3(inputs["budget"], output)
    generate_figure_4(inputs["ablation"], output)
    table_1 = generate_table_1(inputs["counts"], inputs["checks"], output)
    table_2 = generate_table_2(
        inputs["metrics"],
        inputs["noninferiority"],
        inputs["secondary"],
        output,
    )

    figure_stems = {
        "figure_1_chronological_study_design",
        "figure_2_feature_set_stability",
        "figure_3_future_performance",
        "figure_4_ablation_sensitivity",
    }
    table_stems = {
        "table_1_dataset_composition_quality_control",
        "table_2_confirmatory_performance_efficiency",
    }
    for stem in figure_stems | table_stems:
        for extension in ("pdf", "svg", "png"):
            artifact = output / f"{stem}.{extension}"
            assert artifact.stat().st_size > 100
    for stem in table_stems:
        assert (output / f"{stem}.csv").is_file()
        assert (output / f"{stem}.tex").is_file()
    assert table_1["Total"].iloc[-1] == 120
    assert len(table_2) == 3 * len(PREDICTIVE_METRICS + RESOURCE_METRICS)


def test_svg_output_is_reproducible(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_figure_3(inputs["budget"], first)
    generate_figure_3(inputs["budget"], second)
    name = "figure_3_future_performance.svg"
    assert (first / name).read_bytes() == (second / name).read_bytes()
