from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcfs.evaluation import (
    PREDICTIVE_METRICS,
    RESOURCE_METRICS,
    evaluate_feature_stability,
    evaluate_predictions,
    profile_model,
    save_evaluation_results,
)


class ProbabilityModel:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = np.clip(np.asarray(X)[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


def _evaluation_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    metadata_rows = []
    labels = np.array([0] * 10 + [1] * 10)
    perfect_probability = np.linspace(0.01, 0.99, len(labels))
    for file_type in ("APK", "ELF", "PDF"):
        for method, feature_count in (("tcfs", 4), ("all_features", 20)):
            for row_id, (label, probability) in enumerate(
                zip(labels, perfect_probability, strict=True)
            ):
                prediction_rows.append(
                    {
                        "file_type": file_type,
                        "method": method,
                        "row_id": row_id,
                        "y_true": label,
                        "probability": probability,
                        "prediction": int(probability >= 0.5),
                    }
                )
            metadata_rows.append(
                {
                    "file_type": file_type,
                    "method": method,
                    "feature_count": feature_count,
                    "model_size_bytes": 1_000 + feature_count,
                    "inference_latency_ms_per_sample": 0.1 + feature_count / 1_000,
                }
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metadata_rows)


def test_all_metrics_bootstrap_noninferiority_and_class_counts() -> None:
    predictions, metadata = _evaluation_inputs()
    results = evaluate_predictions(
        predictions,
        metadata,
        n_bootstrap=99,
        random_state=7,
        comparisons=(("tcfs", "all_features"),),
    )

    assert set(results.metrics["file_type"]) == {"APK", "ELF", "PDF"}
    assert set(PREDICTIVE_METRICS + RESOURCE_METRICS) == set(
        results.metrics["metric"]
    )
    assert results.metrics["n_samples"].eq(20).all()
    assert results.metrics["n_positive"].eq(10).all()
    assert results.metrics["n_negative"].eq(10).all()
    assert results.method_differences["bootstrap_samples"].eq(99).all()
    assert results.noninferiority["noninferiority_margin"].eq(-0.01).all()
    assert results.noninferiority["noninferior_at_0_05"].all()
    assert results.secondary_comparisons["holm_adjusted_p_value"].between(
        0.0, 1.0
    ).all()
    assert (
        results.secondary_comparisons["holm_adjusted_p_value"]
        >= results.secondary_comparisons["p_value"]
    ).all()


def test_paired_bootstrap_is_reproducible() -> None:
    predictions, metadata = _evaluation_inputs()
    first = evaluate_predictions(
        predictions, metadata, n_bootstrap=40, random_state=19
    )
    second = evaluate_predictions(
        predictions, metadata, n_bootstrap=40, random_state=19
    )

    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    pd.testing.assert_frame_equal(
        first.method_differences, second.method_differences
    )
    pd.testing.assert_frame_equal(first.noninferiority, second.noninferiority)


def test_methods_must_be_row_aligned() -> None:
    predictions, metadata = _evaluation_inputs()
    mask = (
        predictions["file_type"].eq("APK")
        & predictions["method"].eq("tcfs")
        & predictions["row_id"].eq(0)
    )
    predictions = predictions.loc[~mask]
    with pytest.raises(ValueError, match="aligned row IDs"):
        evaluate_predictions(predictions, metadata, n_bootstrap=10)


def test_pairwise_feature_stability_and_rank_correlation() -> None:
    feature_results = pd.DataFrame(
        {
            "file_type": ["APK"] * 6,
            "method": ["tcfs"] * 6,
            "replicate": ["window_1"] * 3 + ["window_2"] * 3,
            "feature_name": ["a", "b", "c"] * 2,
            "selected": [True, True, False, False, True, True],
            "normalized_rank": [1.0, 0.5, 0.0, 0.0, 0.5, 1.0],
        }
    )
    counts = pd.DataFrame(
        {"file_type": ["APK"], "n_samples": [20], "n_positive": [8], "n_negative": [12]}
    )

    jaccard, spearman = evaluate_feature_stability(feature_results, counts)
    assert jaccard.loc[0, "jaccard"] == pytest.approx(1 / 3)
    assert spearman.loc[0, "spearman_rho"] == pytest.approx(-1.0)
    assert jaccard.loc[0, "n_samples"] == 20
    assert spearman.loc[0, "n_positive"] == 8


def test_model_profile_and_result_persistence(tmp_path: Path) -> None:
    profile = profile_model(
        ProbabilityModel(), np.linspace(0.0, 1.0, 20).reshape(-1, 1), timed_runs=2
    )
    assert profile["feature_count"] == 1
    assert profile["model_size_bytes"] > 0
    assert profile["inference_latency_ms_per_sample"] >= 0.0

    predictions, metadata = _evaluation_inputs()
    results = evaluate_predictions(predictions, metadata, n_bootstrap=10)
    save_evaluation_results(results, tmp_path)
    expected = {
        "metrics.csv",
        "method_differences.csv",
        "noninferiority.csv",
        "secondary_comparisons.csv",
        "jaccard_stability.csv",
        "spearman_stability.csv",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
