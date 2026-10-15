from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcfs.publication_aggregation import (
    aggregate_ablation_sensitivity,
    aggregate_budget_metrics,
    save_publication_aggregate,
)
from tcfs.temporal_split import PROTOCOL_VERSION, TemporalLeakageError


def _base_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    for file_type in ("APK", "ELF", "PDF"):
        for row_id, label in enumerate(labels):
            rows.append(
                {
                    "file_type": file_type,
                    "row_id": row_id,
                    "y_true": int(label),
                    "protocol_role": "validation",
                    "protocol_week": 41 + row_id % 12,
                    "protocol_version": PROTOCOL_VERSION,
                }
            )
    return rows


def _budget_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for base in _base_rows():
        label = int(base["y_true"])
        for method, budget in (("tcfs", 0.1), ("all_features", 0.6)):
            predicted = label if method == "tcfs" else int(base["row_id"] != 0)
            rows.append(
                {
                    **base,
                    "method": method,
                    "feature_budget": budget,
                    "prediction": predicted,
                }
            )
    return pd.DataFrame(rows)


def _analysis_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rho_values = (0.0, 0.25, 0.5, 1.0)
    for base in _base_rows():
        label = int(base["y_true"])
        row_id = int(base["row_id"])
        for setting, predicted in (
            ("full_tcfs", label),
            ("without_frequency", int(row_id not in {0, 4})),
        ):
            rows.append(
                {
                    **base,
                    "analysis": "ablation",
                    "setting": setting,
                    "x_value": np.nan,
                    "prediction": predicted,
                }
            )
        for rho in rho_values:
            rows.append(
                {
                    **base,
                    "analysis": "sensitivity",
                    "setting": f"rho={rho:g}",
                    "x_value": rho,
                    "prediction": label if rho < 1 else int(row_id != 0),
                }
            )
    return pd.DataFrame(rows)


def test_budget_metrics_are_deterministic_and_complete() -> None:
    predictions = _budget_predictions()
    first = aggregate_budget_metrics(predictions, n_bootstrap=31, random_state=7)
    second = aggregate_budget_metrics(
        predictions.sample(frac=1, random_state=99),
        n_bootstrap=31,
        random_state=7,
    )

    pd.testing.assert_frame_equal(first, second)
    assert set(first["file_type"]) == {"APK", "ELF", "PDF"}
    assert set(first["metric"]) == {"macro_f1"}
    assert first["n_samples"].eq(8).all()
    assert first["bootstrap_samples"].eq(31).all()
    assert first["estimate"].between(0, 1).all()


def test_ablation_uses_paired_differences_and_rho_values() -> None:
    result = aggregate_ablation_sensitivity(
        _analysis_predictions(), n_bootstrap=31, random_state=11
    )

    ablation = result.loc[result["analysis"].eq("ablation")]
    sensitivity = result.loc[result["analysis"].eq("sensitivity")]
    assert len(ablation) == 3
    assert (ablation["estimate"] < 0).all()
    assert set(sensitivity["x_value"]) == {0.0, 0.25, 0.5, 1.0}
    assert sensitivity.groupby("file_type").size().eq(4).all()
    assert result["n_positive"].eq(4).all()
    assert result["n_negative"].eq(4).all()


@pytest.mark.parametrize("week,role", [(53, "sealed_test"), (40, "discovery")])
def test_publication_aggregations_reject_nonvalidation_rows(
    week: int, role: str
) -> None:
    predictions = _budget_predictions()
    predictions.loc[0, "protocol_week"] = week
    predictions.loc[0, "protocol_role"] = role

    with pytest.raises(TemporalLeakageError, match="validation"):
        aggregate_budget_metrics(predictions, n_bootstrap=5)


def test_aggregate_save_refuses_overwrite(tmp_path: Path) -> None:
    result = aggregate_budget_metrics(
        _budget_predictions(), n_bootstrap=5, random_state=3
    )
    destination = tmp_path / "budget_metrics.csv"

    assert save_publication_aggregate(result, destination) == destination
    with pytest.raises(FileExistsError, match="already exists"):
        save_publication_aggregate(result, destination)
