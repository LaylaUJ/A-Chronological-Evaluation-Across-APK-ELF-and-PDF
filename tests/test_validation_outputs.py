import pandas as pd
import pytest

from tcfs.temporal_split import PROTOCOL_VERSION, TemporalLeakageError
from tcfs.validation_outputs import (
    choose_budget_predictions,
    validate_ablation_predictions,
)


def _predictions() -> pd.DataFrame:
    rows = []
    labels = [0, 0, 1, 1]
    for rho, predicted in ((0.0, [0, 1, 1, 1]), (0.25, labels)):
        for row_id, (label, value) in enumerate(zip(labels, predicted, strict=True)):
            rows.append(
                {
                    "file_type": "APK",
                    "method": "tcfs",
                    "classifier": "lightgbm",
                    "feature_budget": 0.1,
                    "rho": rho,
                    "candidate_id": f"rho-{rho}",
                    "row_id": row_id,
                    "protocol_week": 41 + row_id,
                    "protocol_role": "validation",
                    "protocol_version": PROTOCOL_VERSION,
                    "y_true": label,
                    "prediction": value,
                }
            )
    return pd.DataFrame(rows)


def test_chooses_best_rho_candidate_and_preserves_rows() -> None:
    result = choose_budget_predictions(_predictions())
    assert set(result["candidate_id"]) == {"rho-0.25"}
    assert list(result["row_id"]) == [0, 1, 2, 3]


def test_candidate_ids_are_scoped_by_method_and_budget() -> None:
    first = _predictions()
    second = _predictions().assign(method="shared_tcfs", feature_budget=0.2)
    combined = pd.concat([first, second], ignore_index=True)

    result = choose_budget_predictions(combined)

    assert len(result) == 8
    assert set(result["method"]) == {"tcfs", "shared_tcfs"}
    assert not result.duplicated(
        ["file_type", "method", "feature_budget", "row_id"]
    ).any()


def test_rejects_nonvalidation_rows() -> None:
    frame = _predictions()
    frame.loc[0, "protocol_role"] = "sealed_test"
    frame.loc[0, "protocol_week"] = 53
    with pytest.raises(TemporalLeakageError, match="validation rows"):
        choose_budget_predictions(frame)


def test_validates_complete_ablation_formats() -> None:
    rows = []
    for file_type in ("APK", "ELF", "PDF"):
        for analysis, setting, x_value in (
            ("ablation", "full_tcfs", float("nan")),
            ("sensitivity", "rho=0", 0.0),
        ):
            for row_id, label in enumerate([0, 1]):
                rows.append(
                    {
                        "file_type": file_type,
                        "analysis": analysis,
                        "setting": setting,
                        "x_value": x_value,
                        "row_id": row_id,
                        "protocol_week": 41 + row_id,
                        "protocol_role": "validation",
                        "protocol_version": PROTOCOL_VERSION,
                        "y_true": label,
                        "prediction": label,
                    }
                )
    result = validate_ablation_predictions(pd.DataFrame(rows))
    assert len(result) == 12
    assert set(result["file_type"]) == {"APK", "ELF", "PDF"}
