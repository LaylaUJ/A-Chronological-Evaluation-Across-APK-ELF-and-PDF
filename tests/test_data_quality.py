import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcfs.data_quality import run_data_quality


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [0, 1, 0],
            "sha256": ["a" * 64, "a" * 64, "a" * 64],
            "file_type": ["APK", "APK", "ELF"],
            "label": [0, 0, 1],
            "first_submission_date": pd.to_datetime(
                ["2023-09-24", "2023-09-24", "2024-09-22"], utc=True
            ),
            "last_analysis_date": pd.to_datetime(
                ["2023-09-25", "2023-09-25", "2024-09-23"], utc=True
            ),
            "official_split": ["train", "train", "test"],
            "week_id": [0, 1, 52],
            "derived_week_id": [0, 0, 52],
            "family": [None, None, "example"],
        }
    )


def test_metadata_checks_and_exclusion_ledger_are_non_destructive(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    path = tmp_path / "metadata.parquet"
    metadata.to_parquet(path, index=False)
    output = tmp_path / "quality"

    result = run_data_quality(path, output)

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    exclusions = pd.read_csv(output / "proposed_exclusions.csv")
    assert result.input_rows == 3
    assert summary["input_rows"] == summary["output_rows"] == 3
    assert summary["records_deleted"] == 0
    assert {"DQ003", "DQ004", "DQ008", "DQ010"} <= set(exclusions["rule_id"])
    assert pd.read_parquet(path).equals(metadata)
    assert "No records were changed or deleted" in (
        output / "quality_control_report.md"
    ).read_text(encoding="utf-8")


def test_feature_checks_find_nan_infinity_and_constants(tmp_path: Path) -> None:
    metadata = _metadata().iloc[[0, 2]].copy()
    path = tmp_path / "metadata.parquet"
    metadata.to_parquet(path, index=False)
    train = np.array([[1.0, np.nan, 5.0]], dtype=np.float32)
    test = np.array([[1.0, np.inf, 5.0]], dtype=np.float32)
    train_path = tmp_path / "train.npy"
    test_path = tmp_path / "test.npy"
    np.save(train_path, train)
    np.save(test_path, test)

    result = run_data_quality(
        path,
        tmp_path / "quality",
        feature_paths={"train": train_path, "test": test_path},
    )

    quality = pd.read_csv(tmp_path / "quality" / "feature_quality.csv")
    exclusions = pd.read_csv(tmp_path / "quality" / "proposed_exclusions.csv")
    assert result.proposed_feature_exclusions == 3
    assert quality["constant"].tolist() == [True, False, True]
    assert {"DQ005", "DQ006"} <= set(exclusions["rule_id"])


def test_feature_row_mismatch_stops(tmp_path: Path) -> None:
    metadata = _metadata().iloc[[0, 2]].copy()
    path = tmp_path / "metadata.parquet"
    metadata.to_parquet(path, index=False)
    train_path = tmp_path / "train.npy"
    test_path = tmp_path / "test.npy"
    np.save(train_path, np.zeros((2, 2), dtype=np.float32))
    np.save(test_path, np.zeros((1, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="row counts differ for train"):
        run_data_quality(
            path,
            tmp_path / "quality",
            feature_paths={"train": train_path, "test": test_path},
        )

