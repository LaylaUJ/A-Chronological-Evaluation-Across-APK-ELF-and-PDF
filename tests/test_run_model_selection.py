from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.run_model_selection import _load_aligned_development_data


def test_loads_memmap_and_applies_only_approved_train_exclusions(
    tmp_path: Path,
) -> None:
    features = np.arange(20, dtype=np.float32).reshape(5, 4)
    labels = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    feature_path = tmp_path / "X_train.dat"
    label_path = tmp_path / "y_train.dat"
    features.tofile(feature_path)
    labels.tofile(label_path)
    metadata = pd.DataFrame(
        {
            "row_id": [10, 11, 12, 13, 14],
            "file_type": "APK",
            "official_split": "train",
            "label": labels,
        }
    )
    assignments = pd.DataFrame(
        {
            "row_id": metadata["row_id"],
            "protocol_week": [1, 2, 41, 42, 52],
            "protocol_role": [
                "discovery",
                "discovery",
                "validation",
                "validation",
                "validation",
            ],
        }
    )
    ledger = pd.DataFrame(
        {
            "rule_id": ["DQ003"],
            "decision": ["exclude"],
            "official_split": ["train"],
            "row_id": [11],
        }
    )
    ledger_path = tmp_path / "approved.csv"
    ledger.to_csv(ledger_path, index=False)
    loaded_x, loaded_y, loaded_assignments = _load_aligned_development_data(
        feature_path,
        label_path,
        assignments,
        [f"f{i}" for i in range(4)],
        metadata=metadata,
        exclusions=ledger_path,
        file_type="APK",
    )
    assert np.array_equal(loaded_x, features[[0, 2, 3, 4]])
    assert np.array_equal(loaded_y, labels[[0, 2, 3, 4]])
    assert list(loaded_assignments["row_id"]) == [10, 12, 13, 14]


def test_memmap_rejects_unapproved_ledger(tmp_path: Path) -> None:
    feature_path = tmp_path / "X.dat"
    label_path = tmp_path / "y.dat"
    np.zeros((1, 1), dtype=np.float32).tofile(feature_path)
    np.zeros(1, dtype=np.int32).tofile(label_path)
    metadata = pd.DataFrame(
        {"row_id": [0], "file_type": ["APK"], "official_split": ["train"], "label": [0]}
    )
    assignments = pd.DataFrame(
        {"row_id": [0], "protocol_week": [1], "protocol_role": ["discovery"]}
    )
    ledger = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "rule_id": ["DQ010"],
            "decision": ["exclude"],
            "official_split": ["train"],
            "row_id": [0],
        }
    ).to_csv(ledger, index=False)
    with pytest.raises(ValueError, match="approved DQ003"):
        _load_aligned_development_data(
            feature_path,
            label_path,
            assignments,
            ["f0"],
            metadata=metadata,
            exclusions=ledger,
            file_type="APK",
        )
