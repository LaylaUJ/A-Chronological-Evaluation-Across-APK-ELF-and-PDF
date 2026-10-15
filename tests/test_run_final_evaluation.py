from pathlib import Path

import numpy as np
import pandas as pd
from scripts.run_final_evaluation import _load_aligned_final_data


def test_loads_both_splits_and_applies_approved_dq003(tmp_path: Path) -> None:
    names = ["f0", "f1"]
    train_x = np.arange(8, dtype=np.float32).reshape(4, 2)
    test_x = np.arange(6, dtype=np.float32).reshape(3, 2) + 10
    train_y = np.array([0, 1, 0, 1], dtype=np.int32)
    test_y = np.array([0, 1, 1], dtype=np.int32)
    paths = {}
    for name, values in (
        ("train_x", train_x),
        ("test_x", test_x),
        ("train_y", train_y),
        ("test_y", test_y),
    ):
        path = tmp_path / f"{name}.dat"
        values.tofile(path)
        paths[name] = path
    metadata = pd.DataFrame(
        {
            "row_id": [10, 11, 12, 13, 20, 21, 22],
            "file_type": "APK",
            "official_split": ["train"] * 4 + ["test"] * 3,
            "label": np.concatenate([train_y, test_y]),
        }
    )
    assignments = pd.DataFrame(
        {
            "row_id": metadata["row_id"],
            "protocol_week": [1, 40, 41, 52, 53, 60, 64],
            "protocol_role": [
                "discovery",
                "discovery",
                "validation",
                "validation",
                "sealed_test",
                "sealed_test",
                "sealed_test",
            ],
        }
    )
    ledger = tmp_path / "approved.csv"
    pd.DataFrame(
        {
            "rule_id": ["DQ003", "DQ003"],
            "decision": ["exclude", "exclude"],
            "official_split": ["train", "test"],
            "row_id": [11, 21],
        }
    ).to_csv(ledger, index=False)

    X, y, aligned = _load_aligned_final_data(
        paths["train_x"],
        paths["train_y"],
        paths["test_x"],
        paths["test_y"],
        metadata,
        assignments,
        names,
        ledger,
        "APK",
    )

    assert np.array_equal(X, np.vstack([train_x[[0, 2, 3]], test_x[[0, 2]]]))
    assert np.array_equal(y, np.concatenate([train_y[[0, 2, 3]], test_y[[0, 2]]]))
    assert list(aligned["row_id"]) == [10, 12, 13, 20, 22]
