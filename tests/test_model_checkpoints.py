from pathlib import Path

import pandas as pd

import tcfs.model_checkpoints as checkpoints


def test_merges_distinct_method_checkpoints(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    methods = ("all_features", "tcfs")
    for method in methods:
        root = checkpoint_root / method
        root.mkdir(parents=True)
        rows = []
        for classifier in ("lightgbm", "logistic_regression"):
            experiment_id = f"{method}__{classifier}"
            rows.append(
                {
                    "method": method,
                    "classifier": classifier,
                    "experiment_id": experiment_id,
                }
            )
            experiment = root / experiment_id
            experiment.mkdir()
            (experiment / "marker.txt").write_text(method, encoding="utf-8")
        pd.DataFrame(rows).to_csv(root / "selected_configurations.csv", index=False)
        pd.DataFrame(
            {"method": [method], "row_id": [1]}
        ).to_parquet(root / "validation_variant_predictions.parquet", index=False)
    monkeypatch.setattr(checkpoints, "method_checkpoint_complete", lambda *_: True)

    output = tmp_path / "experiments"
    checkpoints.merge_method_checkpoints(checkpoint_root, output, methods)

    selected = pd.read_csv(output / "selected_configurations.csv")
    predictions = pd.read_parquet(output / "validation_variant_predictions.parquet")
    assert len(selected) == 4
    assert set(predictions["method"]) == set(methods)
    assert (output / "tcfs__lightgbm" / "marker.txt").is_file()
