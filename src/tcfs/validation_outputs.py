"""Deterministic reducers for saved validation-only prediction variants."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

from tcfs.temporal_split import PROTOCOL_VERSION, TemporalLeakageError

FORMATS = {"APK", "ELF", "PDF"}


def choose_budget_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Choose one validation candidate per format/method/budget without test data."""
    required = {
        "file_type",
        "method",
        "classifier",
        "feature_budget",
        "rho",
        "candidate_id",
        "row_id",
        "protocol_week",
        "protocol_role",
        "protocol_version",
        "y_true",
        "prediction",
    }
    if missing := required - set(predictions):
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    frame = predictions.copy()
    if set(frame["classifier"].astype(str)) != {"lightgbm"}:
        raise ValueError("budget curves require primary LightGBM predictions only")
    if set(frame["protocol_role"].astype(str)) != {"validation"}:
        raise TemporalLeakageError("budget selection requires validation rows only")
    weeks = pd.to_numeric(frame["protocol_week"], errors="raise")
    if not weeks.between(41, 52).all():
        raise TemporalLeakageError("budget selection is restricted to weeks 41-52")
    if set(frame["protocol_version"].astype(str)) != {PROTOCOL_VERSION}:
        raise ValueError("predictions use an unexpected protocol version")

    candidate_columns = [
        "file_type",
        "method",
        "feature_budget",
        "rho",
        "candidate_id",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(candidate_columns, sort=True, dropna=False):
        labels = group["y_true"].to_numpy(dtype=int)
        predicted = group["prediction"].to_numpy(dtype=int)
        rows.append(
            {
                **dict(zip(candidate_columns, keys, strict=True)),
                "macro_f1": f1_score(labels, predicted, average="macro"),
                "mcc": matthews_corrcoef(labels, predicted),
            }
        )
    scores = pd.DataFrame(rows)
    selected_keys: list[dict[str, object]] = []
    for _, group in scores.groupby(
        ["file_type", "method", "feature_budget"], sort=True
    ):
        chosen = group.assign(
            rho_sort=pd.to_numeric(group["rho"], errors="coerce").fillna(-1.0)
        ).sort_values(
            ["macro_f1", "mcc", "rho_sort", "candidate_id"],
            ascending=[False, False, True, True],
            kind="stable",
        ).iloc[0]
        selected_keys.append(chosen[candidate_columns].to_dict())
    selected = pd.DataFrame(selected_keys).drop_duplicates(candidate_columns)
    result = frame.merge(
        selected,
        on=candidate_columns,
        how="inner",
        validate="many_to_one",
    )
    keys = ["file_type", "method", "feature_budget", "row_id"]
    if result.duplicated(keys).any():
        raise RuntimeError("budget reduction produced duplicate prediction rows")
    return result.sort_values(keys, kind="stable", ignore_index=True)


def validate_ablation_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate and order the three-format Figure 4 prediction artifact."""
    required = {
        "file_type",
        "analysis",
        "setting",
        "x_value",
        "row_id",
        "protocol_week",
        "protocol_role",
        "protocol_version",
        "y_true",
        "prediction",
    }
    if missing := required - set(predictions):
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["file_type"] = frame["file_type"].astype(str).str.upper()
    if set(frame["file_type"]) != FORMATS:
        raise ValueError("ablation predictions require APK, ELF, and PDF")
    if set(frame["protocol_role"].astype(str)) != {"validation"}:
        raise TemporalLeakageError("ablation predictions require validation rows only")
    weeks = pd.to_numeric(frame["protocol_week"], errors="raise")
    if not weeks.between(41, 52).all():
        raise TemporalLeakageError("ablation predictions are restricted to weeks 41-52")
    if set(frame["protocol_version"].astype(str)) != {PROTOCOL_VERSION}:
        raise ValueError("ablation predictions use an unexpected protocol version")
    expected_analyses = {"ablation", "sensitivity"}
    if set(frame["analysis"].astype(str)) != expected_analyses:
        raise ValueError("ablation and sensitivity settings are both required")
    keys = ["file_type", "analysis", "setting", "row_id"]
    if frame.duplicated(keys).any():
        raise ValueError("ablation predictions contain duplicate setting rows")
    for file_type, format_frame in frame.groupby("file_type", sort=True):
        expected_rows = None
        expected_labels = None
        for _, group in format_frame.groupby(["analysis", "setting"], sort=True):
            ordered = group.sort_values("row_id", kind="stable")
            rows = ordered["row_id"].to_numpy()
            labels = ordered["y_true"].to_numpy()
            if expected_rows is None:
                expected_rows, expected_labels = rows, labels
            elif (
                len(rows) != len(expected_rows)
                or not (rows == expected_rows).all()
                or not (labels == expected_labels).all()
            ):
                raise ValueError(f"unaligned ablation settings for {file_type}")
    return frame.sort_values(keys, kind="stable", ignore_index=True)
