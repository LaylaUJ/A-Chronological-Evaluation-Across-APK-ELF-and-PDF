import pandas as pd
import pytest
from scripts.apply_data_policy import build_approved_dq003_exclusions


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "official_split": ["train", "train", "train", "test", "test"],
            "row_id": [9, 2, 4, 8, 1],
            "sha256": ["a", "a", "b", "z", "z"],
            "file_type": ["APK", "APK", "ELF", "PDF", "PDF"],
        }
    )


def _proposed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rule_id": ["DQ003", "DQ003", "DQ010"],
            "official_split": ["train", "test", "test"],
            "row_id": [9, 8, 99],
            "sha256": ["a", "z", "other"],
        }
    )


def test_approved_policy_retains_lowest_row_id_without_deleting_data() -> None:
    metadata = _metadata()
    result = build_approved_dq003_exclusions(
        metadata, _proposed(), approved_at_utc="2026-01-01T00:00:00+00:00"
    )
    assert list(result["row_id"]) == [8, 9]
    assert set(result["rule_id"]) == {"DQ003"}
    assert set(result["decision"]) == {"exclude"}
    assert len(metadata) == 5


def test_approved_policy_rejects_incomplete_proposal() -> None:
    with pytest.raises(ValueError, match="do not exactly match"):
        build_approved_dq003_exclusions(
            _metadata(),
            _proposed().iloc[:1],
            approved_at_utc="2026-01-01T00:00:00+00:00",
        )
