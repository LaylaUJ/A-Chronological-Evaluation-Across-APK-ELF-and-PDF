import numpy as np
import pandas as pd
import pytest

from tcfs.consensus import aggregate_consensus, evaluate_rho_values
from tcfs.temporal_split import TemporalLeakageError

SELECTORS = ("l1", "lightgbm", "mutual_information")


def _scores(
    feature_values: dict[str, tuple[list[float], list[bool]]],
    *,
    file_type: str = "APK",
) -> pd.DataFrame:
    rows = []
    for feature_name, (ranks, selected) in feature_values.items():
        assert len(ranks) == 12
        assert len(selected) == 12
        for index, (rank, is_selected) in enumerate(zip(ranks, selected, strict=True)):
            rows.append(
                {
                    "file_type": file_type,
                    "feature_name": feature_name,
                    "discovery_window": index // 3 + 1,
                    "selector": SELECTORS[index % 3],
                    "normalized_rank": rank,
                    "selected": is_selected,
                }
            )
    return pd.DataFrame(rows)


def test_manually_calculated_consensus_statistics_and_score() -> None:
    ranks = [0.2, 0.4] * 6
    selected = [True, False] * 6
    result = aggregate_consensus(
        _scores({"alpha": (ranks, selected)}), rho=0.5
    ).iloc[0]

    assert result["mean_normalized_rank"] == pytest.approx(0.3)
    assert result["selection_frequency"] == pytest.approx(0.5)
    assert result["rank_variance"] == pytest.approx(0.01)
    assert result["format_tcfs_score"] == pytest.approx(0.15 / 1.005)
    assert result["shared_tcfs_score"] == pytest.approx(0.15 / 1.005)
    assert result["observation_count"] == 12


def test_zero_variance_has_no_penalty() -> None:
    result = aggregate_consensus(
        _scores({"stable": ([0.8] * 12, [True] * 12)}), rho=1.0
    ).iloc[0]
    assert result["rank_variance"] == 0.0
    assert result["format_tcfs_score"] == pytest.approx(0.8)


def test_ties_use_frequency_then_mean_rank_then_alphabetical_name() -> None:
    scores = _scores(
        {
            "z_low_frequency": ([1.0] * 12, [True, False] * 6),
            "beta": ([0.5] * 12, [True] * 12),
            "alpha": ([0.5] * 12, [True] * 12),
        }
    )
    ranked = aggregate_consensus(scores, rho=0.0)
    assert ranked["feature_name"].tolist() == [
        "alpha",
        "beta",
        "z_low_frequency",
    ]


def test_shared_score_combines_same_feature_across_formats() -> None:
    apk = _scores({"common": ([1.0] * 12, [True] * 12)}, file_type="APK")
    elf = _scores({"common": ([0.5] * 12, [False] * 12)}, file_type="ELF")
    result = aggregate_consensus(pd.concat([apk, elf]), rho=1.0)

    # Across formats: M=.75, F=.5, V=.0625.
    expected = 0.375 / 1.0625
    assert result["shared_tcfs_score"].nunique() == 1
    assert result["shared_tcfs_score"].iloc[0] == pytest.approx(expected)


def test_missing_selector_score_is_rejected() -> None:
    incomplete = _scores({"alpha": ([0.5] * 12, [True] * 12)}).iloc[:-1]
    with pytest.raises(ValueError, match="missing selector scores"):
        aggregate_consensus(incomplete, rho=0.25)


def test_reproducible_for_shuffled_input() -> None:
    scores = _scores(
        {
            "alpha": (np.linspace(0.1, 1.0, 12).tolist(), [True, False] * 6),
            "beta": (np.linspace(1.0, 0.1, 12).tolist(), [False, True] * 6),
        }
    )
    first = aggregate_consensus(scores.sample(frac=1, random_state=1), rho=0.5)
    second = aggregate_consensus(scores.sample(frac=1, random_state=99), rho=0.5)
    pd.testing.assert_frame_equal(first, second)


def test_rho_search_uses_validation_only() -> None:
    scores = _scores(
        {"alpha": ([0.2, 0.8] * 6, [True, False] * 6)}
    )
    validation = pd.DataFrame(
        {"protocol_role": ["validation", "validation"], "label": [0, 1]}
    )

    def evaluator(consensus: pd.DataFrame, data: pd.DataFrame) -> float:
        assert set(data["protocol_role"]) == {"validation"}
        rho = float(consensus["rho"].iloc[0])
        return -abs(rho - 0.25)

    result = evaluate_rho_values(scores, validation, evaluator)
    assert result.best_rho == 0.25
    assert set(result.consensus_tables) == {0.0, 0.25, 0.5, 1.0}
    assert result.evaluations.loc[result.evaluations["selected"], "rho"].item() == 0.25


def test_rho_search_rejects_non_validation_rows() -> None:
    scores = _scores({"alpha": ([0.5] * 12, [True] * 12)})
    mixed = pd.DataFrame(
        {"protocol_role": ["validation", "sealed_test"], "label": [0, 1]}
    )
    with pytest.raises(TemporalLeakageError, match="validation rows only"):
        evaluate_rho_values(scores, mixed, lambda consensus, data: 1.0)
