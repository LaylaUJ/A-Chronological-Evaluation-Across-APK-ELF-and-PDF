import numpy as np
import pandas as pd
from scripts.build_feature_rankings import build_rankings


def _scores(selectors: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for file_type in ("APK", "ELF"):
        for window in range(1, 5):
            for selector in selectors:
                for index, name in enumerate(("a", "b", "c", "d")):
                    rows.append(
                        {
                            "file_type": file_type,
                            "discovery_window": window,
                            "selector": selector,
                            "feature_name": name,
                            "raw_importance": 4 - index,
                            "normalized_rank": 1.0 - index * 0.25,
                            "selected": index == 0,
                        }
                    )
    return pd.DataFrame(rows)


def test_builds_budget_specific_tcfs_and_complete_baselines() -> None:
    selectors = _scores(
        ("mutual_information", "l1_logistic", "lightgbm_permutation")
    )
    controls = _scores(("random_matched",))
    rankings, consensus = build_rankings(selectors, controls, budgets=(0.25, 0.5))
    tcfs = rankings.loc[rankings["method"].eq("tcfs")]
    assert set(tcfs["ranking_budget"]) == {0.25, 0.5}
    assert set(tcfs["rho"]) == {0.0, 0.25, 0.5, 1.0}
    assert len(tcfs) == 2 * 2 * 4 * 4
    assert set(rankings["method"]) == {
        "tcfs",
        "shared_tcfs",
        "mutual_information",
        "l1_logistic",
        "lightgbm_permutation",
        "random_matched",
        "all_features",
    }
    selected_frequency = consensus.loc[
        (np.isclose(consensus["rho"], 0.0))
        & (consensus["file_type"] == "APK")
        & (consensus["feature_name"] == "b")
    ].set_index("feature_budget")["selection_frequency"]
    assert selected_frequency.loc[0.25] == 0.0
    assert selected_frequency.loc[0.5] == 1.0
