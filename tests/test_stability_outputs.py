import pandas as pd

from tcfs.stability_outputs import build_feature_results


def _scores(selectors: tuple[str, ...], formats: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for file_type in formats:
        for window in range(1, 5):
            for selector_index, selector in enumerate(selectors):
                order = ("a", "b", "c", "d")
                if selector_index % 2:
                    order = order[::-1]
                for rank, name in enumerate(order, start=1):
                    rows.append(
                        {
                            "file_type": file_type,
                            "discovery_window": window,
                            "selector": selector,
                            "feature_name": name,
                            "normalized_rank": (5 - rank) / 4,
                        }
                    )
    return pd.DataFrame(rows)


def test_builds_complete_window_replicates_for_all_methods() -> None:
    selectors = _scores(
        ("mutual_information", "l1_logistic", "lightgbm_permutation"),
        ("APK", "ELF"),
    )
    controls = _scores(("random_matched",), ("APK", "ELF"))
    methods = [
        "tcfs",
        "shared_tcfs",
        "mutual_information",
        "l1_logistic",
        "lightgbm_permutation",
        "random_matched",
        "all_features",
    ]
    configurations = pd.DataFrame(
        {
            "file_type": "APK",
            "method": methods,
            "classifier": "lightgbm",
            "feature_budget": [0.5] * 6 + [1.0],
            "rho": [0.25, 0.25] + [float("nan")] * 5,
        }
    )
    result = build_feature_results(selectors, controls, configurations)
    assert set(result["method"]) == set(methods)
    assert set(result["replicate"]) == {
        "window_1",
        "window_2",
        "window_3",
        "window_4",
    }
    assert not result.duplicated(
        ["file_type", "method", "replicate", "feature_name"]
    ).any()
    all_features = result.loc[result["method"].eq("all_features")]
    assert all_features["selected"].all()
