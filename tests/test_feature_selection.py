import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from tcfs.feature_selection import (
    AllFeaturesSelector,
    L1LogisticSelector,
    LightGBMPermutationSelector,
    MutualInformationSelector,
    RandomMatchedSelector,
)


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, np.ndarray]:
    X, y = make_classification(
        n_samples=180,
        n_features=20,
        n_informative=6,
        n_redundant=2,
        random_state=17,
        shuffle=False,
    )
    columns = [f"signal_{index:02d}" for index in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), y


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(0.05, 1), (0.10, 2), (0.20, 4), (0.40, 8), (0.60, 12)],
)
def test_supported_budgets_select_expected_count(
    classification_data: tuple[pd.DataFrame, np.ndarray],
    budget: float,
    expected: int,
) -> None:
    X, y = classification_data
    selector = MutualInformationSelector(feature_budget=budget).fit(X, y)
    assert selector.get_support().sum() == expected
    assert selector.transform(X).shape == (len(X), expected)


def test_complete_ranked_table_and_pandas_transform(
    classification_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = classification_data
    selector = L1LogisticSelector(feature_budget=20, random_state=9).fit(X, y)
    scores = selector.get_feature_scores()

    assert list(scores.columns) == [
        "feature_name",
        "raw_importance",
        "normalized_rank",
        "selected",
    ]
    assert len(scores) == X.shape[1]
    assert scores["feature_name"].is_unique
    assert scores["normalized_rank"].iloc[0] == 1.0
    assert scores["normalized_rank"].is_monotonic_decreasing
    assert scores["raw_importance"].is_monotonic_decreasing
    assert list(selector.transform(X).columns) == list(
        selector.get_feature_names_out()
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MutualInformationSelector(0.20, random_state=11),
        lambda: L1LogisticSelector(0.20, random_state=11),
        lambda: RandomMatchedSelector(0.20, random_state=11),
        lambda: LightGBMPermutationSelector(
            0.20, random_state=11, n_estimators=30, n_repeats=2
        ),
    ],
)
def test_selectors_are_deterministic(
    classification_data: tuple[pd.DataFrame, np.ndarray], factory
) -> None:
    X, y = classification_data
    first = factory().fit(X, y)
    second = factory().fit(X, y)

    np.testing.assert_array_equal(first.get_support(), second.get_support())
    pd.testing.assert_frame_equal(
        first.get_feature_scores(), second.get_feature_scores()
    )


def test_all_features_baseline(classification_data) -> None:
    X, _ = classification_data
    selector = AllFeaturesSelector().fit(X)

    assert selector.get_support().all()
    assert selector.get_support(indices=True).tolist() == list(range(X.shape[1]))
    pd.testing.assert_frame_equal(selector.transform(X), X)
    assert selector.get_feature_scores()["selected"].all()


def test_invalid_budget_is_rejected(classification_data) -> None:
    X, y = classification_data
    with pytest.raises(ValueError, match="feature_budget"):
        MutualInformationSelector(0.30).fit(X, y)


def test_dataframe_column_reordering_is_rejected(classification_data) -> None:
    X, y = classification_data
    selector = MutualInformationSelector().fit(X, y)
    with pytest.raises(ValueError, match="columns differ"):
        selector.transform(X[X.columns[::-1]])
