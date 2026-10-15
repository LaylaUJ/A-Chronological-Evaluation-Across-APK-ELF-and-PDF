"""Deterministic feature selectors used by TCFS experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

SUPPORTED_FEATURE_BUDGETS = (0.05, 0.10, 0.20, 0.40, 0.60)


def _normalize_budget(feature_budget: float | int) -> float:
    """Return a supported budget as a fraction."""
    budget = float(feature_budget)
    if budget > 1:
        budget /= 100
    if not any(np.isclose(budget, allowed) for allowed in SUPPORTED_FEATURE_BUDGETS):
        allowed = ", ".join(f"{value:.0%}" for value in SUPPORTED_FEATURE_BUDGETS)
        raise ValueError(f"feature_budget must be one of: {allowed}")
    return budget


class BaseFeatureSelector(TransformerMixin, BaseEstimator, ABC):
    """Common interface for deterministic, budgeted feature selectors.

    Normalized rank is in ``(0, 1]``: 1.0 denotes the highest-ranked feature.
    Ties are resolved by original column position so rankings remain reproducible.
    """

    def __init__(self, feature_budget: float | int = 0.10) -> None:
        self.feature_budget = feature_budget

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        feature_names: list[str] | tuple[str, ...] | np.ndarray | None = None,
    ) -> BaseFeatureSelector:
        """Fit the selector and construct its complete feature ranking."""
        names = self._resolve_feature_names(X, feature_names)
        if self._requires_y:
            if y is None:
                raise ValueError(f"{type(self).__name__} requires target labels")
            X_checked, y_checked = check_X_y(X, y, dtype="numeric")
        else:
            X_checked = check_array(X, dtype="numeric")
            y_checked = None if y is None else np.asarray(y)

        if X_checked.shape[1] != len(names):
            raise ValueError("feature_names length must match the number of columns")
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")

        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = X_checked.shape[1]
        raw_importance = np.asarray(
            self._compute_importance(X_checked, y_checked), dtype=float
        )
        if raw_importance.shape != (self.n_features_in_,):
            raise RuntimeError(
                "selector returned one importance per feature incorrectly"
            )
        if not np.isfinite(raw_importance).all():
            raise RuntimeError("selector returned non-finite feature importance")

        order = np.lexsort((np.arange(self.n_features_in_), -raw_importance))
        selected_count = self._selected_count(self.n_features_in_)
        support = np.zeros(self.n_features_in_, dtype=bool)
        support[order[:selected_count]] = True

        normalized_rank = np.empty(self.n_features_in_, dtype=float)
        normalized_rank[order] = (
            np.arange(self.n_features_in_, 0, -1) / self.n_features_in_
        )
        self.support_ = support
        self.feature_scores_ = (
            pd.DataFrame(
                {
                    "feature_name": self.feature_names_in_,
                    "raw_importance": raw_importance,
                    "normalized_rank": normalized_rank,
                    "selected": support,
                }
            )
            .iloc[order]
            .reset_index(drop=True)
        )
        return self

    def transform(self, X: Any) -> Any:
        """Return selected columns, preserving pandas input when supplied."""
        check_is_fitted(self, ("support_", "feature_scores_"))
        if isinstance(X, pd.DataFrame):
            actual = np.asarray([str(column) for column in X.columns], dtype=object)
            if not np.array_equal(actual, self.feature_names_in_):
                raise ValueError(
                    "DataFrame columns differ from the columns used in fit"
                )
            check_array(X, dtype="numeric")
            return X.loc[:, self.support_]

        checked = check_array(X, dtype="numeric")
        if checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted data"
            )
        return checked[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return the selection mask, or selected column indices."""
        check_is_fitted(self, "support_")
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()

    def get_feature_scores(self) -> pd.DataFrame:
        """Return a defensive copy of the complete ranked feature table."""
        check_is_fitted(self, "feature_scores_")
        return self.feature_scores_.copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return selected feature names for sklearn pipeline compatibility."""
        check_is_fitted(self, "support_")
        if input_features is not None:
            supplied = np.asarray(
                [str(value) for value in input_features], dtype=object
            )
            if not np.array_equal(supplied, self.feature_names_in_):
                raise ValueError("input_features differs from the columns used in fit")
        return self.feature_names_in_[self.support_].copy()

    @property
    def _requires_y(self) -> bool:
        return True

    def _selected_count(self, n_features: int) -> int:
        return max(1, ceil(_normalize_budget(self.feature_budget) * n_features))

    @staticmethod
    def _resolve_feature_names(X: Any, feature_names: Any) -> list[str]:
        if feature_names is not None:
            return [str(name) for name in feature_names]
        if isinstance(X, pd.DataFrame):
            return [str(column) for column in X.columns]
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        return [f"feature_{index}" for index in range(array.shape[1])]

    @abstractmethod
    def _compute_importance(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        """Compute one raw importance value per input feature."""


class MutualInformationSelector(BaseFeatureSelector):
    """Rank features by classification mutual information."""

    def __init__(
        self,
        feature_budget: float | int = 0.10,
        *,
        random_state: int = 42,
        n_neighbors: int = 3,
    ) -> None:
        super().__init__(feature_budget)
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def _compute_importance(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        assert y is not None
        return mutual_info_classif(
            X,
            y,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
        )


class L1LogisticSelector(BaseFeatureSelector):
    """Rank features by absolute L1-regularized logistic coefficients."""

    def __init__(
        self,
        feature_budget: float | int = 0.10,
        *,
        C: float = 1.0,
        random_state: int = 42,
        max_iter: int = 2_000,
    ) -> None:
        super().__init__(feature_budget)
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter

    def _compute_importance(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        assert y is not None
        version = tuple(int(part) for part in sklearn_version.split(".")[:2])
        l1_parameters = (
            {"l1_ratio": 1.0} if version >= (1, 8) else {"penalty": "l1"}
        )
        self.estimator_ = LogisticRegression(
            solver="liblinear",
            C=self.C,
            random_state=self.random_state,
            max_iter=self.max_iter,
            **l1_parameters,
        ).fit(X, y)
        return np.mean(np.abs(self.estimator_.coef_), axis=0)


class LightGBMPermutationSelector(BaseFeatureSelector):
    """Rank features by held-out LightGBM permutation importance."""

    def __init__(
        self,
        feature_budget: float | int = 0.10,
        *,
        random_state: int = 42,
        evaluation_fraction: float = 0.25,
        n_repeats: int = 5,
        scoring: str = "roc_auc",
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
    ) -> None:
        super().__init__(feature_budget)
        self.random_state = random_state
        self.evaluation_fraction = evaluation_fraction
        self.n_repeats = n_repeats
        self.scoring = scoring
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves

    def _compute_importance(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        assert y is not None
        X_train, X_evaluation, y_train, y_evaluation = train_test_split(
            X,
            y,
            test_size=self.evaluation_fraction,
            random_state=self.random_state,
            stratify=y,
        )
        self.estimator_ = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ).fit(X_train, y_train)
        result = permutation_importance(
            self.estimator_,
            X_evaluation,
            y_evaluation,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=1,
        )
        return result.importances_mean


class RandomMatchedSelector(BaseFeatureSelector):
    """Seeded random selector matched to the requested feature budget."""

    def __init__(
        self, feature_budget: float | int = 0.10, *, random_state: int = 42
    ) -> None:
        super().__init__(feature_budget)
        self.random_state = random_state

    @property
    def _requires_y(self) -> bool:
        return False

    def _compute_importance(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        return np.random.default_rng(self.random_state).random(X.shape[1])


class AllFeaturesSelector(BaseFeatureSelector):
    """Control selector that retains every feature."""

    def __init__(self) -> None:
        super().__init__(feature_budget=0.60)

    @property
    def _requires_y(self) -> bool:
        return False

    def _selected_count(self, n_features: int) -> int:
        return n_features

    def _compute_importance(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        return np.ones(X.shape[1], dtype=float)
