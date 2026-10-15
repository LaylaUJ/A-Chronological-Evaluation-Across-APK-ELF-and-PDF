import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from tcfs.ablation import build_validation_ablation_predictions
from tcfs.modeling import FittedClassifier
from tcfs.temporal_split import PROTOCOL_VERSION, TemporalLeakageError


def _inputs():
    rng = np.random.default_rng(4)
    weeks = np.repeat(np.arange(1, 53), 2)
    labels = np.tile([0, 1], 52).astype(np.int8)
    X = rng.normal(size=(len(weeks), 4))
    X[:, 0] += labels * 2
    assignments = pd.DataFrame(
        {
            "row_id": np.arange(len(weeks)),
            "protocol_week": weeks,
            "protocol_role": np.where(weeks <= 40, "discovery", "validation"),
            "protocol_version": PROTOCOL_VERSION,
        }
    )
    names = ["a", "b", "c", "d"]
    scores = pd.DataFrame(
        {
            "file_type": "APK",
            "feature_name": names * 4,
            "feature_budget": 0.5,
            "rho": np.repeat([0.0, 0.25, 0.5, 1.0], 4),
            "mean_normalized_rank": [1.0, 0.8, 0.6, 0.4] * 4,
            "selection_frequency": [1.0, 0.5, 0.8, 0.2] * 4,
            "rank_variance": [0.0, 0.2, 0.1, 0.3] * 4,
        }
    )
    estimator = LogisticRegression(random_state=7).fit(X[:80, :2], labels[:80])
    model = FittedClassifier(
        estimators=(estimator,),
        feature_indices=np.array([0, 1]),
        feature_names=("a", "b"),
        threshold=0.5,
        classifier="logistic_regression",
        seeds=(7,),
    )
    return X, labels, assignments, names, scores, model


def test_ablation_predictions_are_complete_and_reproducible() -> None:
    args = _inputs()
    first = build_validation_ablation_predictions(
        *args, file_type="APK", feature_budget=0.5, selected_rho=0.25
    )
    second = build_validation_ablation_predictions(
        *args, file_type="APK", feature_budget=0.5, selected_rho=0.25
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first["analysis"]) == {"ablation", "sensitivity"}
    assert first.groupby(["analysis", "setting"]).size().eq(24).all()
    assert set(first.loc[first["analysis"].eq("sensitivity"), "x_value"]) == {
        0.0,
        0.25,
        0.5,
        1.0,
    }
    assert first["protocol_role"].eq("validation").all()


def test_ablation_rejects_test_rows() -> None:
    X, y, assignments, names, scores, model = _inputs()
    assignments.loc[len(assignments)] = [999, 53, "sealed_test", PROTOCOL_VERSION]
    X = np.vstack([X, np.zeros(4)])
    y = np.append(y, 0)
    with pytest.raises(TemporalLeakageError, match="only"):
        build_validation_ablation_predictions(
            X,
            y,
            assignments,
            names,
            scores,
            model,
            file_type="APK",
            feature_budget=0.5,
            selected_rho=0.25,
        )
