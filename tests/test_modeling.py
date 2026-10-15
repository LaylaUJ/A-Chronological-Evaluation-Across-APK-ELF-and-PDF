import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from tcfs.modeling import (
    PRIMARY_CLASSIFIER,
    SECONDARY_CLASSIFIER,
    run_development_experiments,
    select_validation_configurations,
    select_validation_variant_configurations,
)
from tcfs.temporal_split import TemporalLeakageError


def _development_data(
    n_features: int = 20,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    X, y = make_classification(
        n_samples=104,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        random_state=12,
    )
    weeks = np.repeat(np.arange(1, 53), 2)
    assignments = pd.DataFrame(
        {
            "row_id": np.arange(len(X)),
            "protocol_week": weeks,
            "protocol_role": np.where(weeks <= 40, "discovery", "validation"),
        }
    )
    names = [f"feature_{index:02d}" for index in range(n_features)]
    return X, y, assignments, names


def _rankings(names: list[str], *, include_tcfs: bool = True) -> pd.DataFrame:
    rows = [
        {
            "method": "mutual_information",
            "feature_name": name,
            "rho": np.nan,
            "rank": rank,
        }
        for rank, name in enumerate(names, start=1)
    ]
    if include_tcfs:
        for rho in (0.0, 0.25, 0.5, 1.0):
            ordered = names if rho < 0.5 else names[::-1]
            rows.extend(
                {
                    "method": "tcfs",
                    "feature_name": name,
                    "rho": rho,
                    "rank": rank,
                }
                for rank, name in enumerate(ordered, start=1)
            )
    return pd.DataFrame(rows)


def _small_grids() -> dict[str, list[dict[str, float | int]]]:
    return {
        PRIMARY_CLASSIFIER: [
            {"n_estimators": 10, "learning_rate": 0.1, "num_leaves": 7}
        ],
        SECONDARY_CLASSIFIER: [{"C": 1.0}],
    }


def test_parsimony_rule_then_mcc_and_feature_count_ties() -> None:
    common = {
        "method": "tcfs",
        "classifier": PRIMARY_CLASSIFIER,
        "candidate_id": "candidate",
        "rho_sort": 0.0,
        "parameter_id": 0,
        "threshold": 0.5,
        "seed_rule": "seed_1",
    }
    results = pd.DataFrame(
        [
            {
                **common,
                "feature_budget": 0.6,
                "macro_f1": 0.90,
                "mcc": 0.9,
                "n_features": 12,
            },
            {
                **common,
                "feature_budget": 0.2,
                "macro_f1": 0.895,
                "mcc": 0.3,
                "n_features": 5,
            },
            {
                **common,
                "feature_budget": 0.2,
                "macro_f1": 0.895,
                "mcc": 0.4,
                "n_features": 5,
                "candidate_id": "higher_mcc",
            },
            {
                **common,
                "feature_budget": 0.2,
                "macro_f1": 0.895,
                "mcc": 0.4,
                "n_features": 4,
                "candidate_id": "fewer",
            },
            {
                **common,
                "feature_budget": 0.1,
                "macro_f1": 0.88,
                "mcc": 1.0,
                "n_features": 2,
            },
        ]
    )

    selected = select_validation_configurations(results).iloc[0]
    assert selected["feature_budget"] == 0.2
    assert selected["candidate_id"] == "fewer"
    assert selected["best_validation_macro_f1"] == 0.90


def test_selects_one_configuration_per_budget_and_rho_variant() -> None:
    rows = []
    for budget in (0.1, 0.2):
        for rho in (0.0, 0.25):
            for threshold, score in ((0.4, 0.7), (0.5, 0.8)):
                rows.append(
                    {
                        "method": "tcfs",
                        "classifier": PRIMARY_CLASSIFIER,
                        "feature_budget": budget,
                        "rho": rho,
                        "rho_sort": rho,
                        "macro_f1": score,
                        "mcc": score,
                        "n_features": int(20 * budget),
                        "candidate_id": f"{budget}-{rho}",
                        "parameter_id": 0,
                        "threshold": threshold,
                        "seed_rule": "seed_3",
                    }
                )
    selected = select_validation_variant_configurations(pd.DataFrame(rows))
    assert len(selected) == 4
    assert set(selected["threshold"]) == {0.5}


def test_tunes_both_classifiers_with_equal_method_parameter_budget() -> None:
    X, y, assignments, names = _development_data()
    result = run_development_experiments(
        X,
        y,
        assignments,
        names,
        _rankings(names),
        classifier_grids=_small_grids(),
        feature_budgets=(0.05, 0.10),
        thresholds=(0.4, 0.5),
        seeds=(3, 5),
    )

    assert set(result.selected_configurations["classifier"]) == {
        PRIMARY_CLASSIFIER,
        SECONDARY_CLASSIFIER,
    }
    assert set(result.selected_configurations["classifier_role"]) == {
        "primary",
        "secondary",
    }
    assert {
        "method",
        "classifier",
        "feature_budget",
        "rho",
        "row_id",
        "y_true",
        "probability",
        "prediction",
    } <= set(result.validation_variant_predictions)
    grouped = result.tuning_results.groupby(["method", "classifier"])[
        "parameter_id"
    ].nunique()
    assert grouped.nunique() == 1
    assert grouped.iloc[0] == 1
    assert {"seed_3", "seed_5", "mean_probability_ensemble"} <= set(
        result.tuning_results["seed_rule"]
    )
    tcfs_rhos = result.tuning_results.loc[
        result.tuning_results["method"] == "tcfs", "rho"
    ]
    assert set(tcfs_rhos) == {
        0.0,
        0.25,
        0.5,
        1.0,
    }


def test_budget_specific_rankings_are_evaluated_only_at_matching_budget() -> None:
    X, y, assignments, names = _development_data()
    rankings = _rankings(names)
    rankings["ranking_budget"] = np.where(
        rankings["method"].eq("tcfs"), 0.10, np.nan
    )
    result = run_development_experiments(
        X,
        y,
        assignments,
        names,
        rankings,
        classifier_grids=_small_grids(),
        feature_budgets=(0.05, 0.10),
        thresholds=(0.5,),
        seeds=(3,),
    )
    tcfs = result.tuning_results.loc[result.tuning_results["method"].eq("tcfs")]
    assert set(tcfs["feature_budget"]) == {0.10}


def test_rejects_official_test_rows_during_development() -> None:
    X, y, assignments, names = _development_data()
    assignments.loc[0, "protocol_role"] = "sealed_test"
    assignments.loc[0, "protocol_week"] = 53

    with pytest.raises(TemporalLeakageError, match="development experiments"):
        run_development_experiments(
            X,
            y,
            assignments,
            names,
            _rankings(names, include_tcfs=False),
            classifier_grids=_small_grids(),
            feature_budgets=(0.10,),
            thresholds=(0.5,),
            seeds=(3,),
        )


def test_saves_complete_artifacts_for_every_experiment(tmp_path: Path) -> None:
    X, y, assignments, names = _development_data()
    output = tmp_path / "experiments"
    result = run_development_experiments(
        X,
        y,
        assignments,
        names,
        _rankings(names, include_tcfs=False),
        classifier_grids=_small_grids(),
        feature_budgets=(0.10,),
        thresholds=(0.5,),
        seeds=(3,),
        output_dir=output,
    )

    assert len(result.selected_configurations) == 2
    expected_files = {
        "configuration.json",
        "selected_features.csv",
        "model.joblib",
        "validation_predictions.parquet",
        "tuning_results.csv",
        "runtime.json",
        "software_versions.json",
        "freeze_manifest.json",
    }
    assert (output / "validation_variant_predictions.parquet").is_file()
    for experiment_id in result.selected_configurations["experiment_id"]:
        experiment_dir = output / experiment_id
        assert {path.name for path in experiment_dir.iterdir()} == expected_files
        configuration = json.loads(
            (experiment_dir / "configuration.json").read_text(encoding="utf-8")
        )
        assert configuration["trained_through_week"] == 52
        assert configuration["protocol_version"]
        predictions = pd.read_parquet(
            experiment_dir / "validation_predictions.parquet"
        )
        assert set(predictions["protocol_role"]) == {"validation"}

    with pytest.raises(FileExistsError, match="force=True"):
        run_development_experiments(
            X,
            y,
            assignments,
            names,
            _rankings(names, include_tcfs=False),
            classifier_grids=_small_grids(),
            feature_budgets=(0.10,),
            thresholds=(0.5,),
            seeds=(3,),
            output_dir=output,
        )
