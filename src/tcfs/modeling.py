"""Leakage-resistant classifier tuning and experiment persistence."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tcfs.consensus import TCFS_RHO_VALUES
from tcfs.feature_selection import SUPPORTED_FEATURE_BUDGETS
from tcfs.temporal_split import PROTOCOL_VERSION, TemporalLeakageError

PRIMARY_CLASSIFIER = "lightgbm"
SECONDARY_CLASSIFIER = "logistic_regression"
CLASSIFIER_ROLES = {
    PRIMARY_CLASSIFIER: "primary",
    SECONDARY_CLASSIFIER: "secondary",
}
DEFAULT_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
DEFAULT_SEEDS = (17, 42, 73)
DEFAULT_CLASSIFIER_GRIDS: dict[str, tuple[dict[str, Any], ...]] = {
    PRIMARY_CLASSIFIER: (
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
        {"n_estimators": 200, "learning_rate": 0.10, "num_leaves": 31},
    ),
    SECONDARY_CLASSIFIER: (
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
    ),
}
REQUIRED_RANKING_COLUMNS = {"method", "feature_name", "rho", "rank"}


@dataclass(frozen=True)
class FittedClassifier:
    """Serializable final classifier, including ensemble and decision metadata."""

    estimators: tuple[BaseEstimator, ...]
    feature_indices: np.ndarray
    feature_names: tuple[str, ...]
    threshold: float
    classifier: str
    seeds: tuple[int, ...]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return mean positive-class probability across fitted estimators."""
        matrix = np.asarray(X)
        selected = matrix[:, self.feature_indices]
        probabilities = [
            model.predict_proba(selected)[:, 1] for model in self.estimators
        ]
        positive = np.mean(probabilities, axis=0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> np.ndarray:
        """Apply the validation-selected classification threshold."""
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(np.int8)


@dataclass(frozen=True)
class DevelopmentResults:
    """Outputs from validation-only model selection and final development fits."""

    selected_configurations: pd.DataFrame
    tuning_results: pd.DataFrame
    fitted_models: dict[str, FittedClassifier]
    validation_predictions: dict[str, pd.DataFrame]
    validation_variant_predictions: pd.DataFrame
    runtime: pd.DataFrame
    software_versions: dict[str, str]


def run_development_experiments(
    X: Any,
    y: Any,
    assignments: pd.DataFrame,
    feature_names: Sequence[str],
    rankings: pd.DataFrame,
    *,
    classifier_grids: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    feature_budgets: Sequence[float] = SUPPORTED_FEATURE_BUDGETS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    output_dir: str | Path | None = None,
    force: bool = False,
    file_type: str | None = None,
) -> DevelopmentResults:
    """Tune two classifiers using weeks 41-52 and persist final experiments.

    Models used to obtain validation metrics are trained on discovery weeks
    1-40. After every choice is frozen, final models are refit on weeks 1-52.
    Official-test rows are rejected from this development API.
    """
    started = perf_counter()
    matrix, labels, names, discovery, validation = _validate_development_inputs(
        X, y, assignments, feature_names
    )
    ranking_variants = _validate_rankings(rankings, names)
    grids = _validate_search_space(
        classifier_grids or DEFAULT_CLASSIFIER_GRIDS,
        feature_budgets,
        thresholds,
        seeds,
    )

    trial_rows: list[dict[str, Any]] = []
    probability_cache: dict[str, np.ndarray] = {}
    candidate_cache: dict[str, dict[str, Any]] = {}
    fit_runtime: dict[str, float] = {}

    for method, variants in ranking_variants.items():
        for (rho, ranking_budget), ordered_features in variants.items():
            if method == "all_features":
                budgets = (1.0,)
            elif ranking_budget is None:
                budgets = tuple(feature_budgets)
            else:
                budgets = (ranking_budget,)
            for budget in budgets:
                selected_names = _select_budget(ordered_features, budget)
                indices = np.asarray([names.index(name) for name in selected_names])
                X_train = matrix[discovery][:, indices]
                X_validation = matrix[validation][:, indices]
                y_train = labels[discovery]
                y_validation = labels[validation]
                for classifier, parameter_grid in grids.items():
                    for parameter_id, parameters in enumerate(parameter_grid):
                        seed_probabilities: list[np.ndarray] = []
                        seed_times: list[float] = []
                        for seed in seeds:
                            fit_started = perf_counter()
                            estimator = _build_classifier(
                                classifier, parameters, int(seed)
                            )
                            estimator.fit(X_train, y_train)
                            elapsed = perf_counter() - fit_started
                            probabilities = estimator.predict_proba(X_validation)[:, 1]
                            seed_probabilities.append(probabilities)
                            seed_times.append(elapsed)
                            key = _candidate_key(
                                method,
                                classifier,
                                rho,
                                budget,
                                parameter_id,
                                f"seed_{seed}",
                            )
                            probability_cache[key] = probabilities
                            candidate_cache[key] = {
                                "feature_indices": indices,
                                "feature_names": selected_names,
                                "parameters": dict(parameters),
                                "seeds": (int(seed),),
                            }
                            fit_runtime[key] = elapsed
                            _append_threshold_trials(
                                trial_rows,
                                key=key,
                                method=method,
                                classifier=classifier,
                                rho=rho,
                                budget=budget,
                                parameter_id=parameter_id,
                                parameters=parameters,
                                seed_rule=f"seed_{seed}",
                                probabilities=probabilities,
                                y_true=y_validation,
                                thresholds=thresholds,
                                n_features=len(indices),
                                fit_seconds=elapsed,
                            )

                        ensemble_probabilities = np.mean(seed_probabilities, axis=0)
                        ensemble_key = _candidate_key(
                            method,
                            classifier,
                            rho,
                            budget,
                            parameter_id,
                            "mean_probability_ensemble",
                        )
                        probability_cache[ensemble_key] = ensemble_probabilities
                        candidate_cache[ensemble_key] = {
                            "feature_indices": indices,
                            "feature_names": selected_names,
                            "parameters": dict(parameters),
                            "seeds": tuple(int(seed) for seed in seeds),
                        }
                        fit_runtime[ensemble_key] = float(sum(seed_times))
                        _append_threshold_trials(
                            trial_rows,
                            key=ensemble_key,
                            method=method,
                            classifier=classifier,
                            rho=rho,
                            budget=budget,
                            parameter_id=parameter_id,
                            parameters=parameters,
                            seed_rule="mean_probability_ensemble",
                            probabilities=ensemble_probabilities,
                            y_true=y_validation,
                            thresholds=thresholds,
                            n_features=len(indices),
                            fit_seconds=float(sum(seed_times)),
                        )

    tuning_results = pd.DataFrame(trial_rows)
    selected_configurations = select_validation_configurations(tuning_results)
    variant_configurations = select_validation_variant_configurations(
        tuning_results.loc[tuning_results["classifier"].eq(PRIMARY_CLASSIFIER)]
    )
    variant_prediction_frames: list[pd.DataFrame] = []
    for variant in variant_configurations.itertuples(index=False):
        probability = probability_cache[str(variant.candidate_id)]
        frame = pd.DataFrame(
            {
                "row_id": assignments.iloc[validation]["row_id"].to_numpy(),
                "protocol_week": assignments.iloc[validation][
                    "protocol_week"
                ].to_numpy(),
                "protocol_role": "validation",
                "protocol_version": PROTOCOL_VERSION,
                "method": str(variant.method),
                "classifier": str(variant.classifier),
                "feature_budget": float(variant.feature_budget),
                "rho": variant.rho,
                "candidate_id": str(variant.candidate_id),
                "parameter_id": int(variant.parameter_id),
                "seed_rule": str(variant.seed_rule),
                "threshold": float(variant.threshold),
                "y_true": labels[validation],
                "probability": probability,
                "prediction": (
                    probability >= float(variant.threshold)
                ).astype(np.int8),
            }
        )
        if file_type is not None:
            frame.insert(0, "file_type", file_type)
        variant_prediction_frames.append(frame)
    validation_variant_predictions = pd.concat(
        variant_prediction_frames, ignore_index=True
    )
    fitted_models: dict[str, FittedClassifier] = {}
    validation_predictions: dict[str, pd.DataFrame] = {}
    runtime_rows: list[dict[str, Any]] = []
    development = np.concatenate([discovery, validation])

    for row in selected_configurations.itertuples(index=False):
        experiment_id = str(row.experiment_id)
        candidate = candidate_cache[str(row.candidate_id)]
        indices = candidate["feature_indices"]
        final_estimators: list[BaseEstimator] = []
        final_started = perf_counter()
        for seed in candidate["seeds"]:
            estimator = _build_classifier(
                str(row.classifier), candidate["parameters"], seed
            )
            estimator.fit(matrix[development][:, indices], labels[development])
            final_estimators.append(estimator)
        final_seconds = perf_counter() - final_started
        fitted_models[experiment_id] = FittedClassifier(
            estimators=tuple(final_estimators),
            feature_indices=indices.copy(),
            feature_names=tuple(candidate["feature_names"]),
            threshold=float(row.threshold),
            classifier=str(row.classifier),
            seeds=tuple(candidate["seeds"]),
        )

        probability = probability_cache[str(row.candidate_id)]
        validation_predictions[experiment_id] = pd.DataFrame(
            {
                "row_id": assignments.iloc[validation]["row_id"].to_numpy(),
                "protocol_week": assignments.iloc[validation][
                    "protocol_week"
                ].to_numpy(),
                "protocol_role": "validation",
                "y_true": labels[validation],
                "probability": probability,
                "prediction": (probability >= float(row.threshold)).astype(np.int8),
                "threshold": float(row.threshold),
            }
        )
        runtime_rows.append(
            {
                "experiment_id": experiment_id,
                "validation_fit_seconds": fit_runtime[str(row.candidate_id)],
                "final_fit_seconds": final_seconds,
            }
        )

    selected_configurations = selected_configurations.copy()
    selected_configurations["protocol_version"] = PROTOCOL_VERSION
    selected_configurations["trained_through_week"] = 52
    selected_configurations["final_seeds_json"] = selected_configurations[
        "candidate_id"
    ].map(lambda key: json.dumps(candidate_cache[str(key)]["seeds"]))
    total_seconds = perf_counter() - started
    runtime = pd.DataFrame(runtime_rows)
    runtime["development_run_seconds"] = total_seconds
    results = DevelopmentResults(
        selected_configurations=selected_configurations,
        tuning_results=tuning_results,
        fitted_models=fitted_models,
        validation_predictions=validation_predictions,
        validation_variant_predictions=validation_variant_predictions,
        runtime=runtime,
        software_versions=collect_software_versions(),
    )
    if output_dir is not None:
        save_experiment_artifacts(results, output_dir, force=force)
    return results


def select_validation_configurations(tuning_results: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented 0.01 parsimony rule to each method/classifier."""
    required = {
        "method",
        "classifier",
        "feature_budget",
        "macro_f1",
        "mcc",
        "n_features",
        "candidate_id",
    }
    missing = required - set(tuning_results.columns)
    if missing:
        raise ValueError(f"tuning_results is missing columns: {sorted(missing)}")
    selected: list[pd.Series] = []
    for (method, classifier), group in tuning_results.groupby(
        ["method", "classifier"], sort=True
    ):
        best_macro_f1 = float(group["macro_f1"].max())
        eligible = group.loc[group["macro_f1"] >= best_macro_f1 - 0.01 - 1e-12]
        smallest_budget = float(eligible["feature_budget"].min())
        eligible = eligible.loc[
            np.isclose(eligible["feature_budget"], smallest_budget)
        ]
        chosen = eligible.sort_values(
            [
                "macro_f1",
                "mcc",
                "n_features",
                "rho_sort",
                "parameter_id",
                "threshold",
                "seed_rule",
            ],
            ascending=[False, False, True, True, True, True, True],
            kind="stable",
        ).iloc[0]
        chosen = chosen.copy()
        chosen["best_validation_macro_f1"] = best_macro_f1
        chosen["classifier_role"] = CLASSIFIER_ROLES[str(classifier)]
        chosen["experiment_id"] = _experiment_id(str(method), str(classifier))
        selected.append(chosen)
    return pd.DataFrame(selected).reset_index(drop=True)


def select_validation_variant_configurations(
    tuning_results: pd.DataFrame,
) -> pd.DataFrame:
    """Select one prediction-producing candidate per method/budget/rho variant."""
    required = {
        "method",
        "classifier",
        "feature_budget",
        "rho",
        "rho_sort",
        "macro_f1",
        "mcc",
        "n_features",
        "candidate_id",
        "parameter_id",
        "threshold",
        "seed_rule",
    }
    if missing := required - set(tuning_results):
        raise ValueError(f"tuning_results is missing columns: {sorted(missing)}")
    selected: list[pd.Series] = []
    group_columns = ["method", "classifier", "feature_budget", "rho_sort"]
    for _, group in tuning_results.groupby(group_columns, sort=True, dropna=False):
        chosen = group.sort_values(
            [
                "macro_f1",
                "mcc",
                "n_features",
                "parameter_id",
                "threshold",
                "seed_rule",
            ],
            ascending=[False, False, True, True, True, True],
            kind="stable",
        ).iloc[0]
        selected.append(chosen)
    return pd.DataFrame(selected).reset_index(drop=True)


def save_experiment_artifacts(
    results: DevelopmentResults,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> None:
    """Persist complete, self-contained artifacts for every final experiment."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    variants_path = root / "validation_variant_predictions.parquet"
    if variants_path.exists() and not force:
        raise FileExistsError(
            f"Validation variant output already exists: {variants_path}; "
            "use force=True"
        )
    results.validation_variant_predictions.to_parquet(variants_path, index=False)
    for row in results.selected_configurations.itertuples(index=False):
        experiment_id = str(row.experiment_id)
        destination = root / experiment_id
        if destination.exists():
            if not force:
                raise FileExistsError(
                    f"Experiment output already exists: {destination}; use force=True"
                )
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        configuration = _json_ready(row._asdict())
        (destination / "configuration.json").write_text(
            json.dumps(configuration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selected_features = results.fitted_models[experiment_id].feature_names
        pd.DataFrame(
            {
                "selection_order": np.arange(1, len(selected_features) + 1),
                "feature_name": selected_features,
            }
        ).to_csv(destination / "selected_features.csv", index=False)
        joblib.dump(results.fitted_models[experiment_id], destination / "model.joblib")
        results.validation_predictions[experiment_id].to_parquet(
            destination / "validation_predictions.parquet", index=False
        )
        results.tuning_results.loc[
            results.tuning_results["method"].eq(row.method)
            & results.tuning_results["classifier"].eq(row.classifier)
        ].to_csv(destination / "tuning_results.csv", index=False)
        runtime = results.runtime.loc[
            results.runtime["experiment_id"].eq(experiment_id)
        ].iloc[0]
        (destination / "runtime.json").write_text(
            json.dumps(_json_ready(runtime.to_dict()), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "software_versions.json").write_text(
            json.dumps(results.software_versions, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        freeze_experiment_artifacts(destination)


def freeze_experiment_artifacts(experiment_dir: str | Path) -> Path:
    """Write a cryptographic freeze manifest for a finalized experiment."""
    directory = Path(experiment_dir)
    required_files = (
        "configuration.json",
        "selected_features.csv",
        "model.joblib",
    )
    missing = [name for name in required_files if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze missing artifacts: {missing}")
    configuration = json.loads(
        (directory / "configuration.json").read_text(encoding="utf-8")
    )
    required_fields = {
        "protocol_version",
        "parameters_json",
        "threshold",
        "final_seeds_json",
        "trained_through_week",
    }
    missing_fields = required_fields - set(configuration)
    if missing_fields:
        raise ValueError(
            f"configuration cannot be frozen; missing fields: {sorted(missing_fields)}"
        )
    manifest = {
        "frozen": True,
        "protocol_frozen": True,
        "protocol_version": configuration["protocol_version"],
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "frozen_fields": [
            "selected_features",
            "classifier_hyperparameters",
            "classification_threshold",
            "random_seed_or_ensemble_rule",
        ],
        "sha256": {
            name: _sha256_file(directory / name) for name in required_files
        },
    }
    path = directory / "freeze_manifest.json"
    if path.exists():
        raise FileExistsError(f"freeze manifest already exists: {path}")
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def collect_software_versions() -> dict[str, str]:
    """Capture runtime and direct scientific-package versions."""
    packages = [
        "joblib",
        "lightgbm",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
    ]
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:  # pragma: no cover - required dependencies
            versions[package] = "not-installed"
    return versions


def _validate_development_inputs(
    X: Any,
    y: Any,
    assignments: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    matrix = np.asarray(X)
    labels = np.asarray(y)
    names = [str(name) for name in feature_names]
    if matrix.ndim != 2 or labels.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if len(matrix) != len(labels) or len(matrix) != len(assignments):
        raise ValueError("X, y, and assignments must have identical row counts")
    if matrix.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("feature_names must uniquely name every X column")
    required = {"row_id", "protocol_week", "protocol_role"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"assignments is missing columns: {sorted(missing)}")
    roles = set(assignments["protocol_role"].astype(str))
    if not roles <= {"discovery", "validation"} or "validation" not in roles:
        raise TemporalLeakageError(
            "development experiments accept discovery and validation rows only"
        )
    weeks = pd.to_numeric(assignments["protocol_week"], errors="raise")
    discovery = np.flatnonzero(assignments["protocol_role"].eq("discovery"))
    validation = np.flatnonzero(assignments["protocol_role"].eq("validation"))
    if not len(discovery):
        raise ValueError("development experiments require discovery rows")
    if not weeks.iloc[discovery].between(1, 40).all():
        raise TemporalLeakageError("discovery rows must be weeks 1-40")
    if not weeks.iloc[validation].between(41, 52).all():
        raise TemporalLeakageError("rho and model selection require weeks 41-52")
    if set(weeks.iloc[validation]) != set(range(41, 53)):
        raise TemporalLeakageError(
            "validation data must cover every protocol week from 41 through 52"
        )
    if not np.isfinite(matrix).all() or not np.isfinite(labels).all():
        raise ValueError("development features and labels must be finite")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("binary labels 0 and 1 are required")
    return matrix, labels.astype(np.int8), names, discovery, validation


def _validate_rankings(
    rankings: pd.DataFrame, feature_names: Sequence[str]
) -> dict[str, dict[tuple[float | None, float | None], list[str]]]:
    missing = REQUIRED_RANKING_COLUMNS - set(rankings.columns)
    if missing:
        raise ValueError(f"rankings is missing columns: {sorted(missing)}")
    if rankings.empty:
        raise ValueError("rankings must not be empty")
    expected = set(feature_names)
    variants: dict[
        str, dict[tuple[float | None, float | None], list[str]]
    ] = {}
    frame = rankings.copy()
    frame["method"] = frame["method"].astype(str)
    frame["feature_name"] = frame["feature_name"].astype(str)
    if "ranking_budget" not in frame:
        frame["ranking_budget"] = np.nan
    for method, method_rows in frame.groupby("method", sort=True):
        variants[method] = {}
        defined_rho = method_rows["rho"].dropna().astype(float)
        if not defined_rho.empty and set(defined_rho.unique()) != set(TCFS_RHO_VALUES):
            raise ValueError(
                f"rho-aware method {method} must provide {TCFS_RHO_VALUES}"
            )
        if not defined_rho.empty and method_rows["rho"].isna().any():
            raise ValueError(f"method {method} mixes missing and defined rho values")
        budget_values: list[float | None] = (
            [None]
            if method_rows["ranking_budget"].isna().all()
            else sorted(method_rows["ranking_budget"].dropna().astype(float).unique())
        )
        if budget_values != [None] and method_rows["ranking_budget"].isna().any():
            raise ValueError(
                f"method {method} mixes missing and defined ranking budgets"
            )
        for ranking_budget in budget_values:
            budget_rows = (
                method_rows.loc[method_rows["ranking_budget"].isna()]
                if ranking_budget is None
                else method_rows.loc[
                    np.isclose(
                        method_rows["ranking_budget"].astype(float), ranking_budget
                    )
                ]
            )
            rho_values: list[float | None] = (
                [None]
                if budget_rows["rho"].isna().all()
                else sorted(budget_rows["rho"].astype(float).unique())
            )
            for rho in rho_values:
                subset = (
                    budget_rows.loc[budget_rows["rho"].isna()]
                    if rho is None
                    else budget_rows.loc[
                        np.isclose(budget_rows["rho"].astype(float), rho)
                    ]
                )
                if (
                    set(subset["feature_name"]) != expected
                    or len(subset) != len(expected)
                ):
                    raise ValueError(
                        f"method={method}, rho={rho}, budget={ranking_budget} must "
                        "rank every feature exactly once"
                    )
                numeric_rank = pd.to_numeric(subset["rank"], errors="raise")
                if (
                    not np.isfinite(numeric_rank).all()
                    or numeric_rank.duplicated().any()
                ):
                    raise ValueError(
                        "ranking values must be finite and unique per variant"
                    )
                ordered = subset.assign(_rank=numeric_rank).sort_values(
                    ["_rank", "feature_name"], kind="stable"
                )["feature_name"]
                variants[method][(rho, ranking_budget)] = ordered.tolist()
    return variants


def _validate_search_space(
    grids: Mapping[str, Sequence[Mapping[str, Any]]],
    budgets: Sequence[float],
    thresholds: Sequence[float],
    seeds: Sequence[int],
) -> dict[str, tuple[dict[str, Any], ...]]:
    if set(grids) != set(CLASSIFIER_ROLES):
        raise ValueError(f"classifier grids must define {sorted(CLASSIFIER_ROLES)}")
    normalized = {
        name: tuple(dict(parameters) for parameters in grid)
        for name, grid in grids.items()
    }
    if any(not grid for grid in normalized.values()):
        raise ValueError(
            "each classifier requires at least one parameter configuration"
        )
    budget_values = tuple(float(value) for value in budgets)
    if not budget_values or not set(budget_values) <= set(SUPPORTED_FEATURE_BUDGETS):
        raise ValueError(
            f"feature budgets must be drawn from {SUPPORTED_FEATURE_BUDGETS}"
        )
    if not thresholds or not all(0.0 < float(value) < 1.0 for value in thresholds):
        raise ValueError("thresholds must be within (0, 1)")
    if not seeds or len(set(int(value) for value in seeds)) != len(seeds):
        raise ValueError("seeds must contain unique values")
    return normalized


def _build_classifier(
    classifier: str, parameters: Mapping[str, Any], seed: int
) -> BaseEstimator:
    if classifier == PRIMARY_CLASSIFIER:
        return LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            **parameters,
        )
    if classifier == SECONDARY_CLASSIFIER:
        version_parts = tuple(int(part) for part in sklearn_version.split(".")[:2])
        l1_parameters = (
            {"l1_ratio": 0.0} if version_parts >= (1, 8) else {"penalty": "l2"}
        )
        logistic = LogisticRegression(
            solver="liblinear",
            random_state=seed,
            max_iter=2_000,
            **l1_parameters,
            **parameters,
        )
        return Pipeline([("scale", StandardScaler()), ("classifier", logistic)])
    raise ValueError(f"unknown classifier: {classifier}")


def _append_threshold_trials(
    rows: list[dict[str, Any]],
    *,
    key: str,
    method: str,
    classifier: str,
    rho: float | None,
    budget: float,
    parameter_id: int,
    parameters: Mapping[str, Any],
    seed_rule: str,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    thresholds: Sequence[float],
    n_features: int,
    fit_seconds: float,
) -> None:
    for threshold in thresholds:
        prediction = (probabilities >= float(threshold)).astype(np.int8)
        rows.append(
            {
                "candidate_id": key,
                "method": method,
                "classifier": classifier,
                "classifier_role": CLASSIFIER_ROLES[classifier],
                "rho": rho,
                "rho_sort": -1.0 if rho is None else rho,
                "feature_budget": budget,
                "n_features": n_features,
                "parameter_id": parameter_id,
                "parameters_json": json.dumps(parameters, sort_keys=True),
                "seed_rule": seed_rule,
                "threshold": float(threshold),
                "macro_f1": f1_score(y_true, prediction, average="macro"),
                "mcc": matthews_corrcoef(y_true, prediction),
                "validation_rows": len(y_true),
                "fit_seconds": fit_seconds,
            }
        )


def _select_budget(ordered_features: Sequence[str], budget: float) -> list[str]:
    if np.isclose(budget, 1.0):
        count = len(ordered_features)
    else:
        count = max(1, ceil(float(budget) * len(ordered_features)))
    return list(ordered_features[:count])


def _candidate_key(
    method: str,
    classifier: str,
    rho: float | None,
    budget: float,
    parameter_id: int,
    seed_rule: str,
) -> str:
    rho_text = "none" if rho is None else f"{rho:g}"
    return (
        f"{method}|{classifier}|rho={rho_text}|budget={budget:g}|"
        f"params={parameter_id}|{seed_rule}"
    )


def _experiment_id(method: str, classifier: str) -> str:
    safe_method = "".join(char if char.isalnum() else "_" for char in method).strip("_")
    return f"{safe_method}__{classifier}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
