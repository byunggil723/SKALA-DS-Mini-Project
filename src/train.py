"""Train, select, and evaluate battery cycle-life regression models.

Feature extraction lives in ``src.features``. This module consumes the
cell-level feature table, performs leakage-safe model selection on Batch 1,
and evaluates the selected model on external batches.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.features import FEATURE_SETS, extract_feature_table
except ModuleNotFoundError:  # Support `python src/train.py` from the project root.
    from features import FEATURE_SETS, extract_feature_table


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
PAPER_TARGET_MAPE = 9.1
# One-based raw-file indices documented by the authors' public loading code
# for Batch 3 data-quality screening.  Both raw labeled and screened scores
# are reported so the optional result is transparent.
BATCH3_PAPER_EXCLUDED_CELL_NUMBERS = {3, 24, 33, 38, 43, 44}


def make_pipeline(model: BaseEstimator) -> Pipeline:
    steps: list[tuple[str, BaseEstimator]] = [("imputer", SimpleImputer(strategy="median"))]
    if isinstance(model, (LinearRegression, Ridge, ElasticNet)):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def model_specs() -> dict[str, tuple[Pipeline, dict[str, list[Any]], int]]:
    """Return estimator, compact tuning grid, and simplicity rank."""
    return {
        "Dummy": (make_pipeline(DummyRegressor(strategy="median")), {}, 0),
        "Linear": (make_pipeline(LinearRegression()), {}, 1),
        "Ridge": (make_pipeline(Ridge()), {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}, 2),
        "ElasticNet": (
            make_pipeline(ElasticNet(max_iter=100000, random_state=RANDOM_STATE)),
            {"model__alpha": [0.001, 0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]},
            3,
        ),
        "HistGradientBoosting": (
            make_pipeline(HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            {
                "model__learning_rate": [0.03, 0.1],
                "model__max_leaf_nodes": [5, 10],
                "model__min_samples_leaf": [5, 10],
                "model__l2_regularization": [0.0, 1.0],
            },
            4,
        ),
        "RandomForest": (
            make_pipeline(RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE)),
            {
                "model__max_depth": [2, 4, None],
                "model__min_samples_leaf": [2, 4],
                "model__max_features": [0.7, 1.0],
            },
            5,
        ),
    }


def _wrap(estimator: Pipeline, transform: str) -> BaseEstimator:
    if transform == "log":
        return TransformedTargetRegressor(regressor=estimator, func=np.log, inverse_func=np.exp, check_inverse=False)
    return estimator


def _prefix_grid(grid: dict[str, list[Any]], transform: str) -> dict[str, list[Any]]:
    return {f"regressor__{key}": value for key, value in grid.items()} if transform == "log" else grid


def _group_kfold(groups: pd.Series, desired: int = 4) -> GroupKFold:
    return GroupKFold(n_splits=max(2, min(desired, int(groups.nunique()))))


def _nested_cv(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    estimator: BaseEstimator,
    grid: dict[str, list[Any]],
    outer_splits: int = 4,
) -> tuple[float, float, list[float]]:
    fold_scores: list[float] = []
    outer = _group_kfold(groups, outer_splits)
    for train_index, valid_index in outer.split(x, y, groups):
        inner_groups = groups.iloc[train_index]
        search = GridSearchCV(
            clone(estimator), grid or [{}], scoring="neg_mean_absolute_percentage_error",
            cv=_group_kfold(inner_groups, 3), n_jobs=-1, refit=True,
        )
        search.fit(x.iloc[train_index], y.iloc[train_index], groups=inner_groups)
        prediction = search.predict(x.iloc[valid_index])
        fold_scores.append(100 * mean_absolute_percentage_error(y.iloc[valid_index], prediction))
    return float(np.mean(fold_scores)), float(np.std(fold_scores, ddof=1)), fold_scores


def _fit_tuned(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    estimator: BaseEstimator,
    grid: dict[str, list[Any]],
) -> GridSearchCV:
    search = GridSearchCV(
        clone(estimator), grid or [{}], scoring="neg_mean_absolute_percentage_error",
        cv=_group_kfold(groups, 4), n_jobs=-1, refit=True,
    )
    search.fit(x, y, groups=groups)
    return search


def _metrics(y: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    return {
        "mape_percent": 100 * mean_absolute_percentage_error(y, prediction),
        "mae_cycles": mean_absolute_error(y, prediction),
        "rmse_cycles": mean_squared_error(y, prediction) ** 0.5,
        "r2": r2_score(y, prediction),
    }


@dataclass
class SelectedConfig:
    model: str
    transform: str
    feature_set: str
    features: list[str]
    cv_mape: float
    cv_std: float


def run_modeling(feature_table: pd.DataFrame, results_dir: Path) -> SelectedConfig:
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    supervised = feature_table.loc[feature_table["cycle_life"].notna()].copy()
    b1 = supervised.loc[supervised["batch"].eq("Batch 1")].reset_index(drop=True)
    b2 = supervised.loc[supervised["batch"].eq("Batch 2")].reset_index(drop=True)
    b3 = supervised.loc[supervised["batch"].eq("Batch 3")].reset_index(drop=True)
    if len(b1) != 46 or len(b2) != 39:
        warnings.warn(f"Expected 46 Batch 1 and 39 Batch 2 labeled cells; got {len(b1)} and {len(b2)}")

    holdout_split = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=RANDOM_STATE)
    dev_index, holdout_index = next(holdout_split.split(b1, groups=b1["policy_readable"]))
    dev, holdout = b1.iloc[dev_index].reset_index(drop=True), b1.iloc[holdout_index].reset_index(drop=True)
    assert set(dev["policy_readable"]).isdisjoint(set(holdout["policy_readable"]))
    pd.concat([
        dev[["cell_key", "policy_readable"]].assign(split="Development train"),
        holdout[["cell_key", "policy_readable"]].assign(split="Batch 1 hold-out"),
    ], ignore_index=True).to_csv(tables_dir / "split_manifest.csv", index=False)

    specs = model_specs()
    model_rows: list[dict[str, Any]] = []
    for feature_set in ["base_initial_q", "base_mean_q"]:
        features = FEATURE_SETS[feature_set]
        for model_name, (pipeline, parameter_grid, simplicity) in specs.items():
            transforms = ["raw"] if model_name == "Dummy" else ["raw", "log"]
            for transform in transforms:
                estimator = _wrap(pipeline, transform)
                grid = _prefix_grid(parameter_grid, transform)
                cv_mean, cv_std, fold_scores = _nested_cv(
                    dev[features], dev["cycle_life"], dev["policy_readable"], estimator, grid
                )
                fitted = _fit_tuned(
                    dev[features], dev["cycle_life"], dev["policy_readable"], estimator, grid
                )
                valid_mape = _metrics(holdout["cycle_life"], fitted.predict(holdout[features]))["mape_percent"]
                model_rows.append({
                    "model": model_name,
                    "target_transform": transform,
                    "feature_set": feature_set,
                    "cv_mape_percent": cv_mean,
                    "cv_std_percent": cv_std,
                    "holdout_mape_percent": valid_mape,
                    "simplicity_rank": simplicity,
                    "fold_mapes": json.dumps([round(x, 6) for x in fold_scores]),
                    "best_params": json.dumps(fitted.best_params_, ensure_ascii=False),
                })
    comparison = pd.DataFrame(model_rows).sort_values(["cv_mape_percent", "cv_std_percent", "simplicity_rank"])
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)

    # The assignment asks for the optimal model.  Select strictly on Batch 1
    # nested-CV mean MAPE; use fold dispersion and simplicity only as ties.
    # The hold-out and external batches remain untouched by this decision.
    family_choice = comparison.sort_values(
        ["cv_mape_percent", "cv_std_percent", "simplicity_rank"]
    ).iloc[0]
    selected_model_name = str(family_choice["model"])
    selected_transform = str(family_choice["target_transform"])

    pipeline, parameter_grid, simplicity = specs[selected_model_name]
    estimator = _wrap(pipeline, selected_transform)
    grid = _prefix_grid(parameter_grid, selected_transform)
    ablation_rows: list[dict[str, Any]] = []
    for feature_set, features in FEATURE_SETS.items():
        cv_mean, cv_std, fold_scores = _nested_cv(
            dev[features], dev["cycle_life"], dev["policy_readable"], estimator, grid
        )
        fitted = _fit_tuned(dev[features], dev["cycle_life"], dev["policy_readable"], estimator, grid)
        valid_mape = _metrics(holdout["cycle_life"], fitted.predict(holdout[features]))["mape_percent"]
        ablation_rows.append({
            "feature_set": feature_set,
            "features": " | ".join(features),
            "cv_mape_percent": cv_mean,
            "cv_std_percent": cv_std,
            "holdout_mape_percent": valid_mape,
            "fold_mapes": json.dumps([round(x, 6) for x in fold_scores]),
            "best_params": json.dumps(fitted.best_params_, ensure_ascii=False),
        })
    ablation = pd.DataFrame(ablation_rows).sort_values(["cv_mape_percent", "cv_std_percent"])
    ablation.to_csv(tables_dir / "feature_ablation.csv", index=False)
    feature_choice = ablation.iloc[0]
    selected_feature_set = str(feature_choice["feature_set"])
    selected_features = FEATURE_SETS[selected_feature_set]

    dev_search = _fit_tuned(
        dev[selected_features], dev["cycle_life"], dev["policy_readable"], estimator, grid
    )
    valid_prediction = dev_search.predict(holdout[selected_features])
    valid_metrics = _metrics(holdout["cycle_life"], valid_prediction)

    final_search = _fit_tuned(
        b1[selected_features], b1["cycle_life"], b1["policy_readable"], estimator, grid
    )
    test_prediction = final_search.predict(b2[selected_features])
    test_metrics = _metrics(b2["cycle_life"], test_prediction)
    batch3_prediction = final_search.predict(b3[selected_features])
    batch3_metrics = _metrics(b3["cycle_life"], batch3_prediction)
    b3_screened_mask = ~b3["cell_index"].add(1).isin(BATCH3_PAPER_EXCLUDED_CELL_NUMBERS)
    batch3_screened_metrics = _metrics(
        b3.loc[b3_screened_mask, "cycle_life"], batch3_prediction[b3_screened_mask.to_numpy()]
    )

    train_cv = float(feature_choice["cv_mape_percent"])
    performance_rows = [
        {"index": "Train (Batch 1 CV)", "mape_percent": train_cv,
         "note": "Development partition; protocol-grouped nested CV mean"},
        {"index": "Valid (Batch 1 Hold-out)", "mape_percent": valid_metrics["mape_percent"],
         "note": "Unseen charging protocols; model fit on development partition"},
        {"index": "Test (Batch 2)", "mape_percent": test_metrics["mape_percent"],
         "note": "External batch; selected model refit on all Batch 1"},
        {"index": "Gap (Train-Valid)", "mape_percent": valid_metrics["mape_percent"] - train_cv,
         "note": "(+) indicates possible overfitting"},
        {"index": "Gap (Valid-Test)", "mape_percent": test_metrics["mape_percent"] - valid_metrics["mape_percent"],
         "note": "(+) indicates batch generalization degradation"},
        {"index": "Gap (Target-Test)", "mape_percent": test_metrics["mape_percent"] - PAPER_TARGET_MAPE,
         "note": "Paper target: 9.1% MAPE"},
    ]
    pd.DataFrame(performance_rows).to_csv(results_dir / "model_performance.csv", index=False)

    supplementary = pd.DataFrame([
        {"set": "Valid (Batch 1 Hold-out)", **valid_metrics},
        {"set": "Test (Batch 2)", **test_metrics},
        {"set": "Optional Test (Batch 3, all labeled cells)", **batch3_metrics},
        {"set": "Optional Test (Batch 3, paper-screened)", **batch3_screened_metrics},
    ])
    supplementary.to_csv(tables_dir / "supplementary_metrics.csv", index=False)

    prediction_tables = []
    for label, frame, prediction in [
        ("Valid (Batch 1 Hold-out)", holdout, valid_prediction),
        ("Test (Batch 2)", b2, test_prediction),
        ("Optional Test (Batch 3)", b3, batch3_prediction),
    ]:
        table = frame[["cell_key", "batch", "policy_readable", "cycle_life"]].copy()
        table["evaluation_set"] = label
        table["prediction"] = prediction
        table["signed_error"] = table["prediction"] - table["cycle_life"]
        table["absolute_percentage_error"] = 100 * table["signed_error"].abs() / table["cycle_life"]
        prediction_tables.append(table)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions.sort_values(["evaluation_set", "absolute_percentage_error"], ascending=[True, False]).to_csv(
        tables_dir / "predictions_and_errors.csv", index=False
    )

    predictions["life_group"] = pd.cut(
        predictions["cycle_life"], [-np.inf, 499, 1000, np.inf],
        labels=["Short", "Middle", "Long"],
    )
    error_summary = (
        predictions.groupby("evaluation_set", observed=True)
        .agg(
            n_cells=("cell_key", "size"),
            mape_percent=("absolute_percentage_error", "mean"),
            median_ape_percent=("absolute_percentage_error", "median"),
            mean_signed_error_cycles=("signed_error", "mean"),
            overprediction_rate_percent=("signed_error", lambda x: 100 * (x > 0).mean()),
        )
        .reset_index()
    )
    error_summary.to_csv(tables_dir / "error_analysis_summary.csv", index=False)
    (
        predictions.loc[predictions["evaluation_set"].eq("Test (Batch 2)")]
        .groupby("life_group", observed=True)
        .agg(
            n_cells=("cell_key", "size"),
            mape_percent=("absolute_percentage_error", "mean"),
            mean_signed_error_cycles=("signed_error", "mean"),
            overprediction_rate_percent=("signed_error", lambda x: 100 * (x > 0).mean()),
        )
        .reset_index()
        .to_csv(tables_dir / "batch2_error_by_life_group.csv", index=False)
    )

    distribution_rows = []
    distribution_features = ["cycle_life", *selected_features]
    for batch_name, frame in [("Batch 1", b1), ("Batch 2", b2), ("Batch 3", b3)]:
        row: dict[str, Any] = {
            "batch": batch_name,
            "n_labeled": len(frame),
            "short_life_percent": 100 * frame["cycle_life"].lt(500).mean(),
        }
        for feature in distribution_features:
            row[f"{feature}_mean"] = float(frame[feature].mean())
            row[f"{feature}_median"] = float(frame[feature].median())
        distribution_rows.append(row)
    pd.DataFrame(distribution_rows).to_csv(tables_dir / "batch_distribution_shift.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE,
        "selected_model": selected_model_name,
        "selected_target_transform": selected_transform,
        "selected_feature_set": selected_feature_set,
        "selected_features": selected_features,
        "selection_rule": "Lowest Batch 1 nested group-CV mean MAPE; CV dispersion and model simplicity are tie-breakers. Feature set is then selected by the same nested group-CV criterion.",
        "holdout_used_for_selection": False,
        "batch2_used_for_selection": False,
        "final_best_params": final_search.best_params_,
        "sample_counts": {"batch1": len(b1), "development": len(dev), "holdout": len(holdout), "batch2": len(b2), "batch3_labeled": len(b3), "batch3_paper_screened": int(b3_screened_mask.sum())},
        "batch3_paper_excluded_one_based_cell_numbers": sorted(BATCH3_PAPER_EXCLUDED_CELL_NUMBERS),
        "protocol_overlap_development_holdout": sorted(set(dev["policy_readable"]) & set(holdout["policy_readable"])),
        "paper_target_mape_percent": PAPER_TARGET_MAPE,
    }
    (tables_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(final_search.best_estimator_, results_dir / "final_model.joblib")
    return SelectedConfig(
        model=selected_model_name,
        transform=selected_transform,
        feature_set=selected_feature_set,
        features=selected_features,
        cv_mape=train_cv,
        cv_std=float(feature_choice["cv_std_percent"]),
    )


def run_project(data_dir: Path, results_dir: Path, rebuild_features: bool = False) -> SelectedConfig:
    cache_path = results_dir / "feature_table.csv"
    if rebuild_features or not cache_path.exists():
        feature_table = extract_feature_table(data_dir, cache_path)
    else:
        feature_table = pd.read_csv(cache_path)
    return run_modeling(feature_table, results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    selected = run_project(args.data_dir, args.results_dir, args.rebuild_features)
    print(json.dumps(selected.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
