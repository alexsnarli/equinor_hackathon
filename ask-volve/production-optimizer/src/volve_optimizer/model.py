from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

from .data import FEATURE_COLUMNS, TARGET_COLUMNS


BASELINE_COLUMNS = [
    "previous_oil_rate",
    "previous_gas_rate",
    "previous_water_rate",
    "previous_whp",
]


@dataclass
class ModelBundle:
    imputer: SimpleImputer
    model: ExtraTreesRegressor
    target_scale: np.ndarray
    residual_weights: np.ndarray
    metrics: pd.DataFrame
    selected_min_samples_leaf: int
    training_rows: int


def chronological_split(frame: pd.DataFrame) -> pd.Series:
    """Assign chronological train/validation/test partitions inside every well."""
    split = pd.Series(index=frame.index, dtype="object")
    for _, group in frame.groupby("well_label"):
        ordered = group.sort_values("DATEPRD")
        count = len(ordered)
        train_end = max(1, int(count * 0.70))
        validation_end = max(train_end + 1, int(count * 0.85))
        validation_end = min(validation_end, count - 1) if count > 2 else count
        split.loc[ordered.index[:train_end]] = "train"
        split.loc[ordered.index[train_end:validation_end]] = "validation"
        split.loc[ordered.index[validation_end:]] = "test"
    return split


def _fit_model(
    frame: pd.DataFrame,
    min_samples_leaf: int,
    n_estimators: int,
    random_state: int,
) -> tuple[SimpleImputer, ExtraTreesRegressor, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    features = imputer.fit_transform(frame[FEATURE_COLUMNS])
    targets = frame[TARGET_COLUMNS].to_numpy(dtype=float)
    baseline = _baseline_values(frame)
    residuals = targets - baseline
    target_scale = np.nanstd(residuals, axis=0)
    target_scale[target_scale == 0] = 1.0
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=0.85,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(features, residuals / target_scale)
    return imputer, model, target_scale


def _baseline_values(features: pd.DataFrame) -> np.ndarray:
    return np.clip(
        features[BASELINE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float),
        0,
        None,
    )


def _predict_components(
    imputer: SimpleImputer,
    model: ExtraTreesRegressor,
    target_scale: np.ndarray,
    features: pd.DataFrame,
) -> np.ndarray:
    transformed = imputer.transform(features[FEATURE_COLUMNS])
    residual = model.predict(transformed) * target_scale
    return np.clip(_baseline_values(features) + residual, 0, None)


def _normalized_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    scales = np.maximum(np.nanmedian(np.abs(actual), axis=0), 1.0)
    return float(np.mean(np.mean(np.abs(actual - predicted), axis=0) / scales))


def train_model(
    frame: pd.DataFrame,
    random_state: int = 42,
    final_estimators: int = 220,
) -> ModelBundle:
    if frame.empty:
        raise ValueError("No active producer rows were available for model training.")

    frame = frame.copy()
    frame["split"] = chronological_split(frame)
    train = frame[frame["split"].eq("train")]
    validation = frame[frame["split"].eq("validation")]
    test = frame[frame["split"].eq("test")]

    best_leaf = 5
    best_score = float("inf")
    for leaf in (2, 5, 10):
        imputer, model, scale = _fit_model(train, leaf, 80, random_state)
        predicted = _predict_components(imputer, model, scale, validation)
        score = _normalized_mae(validation[TARGET_COLUMNS].to_numpy(), predicted)
        if score < best_score:
            best_score = score
            best_leaf = leaf

    development = frame[frame["split"].isin(["train", "validation"])]
    tuning_imputer, tuning_model, tuning_scale = _fit_model(
        train, best_leaf, final_estimators, random_state
    )
    validation_raw = _predict_components(
        tuning_imputer, tuning_model, tuning_scale, validation
    )
    validation_actual = validation[TARGET_COLUMNS].to_numpy(dtype=float)
    validation_baseline = _baseline_values(validation)
    validation_residual = validation_raw - validation_baseline
    residual_weights = []
    for target_index in range(len(TARGET_COLUMNS)):
        candidates = np.array([0.0, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0])
        errors = [
            mean_absolute_error(
                validation_actual[:, target_index],
                validation_baseline[:, target_index]
                + weight * validation_residual[:, target_index],
            )
            for weight in candidates
        ]
        # Preserve a small, conservative response for scenario exploration even when
        # persistence wins the observational holdout. The UI labels this as an estimate.
        residual_weights.append(max(0.05, float(candidates[int(np.argmin(errors))])))
    residual_weights_array = np.array(residual_weights)

    eval_imputer, eval_model, eval_scale = _fit_model(
        development, best_leaf, final_estimators, random_state
    )
    test_raw = _predict_components(eval_imputer, eval_model, eval_scale, test)
    test_baseline = _baseline_values(test)
    test_predictions = np.clip(
        test_baseline + residual_weights_array * (test_raw - test_baseline), 0, None
    )

    metric_rows: list[dict[str, Any]] = []
    for index, target in enumerate(TARGET_COLUMNS):
        actual = test[target].to_numpy(dtype=float)
        predicted = test_predictions[:, index]
        baseline = _baseline_values(test)[:, index]
        metric_rows.append(
            {
                "target": target.removeprefix("target_"),
                "test_rows": len(test),
                "model_mae": mean_absolute_error(actual, predicted),
                "persistence_mae": mean_absolute_error(actual, baseline),
                "mae_improvement_pct": 100
                * (mean_absolute_error(actual, baseline) - mean_absolute_error(actual, predicted))
                / max(mean_absolute_error(actual, baseline), 1e-9),
                "r2": r2_score(actual, predicted),
            }
        )

    final_imputer, final_model, final_scale = _fit_model(
        frame, best_leaf, final_estimators, random_state
    )
    return ModelBundle(
        imputer=final_imputer,
        model=final_model,
        target_scale=final_scale,
        residual_weights=residual_weights_array,
        metrics=pd.DataFrame(metric_rows),
        selected_min_samples_leaf=best_leaf,
        training_rows=len(frame),
    )


def predict(bundle: ModelBundle, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    transformed = bundle.imputer.transform(features[FEATURE_COLUMNS])
    tree_predictions = np.stack(
        [tree.predict(transformed) for tree in bundle.model.estimators_], axis=0
    )
    residual_mean = tree_predictions.mean(axis=0) * bundle.target_scale
    mean = np.clip(
        _baseline_values(features) + bundle.residual_weights * residual_mean, 0, None
    )
    standard_deviation = (
        tree_predictions.std(axis=0) * bundle.target_scale * bundle.residual_weights
    )
    return mean, standard_deviation
