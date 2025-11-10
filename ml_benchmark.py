"""Data-driven benchmark models for HydroLearn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from utils.performance import (
    nash_sutcliffe_efficiency,
    percent_bias,
    root_mean_squared_error,
)


@dataclass
class RandomForestBenchmarkResult:
    """Container holding the fitted model, predictions, and diagnostics."""

    model: RandomForestRegressor
    full_series: np.ndarray
    metrics: Dict[str, float]
    feature_names: Tuple[str, ...]


def _build_supervised_matrix(
    precipitation: np.ndarray,
    discharge: np.ndarray,
    n_lags: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Create the feature matrix used by the benchmark regressor."""

    precipitation = np.asarray(precipitation, dtype=float)
    discharge = np.asarray(discharge, dtype=float)

    if precipitation.size != discharge.size:
        raise ValueError("Precipitation and discharge arrays must share the same length.")

    if precipitation.size <= n_lags:
        raise ValueError("Time series is too short for the requested number of lags.")

    rows = []
    targets = []
    precip_features = ["P_t" if lag == 0 else f"P_t-{lag}" for lag in range(n_lags + 1)]
    flow_features = [f"Q_t-{lag}" for lag in range(1, n_lags + 1)]
    feature_names = tuple(precip_features + flow_features)

    for idx in range(n_lags, precipitation.size):
        row = [precipitation[idx - lag] for lag in range(0, n_lags + 1)]
        row.extend(discharge[idx - lag] for lag in range(1, n_lags + 1))
        rows.append(row)
        targets.append(discharge[idx])

    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), feature_names


def run_random_forest_benchmark(
    precipitation: np.ndarray,
    potential_evap: np.ndarray,
    observed_discharge: np.ndarray,
    n_lags: int = 2,
    test_fraction: float = 0.3,
    random_state: int | None = 42,
) -> RandomForestBenchmarkResult:
    """Train a RandomForest baseline and compute hydrological skill metrics."""

    del potential_evap  # retained for API symmetry with conceptual models

    features, targets, feature_names = _build_supervised_matrix(
        precipitation, observed_discharge, n_lags
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=test_fraction,
        shuffle=False,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "NSE": nash_sutcliffe_efficiency(y_test, predictions),
        "RMSE": root_mean_squared_error(y_test, predictions),
        "PBIAS": percent_bias(y_test, predictions),
    }

    # Construct full-length prediction series for plotting/summary usage.
    aligned_predictions = np.full(observed_discharge.shape, np.nan, dtype=float)
    aligned_predictions[n_lags:] = model.predict(features)

    return RandomForestBenchmarkResult(
        model=model,
        full_series=aligned_predictions,
        metrics=metrics,
        feature_names=feature_names,
    )


__all__ = ["RandomForestBenchmarkResult", "run_random_forest_benchmark"]
