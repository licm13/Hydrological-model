"""Calibration and validation example for the GR4J hydrological model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from gr4j_model import GR4J


def calculate_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Nash-Sutcliffe Efficiency between observed and simulated data."""
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)

    if observed.size == 0:
        raise ValueError("Observed series is empty.")

    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        return -np.inf

    numerator = np.sum((observed - simulated) ** 2)
    return 1.0 - numerator / denominator


def objective_function(params: np.ndarray, P: np.ndarray, E: np.ndarray, Q_obs: np.ndarray) -> float:
    """Objective function for GR4J calibration using NSE as the performance metric."""
    X1, X2, X3, X4 = params

    if X1 <= 0 or X3 <= 0 or X4 <= 0:
        return 1e6

    model = GR4J(X1=X1, X2=X2, X3=X3, X4=X4)
    simulation = model.run(P, E)
    Q_sim = simulation['Q']

    nse = calculate_nse(Q_obs, Q_sim)
    return 1.0 - nse


def load_dataset(path: str) -> pd.DataFrame:
    """Load sample hydrological dataset for calibration and validation."""
    data = pd.read_csv(path, parse_dates=['Date'])
    expected_columns = {'Date', 'Precipitation', 'Evapotranspiration', 'Observed_Flow'}
    missing_columns = expected_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return data.sort_values('Date').reset_index(drop=True)


if __name__ == "__main__":
    dataset = load_dataset('data/sample_data.csv')

    P = dataset['Precipitation'].to_numpy(dtype=float)
    E = dataset['Evapotranspiration'].to_numpy(dtype=float)
    Q_obs = dataset['Observed_Flow'].to_numpy(dtype=float)

    split_index = int(len(dataset) * 0.7)
    calibration_slice = slice(0, split_index)
    validation_slice = slice(split_index, None)

    P_cal, E_cal, Q_cal = P[calibration_slice], E[calibration_slice], Q_obs[calibration_slice]
    P_val, E_val, Q_val = P[validation_slice], E[validation_slice], Q_obs[validation_slice]

    initial_params = np.array([300.0, 0.0, 80.0, 1.5], dtype=float)
    bounds = [(100.0, 1200.0), (-5.0, 5.0), (20.0, 400.0), (1.0, 4.0)]

    print("Starting calibration using Nash-Sutcliffe Efficiency (NSE)...")
    result = minimize(
        objective_function,
        x0=initial_params,
        args=(P_cal, E_cal, Q_cal),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200}
    )

    if not result.success:
        raise RuntimeError(f"Calibration did not converge: {result.message}")

    optimal_params = result.x
    calibrated_model = GR4J(*optimal_params)

    Q_cal_sim = calibrated_model.run(P_cal, E_cal)['Q']
    Q_val_sim = calibrated_model.run(P_val, E_val)['Q']

    nse_cal = calculate_nse(Q_cal, Q_cal_sim)
    nse_val = calculate_nse(Q_val, Q_val_sim)

    print("Calibration completed.")
    print(f"Optimal parameters: X1={optimal_params[0]:.2f}, X2={optimal_params[1]:.2f}, "
          f"X3={optimal_params[2]:.2f}, X4={optimal_params[3]:.2f}")
    print(f"Calibration NSE: {nse_cal:.3f}")
    print(f"Validation NSE: {nse_val:.3f}")

    days = np.arange(len(dataset))
    Q_sim_full = np.concatenate([Q_cal_sim, Q_val_sim])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 10))

    ax1.bar(days, P, color='steelblue')
    ax1.set_ylabel('Precipitation (mm/day)')
    ax1.invert_yaxis()
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2.plot(days, Q_obs, label='Observed Flow', color='black', linestyle='--')
    ax2.plot(days, Q_sim_full, label='GR4J Simulated Flow', color='firebrick')
    ax2.axvline(split_index, color='gray', linestyle=':', label='Calibration/Validation Split')
    ax2.set_ylabel('Discharge (mm/day)')
    ax2.set_xlabel('Day')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend()

    fig.suptitle('GR4J Calibration and Validation using Sample Dataset')
    plt.tight_layout()
    plt.show()
