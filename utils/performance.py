"""
Performance Metrics Module

This module provides functions to compute common hydrological model
performance metrics including NSE, RMSE, PBIAS, and basic runoff statistics.
"""

import numpy as np


def nash_sutcliffe_efficiency(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Compute Nash-Sutcliffe Efficiency (NSE) between observed and simulated flows.

    The Nash-Sutcliffe efficiency coefficient (NSE) is a normalized statistic
    that determines the relative magnitude of the residual variance compared
    to the measured data variance. NSE ranges from -∞ to 1, where 1 indicates
    perfect agreement.

    Parameters
    ----------
    observed : np.ndarray
        Array of observed flow values
    simulated : np.ndarray
        Array of simulated flow values

    Returns
    -------
    float
        Nash-Sutcliffe Efficiency value. Returns np.nan if computation is invalid
        (e.g., empty arrays or zero denominator).

    Examples
    --------
    >>> obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> sim = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
    >>> nse = nash_sutcliffe_efficiency(obs, sim)
    >>> print(f"NSE: {nse:.3f}")
    """
    # Guard against empty arrays
    if observed.size == 0 or simulated.size == 0:
        return np.nan

    # Guard against NaN values
    if np.any(np.isnan(observed)) or np.any(np.isnan(simulated)):
        return np.nan

    # Compute denominator (variance of observations)
    denominator = np.sum((observed - np.mean(observed)) ** 2)

    # Guard against zero denominator
    if denominator == 0:
        return np.nan

    # Compute NSE
    numerator = np.sum((observed - simulated) ** 2)
    nse = 1.0 - (numerator / denominator)

    return float(nse)


def root_mean_squared_error(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE) between observed and simulated flows.

    RMSE measures the average magnitude of the error between observed and
    simulated values. Lower values indicate better model performance.

    Parameters
    ----------
    observed : np.ndarray
        Array of observed flow values
    simulated : np.ndarray
        Array of simulated flow values

    Returns
    -------
    float
        Root Mean Squared Error value. Returns np.nan if arrays contain NaN values.

    Examples
    --------
    >>> obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> sim = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
    >>> rmse = root_mean_squared_error(obs, sim)
    >>> print(f"RMSE: {rmse:.3f}")
    """
    # Guard against empty arrays
    if observed.size == 0 or simulated.size == 0:
        return np.nan

    # Guard against NaN values
    if np.any(np.isnan(observed)) or np.any(np.isnan(simulated)):
        return np.nan

    # Compute RMSE
    mse = np.mean((observed - simulated) ** 2)
    rmse = np.sqrt(mse)

    return float(rmse)


def percent_bias(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Compute Percent Bias (PBIAS) between observed and simulated flows.

    PBIAS measures the average tendency of simulated values to be larger or
    smaller than observed values. Optimal value is 0.0, with positive values
    indicating model underestimation and negative values indicating overestimation.

    Parameters
    ----------
    observed : np.ndarray
        Array of observed flow values
    simulated : np.ndarray
        Array of simulated flow values

    Returns
    -------
    float
        Percent Bias value. Returns np.nan if sum of observed is zero or
        arrays contain NaN values.

    Examples
    --------
    >>> obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> sim = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
    >>> pbias = percent_bias(obs, sim)
    >>> print(f"PBIAS: {pbias:.2f}%")
    """
    # Guard against empty arrays
    if observed.size == 0 or simulated.size == 0:
        return np.nan

    # Guard against NaN values
    if np.any(np.isnan(observed)) or np.any(np.isnan(simulated)):
        return np.nan

    # Guard against zero denominator
    obs_sum = np.sum(observed)
    if obs_sum == 0:
        return np.nan

    # Compute PBIAS
    pbias = 100.0 * np.sum(simulated - observed) / obs_sum

    return float(pbias)


def basic_runoff_statistics(simulated_flow: np.ndarray) -> dict:
    """
    Compute basic runoff statistics from simulated flow data.

    This function calculates common statistical metrics for a simulated
    flow time series including total volume, mean, peak, and minimum values.

    Parameters
    ----------
    simulated_flow : np.ndarray
        Array of simulated flow values

    Returns
    -------
    dict
        Dictionary containing:
        - 'total': Total volume (sum of all values)
        - 'mean': Mean flow value
        - 'peak': Maximum flow value
        - 'min': Minimum flow value
        - 'std': Standard deviation of flow values

    Examples
    --------
    >>> flow = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> stats = basic_runoff_statistics(flow)
    >>> print(f"Total: {stats['total']:.2f}, Mean: {stats['mean']:.2f}")
    """
    # Guard against empty arrays
    if simulated_flow.size == 0:
        return {
            'total': np.nan,
            'mean': np.nan,
            'peak': np.nan,
            'min': np.nan,
            'std': np.nan
        }

    # Guard against NaN values
    if np.any(np.isnan(simulated_flow)):
        return {
            'total': np.nan,
            'mean': np.nan,
            'peak': np.nan,
            'min': np.nan,
            'std': np.nan
        }

    return {
        'total': float(np.sum(simulated_flow)),
        'mean': float(np.mean(simulated_flow)),
        'peak': float(np.max(simulated_flow)),
        'min': float(np.min(simulated_flow)),
        'std': float(np.std(simulated_flow))
    }
