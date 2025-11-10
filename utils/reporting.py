"""
Reporting Module

This module provides functions for computing water balance, formatting model
summaries, and printing tabular output for hydrological model results.
"""

import numpy as np
import logging
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta

from .performance import (
    nash_sutcliffe_efficiency,
    root_mean_squared_error,
    percent_bias
)

# Get module logger
logger = logging.getLogger(__name__)


def compute_water_balance(
    precipitation: np.ndarray,
    potential_evap: np.ndarray,
    model_results: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute water balance components from model results.

    This function calculates the water balance including total inputs, outputs,
    storage changes, and balance error for a hydrological model simulation.

    Parameters
    ----------
    precipitation : np.ndarray
        Array of precipitation values (mm)
    potential_evap : np.ndarray
        Array of potential evapotranspiration values (mm)
    model_results : Dict[str, np.ndarray]
        Dictionary containing model output arrays. Must include 'Q' (discharge).
        May optionally include 'ET' (actual evapotranspiration) and storage components.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'total_input': Total precipitation (mm)
        - 'total_output': Total discharge + actual ET (mm)
        - 'storage_change': Estimated change in storage (mm)
        - 'balance_error': Water balance error (mm)

    Examples
    --------
    >>> P = np.array([10.0, 5.0, 0.0, 15.0])
    >>> ET = np.array([3.0, 3.0, 3.0, 3.0])
    >>> results = {'Q': np.array([2.0, 3.0, 1.0, 5.0]), 'ET': np.array([3.0, 2.5, 2.8, 2.9])}
    >>> balance = compute_water_balance(P, ET, results)
    """
    # Guard against empty arrays or NaN values
    if precipitation.size == 0 or potential_evap.size == 0:
        return {
            'total_input': np.nan,
            'total_output': np.nan,
            'storage_change': np.nan,
            'balance_error': np.nan
        }

    # Total input (precipitation)
    total_input = float(np.sum(precipitation))

    # Total output (discharge + actual ET if available)
    total_discharge = float(np.sum(model_results.get('Q', np.array([0.0]))))

    # Use actual ET if available, otherwise use potential ET as approximation
    if 'ET' in model_results:
        total_et = float(np.sum(model_results['ET']))
    else:
        total_et = float(np.sum(potential_evap))

    total_output = total_discharge + total_et

    # Storage change (input - output)
    storage_change = total_input - total_output

    # Balance error: For simple lumped models without explicit storage tracking,
    # the balance error equals the storage change. Models with detailed storage
    # components could extend this calculation.
    balance_error = storage_change

    return {
        'total_input': total_input,
        'total_output': total_output,
        'storage_change': storage_change,
        'balance_error': balance_error
    }


def format_model_summary(
    name: str,
    model_results: Dict[str, np.ndarray],
    precipitation: np.ndarray,
    potential_evap: np.ndarray,
    observed_flow: Optional[np.ndarray] = None
) -> str:
    """
    Format a comprehensive summary for a hydrological model.

    Creates a formatted string containing key performance metrics and statistics
    for a model simulation.

    Parameters
    ----------
    name : str
        Name of the model
    model_results : Dict[str, np.ndarray]
        Dictionary containing model outputs (must include 'Q' for discharge)
    precipitation : np.ndarray
        Array of precipitation values (mm)
    potential_evap : np.ndarray
        Array of potential evapotranspiration values (mm)
    observed_flow : Optional[np.ndarray]
        Array of observed flow values for performance metric calculation.
        If None, performance metrics (NSE, RMSE, PBIAS) will not be computed.

    Returns
    -------
    str
        Formatted summary string containing model statistics

    Examples
    --------
    >>> P = np.array([10.0, 5.0, 0.0, 15.0])
    >>> ET = np.array([3.0, 3.0, 3.0, 3.0])
    >>> results = {'Q': np.array([2.0, 3.0, 1.0, 5.0])}
    >>> summary = format_model_summary("TestModel", results, P, ET)
    >>> print(summary)
    """
    Q = model_results.get('Q', np.array([]))

    if Q.size == 0:
        return f"{name:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<12} {'N/A':<12} {'N/A':<12}"

    # Calculate basic statistics
    total_Q = float(np.nansum(Q))
    total_P = float(np.sum(precipitation))
    runoff_coef = total_Q / total_P if total_P > 0 else np.nan
    if np.all(np.isnan(Q)):
        peak_Q = float('nan')
        mean_Q = float('nan')
    else:
        peak_Q = float(np.nanmax(Q))
        mean_Q = float(np.nanmean(Q))

    # Calculate performance metrics if observed flow is provided
    if observed_flow is not None and observed_flow.size > 0:
        mask = ~np.isnan(observed_flow) & ~np.isnan(Q)
        if np.any(mask):
            nse = nash_sutcliffe_efficiency(observed_flow[mask], Q[mask])
            rmse = root_mean_squared_error(observed_flow[mask], Q[mask])
            pbias = percent_bias(observed_flow[mask], Q[mask])
        else:
            nse = np.nan
            rmse = np.nan
            pbias = np.nan
    else:
        nse = np.nan
        rmse = np.nan
        pbias = np.nan

    # Format the summary line
    summary = "{:<20} {:<15.2f} {:<15.3f} {:<15.2f} {:<15.2f} {:<12.3f} {:<12.2f} {:<12.2f}".format(
        name, total_Q, runoff_coef, peak_Q, mean_Q, nse, rmse, pbias
    )

    return summary


def print_first_n_days_table(
    start_date: datetime,
    precipitation: np.ndarray,
    potential_evap: np.ndarray,
    model_results: Dict[str, np.ndarray],
    n: int = 20,
    date_formatter: Optional[Callable[[datetime], str]] = None
) -> str:
    """
    Print a formatted table of the first N days of model results.

    Creates a tabular view of daily hydrological variables including precipitation,
    potential ET, and discharge for the first N days of simulation.

    Parameters
    ----------
    start_date : datetime
        Starting date of the simulation
    precipitation : np.ndarray
        Array of precipitation values (mm)
    potential_evap : np.ndarray
        Array of potential evapotranspiration values (mm)
    model_results : Dict[str, np.ndarray]
        Dictionary containing model outputs (must include 'Q' for discharge)
    n : int, optional
        Number of days to display (default: 20)
    date_formatter : Optional[Callable[[datetime], str]], optional
        Custom date formatting function. If None, uses ISO format (YYYY-MM-DD).

    Returns
    -------
    str
        Formatted table string

    Examples
    --------
    >>> from datetime import datetime
    >>> start = datetime(2020, 1, 1)
    >>> P = np.array([10.0, 5.0, 0.0, 15.0, 8.0])
    >>> ET = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    >>> results = {'Q': np.array([2.0, 3.0, 1.0, 5.0, 4.0])}
    >>> table = print_first_n_days_table(start, P, ET, results, n=5)
    >>> print(table)
    """
    # Default date formatter if none provided
    if date_formatter is None:
        def default_date_formatter(d):
            return d.strftime('%Y-%m-%d')
        date_formatter = default_date_formatter

    # Extract discharge from model results
    Q = model_results.get('Q', np.array([]))

    # Determine actual number of days to display
    n_days = min(n, len(precipitation), len(potential_evap), len(Q))

    if n_days == 0:
        return "No data available to display."

    # Build the table header
    lines = []
    lines.append("\nFirst {} days of simulation:".format(n_days))
    lines.append("{:<12} {:<12} {:<12} {:<12}".format(
        "Date", "P (mm)", "ET (mm)", "Q (mm)"
    ))
    lines.append("-" * 50)

    # Build table rows
    for i in range(n_days):
        current_date = start_date + timedelta(days=i)
        date_str = date_formatter(current_date)
        p_val = precipitation[i] if i < len(precipitation) else np.nan
        et_val = potential_evap[i] if i < len(potential_evap) else np.nan
        q_val = Q[i] if i < len(Q) else np.nan

        lines.append("{:<12} {:<12.2f} {:<12.2f} {:<12.2f}".format(
            date_str, p_val, et_val, q_val
        ))

    # Join all lines into a single string
    table_str = "\n".join(lines)

    # Log the table using the module logger
    logger.info("\n" + table_str)

    return table_str
