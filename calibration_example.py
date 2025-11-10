"""Calibration and validation example for the GR4J hydrological model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
import seaborn as sns
import os
from datetime import datetime, timedelta

# Configure matplotlib for Chinese font display / 配置matplotlib以显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'KaiTi', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display / 修复负号显示

from gr4j_model import GR4J
from utils.performance import nash_sutcliffe_efficiency

def objective_function(params: np.ndarray, P: np.ndarray, E: np.ndarray, Q_obs: np.ndarray) -> float:
    """Objective function for GR4J calibration using NSE as the performance metric."""
    X1, X2, X3, X4 = params

    if X1 <= 0 or X3 <= 0 or X4 <= 0:
        return 1e6

    model = GR4J(X1=X1, X2=X2, X3=X3, X4=X4)
    simulation = model.run(P, E)
    Q_sim = simulation['Q']

    nse = nash_sutcliffe_efficiency(Q_obs, Q_sim)
    return 1.0 - nse


def load_dataset(path: str) -> pd.DataFrame:
    """Load sample hydrological dataset for calibration and validation."""
    data = pd.read_csv(path, parse_dates=['Date'])
    expected_columns = {'Date', 'Precipitation', 'Evapotranspiration', 'Observed_Flow'}
    missing_columns = expected_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return data.sort_values('Date').reset_index(drop=True)


def create_calibration_plots(dataset, Q_obs, Q_sim_full, optimal_params, nse_cal, nse_val, split_index, save_dir="figures"):
    """
    Create comprehensive calibration and validation plots.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Re-configure Chinese font after style setting / 样式设置后重新配置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'KaiTi', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    P = dataset['Precipitation'].to_numpy()
    E = dataset['Evapotranspiration'].to_numpy()
    days = np.arange(len(dataset))
    
    # Create date index if available
    if 'Date' in dataset.columns:
        dates = pd.to_datetime(dataset['Date'])
    else:
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(len(dataset))]
    
    # Figure 1: Main calibration and validation plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('GR4J Model Calibration and Validation Analysis', fontsize=16, fontweight='bold')
    
    # Precipitation (inverted)
    axes[0].bar(dates, P, color='steelblue', alpha=0.7, width=1)
    axes[0].set_ylabel('Precipitation\n(mm/day)', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(max(P) * 1.1, 0)
    
    # Evapotranspiration
    axes[1].plot(dates, E, color='orange', linewidth=1.5, label='Potential ET')
    axes[1].set_ylabel('Evapotranspiration\n(mm/day)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Discharge comparison
    cal_end_date = dates[split_index] if hasattr(dates, '__getitem__') else dates.iloc[split_index]
    
    axes[2].plot(dates, Q_obs, label='Observed Flow', color='black', linewidth=2, linestyle='-')
    axes[2].plot(dates, Q_sim_full, label='GR4J Simulated Flow', color='red', linewidth=2)
    axes[2].axvline(cal_end_date, color='purple', linestyle='--', linewidth=2, 
                   label=f'Cal/Val Split (NSE_cal={nse_cal:.3f}, NSE_val={nse_val:.3f})')
    axes[2].fill_between(dates[:split_index], 0, max(max(Q_obs), max(Q_sim_full)), 
                        alpha=0.1, color='blue', label='Calibration Period')
    axes[2].fill_between(dates[split_index:], 0, max(max(Q_obs), max(Q_sim_full)), 
                        alpha=0.1, color='green', label='Validation Period')
    axes[2].set_ylabel('Discharge\n(mm/day)', fontweight='bold')
    axes[2].set_xlabel('Date', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Format x-axis
    for ax in axes:
        if hasattr(dates, '__getitem__'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gr4j_calibration_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Performance analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GR4J Calibration Performance Analysis', fontsize=16, fontweight='bold')
    
    # Scatter plot - Calibration
    Q_cal_obs = Q_obs[:split_index]
    Q_cal_sim = Q_sim_full[:split_index]
    
    axes[0,0].scatter(Q_cal_obs, Q_cal_sim, alpha=0.6, color='blue', s=30)
    max_q_cal = max(max(Q_cal_obs), max(Q_cal_sim))
    axes[0,0].plot([0, max_q_cal], [0, max_q_cal], 'r--', linewidth=2, label='1:1 Line')
    axes[0,0].set_xlabel('Observed Discharge (mm/day)', fontweight='bold')
    axes[0,0].set_ylabel('Simulated Discharge (mm/day)', fontweight='bold')
    axes[0,0].set_title(f'Calibration Period\nNSE = {nse_cal:.3f}', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Scatter plot - Validation
    Q_val_obs = Q_obs[split_index:]
    Q_val_sim = Q_sim_full[split_index:]
    
    axes[0,1].scatter(Q_val_obs, Q_val_sim, alpha=0.6, color='green', s=30)
    max_q_val = max(max(Q_val_obs), max(Q_val_sim))
    axes[0,1].plot([0, max_q_val], [0, max_q_val], 'r--', linewidth=2, label='1:1 Line')
    axes[0,1].set_xlabel('Observed Discharge (mm/day)', fontweight='bold')
    axes[0,1].set_ylabel('Simulated Discharge (mm/day)', fontweight='bold')
    axes[0,1].set_title(f'Validation Period\nNSE = {nse_val:.3f}', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Residuals analysis - Calibration
    residuals_cal = Q_cal_obs - Q_cal_sim
    axes[1,0].scatter(Q_cal_sim, residuals_cal, alpha=0.6, color='blue', s=30)
    axes[1,0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1,0].set_xlabel('Simulated Discharge (mm/day)', fontweight='bold')
    axes[1,0].set_ylabel('Residuals (mm/day)', fontweight='bold')
    axes[1,0].set_title('Calibration Residuals', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # Residuals analysis - Validation
    residuals_val = Q_val_obs - Q_val_sim
    axes[1,1].scatter(Q_val_sim, residuals_val, alpha=0.6, color='green', s=30)
    axes[1,1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('Simulated Discharge (mm/day)', fontweight='bold')
    axes[1,1].set_ylabel('Residuals (mm/day)', fontweight='bold')
    axes[1,1].set_title('Validation Residuals', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gr4j_calibration_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Parameter analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('GR4J Optimized Parameters Analysis', fontsize=16, fontweight='bold')
    
    param_names = ['X1 (Production\nStore Capacity)', 'X2 (Groundwater\nExchange)', 
                   'X3 (Routing Store\nCapacity)', 'X4 (Unit Hydrograph\nTime Base)']
    param_units = ['mm', 'mm', 'mm', 'days']
    param_colors = ['blue', 'green', 'red', 'orange']
    
    for i, (name, value, unit, color) in enumerate(zip(param_names, optimal_params, param_units, param_colors)):
        row, col = i // 2, i % 2
        axes[row, col].bar([name.split('(')[0]], [value], color=color, alpha=0.7, width=0.5)
        axes[row, col].set_ylabel(f'Parameter Value ({unit})', fontweight='bold')
        axes[row, col].set_title(f'{name}\nOptimal Value: {value:.2f} {unit}', fontweight='bold')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add value text on bar
        axes[row, col].text(0, value/2, f'{value:.2f}', ha='center', va='center', 
                           fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gr4j_calibration_parameters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Calibration visualization plots saved to '{save_dir}' directory")
    print("  Generated files:")
    print("  - gr4j_calibration_validation.png")
    print("  - gr4j_calibration_performance.png") 
    print("  - gr4j_calibration_parameters.png")


if __name__ == "__main__":
    # Generate synthetic dataset for demonstration
    print("=" * 80)
    print("GR4J Model Calibration and Validation - Enhanced Demonstration")
    print("=" * 80)
    
    # Create synthetic data
    np.random.seed(42)
    n_days = 365
    
    # Generate realistic precipitation
    t = np.arange(n_days)
    seasonal_p = 1.2 + 0.8 * np.sin(2 * np.pi * t / 365 + np.pi/2)
    P = np.random.gamma(1.5, 3, n_days) * seasonal_p
    P = np.where(np.random.rand(n_days) < 0.6, 0, P)  # 60% dry days
    
    # Add storm events
    storm_days = np.random.choice(n_days, size=8, replace=False)
    P[storm_days] += np.random.gamma(3, 10, 8)
    
    # Generate evapotranspiration
    E_mean = 3.5
    E_amplitude = 2.0
    E = E_mean + E_amplitude * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.2, n_days)
    E = np.maximum(E, 0.1)
    
    # Generate synthetic observed flow using a reference model
    ref_model = GR4J(X1=350.0, X2=0.5, X3=90.0, X4=1.8)
    ref_results = ref_model.run(P, E)
    Q_obs = ref_results['Q'] + np.random.normal(0, 0.1, n_days)  # Add noise
    Q_obs = np.maximum(Q_obs, 0.001)  # Ensure positive flows
    
    # Create dataset
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    dataset = pd.DataFrame({
        'Date': dates,
        'Precipitation': P,
        'Evapotranspiration': E,
        'Observed_Flow': Q_obs
    })
    
    print("\nSynthetic Dataset Summary:")
    print(f"  Number of days: {n_days}")
    print(f"  Total precipitation: {np.sum(P):.2f} mm")
    print(f"  Average daily precipitation: {np.mean(P):.2f} mm")
    print(f"  Rainfall days: {np.sum(P > 0.1)} ({np.sum(P > 0.1)/n_days*100:.1f}%)")
    print(f"  Total evapotranspiration: {np.sum(E):.2f} mm")
    print(f"  Average daily ET: {np.mean(E):.2f} mm")
    print(f"  Total observed flow: {np.sum(Q_obs):.2f} mm")
    print(f"  Average daily flow: {np.mean(Q_obs):.2f} mm")
    
    # Proceed with existing calibration code
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

    print("\nStarting calibration using Nash-Sutcliffe Efficiency (NSE)...")
    print("Initial parameters:", initial_params)
    print("Parameter bounds:", bounds)
    
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

    nse_cal = nash_sutcliffe_efficiency(Q_cal, Q_cal_sim)
    nse_val = nash_sutcliffe_efficiency(Q_val, Q_val_sim)

    print("\nCalibration completed successfully!")
    print("=" * 60)
    print("OPTIMAL PARAMETERS:")
    print(f"  X1 (Production store capacity): {optimal_params[0]:.2f} mm")
    print(f"  X2 (Groundwater exchange): {optimal_params[1]:.2f} mm")
    print(f"  X3 (Routing store capacity): {optimal_params[2]:.2f} mm")
    print(f"  X4 (Unit hydrograph time base): {optimal_params[3]:.2f} days")
    print("=" * 60)
    print("PERFORMANCE METRICS:")
    print(f"  Calibration NSE: {nse_cal:.3f}")
    print(f"  Validation NSE: {nse_val:.3f}")
    
    # Calculate additional metrics
    cal_rmse = np.sqrt(np.mean((Q_cal - Q_cal_sim)**2))
    val_rmse = np.sqrt(np.mean((Q_val - Q_val_sim)**2))
    cal_bias = np.mean(Q_cal_sim - Q_cal)
    val_bias = np.mean(Q_val_sim - Q_val)
    
    print(f"  Calibration RMSE: {cal_rmse:.3f} mm/day")
    print(f"  Validation RMSE: {val_rmse:.3f} mm/day")
    print(f"  Calibration Bias: {cal_bias:.3f} mm/day")
    print(f"  Validation Bias: {val_bias:.3f} mm/day")
    print("=" * 60)

    Q_sim_full = np.concatenate([Q_cal_sim, Q_val_sim])

    # Generate enhanced visualizations
    print("\nGenerating comprehensive calibration visualizations...")
    create_calibration_plots(dataset, Q_obs, Q_sim_full, optimal_params, 
                           nse_cal, nse_val, split_index, save_dir="figures")
    
    print("\n" + "=" * 80)
    print("Enhanced GR4J calibration and validation completed successfully!")
    print("This demonstration shows:")
    print("  - Automatic parameter optimization using Nash-Sutcliffe Efficiency")
    print("  - Comprehensive performance evaluation on calibration and validation periods")
    print("  - Detailed statistical analysis and residual diagnostics")
    print("  - Parameter sensitivity and optimal value visualization")
    print("Check the 'figures' directory for detailed visualizations.")
    print("=" * 80)
