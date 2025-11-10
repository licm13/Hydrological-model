"""
Comprehensive Examples for All Hydrological Models

This script demonstrates the usage of all implemented hydrological models
with various scenarios including:
- Random data generation
- Model comparison
- Visualization
- Real-world data structure examples

Models included:
1. Xinanjiang Model (新安江模型)
2. Tank Model (タンクモデル) - 1D, 2D, 3D
3. GR4J Model
4. Sacramento Model (SAC-SMA)

Author: LCM
Date: 2025-11-04
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence, TYPE_CHECKING

# Configure matplotlib for Chinese font display / 配置matplotlib以显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'KaiTi', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display / 修复负号显示

# Import all models / 导入所有模型
from xinanjiang_model import XinanjiangModel
from tank_model import TankModel1D, TankModel2D, TankModel3D
from gr4j_model import GR4J
from sacramento_model import SacramentoModel
from hbv_model import HBVModel
from event_model_scs_uh import EventModel, create_event_plots
from topmodel import Topmodel

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ml_benchmark import RandomForestBenchmarkResult
# Note: event_model_scs_uh requires hourly data and is event-based (see separate usage)
# 注意: event_model_scs_uh需要小时数据，是基于事件的(参见单独使用说明)

# Import utility modules / 导入工具模块
from utils.performance import (
    nash_sutcliffe_efficiency,
    root_mean_squared_error,
    percent_bias,
    basic_runoff_statistics
)
from utils.reporting import (
    format_model_summary,
    compute_water_balance,
    print_first_n_days_table
)


def generate_synthetic_data(n_days: int = 365, seed: int = 42, climate: str = "baseline") -> Dict[str, np.ndarray]:
    """
    Generate synthetic hydrological data for testing.
    生成用于测试的合成水文数据。
    
This function creates realistic precipitation, evapotranspiration, and
temperature time series that can be used as input for hydrological models.
The ``climate`` argument enables quick experiments with arid and humid
catchments for classroom demonstrations.
此函数创建可用作水文模型输入的真实降水、蒸散发和温度时间序列，并可通过
``climate`` 参数快速构建干旱或湿润流域情景以用于教学。
    
    Parameters / 参数:
    -----------
    n_days : int
        Number of days to simulate / 模拟天数
    seed : int
        Random seed for reproducibility / 用于可重复性的随机种子
        
    climate : str
        Scenario flag. Supported values: ``"baseline"``, ``"arid"``, ``"humid"``.
        场景标记，可选：``"baseline"``、``"arid"``、``"humid"``。

    Returns / 返回:
    --------
    dict : Dictionary containing P (precipitation), ET (evapotranspiration), and T (temperature)
           包含P(降水)、ET(蒸散发)和T(温度)的字典
    """
    climate = climate.lower()
    rng = np.random.default_rng(seed)

    climate_settings = {
        "baseline": {"precip_scale": 1.0, "dry_fraction": 0.6, "pet_multiplier": 1.0, "temp_shift": 0.0},
        "arid": {"precip_scale": 0.45, "dry_fraction": 0.8, "pet_multiplier": 1.25, "temp_shift": 2.0},
        "humid": {"precip_scale": 1.35, "dry_fraction": 0.35, "pet_multiplier": 0.85, "temp_shift": -1.0},
    }
    settings = climate_settings.get(climate, climate_settings["baseline"])

    # Precipitation generation
    base_precip = rng.gamma(shape=2.0, scale=5.0, size=n_days)
    dry_mask = rng.random(n_days) < settings["dry_fraction"]
    base_precip[dry_mask] = 0.0
    P = base_precip * settings["precip_scale"]

    # Potential evapotranspiration with seasonal modulation
    t = np.arange(n_days)
    ET = 3.0 + 2.0 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
    ET = np.maximum(ET, 0.5) * settings["pet_multiplier"]

    # Temperature signal, shifted to reflect the scenario
    T_mean = 10.0 + settings["temp_shift"]
    T_amplitude = 10.0
    T = T_mean + T_amplitude * np.sin(2 * np.pi * t / 365 - np.pi / 2) + rng.normal(0, 2, n_days)

    return {
        'P': P,
        'ET': ET,
        'T': T,  # Temperature for HBV model / HBV模型的温度
        'days': n_days
    }


def generate_extreme_event_data(
    n_days: int = 180,
    drought_length: int = 90,
    storm_depth: float = 160.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Create a drought-then-deluge forcing sequence for classroom demos."""

    data = generate_synthetic_data(n_days=n_days, seed=seed, climate="baseline")
    P = np.zeros(n_days, dtype=float)
    ET = np.full(n_days, data['ET'].mean() * 1.2, dtype=float)
    T = data['T'] + 1.5

    drought_days = min(drought_length, n_days - 5)
    rng = np.random.default_rng(seed + 1)
    P[:drought_days] = rng.uniform(0.0, 0.3, size=drought_days)
    ET[:drought_days] *= 1.1

    storm_duration = min(6, n_days - drought_days)
    hyetograph = np.array([0.05, 0.1, 0.2, 0.25, 0.2, 0.2])[:storm_duration]
    hyetograph = hyetograph / hyetograph.sum()
    P[drought_days:drought_days + storm_duration] = hyetograph * storm_depth

    recovery_start = drought_days + storm_duration
    ET[recovery_start:] = data['ET'][recovery_start:] * 0.8
    T[recovery_start:] = data['T'][recovery_start:] - 1.0

    return {'P': P, 'ET': ET, 'T': T, 'days': n_days}


def apply_linear_reservoir_filter(flow: np.ndarray, residence_time: float) -> np.ndarray:
    """Apply a simple linear reservoir routing to mimic upstream storage."""

    if residence_time <= 0:
        raise ValueError("Residence time must be positive.")

    flow = np.asarray(flow, dtype=float)
    routed = np.zeros_like(flow)
    routed[0] = flow[0]
    coefficient = 1.0 / residence_time

    for idx in range(1, flow.size):
        routed[idx] = routed[idx - 1] + coefficient * (flow[idx] - routed[idx - 1])

    return routed


def run_event_sensitivity(cn_values: Sequence[int] = (55, 70, 85, 95)) -> Dict[int, Dict[str, np.ndarray]]:
    """Generate runoff hydrographs for multiple land-use CN scenarios."""

    dt = 1.0
    duration_hours = 24
    P = np.zeros(duration_hours)
    P[5:10] = [5, 15, 25, 15, 5]
    scenario_results: Dict[int, Dict[str, np.ndarray]] = {}

    for cn in cn_values:
        model = EventModel(CN=cn, AMC='II', Tp=3.0, dt=dt)
        results = model.run(P)
        scenario_results[cn] = results

    return scenario_results


def real_world_data_structure():
    """
    Example of how to structure real-world hydrological data.
    
    This function demonstrates the expected data format for using
    these models with actual observations.
    """
    print("=" * 80)
    print("Real-World Data Structure Example")
    print("=" * 80)
    
    print("\nExpected data format for model input:")
    print("\n1. Time Series Data (recommended format: pandas DataFrame or numpy arrays)")
    print("   - Date/Time: datetime index")
    print("   - Precipitation (P): mm/day")
    print("   - Evapotranspiration (ET or PET): mm/day")
    print("   - Optional: Temperature, Humidity, Wind Speed, etc.")
    
    print("\n2. Example CSV structure:")
    print("   Date,Precipitation,Evapotranspiration")
    print("   2024-01-01,5.2,2.1")
    print("   2024-01-02,0.0,2.3")
    print("   2024-01-03,12.5,1.9")
    print("   ...")
    
    print("\n3. Loading data (example code):")
    print("   ```python")
    print("   import pandas as pd")
    print("   import numpy as np")
    print("   ")
    print("   # Load from CSV")
    print("   data = pd.read_csv('hydrological_data.csv', parse_dates=['Date'])")
    print("   ")
    print("   # Extract arrays for model input")
    print("   P = data['Precipitation'].values")
    print("   ET = data['Evapotranspiration'].values")
    print("   ")
    print("   # Run model")
    print("   from xinanjiang_model import XinanjiangModel")
    print("   model = XinanjiangModel()")
    print("   results = model.run(P, ET)")
    print("   ```")
    
    print("\n4. Data quality checks:")
    print("   - Ensure no missing values (or handle them appropriately)")
    print("   - Check for negative values")
    print("   - Verify units (mm/day for daily models)")
    print("   - Check data consistency and temporal alignment")
    
    print("\n" + "=" * 80)


def compare_all_models(
    n_days: int = 365,
    scenario: str = "baseline",
    include_ml: bool = True,
    include_topmodel: bool = True,
    reservoir_residence_time: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compare the continuous models (and optional ML baseline) on one dataset."""

    print("=" * 80)
    print("Comparing All Hydrological Models / 比较所有水文模型")
    print("=" * 80)

    scenario_key = scenario.lower()
    if scenario_key == "extreme_event":
        data = generate_extreme_event_data(n_days=n_days, seed=seed)
        print("Using drought → deluge extreme event forcing.")
    else:
        data = generate_synthetic_data(n_days=n_days, seed=seed, climate=scenario_key)
        print(f"Using {scenario_key} climate scenario forcings.")

    P = data['P']
    ET = data['ET']
    T = data['T']

    print(f"\nInput Data / 输入数据: {data['days']} days / 天")
    print(f"Total Precipitation / 总降水: {np.sum(P):.2f} mm")
    print(f"Total Potential ET / 总潜在蒸散发: {np.sum(ET):.2f} mm")
    print(f"Average Daily P / 平均日降水: {np.mean(P):.2f} mm")
    print(f"Average Daily ET / 平均日蒸散发: {np.mean(ET):.2f} mm")
    print(f"Average Temperature / 平均温度: {np.mean(T):.2f} °C")

    results: Dict[str, Dict[str, np.ndarray]] = {}

    xaj_kwargs: Dict[str, float] = {}
    hbv_kwargs: Dict[str, float] = {}
    if scenario_key == "arid":
        xaj_kwargs.update({"WM": 110.0, "SM": 18.0, "K": 1.1})
        hbv_kwargs.update({"FC": 160.0, "BETA": 1.6, "LP": 0.85})
    elif scenario_key == "humid":
        xaj_kwargs.update({"WM": 220.0, "SM": 45.0, "K": 0.9})
        hbv_kwargs.update({"FC": 320.0, "BETA": 2.6, "LP": 0.6})

    print("\n" + "-" * 80)
    print("1. Running Xinanjiang Model / 运行新安江模型...")
    xaj = XinanjiangModel(**xaj_kwargs)
    results['Xinanjiang'] = xaj.run(P, ET)
    rng = np.random.default_rng(seed + 100)
    observed_Q = np.clip(
        results['Xinanjiang']['Q'] + rng.normal(0.0, 0.5, size=len(P)),
        a_min=0.0,
        a_max=None
    )
    print(f"   Total discharge / 总径流: {np.sum(results['Xinanjiang']['Q']):.2f} mm")
    print(f"   Runoff coefficient / 径流系数: {np.sum(results['Xinanjiang']['Q'])/np.sum(P):.3f}")

    print("\n" + "-" * 80)
    print("2. Running Tank Model 1D / 运行Tank模型1D...")
    tank1d = TankModel1D()
    results['Tank_1D'] = tank1d.run(P, ET)

    print("\n" + "-" * 80)
    print("3. Running Tank Model 2D / 运行Tank模型2D...")
    tank2d = TankModel2D()
    results['Tank_2D'] = tank2d.run(P, ET)

    print("\n" + "-" * 80)
    print("4. Running Tank Model 3D / 运行Tank模型3D...")
    tank3d = TankModel3D()
    results['Tank_3D'] = tank3d.run(P, ET)

    print("\n" + "-" * 80)
    print("5. Running GR4J Model / 运行GR4J模型...")
    gr4j = GR4J()
    results['GR4J'] = gr4j.run(P, ET)

    print("\n" + "-" * 80)
    print("6. Running Sacramento Model / 运行Sacramento模型...")
    sac = SacramentoModel()
    results['Sacramento'] = sac.run(P, ET)

    print("\n" + "-" * 80)
    print("7. Running HBV Model / 运行HBV模型...")
    hbv = HBVModel(**hbv_kwargs)
    results['HBV'] = hbv.run(P, T, ET)

    if include_topmodel:
        print("\n" + "-" * 80)
        print("8. Running TOPMODEL-inspired benchmark / 运行TOPMODEL基准...")
        topmodel = Topmodel()
        results['TOPMODEL'] = topmodel.run(P, ET)

    if include_ml:
        print("\n" + "-" * 80)
        print("9. Training Random Forest benchmark / 训练随机森林基准...")
        from ml_benchmark import run_random_forest_benchmark
        ml_result = run_random_forest_benchmark(P, ET, observed_Q)
        results['RandomForest'] = {'Q': ml_result.full_series}
        print(f"   Validation metrics: {ml_result.metrics}")

    if reservoir_residence_time is not None:
        print("\nApplying linear reservoir routing to illustrate storage effects...")
        routed_results: Dict[str, Dict[str, np.ndarray]] = {}
        for name, res in results.items():
            routed = apply_linear_reservoir_filter(res['Q'], residence_time=reservoir_residence_time)
            routed_results[f"{name}_Reservoir"] = {**res, 'Q': routed}
        results.update(routed_results)

    print("\n" + "=" * 80)
    print("Model Comparison Summary / 模型比较摘要")
    print("=" * 80)
    print("\n{:<20} {:<15} {:<15} {:<15} {:<15} {:<12} {:<12} {:<12}".format(
        "Model / 模型", "Total Q (mm)", "Runoff Coef", "Peak Q (mm)", "Mean Q (mm)",
        "NSE", "RMSE", "PBIAS"))
    print("-" * 80)

    for model_name, result in results.items():
        summary_line = format_model_summary(model_name, result, P, ET, observed_Q)
        print(summary_line)

    print("\n" + "=" * 80)

    create_model_comparison_plots(P, ET, observed_Q, results, save_dir="figures")

    return results


def create_model_comparison_plots(P, ET, observed_Q, results, save_dir="figures"):
    """
    Create comprehensive comparison plots for all hydrological models.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Re-configure Chinese font after style setting / 样式设置后重新配置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'KaiTi', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    n_days = len(P)
    days = np.arange(n_days)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Figure 1: Main model comparison
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('Comprehensive Hydrological Models Comparison', fontsize=16, fontweight='bold')

    # Precipitation (inverted)
    axes[0].bar(dates, P, color='steelblue', alpha=0.7, width=1)
    axes[0].set_ylabel('Precipitation\n(mm/day)', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(max(P) * 1.1, 0)

    # Evapotranspiration
    axes[1].plot(dates, ET, color='orange', linewidth=1.5, label='Potential ET')
    axes[1].set_ylabel('Evapotranspiration\n(mm/day)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Discharge comparison
    axes[2].plot(dates, observed_Q, label='Synthetic Observed', color='black', linewidth=2, linestyle='--')
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    for i, (model_name, result) in enumerate(results.items()):
        axes[2].plot(dates, result['Q'], label=model_name, color=colors[i % len(colors)], linewidth=1.5)

    axes[2].set_ylabel('Discharge\n(mm/day)', fontweight='bold')
    axes[2].set_xlabel('Date', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'models_comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Performance analysis
    model_names = list(results.keys())
    n_models = len(model_names)
    
    # Generate enough colors for all models / 为所有模型生成足够的颜色
    from matplotlib import colormaps as mcm
    cmap = mcm.get_cmap('tab10')
    colors_extended = [cmap(i/10) for i in range(10)]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Models Performance Analysis', fontsize=16, fontweight='bold')

    # Performance metrics
    nse_values = []
    rmse_values = []
    runoff_coeffs = []
    total_discharges = []
    
    for model_name, result in results.items():
        Q = result['Q']
        nse_values.append(nash_sutcliffe_efficiency(observed_Q, Q))
        rmse_values.append(root_mean_squared_error(observed_Q, Q))
        runoff_coeffs.append(np.sum(Q) / np.sum(P))
        total_discharges.append(np.sum(Q))

    # NSE comparison
    bars1 = axes[0,0].bar(model_names, nse_values, color=colors_extended[:n_models], alpha=0.7)
    axes[0,0].set_ylabel('Nash-Sutcliffe Efficiency', fontweight='bold')
    axes[0,0].set_title('Model Performance (NSE)', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, nse in zip(bars1, nse_values):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{nse:.3f}', ha='center', va='bottom', fontweight='bold')

    # RMSE comparison
    bars2 = axes[0,1].bar(model_names, rmse_values, color=colors_extended[:n_models], alpha=0.7)
    axes[0,1].set_ylabel('Root Mean Square Error (mm/day)', fontweight='bold')
    axes[0,1].set_title('Model Error (RMSE)', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rmse in zip(bars2, rmse_values):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                      f'{rmse:.3f}', ha='center', va='bottom', fontweight='bold')

    # Runoff coefficient comparison
    bars3 = axes[1,0].bar(model_names, runoff_coeffs, color=colors_extended[:n_models], alpha=0.7)
    axes[1,0].set_ylabel('Runoff Coefficient', fontweight='bold')
    axes[1,0].set_title('Water Balance (Runoff Coefficient)', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, coeff in zip(bars3, runoff_coeffs):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                      f'{coeff:.3f}', ha='center', va='bottom', fontweight='bold')

    # Total discharge comparison
    bars4 = axes[1,1].bar(model_names, total_discharges, color=colors_extended[:n_models], alpha=0.7)
    axes[1,1].set_ylabel('Total Discharge (mm)', fontweight='bold')
    axes[1,1].set_title('Total Water Yield', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, discharge in zip(bars4, total_discharges):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{discharge:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'models_performance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Detailed time series analysis
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 3*n_models), sharex=True)
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Individual Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # Generate enough colors for all models / 为所有模型生成足够的颜色
    from matplotlib import colormaps as mcm
    cmap = mcm.get_cmap('tab10')
    colors_extended = [cmap(i/10) for i in range(10)]  # Use tab10 colormap / 使用tab10色图

    for i, (model_name, result) in enumerate(results.items()):
        Q_sim = result['Q']
        
        axes[i].plot(dates, observed_Q, label='Observed', color='black', linewidth=2, linestyle='--')
        axes[i].plot(dates, Q_sim, label=f'{model_name}', color=colors_extended[i % len(colors_extended)], linewidth=1.5)
        
        # Fill area between curves / 填充曲线间区域
        axes[i].fill_between(dates, observed_Q, Q_sim, alpha=0.2, color=colors_extended[i % len(colors_extended)])
        
        nse = nash_sutcliffe_efficiency(observed_Q, Q_sim)
        rmse = root_mean_squared_error(observed_Q, Q_sim)
        
        axes[i].set_ylabel('Discharge\n(mm/day)', fontweight='bold')
        axes[i].set_title(f'{model_name} - NSE: {nse:.3f}, RMSE: {rmse:.3f}', fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    axes[-1].set_xlabel('Date', fontweight='bold')
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'models_individual_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Model comparison visualization plots saved to '{save_dir}' directory")
    print("  Generated files:")
    print("  - models_comprehensive_comparison.png")
    print("  - models_performance_analysis.png")
    print("  - models_individual_analysis.png")


def sensitivity_analysis_example():
    """
    Example of sensitivity analysis for model parameters.
    """
    print("=" * 80)
    print("Parameter Sensitivity Analysis Example - Xinanjiang Model")
    print("=" * 80)
    
    # Generate input data
    data = generate_synthetic_data(n_days=365, seed=42)
    P = data['P']
    ET = data['ET']
    
    print("\nTesting sensitivity to parameter B (tension water distribution curve)")
    print("B parameter controls the spatial variability of soil moisture capacity")
    
    B_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("\n{:<10} {:<15} {:<15} {:<15}".format(
        "B value", "Total Q (mm)", "Runoff Coef", "Peak Q (mm)"))
    print("-" * 60)
    
    for B in B_values:
        model = XinanjiangModel(B=B)
        results = model.run(P, ET)
        Q = results['Q']
        
        print("{:<10.2f} {:<15.2f} {:<15.3f} {:<15.2f}".format(
            B, np.sum(Q), np.sum(Q)/np.sum(P), np.max(Q)))
    
    print("\nObservation:")
    print("- Higher B values typically lead to more runoff generation")
    print("- B represents the heterogeneity of the catchment")
    print("- Should be calibrated based on observed streamflow data")
    
    print("\n" + "=" * 80)


def storm_event_example():
    """
    Example simulating a specific storm event.
    """
    print("=" * 80)
    print("Storm Event Simulation Example")
    print("=" * 80)
    
    # Create a storm event: dry period, intense rain, recession
    n_days = 30
    P = np.zeros(n_days)
    
    # Dry period (days 0-9)
    P[0:10] = 0
    
    # Storm event (days 10-13)
    P[10] = 25.0  # Pre-storm rain
    P[11] = 80.0  # Main storm
    P[12] = 45.0  # Continuing rain
    P[13] = 15.0  # Light rain
    
    # Recession period (days 14-29)
    P[14:] = 0
    
    # Evapotranspiration (constant)
    ET = np.ones(n_days) * 3.0
    
    print("\nStorm Event Pattern:")
    print("- Days 0-9: Dry period")
    print("- Days 10-13: Storm event (total: {:.1f} mm)".format(np.sum(P[10:14])))
    print("- Days 14-29: Recession period")
    
    # Run different models
    print("\n" + "-" * 80)
    print("Comparing Model Responses to Storm Event")
    print("-" * 80)
    
    # Xinanjiang
    xaj = XinanjiangModel()
    res_xaj = xaj.run(P, ET)
    
    # Tank 3D
    tank3d = TankModel3D()
    res_tank = tank3d.run(P, ET)
    
    # GR4J
    gr4j = GR4J()
    res_gr4j = gr4j.run(P, ET)
    
    print("\nPeak Discharge Comparison:")
    print(f"  Xinanjiang: {np.max(res_xaj['Q']):.2f} mm/day on day {np.argmax(res_xaj['Q'])+1}")
    print(f"  Tank 3D: {np.max(res_tank['Q']):.2f} mm/day on day {np.argmax(res_tank['Q'])+1}")
    print(f"  GR4J: {np.max(res_gr4j['Q']):.2f} mm/day on day {np.argmax(res_gr4j['Q'])+1}")
    
    print("\nHydrograph during storm (Days 9-16):")
    print("\nDay |   P    |  XAJ   |  Tank  |  GR4J  |")
    print("----|--------|--------|--------|--------|")
    for i in range(9, 17):
        print(f" {i+1:2d} | {P[i]:6.2f} | {res_xaj['Q'][i]:6.2f} | "
              f"{res_tank['Q'][i]:6.2f} | {res_gr4j['Q'][i]:6.2f} |")
    
    print("\n" + "=" * 80)


def seasonal_pattern_example():
    """
    Example showing seasonal patterns in runoff.
    """
    print("=" * 80)
    print("Seasonal Pattern Analysis (2 Years)")
    print("=" * 80)
    
    # Generate 2 years of data with seasonal patterns
    n_days = 730  # 2 years
    t = np.arange(n_days)
    
    np.random.seed(42)
    
    # Seasonal precipitation (higher in summer)
    P_base = 5 + 5 * np.sin(2 * np.pi * t / 365)
    P_random = np.random.gamma(2, 2, n_days)
    P = P_base + P_random
    P[np.random.rand(n_days) > 0.4] = 0  # Still have dry days
    
    # Seasonal ET (higher in summer)
    ET = 2.5 + 2.5 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
    ET = np.maximum(ET, 0.5)
    
    print(f"\nSimulating {n_days} days (2 years)")
    print(f"Total precipitation: {np.sum(P):.2f} mm")
    print(f"Total potential ET: {np.sum(ET):.2f} mm")
    
    # Run model
    model = XinanjiangModel()
    results = model.run(P, ET)
    Q = results['Q']
    
    # Calculate seasonal statistics
    # Split into 4 seasons per year (91-day seasons)
    print("\nSeasonal Runoff Analysis:")
    print("\n{:<15} {:<15} {:<15} {:<15}".format(
        "Season", "Total P (mm)", "Total Q (mm)", "Runoff Coef"))
    print("-" * 60)
    
    seasons = ['Winter Y1', 'Spring Y1', 'Summer Y1', 'Fall Y1',
               'Winter Y2', 'Spring Y2', 'Summer Y2', 'Fall Y2']
    
    for i in range(8):
        start = i * 91
        end = min(start + 91, n_days)
        
        P_season = np.sum(P[start:end])
        Q_season = np.sum(Q[start:end])
        coef = Q_season / P_season if P_season > 0 else 0
        
        print("{:<15} {:<15.2f} {:<15.2f} {:<15.3f}".format(
            seasons[i], P_season, Q_season, coef))
    
    print("\n" + "=" * 80)


def generate_hourly_rain_event(total_depth_mm: float = 80.0,
                               warmup_hours: int = 24,
                               storm_hours: int = 24,
                               recession_hours: int = 24,
                               peak_skew: float = 3.0,
                               seed: int = 42):
    """
    Generate an hourly storm hyetograph (mm/hour).
    生成一个小时尺度暴雨过程(mm/h)。
    
    Parameters:
    - total_depth_mm: total storm depth 总雨量
    - warmup_hours: pre-storm dry hours 暴雨前干旱小时数
    - storm_hours: storm duration 暴雨历时
    - recession_hours: post-storm dry hours 暴雨后衰减/干旱小时数
    - peak_skew: shape of hyetograph (gamma-like) 峰型偏度
    - seed: RNG seed
    
    Returns:
    - dict with keys: 'P_hourly' (np.ndarray mm/h), 'datetimes' (list[datetime]),
                      'dt_hours' (float), 'total_depth_mm' (float)
    """
    import numpy as np
    from datetime import datetime, timedelta

    rng = np.random.default_rng(seed)

    # Build a skewed intensity shape for the storm window
    h = np.arange(storm_hours, dtype=float) + 1.0
    # Gamma-like shape, then add small noise
    shape = np.power(h, peak_skew - 1) * np.exp(-h / (storm_hours / 3.0))
    shape = np.maximum(shape + rng.normal(0, 0.02, size=shape.size), 0)
    if shape.sum() == 0:
        shape[:] = 1.0
    intensities = shape / shape.sum() * total_depth_mm  # mm distributed over storm_hours
    # Convert to hourly intensities (already mm/h per hour bin)

    # Assemble full series with dry periods
    P_hourly = np.concatenate([
        np.zeros(warmup_hours),
        intensities,
        np.zeros(recession_hours),
    ])

    start = datetime(2020, 1, 1, 0, 0, 0)
    datetimes = [start + timedelta(hours=i) for i in range(P_hourly.size)]

    return {
        'P_hourly': P_hourly.astype(float),
        'datetimes': datetimes,
        'dt_hours': 1.0,
        'total_depth_mm': float(P_hourly.sum())
    }


def hourly_rain_demo(save_dir: str = "figures", out_csv: str = "data/hourly_rain_event.csv"):
    """
    Create and save an hourly rainfall event for event-based models.
    生成并保存小时降雨事件，供事件型模型(SCS-CN+UH)使用。
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    data = generate_hourly_rain_event(
        total_depth_mm=80.0, warmup_hours=24, storm_hours=24, recession_hours=24, peak_skew=3.0
    )
    P_hourly = data['P_hourly']
    datetimes = data['datetimes']

    # Save CSV: Datetime, Precipitation(mm/h)
    df = pd.DataFrame({
        "Datetime": datetimes,
        "Precipitation_mm_per_h": P_hourly
    })
    df.to_csv(out_csv, index=False)

    # Plot hyetograph
    plt.figure(figsize=(12, 4))
    plt.bar(datetimes, P_hourly, width=0.03, color='steelblue', alpha=0.8)
    plt.gca().invert_yaxis()  # common hydro convention
    plt.title("Hourly Storm Hyetograph (mm/h)")
    plt.ylabel("Precipitation (mm/h)")
    plt.xlabel("Datetime")
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "hourly_rain_event.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ Hourly rainfall CSV saved: {out_csv}")
    print(f"✓ Hyetograph figure saved: {fig_path}")
    print(f"Total depth: {P_hourly.sum():.1f} mm, Duration: {len(P_hourly)} hours")


# ========================= NEW: Long hourly forcings =========================
def generate_long_hourly_forcings(years: int = 2, seed: int = 123) -> dict:
    """
    Generate a long hourly forcing dataset: precipitation (P), potential ET (PET), and temperature (T).
    生成长时间(小时尺度)的驱动数据：降水P、潜在蒸散PET、气温T。

    Notes:
    - P_h: Intermittent gamma rainfall with seasonal and mild diurnal modulation
    - PET_h: Seasonal + diurnal cycle, typical daily total 2–5 mm/d
    - T_h: Seasonal + diurnal temperature (°C)
    """
    import numpy as np
    from datetime import datetime, timedelta

    rng = np.random.default_rng(seed)

    n_hours = int(years * 365 * 24)
    start = datetime(2020, 1, 1, 0, 0, 0)
    times = [start + timedelta(hours=i) for i in range(n_hours)]

    # Time indices
    h = np.arange(n_hours)
    day_of_year = (h // 24) % 365
    hour_of_day = h % 24

    # Seasonal factors
    seasonal_precip = 1.2 + 0.8 * np.sin(2 * np.pi * (day_of_year - 30) / 365.0)  # wetter in summer
    seasonal_pet = 3.0 + 2.0 * np.sin(2 * np.pi * (day_of_year - 90) / 365.0)     # peak in summer
    seasonal_temp = 10.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 90) / 365.0)

    # Diurnal factors
    diurnal_pet = np.clip(np.sin(np.pi * (hour_of_day - 6) / 12.0), 0, None)  # day-time PET
    diurnal_temp = 3.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)         # +/- 3°C diurnal amplitude

    # Temperature (°C)
    T_h = seasonal_temp + diurnal_temp + rng.normal(0, 1.0, n_hours)

    # Potential ET (mm/h): ensure positive, daily total roughly 2–5 mm
    PET_base = (seasonal_pet / 24.0) * (diurnal_pet / (diurnal_pet.mean() + 1e-6))
    PET_h = np.maximum(0.0, PET_base + rng.normal(0, 0.02, n_hours))

    # Hourly precipitation (mm/h): intermittent gamma with seasonal modulation
    dry_prob = 0.80 - 0.15 * np.sin(2 * np.pi * (day_of_year - 60) / 365.0)   # fewer dry hours in summer
    wet_mask = rng.random(n_hours) > dry_prob
    shape_k, scale_theta = 1.5, 1.2  # gamma parameters -> mean ~1.8 mm
    P_h = np.zeros(n_hours)
    P_h[wet_mask] = rng.gamma(shape_k, scale_theta, wet_mask.sum())
    # Seasonal modulation
    P_h *= seasonal_precip
    # Add a mild clustering effect by smoothing
    kernel = np.array([0.2, 0.6, 0.2])
    P_h = np.convolve(P_h, kernel, mode='same')

    return {
        'times': times,
        'P_hourly': P_h.astype(float),
        'PET_hourly': PET_h.astype(float),
        'T_hourly': T_h.astype(float),
        'dt_hours': 1.0
    }


def aggregate_hourly_to_daily(times, P_h, PET_h, T_h):
    """
    Aggregate hourly series to daily for daily models.
    - P_daily = sum of hourly P (mm)
    - PET_daily = sum of hourly PET (mm)
    - T_daily = mean of hourly T (°C)
    """
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({
        'Datetime': times,
        'P': P_h,
        'PET': PET_h,
        'T': T_h,
    }).set_index('Datetime')

    daily = df.resample('D').agg({'P': 'sum', 'PET': 'sum', 'T': 'mean'})
    # Ensure plain Python datetimes and numpy arrays
    dates = list(daily.index)  # pandas Timestamps are fine for printing/plotting
    P_d = daily['P'].to_numpy(dtype=float)
    PET_d = daily['PET'].to_numpy(dtype=float)
    T_d = daily['T'].to_numpy(dtype=float)
    return dates, P_d, PET_d, T_d


def _extract_storm_from_hourly(P_h: np.ndarray, window_hours: int = 36) -> tuple:
    """
    Extract a single storm window from a long hourly rainfall series by maximizing rolling sum.
    从长小时序列中提取一个暴雨事件窗口(最大滚动和)。
    Returns (start_idx, end_idx) inclusive-exclusive.
    """
    import numpy as np
    n = len(P_h)
    w = max(6, int(window_hours))
    if n < w:
        return 0, n
    cumsum = np.cumsum(np.insert(P_h, 0, 0.0))
    roll = cumsum[w:] - cumsum[:-w]
    center = int(np.argmax(roll))
    start = max(0, center)
    end = min(n, center + w)
    # Ensure there is enough rain; if not, return a fallback short window around max hour
    if np.sum(P_h[start:end]) < 1e-3:
        peak = int(np.argmax(P_h))
        start = max(0, peak - w // 2)
        end = min(n, peak + w // 2)
    return start, end


def run_all_models_with_hourly(years: int = 2,
                               save_hourly_csv: str = 'data/hourly_forcings.csv',
                               save_dir: str = 'figures'):
    """
    Use a long hourly dataset to drive all models:
    - Aggregate to daily for continuous models (Xinanjiang, Tank, GR4J, Sacramento, HBV)
    - Select one storm from hourly data for SCS-CN+UH event model
    保存输入与关键图件。
    """
    import os
    import pandas as pd
    os.makedirs(os.path.dirname(save_hourly_csv), exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    forc = generate_long_hourly_forcings(years=years, seed=2025)
    times_h = forc['times']
    P_h = forc['P_hourly']
    PET_h = forc['PET_hourly']
    T_h = forc['T_hourly']

    # Save hourly CSV
    df_h = pd.DataFrame({
        'Datetime': times_h,
        'P_mm_per_h': P_h,
        'PET_mm_per_h': PET_h,
        'T_C': T_h,
    })
    df_h.to_csv(save_hourly_csv, index=False)
    print(f"✓ Saved long hourly forcings: {save_hourly_csv} ({len(df_h)} rows)")

    # Aggregate to daily for continuous models
    dates_d, P_d, PET_d, T_d = aggregate_hourly_to_daily(times_h, P_h, PET_h, T_h)
    print(f"✓ Aggregated to daily: {len(P_d)} days from {dates_d[0].date()} to {dates_d[-1].date()}")

    # Run continuous models
    results = {}
    print("\nRunning continuous models with daily inputs (aggregated from hourly)...")

    xaj = XinanjiangModel()
    results['Xinanjiang'] = xaj.run(P_d, PET_d)

    tank1d = TankModel1D()
    results['Tank_1D'] = tank1d.run(P_d, PET_d)

    tank2d = TankModel2D()
    results['Tank_2D'] = tank2d.run(P_d, PET_d)

    tank3d = TankModel3D()
    results['Tank_3D'] = tank3d.run(P_d, PET_d)

    gr4j = GR4J()
    results['GR4J'] = gr4j.run(P_d, PET_d)

    sac = SacramentoModel()
    results['Sacramento'] = sac.run(P_d, PET_d)

    hbv = HBVModel()
    results['HBV'] = hbv.run(P_d, T_d, PET_d)

    # Build a synthetic observed series to compute metrics (same approach as compare_all_models)
    rng = np.random.default_rng(123)
    observed_Q = np.clip(results['Xinanjiang']['Q'] + rng.normal(0.0, 0.5, size=len(P_d)), a_min=0.0, a_max=None)

    # Summary table
    print("\n" + "=" * 80)
    print("Hourly-driven (aggregated daily) Model Summary / 基于小时驱动(汇聚到每日)的模型汇总")
    print("=" * 80)
    print("\n{:<20} {:<15} {:<15} {:<15} {:<15} {:<12} {:<12} {:<12}".format(
        "Model / 模型", "Total Q (mm)", "Runoff Coef", "Peak Q (mm)", "Mean Q (mm)",
        "NSE", "RMSE", "PBIAS"))
    print("-" * 80)
    for name, res in results.items():
        summary_line = format_model_summary(name, res, P_d, PET_d, observed_Q)
        print(summary_line)

    # Reuse existing comparison plots pipeline
    try:
        create_model_comparison_plots(P_d, PET_d, observed_Q, results, save_dir=save_dir)
    except Exception as e:
        print(f"(Plotting skipped due to error: {e})")

    # Run SCS-CN+UH on one extracted storm from hourly series
    print("\nRunning SCS-CN + UH event model on extracted hourly storm...")
    s, e = _extract_storm_from_hourly(P_h, window_hours=36)
    P_event = P_h[s:e]
    if P_event.size < 6 or np.sum(P_event) < 1e-3:
        # Fallback to a synthetic event if the long series is too dry
        evt = generate_hourly_rain_event(total_depth_mm=60.0, storm_hours=18)
        P_event = evt['P_hourly']
        t_event = evt['datetimes']
    else:
        t_event = times_h[s:e]

    # Choose reasonable parameters for a mid-size catchment
    em = EventModel(CN=75, AMC='II', Tp=3.0, dt=1.0, uh_type='triangular')
    evt_results = em.run(P_event)
    # Plot and save
    try:
        create_event_plots(em, evt_results, save_dir=save_dir)
    except Exception as e:
        print(f"(Event plotting skipped due to error: {e})")

    print(f"Event total rainfall: {np.sum(P_event):.1f} mm; Peak Q (UH): {np.max(evt_results['Q_hydrograph']):.2f} mm/h")
    print("✓ Hourly-driven analysis completed.")


def main():
    """
    Main function to run all examples.
    """
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  COMPREHENSIVE HYDROLOGICAL MODEL EXAMPLES".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")
    
    # Example 0: Create an hourly rainfall dataset for event-based models (SCS-CN+UH)
    # 事件型模型所需的小时降雨示例（如需生成，请取消下一行注释）
    # hourly_rain_demo()

    # Example 1: Real-world data structure
    real_world_data_structure()
    
    # input("\nPress Enter to continue to model comparison...")
    
    # Example 2: Compare all models
    compare_all_models()
    
    # input("\nPress Enter to continue to sensitivity analysis...")
    
    # Example 2b: Drive all models with long hourly data (aggregated for daily models)
    run_all_models_with_hourly(years=2)

    # Example 3: Sensitivity analysis
    # sensitivity_analysis_example()
    
    # input("\nPress Enter to continue to storm event simulation...")
    
    # Example 4: Storm event
    # storm_event_example()
    
    # input("\nPress Enter to continue to seasonal analysis...")
    
    # Example 5: Seasonal patterns
    # seasonal_pattern_example()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully! / 所有示例成功完成!")
    print("=" * 80)
    print("\nFor more information, see individual model files / 更多信息请查看各模型文件:")
    print("  - xinanjiang_model.py")
    print("  - tank_model.py")
    print("  - gr4j_model.py")
    print("  - sacramento_model.py")
    print("  - hbv_model.py (NEW - with temperature input / 新增 - 需要温度输入)")
    print("  - event_model_scs_uh.py (NEW - event-based model / 新增 - 基于事件的模型)")
    print("\nNote / 注意:")
    print("  • HBV model requires temperature data (see hbv_model.py for details)")
    print("  • HBV模型需要温度数据(详见hbv_model.py)")
    print("  • SCS-CN+UH is event-based and requires hourly rainfall data")
    print("  • SCS-CN+UH是基于事件的，需要小时降雨数据")
    print("  • Run each model file directly for standalone demonstrations")
    print("  • 直接运行每个模型文件以获得独立演示")
    print("\n")


if __name__ == "__main__":
    main()
