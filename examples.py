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

Author: [Your Name]
Date: 2024
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from datetime import datetime, timedelta

# Import all models / 导入所有模型
from xinanjiang_model import XinanjiangModel
from tank_model import TankModel1D, TankModel2D, TankModel3D
from gr4j_model import GR4J
from sacramento_model import SacramentoModel
from hbv_model import HBVModel
# Note: event_model_scs_uh requires hourly data and is event-based (see separate usage)
# 注意: event_model_scs_uh需要小时数据，是基于事件的(参见单独使用说明)


def calculate_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Compute Nash-Sutcliffe Efficiency between observed and simulated flows."""
    if observed.size == 0:
        return np.nan
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        return np.nan
    return 1.0 - np.sum((observed - simulated) ** 2) / denominator


def calculate_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Compute the root mean squared error."""
    return float(np.sqrt(np.mean((observed - simulated) ** 2)))


def calculate_pbias(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Compute percent bias between observed and simulated flows."""
    if np.sum(observed) == 0:
        return np.nan
    return float(100.0 * np.sum(simulated - observed) / np.sum(observed))


def generate_synthetic_data(n_days: int = 365, seed: int = 42):
    """
    Generate synthetic hydrological data for testing.
    生成用于测试的合成水文数据。
    
    This function creates realistic precipitation, evapotranspiration, and
    temperature time series that can be used as input for hydrological models.
    此函数创建可用作水文模型输入的真实降水、蒸散发和温度时间序列。
    
    Parameters / 参数:
    -----------
    n_days : int
        Number of days to simulate / 模拟天数
    seed : int
        Random seed for reproducibility / 用于可重复性的随机种子
        
    Returns / 返回:
    --------
    dict : Dictionary containing P (precipitation), ET (evapotranspiration), and T (temperature)
           包含P(降水)、ET(蒸散发)和T(温度)的字典
    """
    np.random.seed(seed)
    
    # Precipitation: Gamma distribution with intermittent rain
    # 降水：具有间歇性降雨的伽马分布
    # Typical pattern: some days with heavy rain, many dry days
    # 典型模式：一些大雨天，许多干旱天
    P = np.random.gamma(2, 5, n_days)
    dry_days = np.random.rand(n_days) > 0.4  # 60% of days are dry / 60%的天数是干旱的
    P[dry_days] = 0
    
    # Potential Evapotranspiration: Seasonal sinusoidal pattern
    # 潜在蒸散发：季节性正弦模式
    # Higher in summer, lower in winter / 夏季较高，冬季较低
    t = np.arange(n_days)
    ET = 3.0 + 2.0 * np.sin(2 * np.pi * t / 365 - np.pi / 2)  # Peak in day 91 (summer) / 第91天达到峰值(夏季)
    ET = np.maximum(ET, 0.5)  # Minimum 0.5 mm/day / 最小0.5 mm/天
    
    # Temperature: Seasonal cycle with daily variation
    # 温度：具有日变化的季节周期
    # For models like HBV that require temperature
    # 用于像HBV这样需要温度的模型
    T_mean = 10.0  # Annual mean temperature / 年平均温度 (°C)
    T_amplitude = 10.0  # Seasonal amplitude / 季节振幅 (°C)
    T = T_mean + T_amplitude * np.sin(2 * np.pi * t / 365 - np.pi / 2) + np.random.normal(0, 2, n_days)
    
    return {
        'P': P,
        'ET': ET,
        'T': T,  # Temperature for HBV model / HBV模型的温度
        'days': n_days
    }


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


def compare_all_models():
    """
    Compare all hydrological models with the same input data.
    使用相同的输入数据比较所有水文模型。
    
    This function runs all available continuous hydrological models and
    compares their performance with the same synthetic data.
    此函数运行所有可用的连续水文模型，并使用相同的合成数据比较它们的性能。
    
    Note: Event-based models (SCS-CN+UH) require different data structure
    and are demonstrated separately.
    注意：基于事件的模型(SCS-CN+UH)需要不同的数据结构，单独演示。
    """
    print("=" * 80)
    print("Comparing All Hydrological Models / 比较所有水文模型")
    print("=" * 80)
    
    # Generate common input data / 生成通用输入数据
    data = generate_synthetic_data(n_days=365, seed=42)
    P = data['P']
    ET = data['ET']
    T = data['T']  # Temperature for HBV / HBV的温度
    
    print(f"\nInput Data / 输入数据: {data['days']} days / 天")
    print(f"Total Precipitation / 总降水: {np.sum(P):.2f} mm")
    print(f"Total Potential ET / 总潜在蒸散发: {np.sum(ET):.2f} mm")
    print(f"Average Daily P / 平均日降水: {np.mean(P):.2f} mm")
    print(f"Average Daily ET / 平均日蒸散发: {np.mean(ET):.2f} mm")
    print(f"Average Temperature / 平均温度: {np.mean(T):.2f} °C")
    
    results = {}

    # 1. Xinanjiang Model / 新安江模型
    print("\n" + "-" * 80)
    print("1. Running Xinanjiang Model / 运行新安江模型...")
    xaj = XinanjiangModel()
    results['Xinanjiang'] = xaj.run(P, ET)
    rng = np.random.default_rng(123)
    observed_Q = np.clip(
        results['Xinanjiang']['Q'] + rng.normal(0.0, 0.5, size=len(P)),
        a_min=0.0,
        a_max=None
    )
    print(f"   Total discharge / 总径流: {np.sum(results['Xinanjiang']['Q']):.2f} mm")
    print(f"   Runoff coefficient / 径流系数: {np.sum(results['Xinanjiang']['Q'])/np.sum(P):.3f}")
    
    # 2. Tank Model 1D / Tank模型1D
    print("\n" + "-" * 80)
    print("2. Running Tank Model 1D / 运行Tank模型1D...")
    tank1d = TankModel1D()
    results['Tank_1D'] = tank1d.run(P, ET)
    print(f"   Total discharge / 总径流: {np.sum(results['Tank_1D']['Q']):.2f} mm")
    print(f"   Runoff coefficient / 径流系数: {np.sum(results['Tank_1D']['Q'])/np.sum(P):.3f}")
    
    # 3. Tank Model 2D / Tank模型2D
    print("\n" + "-" * 80)
    print("3. Running Tank Model 2D / 运行Tank模型2D...")
    tank2d = TankModel2D()
    results['Tank_2D'] = tank2d.run(P, ET)
    print(f"   Total discharge / 总径流: {np.sum(results['Tank_2D']['Q']):.2f} mm")
    print(f"   Runoff coefficient / 径流系数: {np.sum(results['Tank_2D']['Q'])/np.sum(P):.3f}")
    
    # 4. Tank Model 3D / Tank模型3D
    print("\n" + "-" * 80)
    print("4. Running Tank Model 3D / 运行Tank模型3D...")
    tank3d = TankModel3D()
    results['Tank_3D'] = tank3d.run(P, ET)
    print(f"   Total discharge / 总径流: {np.sum(results['Tank_3D']['Q']):.2f} mm")
    print(f"   Runoff coefficient / 径流系数: {np.sum(results['Tank_3D']['Q'])/np.sum(P):.3f}")
    
    # 5. GR4J Model / GR4J模型
    print("\n" + "-" * 80)
    print("5. Running GR4J Model / 运行GR4J模型...")
    gr4j = GR4J()
    results['GR4J'] = gr4j.run(P, ET)
    print(f"   Total discharge / 总径流: {np.sum(results['GR4J']['Q']):.2f} mm")
    print(f"   Runoff coefficient / 径流系数: {np.sum(results['GR4J']['Q'])/np.sum(P):.3f}")
    
    # 6. Sacramento Model / Sacramento模型
    print("\n" + "-" * 80)
    print("6. Running Sacramento Model / 运行Sacramento模型...")
    sac = SacramentoModel()
    results['Sacramento'] = sac.run(P, ET)
    print(f"   Total discharge / 总径流: {np.sum(results['Sacramento']['Q']):.2f} mm")
    print(f"   Runoff coefficient / 径流系数: {np.sum(results['Sacramento']['Q'])/np.sum(P):.3f}")
    
    # 7. HBV Model / HBV模型
    print("\n" + "-" * 80)
    print("7. Running HBV Model / 运行HBV模型...")
    hbv = HBVModel()
    results['HBV'] = hbv.run(P, T, ET)  # HBV需要温度 / HBV requires temperature
    print(f"   Total discharge / 总径流: {np.sum(results['HBV']['Q']):.2f} mm")
    print(f"   Runoff coefficient / 径流系数: {np.sum(results['HBV']['Q'])/np.sum(P):.3f}")
    
    # Comparison Table / 比较表
    print("\n" + "=" * 80)
    print("Model Comparison Summary / 模型比较摘要")
    print("=" * 80)
    print("\n{:<20} {:<15} {:<15} {:<15} {:<15} {:<12} {:<12} {:<12}".format(
        "Model / 模型", "Total Q (mm)", "Runoff Coef", "Peak Q (mm)", "Mean Q (mm)",
        "NSE", "RMSE", "PBIAS"))
    print("-" * 80)

    for model_name, result in results.items():
        Q = result['Q']
        total_Q = np.sum(Q)
        runoff_coef = total_Q / np.sum(P)
        peak_Q = np.max(Q)
        mean_Q = np.mean(Q)
        nse = calculate_nse(observed_Q, Q)
        rmse = calculate_rmse(observed_Q, Q)
        pbias = calculate_pbias(observed_Q, Q)

        print("{:<20} {:<15.2f} {:<15.3f} {:<15.2f} {:<15.2f} {:<12.3f} {:<12.2f} {:<12.2f}".format(
            model_name, total_Q, runoff_coef, peak_Q, mean_Q, nse, rmse, pbias))

    print("\n" + "=" * 80)

    # Enhanced comprehensive visualization / 增强的综合可视化
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
    import matplotlib.cm as cm
    colors_extended = [cm.tab10(i/10) for i in range(10)]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Models Performance Analysis', fontsize=16, fontweight='bold')

    # Performance metrics
    nse_values = []
    rmse_values = []
    runoff_coeffs = []
    total_discharges = []
    
    for model_name, result in results.items():
        Q = result['Q']
        nse_values.append(calculate_nse(observed_Q, Q))
        rmse_values.append(calculate_rmse(observed_Q, Q))
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
    import matplotlib.cm as cm
    colors_extended = [cm.tab10(i/10) for i in range(10)]  # Use tab10 colormap / 使用tab10色图

    for i, (model_name, result) in enumerate(results.items()):
        Q_sim = result['Q']
        
        axes[i].plot(dates, observed_Q, label='Observed', color='black', linewidth=2, linestyle='--')
        axes[i].plot(dates, Q_sim, label=f'{model_name}', color=colors_extended[i % len(colors_extended)], linewidth=1.5)
        
        # Fill area between curves / 填充曲线间区域
        axes[i].fill_between(dates, observed_Q, Q_sim, alpha=0.2, color=colors_extended[i % len(colors_extended)])
        
        nse = calculate_nse(observed_Q, Q_sim)
        rmse = calculate_rmse(observed_Q, Q_sim)
        
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
    
    # Example 1: Real-world data structure
    real_world_data_structure()
    
    # input("\nPress Enter to continue to model comparison...")
    
    # Example 2: Compare all models
    compare_all_models()
    
    # input("\nPress Enter to continue to sensitivity analysis...")
    
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
