"""
TOPMODEL 示例脚本 / TOPMODEL Example Script

This script demonstrates the usage of the TOPMODEL-inspired hydrological model
with comprehensive visualization capabilities.

TOPMODEL (Beven & Kirkby, 1979) is a physically-based, semi-distributed 
hydrological model that focuses on the role of topography in controlling
subsurface flow and runoff generation.

作者 / Author: LCM
日期 / Date: 2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Configure matplotlib for Chinese font display / 配置matplotlib以显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'KaiTi', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display / 修复负号显示

from topmodel import Topmodel, TopmodelConfig, create_topmodel_plots


def generate_test_data(n_days=365):
    """
    Generate synthetic meteorological data for TOPMODEL testing.
    
    Parameters
    ----------
    n_days : int
        Number of days to simulate (default: 365)
        
    Returns
    -------
    tuple
        (precipitation, evapotranspiration) arrays in mm/day
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate seasonal precipitation pattern
    day_of_year = np.arange(n_days) % 365
    seasonal_precip = 3.0 + 2.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365)  # Peak in summer
    
    # Add random weather events
    random_events = np.random.exponential(scale=2.0, size=n_days)
    wet_days = np.random.random(n_days) < 0.3  # 30% wet days
    
    precipitation = np.where(wet_days, seasonal_precip + random_events, 0.0)
    
    # Generate evapotranspiration (temperature dependent)
    seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer
    evapotranspiration = np.maximum(0.1, seasonal_temp * 0.15 + np.random.normal(0, 0.5, n_days))
    
    return precipitation, evapotranspiration


def run_topmodel_sensitivity_analysis():
    """
    Run TOPMODEL with different parameter settings to demonstrate sensitivity.
    """
    print("=" * 80)
    print("TOPMODEL 敏感性分析 / TOPMODEL Sensitivity Analysis")
    print("=" * 80)
    
    # Generate test data
    P, ET = generate_test_data(n_days=365)
    
    # Define different parameter sets for sensitivity analysis
    param_configs = {
        'Default': TopmodelConfig(),
        'High_m': TopmodelConfig(m=0.1, transmissivity_sat=3.0),
        'Low_m': TopmodelConfig(m=0.02, transmissivity_sat=3.0),
        'High_T': TopmodelConfig(m=0.05, transmissivity_sat=6.0),
        'Low_T': TopmodelConfig(m=0.05, transmissivity_sat=1.5),
    }
    
    # Create different topographic index distributions
    topographic_indices = {
        'Uniform': np.linspace(5.0, 9.0, 20),
        'Normal': np.random.normal(7.0, 1.0, 30),
        'Skewed': np.concatenate([np.random.exponential(2.0, 15) + 5.0, 
                                 np.random.normal(8.0, 0.5, 15)]),
    }
    
    results = {}
    models = {}
    
    # Run simulations for different configurations
    for topo_name, topo_index in topographic_indices.items():
        for param_name, config in param_configs.items():
            if param_name == 'Default':  # Only run default parameters for all topographies
                model_name = f"{topo_name}_Topo"
                model = Topmodel(topographic_index=topo_index, config=config)
                results[model_name] = model.run(P, ET)
                models[model_name] = model
                print(f"✓ 完成 {model_name} 配置")
    
    # Run parameter sensitivity with uniform topography
    topo_uniform = topographic_indices['Uniform']
    for param_name, config in param_configs.items():
        if param_name != 'Default':
            model_name = f"Param_{param_name}"
            model = Topmodel(topographic_index=topo_uniform, config=config)
            results[model_name] = model.run(P, ET)
            models[model_name] = model
            print(f"✓ 完成 {model_name} 配置")
    
    # Create individual plots for each configuration
    os.makedirs("figures", exist_ok=True)
    
    for model_name, result in results.items():
        print(f"\n创建 {model_name} 分析图表...")
        create_topmodel_plots(models[model_name], P, ET, result, 
                             title_prefix=f"TOPMODEL-{model_name}", 
                             save_dir="figures")
    
    # Create comparison plot
    create_sensitivity_comparison_plot(P, ET, results, save_dir="figures")
    
    return results, models


def create_sensitivity_comparison_plot(P, ET, results, save_dir="figures"):
    """
    Create a comparison plot showing sensitivity analysis results.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Set up the plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('TOPMODEL 敏感性分析对比 / TOPMODEL Sensitivity Analysis Comparison', 
                 fontsize=16, fontweight='bold')
    
    n_days = len(P)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Colors for different configurations
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Precipitation
    axes[0].bar(dates, P, color='steelblue', alpha=0.6, width=1)
    axes[0].set_ylabel('降水量 / Precipitation\n(mm/day)', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(max(P) * 1.1, 0)
    axes[0].set_title('气象强迫数据 / Meteorological Forcing')
    
    # Total discharge comparison
    for i, (config_name, result) in enumerate(results.items()):
        color = colors[i % len(colors)]
        axes[1].plot(dates, result['Q'], label=config_name, color=color, linewidth=1.5)
    
    axes[1].set_ylabel('总径流 / Total Runoff\n(mm/day)', fontweight='bold')
    axes[1].set_title('不同配置下的径流对比 / Runoff Comparison Under Different Configurations')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Baseflow vs Surface runoff ratio
    for i, (config_name, result) in enumerate(results.items()):
        color = colors[i % len(colors)]
        baseflow_ratio = result['baseflow'] / (result['Q'] + 1e-6)  # Avoid division by zero
        axes[2].plot(dates, baseflow_ratio, label=config_name, color=color, linewidth=1.5)
    
    axes[2].set_ylabel('基流比例 / Baseflow Ratio', fontweight='bold')
    axes[2].set_xlabel('日期 / Date', fontweight='bold')
    axes[2].set_title('基流占总径流比例 / Baseflow to Total Runoff Ratio')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'topmodel_sensitivity_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 敏感性分析对比图已保存")


def run_simple_topmodel_example():
    """
    Run a simple TOPMODEL example with default settings.
    """
    print("=" * 80)
    print("简单 TOPMODEL 示例 / Simple TOPMODEL Example")
    print("=" * 80)
    
    # Generate test data
    P, ET = generate_test_data(n_days=365)
    
    # Create TOPMODEL with default configuration
    print("创建 TOPMODEL 实例...")
    model = Topmodel()
    
    # Display model configuration
    print(f"\n模型配置:")
    print(f"  指数衰减参数 m: {model.config.m}")
    print(f"  饱和导水率: {model.config.transmissivity_sat} mm/day")
    print(f"  初始平均饱和亏缺: {model.config.initial_mean_deficit} mm")
    print(f"  地形指数范围: {np.min(model.topographic_index):.2f} - {np.max(model.topographic_index):.2f}")
    
    # Run simulation
    print("\n运行模型模拟...")
    results = model.run(P, ET)
    
    # Display basic statistics
    total_P = np.sum(P)
    total_ET = np.sum(ET)
    total_Q = np.sum(results['Q'])
    runoff_coeff = total_Q / max(total_P, 1e-6) if total_P > 0 else 0.0
    baseflow_ratio = np.sum(results['baseflow']) / max(total_Q, 1e-6) if total_Q > 0 else 0.0
    
    print(f"\n模拟结果统计:")
    print(f"  总降水量: {total_P:.1f} mm")
    print(f"  总蒸散发: {total_ET:.1f} mm")
    print(f"  总径流量: {total_Q:.1f} mm")
    print(f"  径流系数: {runoff_coeff:.3f}")
    print(f"  基流比例: {baseflow_ratio:.3f}")
    
    # Create plots
    print("\n创建分析图表...")
    os.makedirs("figures", exist_ok=True)
    create_topmodel_plots(model, P, ET, results, 
                         title_prefix="TOPMODEL", save_dir="figures")
    
    print("\n✓ 简单示例完成!")
    return model, results


def main():
    """
    Main function to run TOPMODEL examples.
    """
    print("TOPMODEL 水文模型示例 / TOPMODEL Hydrological Model Examples")
    print("=" * 80)
    
    # Run simple example
    model, results = run_simple_topmodel_example()
    
    print("\n" + "=" * 80)
    
    # Run sensitivity analysis
    sensitivity_results, sensitivity_models = run_topmodel_sensitivity_analysis()
    
    print("\n" + "=" * 80)
    print("所有分析完成! / All analyses completed!")
    print("图表保存在 'figures' 目录中 / Plots saved in 'figures' directory")
    print("=" * 80)


if __name__ == "__main__":
    main()