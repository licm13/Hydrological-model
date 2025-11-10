"""Topmodel-inspired hydrological model skeleton.

This module provides a compact, teaching-focused implementation of the
TOPMODEL concept (Beven & Kirkby, 1979). It mirrors the API used by the
other models in HydroLearn so that instructors can easily plug it into the
existing example workflows.

The implementation purposefully favours clarity over numerical efficiency.
It exposes the key ideas of TOPMODEL: a distribution of saturation deficits
derived from the topographic index, exponential transmissivity decay, and
the partitioning of rainfall into saturation-excess runoff versus subsurface
flow. The class can therefore be used to highlight the contrast between
topography-driven runoff production and the soil moisture concepts employed
by the other models in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TopmodelConfig:
    """Configuration container for :class:`Topmodel`.

    Parameters
    ----------
    m : float
        Exponential decay parameter for transmissivity with deficit depth (m).
    transmissivity_sat : float
        Saturated lateral transmissivity (mm/day) at the surface.
    initial_mean_deficit : float
        Initial catchment-mean saturation deficit (mm).
    area_weights : np.ndarray | None
        Optional fractional contributing area associated with each
        topographic index entry. When ``None`` the contributing area is
        assumed uniform.
    """

    m: float = 0.05
    transmissivity_sat: float = 3.0
    initial_mean_deficit: float = 50.0
    area_weights: np.ndarray | None = None


class Topmodel:
    """Simplified TOPMODEL implementation.

    The class exposes ``run_timestep`` and ``run`` methods to maintain
    compatibility with the rest of HydroLearn's API. The pedagogical goal is
    to emphasise the link between the spatial distribution of saturation
    deficits and catchment-scale runoff response.
    """

    def __init__(
        self,
        topographic_index: np.ndarray | None = None,
        config: TopmodelConfig | None = None,
    ) -> None:
        self.topographic_index = (
            np.asarray(topographic_index, dtype=float)
            if topographic_index is not None
            else np.linspace(5.5, 8.5, num=20)
        )
        self.config = config or TopmodelConfig()

        if self.topographic_index.ndim != 1:
            raise ValueError("`topographic_index` must be one-dimensional.")

        n_cells = self.topographic_index.size
        if self.config.area_weights is None:
            self.area_weights = np.full(n_cells, 1.0 / n_cells)
        else:
            weights = np.asarray(self.config.area_weights, dtype=float)
            if weights.shape != (n_cells,):
                raise ValueError("Area weights must match the topographic index shape.")
            self.area_weights = weights / np.sum(weights)

        self._mean_deficit = float(self.config.initial_mean_deficit)
        self._cell_deficits = np.full(n_cells, self._mean_deficit, dtype=float)
        self._mean_ti = np.mean(self.topographic_index)

    # ------------------------------------------------------------------
    # Core hydrological routines
    # ------------------------------------------------------------------
    def run_timestep(self, precipitation: float, evapotranspiration: float) -> dict[str, float]:
        """Advance the model by one time step.

        Parameters
        ----------
        precipitation : float
            Liquid water input for the current step (mm).
        evapotranspiration : float
            Actual evapotranspiration demand (mm). In this simplified
            formulation we subtract it uniformly from the saturation deficits.
        """

        # Evapotranspiration increases the saturation deficit.
        self._cell_deficits += evapotranspiration

        # Effective deficit for each index value.
        effective_deficit = self._cell_deficits - self.config.m * (
            self.topographic_index - np.mean(self.topographic_index)
        )

        # Saturation-excess runoff occurs when precipitation exceeds the deficit.
        saturation_excess = np.maximum(precipitation - effective_deficit, 0.0)
        surface_runoff = float(np.sum(saturation_excess * self.area_weights))

        # Update local deficits after the rainfall event.
        infiltration = precipitation - saturation_excess
        self._cell_deficits = np.maximum(self._cell_deficits - infiltration, 0.0)

        # Exponential transmissivity relationship yields baseflow response.
        baseflow = float(
            self.config.transmissivity_sat
            * np.exp(-self._mean_deficit / max(self.config.m, 1e-6))
        )

        # Update the mean deficit used in the exponent.
        self._mean_deficit = float(
            np.sum(self._cell_deficits * self.area_weights)
        )

        discharge = surface_runoff + baseflow
        return {
            "Q": discharge,
            "surface": surface_runoff,
            "baseflow": baseflow,
            "mean_deficit": self._mean_deficit,
        }

    def run(self, precipitation: np.ndarray, evapotranspiration: np.ndarray) -> dict[str, np.ndarray]:
        """Simulate a full time series of forcings."""

        precipitation = np.asarray(precipitation, dtype=float)
        evapotranspiration = np.asarray(evapotranspiration, dtype=float)

        if precipitation.shape != evapotranspiration.shape:
            raise ValueError("Precipitation and evapotranspiration arrays must share the same shape.")

        n_steps = precipitation.size
        discharge = np.zeros(n_steps, dtype=float)
        surface = np.zeros_like(discharge)
        base = np.zeros_like(discharge)
        mean_deficit = np.zeros_like(discharge)

        for idx in range(n_steps):
            outputs = self.run_timestep(precipitation[idx], evapotranspiration[idx])
            discharge[idx] = outputs["Q"]
            surface[idx] = outputs["surface"]
            base[idx] = outputs["baseflow"]
            mean_deficit[idx] = outputs["mean_deficit"]

        return {
            "Q": discharge,
            "surface": surface,
            "baseflow": base,
            "mean_deficit": mean_deficit,
        }


def create_topmodel_plots(model, precipitation, evapotranspiration, results, 
                         title_prefix="TOPMODEL", save_dir="figures"):
    """
    Create comprehensive visualization plots for TOPMODEL analysis.
    
    Parameters
    ----------
    model : Topmodel
        The TOPMODEL instance used for simulation.
    precipitation : np.ndarray
        Precipitation time series (mm).
    evapotranspiration : np.ndarray
        Evapotranspiration time series (mm).
    results : dict
        Results dictionary from model.run() containing Q, surface, baseflow, mean_deficit.
    title_prefix : str, optional
        Prefix for plot titles (default: "TOPMODEL").
    save_dir : str, optional
        Directory to save the plots (default: "figures").
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    import os
    from datetime import datetime, timedelta
    
    # Configure matplotlib for Chinese font display / 配置matplotlib以显示中文
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'KaiTi', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display / 修复负号显示
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    n_days = len(precipitation)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Figure 1: Time series analysis
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'{title_prefix} 模型分析 / {title_prefix} Model Analysis', fontsize=16, fontweight='bold')
    
    # Precipitation (inverted)
    axes[0].bar(dates, precipitation, color='steelblue', alpha=0.7, width=1)
    axes[0].set_ylabel('降水量\nPrecipitation\n(mm/day)', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(max(precipitation) * 1.1, 0)
    axes[0].set_title('气象强迫数据 / Meteorological Forcing Data')
    
    # Evapotranspiration
    axes[1].plot(dates, evapotranspiration, color='orange', linewidth=2, label='蒸散发 / ET')
    axes[1].set_ylabel('蒸散发\nEvapotranspiration\n(mm/day)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Discharge components
    axes[2].plot(dates, results['Q'], label='总径流 / Total Runoff', color='blue', linewidth=2)
    axes[2].plot(dates, results['surface'], label='地表径流 / Surface Runoff', 
                color='red', linewidth=1.5, linestyle='--')
    axes[2].plot(dates, results['baseflow'], label='基流 / Baseflow', 
                color='green', linewidth=1.5, linestyle=':')
    axes[2].set_ylabel('径流量\nRunoff\n(mm/day)', fontweight='bold')
    axes[2].set_title('径流组分分离 / Runoff Component Separation')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Saturation deficit
    axes[3].plot(dates, results['mean_deficit'], color='brown', linewidth=2, 
                label='平均饱和亏缺 / Mean Saturation Deficit')
    axes[3].set_ylabel('饱和亏缺\nSaturation Deficit\n(mm)', fontweight='bold')
    axes[3].set_xlabel('日期 / Date', fontweight='bold')
    axes[3].set_title('流域平均饱和亏缺变化 / Mean Catchment Saturation Deficit')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'topmodel_timeseries_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Water balance and performance analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix} 水量平衡与性能分析 / Water Balance & Performance Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Water balance components (handle NaN values)
    total_precip = np.nansum(precipitation)
    total_et = np.nansum(evapotranspiration)
    total_runoff = np.nansum(results['Q'])
    total_surface = np.nansum(results['surface'])
    total_baseflow = np.nansum(results['baseflow'])
    
    water_balance = [total_precip, total_et, total_runoff]
    components = ['降水 / Precipitation', '蒸散发 / ET', '径流 / Runoff']
    colors = ['steelblue', 'orange', 'green']
    
    bars = axes[0,0].bar(components, water_balance, color=colors, alpha=0.7)
    axes[0,0].set_ylabel('水量 / Water Amount (mm)', fontweight='bold')
    axes[0,0].set_title('水量平衡 / Water Balance')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=15)
    
    # Add value labels on bars
    for bar, value in zip(bars, water_balance):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + max(water_balance)*0.01,
                      f'{value:.1f}mm', ha='center', va='bottom', fontweight='bold')
    
    # Runoff components - handle NaN and zero values properly
    surface_clean = 0.0 if np.isnan(total_surface) or total_surface < 0 else total_surface
    baseflow_clean = 0.0 if np.isnan(total_baseflow) or total_baseflow < 0 else total_baseflow
    
    runoff_components = [surface_clean, baseflow_clean]
    runoff_labels = ['地表径流 / Surface', '基流 / Baseflow']
    runoff_colors = ['red', 'green']
    
    # Check if we have valid data for pie chart
    total_components = sum(runoff_components)
    if total_components > 1e-6:  # Use a small threshold instead of zero
        # Filter out zero components
        non_zero_components = []
        non_zero_labels = []
        non_zero_colors = []
        for comp, label, color in zip(runoff_components, runoff_labels, runoff_colors):
            if comp > 1e-6:
                non_zero_components.append(comp)
                non_zero_labels.append(label)
                non_zero_colors.append(color)
        
        if len(non_zero_components) > 0:
            wedges, texts, autotexts = axes[0,1].pie(non_zero_components, labels=non_zero_labels, 
                                                    colors=non_zero_colors, autopct='%1.1f%%',
                                                    startangle=90)
        else:
            axes[0,1].text(0.5, 0.5, '无径流产生\nNo Runoff Generated', 
                          ha='center', va='center', transform=axes[0,1].transAxes,
                          fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
    else:
        # If no valid data, show text message
        axes[0,1].text(0.5, 0.5, '无有效径流数据\nNo Valid Runoff Data', 
                      ha='center', va='center', transform=axes[0,1].transAxes,
                      fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    axes[0,1].set_title('径流组分 / Runoff Components')
    
    # Topographic index distribution
    if hasattr(model, 'topographic_index') and model.topographic_index is not None:
        axes[1,0].hist(model.topographic_index, bins=15, alpha=0.7, color='purple', 
                      edgecolor='black')
        axes[1,0].set_xlabel('地形指数 / Topographic Index', fontweight='bold')
        axes[1,0].set_ylabel('频率 / Frequency', fontweight='bold')
        axes[1,0].set_title('地形指数分布 / Topographic Index Distribution')
        axes[1,0].grid(True, alpha=0.3)
        
        # Add statistics
        ti_mean = np.mean(model.topographic_index)
        ti_std = np.std(model.topographic_index)
        axes[1,0].axvline(ti_mean, color='red', linestyle='--', linewidth=2, 
                         label=f'平均值 / Mean: {ti_mean:.2f}')
        axes[1,0].legend()
    
    # Model parameters display (handle division by zero)
    runoff_coeff = total_runoff / max(total_precip, 1e-6) if total_precip > 0 else 0.0
    surface_ratio = total_surface / max(total_runoff, 1e-6) if total_runoff > 0 else 0.0
    base_ratio = total_baseflow / max(total_runoff, 1e-6) if total_runoff > 0 else 0.0
    
    params_text = f"""模型参数 / Model Parameters:
    
指数衰减参数 m: {model.config.m:.3f} m
饱和导水率: {model.config.transmissivity_sat:.1f} mm/day
初始平均亏缺: {model.config.initial_mean_deficit:.1f} mm

径流系数: {runoff_coeff:.3f}
地表径流比例: {surface_ratio:.3f}
基流比例: {base_ratio:.3f}"""
    
    axes[1,1].text(0.05, 0.95, params_text, transform=axes[1,1].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    axes[1,1].set_title('模型参数与统计 / Model Parameters & Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'topmodel_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Deficit dynamics and transmissivity
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'{title_prefix} 饱和亏缺动态与导水率分析 / Deficit Dynamics & Transmissivity Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Deficit variation over time
    axes[0].fill_between(dates, 0, results['mean_deficit'], 
                        color='brown', alpha=0.3, label='饱和亏缺 / Saturation Deficit')
    axes[0].plot(dates, results['mean_deficit'], color='brown', linewidth=2)
    axes[0].set_ylabel('饱和亏缺 / Deficit (mm)', fontweight='bold', color='brown')
    axes[0].set_title('流域平均饱和亏缺随时间变化 / Catchment Mean Deficit Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Overlay precipitation bars (inverted)
    ax0_twin = axes[0].twinx()
    ax0_twin.bar(dates, precipitation, color='steelblue', alpha=0.5, width=1)
    ax0_twin.set_ylabel('降水量 / Precipitation (mm)', fontweight='bold', color='steelblue')
    ax0_twin.invert_yaxis()
    ax0_twin.set_ylim(max(precipitation) * 1.5, 0)
    
    # Calculate and plot effective transmissivity over time
    transmissivity = model.config.transmissivity_sat * np.exp(-results['mean_deficit'] / 
                                                              max(model.config.m, 1e-6))
    
    axes[1].plot(dates, transmissivity, color='green', linewidth=2, 
                label='有效导水率 / Effective Transmissivity')
    axes[1].set_ylabel('导水率 / Transmissivity\n(mm/day)', fontweight='bold')
    axes[1].set_xlabel('日期 / Date', fontweight='bold')
    axes[1].set_title('有效导水率变化 / Effective Transmissivity Variation')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'topmodel_deficit_dynamics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ TOPMODEL 分析图表已保存到 '{save_dir}' 目录")
    print(f"✓ TOPMODEL analysis plots saved to '{save_dir}' directory")
    print("  - topmodel_timeseries_analysis.png: 时间序列分析")
    print("  - topmodel_analysis.png: 水量平衡与性能分析")  
    print("  - topmodel_deficit_dynamics.png: 饱和亏缺动态分析")


__all__ = ["Topmodel", "TopmodelConfig", "create_topmodel_plots"]
