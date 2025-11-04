"""
SCS Curve Number (CN) Method with Triangular Unit Hydrograph
SCS曲线数法与三角形单位线

This module implements the SCS Curve Number method for rainfall-runoff estimation
combined with a triangular unit hydrograph for flood event simulation. This is a
simplified, educational implementation suitable for teaching event-based hydrology.

该模块实现了用于降雨-径流估算的SCS曲线数法，并结合三角形单位线进行洪水事件模拟。
这是一个简化的教学实现，适合用于教授事件水文学。

Mathematical Foundation / 数学基础:
=======================

1. SCS Curve Number Method / SCS曲线数法:
   
   The SCS-CN method estimates direct runoff from rainfall based on land use,
   soil type, and antecedent moisture conditions.
   
   SCS-CN法基于土地利用、土壤类型和前期土壤水分条件估算降雨产生的直接径流。
   
   Potential maximum retention / 潜在最大滞留量:
   S = (25400 / CN) - 254  (mm)
   
   Initial abstraction / 初损:
   Ia = 0.2 * S  (mm)
   
   Direct runoff / 直接径流:
   Q = 0                           if P <= Ia
   Q = (P - Ia)² / (P - Ia + S)   if P > Ia
   
   Where / 其中:
   - CN: Curve Number (0-100) / 曲线数 (0-100)
   - P: Total rainfall depth (mm) / 总降雨深度 (毫米)
   - S: Potential maximum retention (mm) / 潜在最大滞留量 (毫米)
   - Ia: Initial abstraction (mm) / 初损 (毫米)
   - Q: Direct runoff depth (mm) / 直接径流深度 (毫米)

2. Triangular Unit Hydrograph / 三角形单位线:
   
   A triangular unit hydrograph (UH) is a simplified representation of the
   catchment response to unit rainfall excess.
   
   三角形单位线是流域对单位净雨响应的简化表示。
   
   Peak flow / 峰值流量:
   Qp = 0.208 * A / Tp  (m³/s per mm of runoff)
   
   Time to peak / 峰现时间:
   Tp = Dt/2 + 0.6 * Tc
   
   Base time / 基底时间:
   Tb = 2.67 * Tp
   
   Where / 其中:
   - Qp: Peak discharge per unit runoff (m³/s/mm) / 单位径流的峰值流量
   - A: Catchment area (km²) / 流域面积 (平方公里)
   - Tp: Time to peak (hours) / 峰现时间 (小时)
   - Dt: Duration of rainfall excess (hours) / 净雨历时 (小时)
   - Tc: Time of concentration (hours) / 汇流时间 (小时)
   - Tb: Base time (hours) / 基底时间 (小时)
   
   The triangular UH has three segments:
   - Rising limb: 0 to Tp (linear increase)
   - Falling limb: Tp to Tb (linear decrease)
   
   三角形单位线有三个部分：
   - 上升段：0到Tp（线性增加）
   - 下降段：Tp到Tb（线性下降）

References / 参考文献:
-----------
USDA Soil Conservation Service (1986). Urban Hydrology for Small Watersheds.
Technical Release 55 (TR-55).

Chow, V. T., Maidment, D. R., & Mays, L. W. (1988). Applied Hydrology.
McGraw-Hill.

Author: Educational implementation for HydroLearn
Date: 2024
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from scipy import signal


class SCS_CN_Model:
    """
    SCS Curve Number model for direct runoff estimation.
    SCS曲线数模型用于直接径流估算。
    
    Parameters / 参数:
    -----------
    CN : float
        Curve Number (0-100), represents watershed characteristics
        曲线数 (0-100)，代表流域特征
        - CN=30-50: Permeable soils, good cover / 透水性土壤，良好覆盖
        - CN=50-70: Moderate conditions / 中等条件
        - CN=70-85: Less permeable soils / 较低透水性土壤
        - CN=85-98: Impervious or nearly impervious / 不透水或接近不透水
    Ia_ratio : float, optional
        Initial abstraction ratio (Ia/S), default=0.2
        初损比 (Ia/S)，默认值=0.2
    """
    
    def __init__(self, CN: float = 75.0, Ia_ratio: float = 0.2):
        """Initialize SCS-CN model / 初始化SCS-CN模型"""
        if not 0 < CN <= 100:
            raise ValueError("CN must be between 0 (exclusive) and 100 (inclusive) / CN必须在0到100之间（0不包括）")
        
        self.CN = CN
        self.Ia_ratio = Ia_ratio
        
        # Calculate potential maximum retention S (mm) / 计算潜在最大滞留量S（毫米）
        self.S = (25400.0 / CN) - 254.0
        
        # Calculate initial abstraction Ia (mm) / 计算初损Ia（毫米）
        self.Ia = self.Ia_ratio * self.S
        
    def calculate_runoff(self, P: float) -> float:
        """
        Calculate direct runoff from rainfall using SCS-CN method.
        使用SCS-CN法计算降雨产生的直接径流。
        
        Parameters / 参数:
        -----------
        P : float
            Total rainfall depth (mm) / 总降雨深度（毫米）
            
        Returns / 返回:
        --------
        Q : float
            Direct runoff depth (mm) / 直接径流深度（毫米）
        """
        if P <= self.Ia:
            # Rainfall does not exceed initial abstraction / 降雨未超过初损
            return 0.0
        else:
            # Calculate direct runoff / 计算直接径流
            Q = ((P - self.Ia) ** 2) / (P - self.Ia + self.S)
            return Q
    
    def calculate_runoff_series(self, P: np.ndarray) -> np.ndarray:
        """
        Calculate direct runoff for a rainfall series.
        为降雨序列计算直接径流。
        
        Parameters / 参数:
        -----------
        P : np.ndarray
            Rainfall depth series (mm) / 降雨深度序列（毫米）
            
        Returns / 返回:
        --------
        Q : np.ndarray
            Direct runoff series (mm) / 直接径流序列（毫米）
        """
        return np.array([self.calculate_runoff(p) for p in P])


class TriangularUnitHydrograph:
    """
    Triangular Unit Hydrograph for routing rainfall excess to streamflow.
    三角形单位线用于将净雨转化为河道流量。
    
    Parameters / 参数:
    -----------
    area : float
        Catchment area (km²) / 流域面积（平方公里）
    Tc : float
        Time of concentration (hours) / 汇流时间（小时）
    dt : float
        Time step (hours), default=1.0
        时间步长（小时），默认值=1.0
    """
    
    def __init__(self, area: float, Tc: float, dt: float = 1.0):
        """Initialize Triangular Unit Hydrograph / 初始化三角形单位线"""
        if area <= 0:
            raise ValueError("Area must be positive / 面积必须为正")
        if Tc <= 0:
            raise ValueError("Time of concentration must be positive / 汇流时间必须为正")
        if dt <= 0:
            raise ValueError("Time step must be positive / 时间步长必须为正")
            
        self.area = area  # km²
        self.Tc = Tc      # hours
        self.dt = dt      # hours
        
        # Calculate time to peak / 计算峰现时间
        # Tp = Dt/2 + 0.6*Tc, where Dt ≈ dt for discrete rainfall
        self.Tp = self.dt / 2.0 + 0.6 * self.Tc
        
        # Calculate base time / 计算基底时间
        self.Tb = 2.67 * self.Tp
        
        # Calculate peak discharge per unit runoff (m³/s per mm) / 计算单位径流的峰值流量
        # Qp = 0.208 * A / Tp
        self.Qp = 0.208 * self.area / self.Tp
        
        # Generate unit hydrograph ordinates / 生成单位线纵标值
        self.uh_ordinates = self._generate_unit_hydrograph()
        
    def _generate_unit_hydrograph(self) -> np.ndarray:
        """
        Generate triangular unit hydrograph ordinates.
        生成三角形单位线纵标值。
        
        Returns / 返回:
        --------
        uh : np.ndarray
            Unit hydrograph ordinates (m³/s per mm of runoff)
            单位线纵标值（单位径流毫米对应的立方米/秒）
        """
        # Number of time steps / 时间步数
        n_steps = int(np.ceil(self.Tb / self.dt)) + 1
        
        # Time array / 时间数组
        t = np.arange(n_steps) * self.dt
        
        # Initialize unit hydrograph / 初始化单位线
        uh = np.zeros(n_steps)
        
        for i, ti in enumerate(t):
            if ti <= self.Tp:
                # Rising limb / 上升段
                uh[i] = self.Qp * (ti / self.Tp)
            elif ti <= self.Tb:
                # Falling limb / 下降段
                uh[i] = self.Qp * (self.Tb - ti) / (self.Tb - self.Tp)
            else:
                # After base time / 基底时间之后
                uh[i] = 0.0
                
        return uh
    
    def convolve(self, runoff: np.ndarray) -> np.ndarray:
        """
        Convolve runoff with unit hydrograph to generate streamflow.
        将径流与单位线卷积以生成河道流量。
        
        Parameters / 参数:
        -----------
        runoff : np.ndarray
            Direct runoff series (mm per time step) / 直接径流序列（毫米/时间步）
            
        Returns / 返回:
        --------
        streamflow : np.ndarray
            Streamflow hydrograph (m³/s) / 河道流量过程（立方米/秒）
        """
        # Convolve runoff with unit hydrograph / 径流与单位线卷积
        streamflow = signal.convolve(runoff, self.uh_ordinates, mode='full')
        
        # Truncate to original length plus base time / 截断到原始长度加基底时间
        n_output = len(runoff) + len(self.uh_ordinates) - 1
        
        return streamflow[:n_output]


class SCS_UH_EventModel:
    """
    Combined SCS-CN and Triangular Unit Hydrograph event model.
    组合的SCS-CN和三角形单位线事件模型。
    
    This model combines the SCS Curve Number method for runoff generation
    with a triangular unit hydrograph for routing.
    
    该模型结合SCS曲线数法进行产流计算和三角形单位线进行汇流计算。
    
    Parameters / 参数:
    -----------
    CN : float
        SCS Curve Number (0-100) / SCS曲线数 (0-100)
    area : float
        Catchment area (km²) / 流域面积（平方公里）
    Tc : float
        Time of concentration (hours) / 汇流时间（小时）
    dt : float
        Time step (hours), default=1.0 / 时间步长（小时），默认值=1.0
    Ia_ratio : float
        Initial abstraction ratio, default=0.2 / 初损比，默认值=0.2
    """
    
    def __init__(
        self,
        CN: float = 75.0,
        area: float = 100.0,
        Tc: float = 3.0,
        dt: float = 1.0,
        Ia_ratio: float = 0.2
    ):
        """Initialize combined SCS-CN + UH model / 初始化组合的SCS-CN+UH模型"""
        # Initialize SCS-CN model / 初始化SCS-CN模型
        self.cn_model = SCS_CN_Model(CN=CN, Ia_ratio=Ia_ratio)
        
        # Initialize Unit Hydrograph / 初始化单位线
        self.uh_model = TriangularUnitHydrograph(area=area, Tc=Tc, dt=dt)
        
        # Store parameters / 存储参数
        self.CN = CN
        self.area = area
        self.Tc = Tc
        self.dt = dt
        
    def run(self, rainfall: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run the event model: rainfall -> runoff -> streamflow.
        运行事件模型：降雨 -> 径流 -> 河道流量。
        
        Parameters / 参数:
        -----------
        rainfall : np.ndarray
            Rainfall time series (mm per time step) / 降雨时间序列（毫米/时间步）
            
        Returns / 返回:
        --------
        results : dict
            Dictionary containing:
            包含以下内容的字典：
            - 'rainfall': input rainfall (mm) / 输入降雨（毫米）
            - 'runoff': direct runoff (mm) / 直接径流（毫米）
            - 'streamflow': streamflow hydrograph (m³/s) / 河道流量过程（立方米/秒）
            - 'time': time array (hours) / 时间数组（小时）
        """
        # Calculate direct runoff from rainfall / 从降雨计算直接径流
        runoff = self.cn_model.calculate_runoff_series(rainfall)
        
        # Route runoff through unit hydrograph / 通过单位线汇流径流
        streamflow = self.uh_model.convolve(runoff)
        
        # Create time arrays / 创建时间数组
        n_rainfall = len(rainfall)
        n_streamflow = len(streamflow)
        time_rainfall = np.arange(n_rainfall) * self.dt
        time_streamflow = np.arange(n_streamflow) * self.dt
        
        return {
            'rainfall': rainfall,
            'runoff': runoff,
            'streamflow': streamflow,
            'time_rainfall': time_rainfall,
            'time_streamflow': time_streamflow
        }


def calculate_peak_flow(streamflow: np.ndarray) -> Tuple[float, int]:
    """
    Find peak flow and its time index.
    查找峰值流量及其时间索引。
    
    Parameters / 参数:
    -----------
    streamflow : np.ndarray
        Streamflow series (m³/s) / 流量序列（立方米/秒）
        
    Returns / 返回:
    --------
    peak_flow : float
        Peak discharge (m³/s) / 峰值流量（立方米/秒）
    peak_index : int
        Index of peak flow / 峰值流量的索引
    """
    peak_index = np.argmax(streamflow)
    peak_flow = streamflow[peak_index]
    return peak_flow, peak_index


# Demo section for educational purposes / 教学演示部分
if __name__ == '__main__':
    """
    Educational demo: SCS-CN + UH event model with synthetic storm.
    教学演示：使用合成暴雨的SCS-CN+UH事件模型。
    """
    print("=" * 70)
    print("SCS-CN + Unit Hydrograph Event Model Demo")
    print("SCS-CN + 单位线事件模型演示")
    print("=" * 70)
    
    # Set random seed for reproducibility / 设置随机种子以确保可重现性
    np.random.seed(42)
    
    # Define catchment parameters / 定义流域参数
    CN = 75.0          # Curve Number / 曲线数
    area = 50.0        # km² / 平方公里
    Tc = 4.0           # hours / 小时
    dt = 1.0           # hours / 小时
    
    print(f"\nCatchment Parameters / 流域参数:")
    print(f"  Curve Number (CN) / 曲线数: {CN}")
    print(f"  Area / 面积: {area} km²")
    print(f"  Time of Concentration (Tc) / 汇流时间: {Tc} hours")
    print(f"  Time Step (dt) / 时间步长: {dt} hour")
    
    # Create synthetic storm event (24 hours) / 创建合成暴雨事件（24小时）
    n_hours = 24
    rainfall = np.zeros(n_hours)
    
    # Concentrated rainfall in hours 6-12 / 集中降雨在6-12小时
    # Using a bell-shaped pattern / 使用钟形模式
    storm_start = 6
    storm_duration = 6
    peak_hour = storm_start + storm_duration // 2
    
    for i in range(storm_start, storm_start + storm_duration):
        # Create bell-shaped storm / 创建钟形暴雨
        distance_from_peak = abs(i - peak_hour)
        rainfall[i] = 20.0 * np.exp(-0.5 * (distance_from_peak / 1.5) ** 2)
    
    print(f"\nStorm Characteristics / 暴雨特征:")
    print(f"  Total Rainfall / 总降雨量: {np.sum(rainfall):.1f} mm")
    print(f"  Peak Intensity / 峰值强度: {np.max(rainfall):.1f} mm/hour")
    print(f"  Storm Duration / 暴雨历时: {storm_duration} hours")
    
    # Initialize and run model / 初始化并运行模型
    print("\nInitializing SCS-CN + UH model...")
    print("初始化SCS-CN + UH模型...")
    
    model = SCS_UH_EventModel(CN=CN, area=area, Tc=Tc, dt=dt)
    
    print("\nRunning event simulation...")
    print("运行事件模拟...")
    results = model.run(rainfall)
    
    # Extract results / 提取结果
    runoff = results['runoff']
    streamflow = results['streamflow']
    
    # Calculate statistics / 计算统计量
    total_runoff = np.sum(runoff)
    runoff_coef = total_runoff / np.sum(rainfall) if np.sum(rainfall) > 0 else 0
    peak_flow, peak_time_idx = calculate_peak_flow(streamflow)
    peak_time = peak_time_idx * dt
    
    print(f"\nSimulation Results / 模拟结果:")
    print(f"  Total Runoff / 总径流量: {total_runoff:.1f} mm")
    print(f"  Runoff Coefficient / 径流系数: {runoff_coef:.2f}")
    print(f"  Peak Flow / 峰值流量: {peak_flow:.2f} m³/s")
    print(f"  Time to Peak / 峰现时间: {peak_time:.1f} hours")
    
    # Create visualization / 创建可视化图表
    print("\nGenerating visualization...")
    print("生成可视化图表...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('SCS-CN + UH Event Model Simulation / SCS-CN + UH事件模型模拟', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Rainfall hyetograph / 降雨柱状图
    ax1 = axes[0]
    time_rain = results['time_rainfall']
    ax1.bar(time_rain, rainfall, width=dt*0.8, alpha=0.7, color='blue', 
            edgecolor='darkblue', linewidth=1.5)
    ax1.set_ylabel('Rainfall (mm/hour)\n降雨量 (毫米/小时)', fontsize=11)
    ax1.set_xlim(0, n_hours)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('Rainfall Hyetograph / 降雨过程线', fontsize=12)
    
    # Add text box with storm info / 添加暴雨信息文本框
    textstr = f'Total: {np.sum(rainfall):.1f} mm\nPeak: {np.max(rainfall):.1f} mm/h'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.98, 0.97, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Plot 2: Direct runoff / 直接径流
    ax2 = axes[1]
    ax2.bar(time_rain, runoff, width=dt*0.8, alpha=0.7, color='green', 
            edgecolor='darkgreen', linewidth=1.5)
    ax2.set_ylabel('Direct Runoff (mm/hour)\n直接径流 (毫米/小时)', fontsize=11)
    ax2.set_xlim(0, n_hours)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('Direct Runoff (SCS-CN Method) / 直接径流 (SCS-CN法)', fontsize=12)
    
    # Add text box with runoff info / 添加径流信息文本框
    textstr = f'Total: {total_runoff:.1f} mm\nCoeff: {runoff_coef:.2f}\nCN: {CN:.0f}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Plot 3: Streamflow hydrograph / 河道流量过程线
    ax3 = axes[2]
    time_flow = results['time_streamflow']
    ax3.plot(time_flow, streamflow, linewidth=2.5, color='red', label='Streamflow / 流量')
    ax3.fill_between(time_flow, 0, streamflow, alpha=0.3, color='red')
    ax3.scatter([peak_time], [peak_flow], color='darkred', s=100, zorder=5, 
                label=f'Peak / 峰值: {peak_flow:.1f} m³/s')
    ax3.axvline(x=peak_time, color='darkred', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Streamflow (m³/s)\n流量 (立方米/秒)', fontsize=11)
    ax3.set_xlabel('Time (hours) / 时间 (小时)', fontsize=11)
    ax3.set_xlim(0, max(time_flow))
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_title('Streamflow Hydrograph (Triangular UH) / 流量过程线 (三角形UH)', fontsize=12)
    
    # Add text box with flow info / 添加流量信息文本框
    textstr = f'Peak: {peak_flow:.1f} m³/s\nTime: {peak_time:.1f} h\nTc: {Tc:.1f} h'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax3.text(0.98, 0.97, textstr, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # Save figure / 保存图表
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'scs_uh_event_model_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to / 图表已保存至: {output_path}")
    
    # Demonstrate effect of different CN values / 演示不同CN值的效果
    print("\n" + "=" * 70)
    print("Sensitivity Analysis: Effect of Curve Number")
    print("敏感性分析：曲线数的影响")
    print("=" * 70)
    
    CN_values = [60, 75, 90]
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for cn_val in CN_values:
        model_test = SCS_UH_EventModel(CN=cn_val, area=area, Tc=Tc, dt=dt)
        results_test = model_test.run(rainfall)
        Q_test = results_test['streamflow']
        time_test = results_test['time_streamflow']
        
        total_runoff_test = np.sum(results_test['runoff'])
        peak_test = np.max(Q_test)
        
        ax.plot(time_test, Q_test, linewidth=2, label=f'CN={cn_val:.0f} (Peak={peak_test:.1f} m³/s)')
        
        print(f"\nCN = {cn_val}:")
        print(f"  Total Runoff / 总径流: {total_runoff_test:.1f} mm")
        print(f"  Peak Flow / 峰值流量: {peak_test:.1f} m³/s")
    
    ax.set_xlabel('Time (hours) / 时间 (小时)', fontsize=11)
    ax.set_ylabel('Streamflow (m³/s) / 流量 (立方米/秒)', fontsize=11)
    ax.set_title('Effect of Curve Number on Hydrograph / 曲线数对过程线的影响', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_path2 = os.path.join(output_dir, 'scs_uh_sensitivity_cn.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"\nSensitivity figure saved to / 敏感性图表已保存至: {output_path2}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully! / 演示成功完成！")
    print("=" * 70)
