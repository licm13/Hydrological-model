"""
SCS-CN (Soil Conservation Service Curve Number) + Unit Hydrograph Event Model
SCS-CN (土壤保持局曲线数) + 单位线事件模型

This module implements the SCS Curve Number method for runoff estimation
combined with Unit Hydrograph routing for event-based rainfall-runoff modeling.
It is particularly useful for teaching and analyzing individual storm events.

本模块实现了用于径流估算的SCS曲线数法，结合单位线汇流进行基于事件的降雨径流建模。
特别适用于教学和分析单个暴雨事件。

Mathematical Foundation / 数学基础:
=====================================

1. SCS Curve Number Method / SCS曲线数法:
   -----------------------------------------
   The SCS-CN method estimates direct runoff from rainfall using a curve number (CN)
   that represents the hydrologic soil-cover complex.
   
   SCS-CN法使用代表水文土壤-覆盖复合体的曲线数(CN)从降雨估算直接径流。
   
   Basic equations / 基本方程:
   
   Q = (P - Ia)² / (P - Ia + S)  for P > Ia
   Q = 0                          for P ≤ Ia
   
   Where / 其中:
   - Q: Direct runoff / 直接径流 (mm or inches)
   - P: Rainfall / 降雨 (mm or inches)  
   - Ia: Initial abstraction / 初损 = 0.2 * S (typical approximation / 典型近似)
   - S: Potential maximum retention / 潜在最大持留 = (25400/CN - 254) mm
     or S = (1000/CN - 10) inches

2. Curve Number (CN) / 曲线数 (CN):
   -----------------------------------
   CN ranges from 0 to 100 / CN范围从0到100:
   - Higher CN → More runoff (impervious surfaces) / 更高CN → 更多径流(不透水表面)
   - Lower CN → Less runoff (pervious surfaces) / 更低CN → 更少径流(可渗透表面)
   
   Typical CN values / 典型CN值:
   - Urban areas / 城区: 70-98
   - Agricultural land / 农田: 60-85
   - Forest / 森林: 30-70
   - Water bodies / 水体: 100

3. Antecedent Moisture Conditions (AMC) / 前期湿度条件 (AMC):
   ------------------------------------------------------------
   The SCS method adjusts CN based on antecedent moisture:
   SCS法根据前期湿度调整CN:
   
   - AMC I (Dry) / AMC I (干旱): CN(I) = CN / (2.281 - 0.01281*CN)
   - AMC II (Average) / AMC II (平均): CN(II) = CN (reference condition / 参考条件)
   - AMC III (Wet) / AMC III (湿润): CN(III) = CN / (0.427 + 0.00573*CN)

4. Unit Hydrograph / 单位线:
   ---------------------------
   A unit hydrograph represents the catchment response to 1 unit of effective rainfall
   uniformly distributed over the catchment during a unit time period.
   
   单位线表示流域对在单位时间段内均匀分布在流域上的1单位有效降雨的响应。
   
   Triangular Unit Hydrograph / 三角形单位线:
   
   Peak flow / 峰值流量: Qp = (0.208 * A) / Tp
   Time to peak / 峰现时间: Tp = Δt/2 + tlag
   Base time / 基流时间: Tb = 2.67 * Tp
   
   Where / 其中:
   - A: Catchment area / 流域面积 (km²)
   - Δt: Rainfall duration / 降雨历时 (hours)
   - tlag: Lag time / 滞时 (hours) = 0.6 * Tc
   - Tc: Time of concentration / 汇流时间 (hours)
   
   Snyder Unit Hydrograph / Snyder单位线:
   
   Tp = Ct * (L * Lca)^0.3
   Qp = Cp * A / Tp
   
   Where / 其中:
   - L: Length of main stream / 主河道长度 (km)
   - Lca: Distance to centroid / 到重心距离 (km)
   - Ct, Cp: Regional coefficients / 区域系数

5. Convolution / 卷积:
   --------------------
   Direct runoff hydrograph is obtained by convolving effective rainfall with UH:
   直接径流过程线通过有效降雨与单位线的卷积获得:
   
   Q(t) = Σ Pe(i) * UH(t - i + 1)
   
   Where / 其中:
   - Q(t): Direct runoff at time t / t时刻的直接径流
   - Pe(i): Effective rainfall at time i / i时刻的有效降雨
   - UH(t): Unit hydrograph ordinate / 单位线纵坐标

References / 参考文献:
========================
USDA-NRCS (2004). National Engineering Handbook, Part 630 Hydrology.
Chapter 10: Estimation of Direct Runoff from Storm Rainfall.

McCuen, R. H. (2004). Hydrologic Analysis and Design (3rd ed.). 
Prentice Hall.

Chow, V. T., Maidment, D. R., & Mays, L. W. (1988). 
Applied Hydrology. McGraw-Hill.

Author: Implementation for HydroLearn教学库
Date: 2024
"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os


class SCSCurveNumber:
    """
    SCS Curve Number method for estimating direct runoff from rainfall.
    SCS曲线数法用于从降雨估算直接径流。
    
    This class implements the SCS-CN method with support for different
    antecedent moisture conditions.
    本类实现了SCS-CN法，支持不同的前期湿度条件。
    
    Parameters / 参数:
    -------------------
    CN : float
        Curve Number under AMC II conditions / AMC II条件下的曲线数
        Range / 范围: 0-100
        Typical values / 典型值:
        - Urban impervious / 城市不透水: 98
        - Urban pervious / 城市可透水: 70-85
        - Agricultural / 农田: 60-80
        - Forest / 森林: 30-60
        - Water / 水体: 100
        
    AMC : Literal['I', 'II', 'III'], optional
        Antecedent Moisture Condition / 前期湿度条件
        - 'I': Dry / 干旱
        - 'II': Average (default) / 平均(默认)
        - 'III': Wet / 湿润
        Default: 'II'
    """
    
    def __init__(self, CN: float, AMC: Literal['I', 'II', 'III'] = 'II'):
        if not 0 <= CN <= 100:
            raise ValueError(f"CN must be between 0 and 100, got {CN}")
        
        self.CN_II = CN  # Reference CN under AMC II / AMC II下的参考CN
        self.AMC = AMC
        
        # Adjust CN based on AMC / 根据AMC调整CN
        self.CN = self._adjust_CN_for_AMC(CN, AMC)
        
        # Calculate potential maximum retention S (mm) / 计算潜在最大持留S(mm)
        # S = (25400/CN - 254) for metric units
        self.S = 25400.0 / self.CN - 254.0
        
        # Initial abstraction Ia (mm) / 初损Ia(mm)
        # Typically Ia = 0.2 * S / 通常Ia = 0.2 * S
        self.Ia = 0.2 * self.S
    
    def _adjust_CN_for_AMC(self, CN: float, AMC: str) -> float:
        """
        Adjust CN for antecedent moisture conditions.
        根据前期湿度条件调整CN。
        
        Parameters / 参数:
        -------------------
        CN : float
            Curve number under AMC II / AMC II下的曲线数
        AMC : str
            'I' (dry), 'II' (average), or 'III' (wet)
            
        Returns / 返回:
        ----------------
        float : Adjusted CN / 调整后的CN
        """
        if AMC == 'II':
            return CN
        elif AMC == 'I':
            # Dry conditions - reduce CN / 干旱条件-减小CN
            return CN / (2.281 - 0.01281 * CN)
        elif AMC == 'III':
            # Wet conditions - increase CN / 湿润条件-增大CN
            return CN / (0.427 + 0.00573 * CN)
        else:
            raise ValueError(f"AMC must be 'I', 'II', or 'III', got {AMC}")
    
    def calculate_runoff(self, P: float) -> float:
        """
        Calculate direct runoff for a given rainfall amount.
        计算给定降雨量的直接径流。
        
        Parameters / 参数:
        -------------------
        P : float
            Rainfall depth (mm) / 降雨深度(mm)
            
        Returns / 返回:
        ----------------
        float : Direct runoff depth (mm) / 直接径流深度(mm)
        """
        if P <= self.Ia:
            return 0.0
        else:
            # SCS runoff equation / SCS径流方程
            Q = (P - self.Ia) ** 2 / (P - self.Ia + self.S)
            return Q
    
    def calculate_runoff_timeseries(self, P: np.ndarray) -> np.ndarray:
        """
        Calculate direct runoff for a rainfall time series.
        计算降雨时间序列的直接径流。
        
        For event-based modeling, P should be cumulative rainfall.
        对于基于事件的建模，P应该是累积降雨。
        
        Parameters / 参数:
        -------------------
        P : np.ndarray
            Rainfall depths (mm) / 降雨深度(mm)
            
        Returns / 返回:
        ----------------
        np.ndarray : Direct runoff depths (mm) / 直接径流深度(mm)
        """
        Q = np.zeros_like(P)
        for i in range(len(P)):
            Q[i] = self.calculate_runoff(P[i])
        return Q


class UnitHydrograph:
    """
    Unit Hydrograph for routing effective rainfall to direct runoff hydrograph.
    单位线用于将有效降雨汇流为直接径流过程线。
    
    This class provides implementations of triangular and Snyder unit hydrographs.
    本类提供三角形和Snyder单位线的实现。
    """
    
    @staticmethod
    def triangular_uh(Tp: float, Tb: float, dt: float = 1.0) -> np.ndarray:
        """
        Create triangular unit hydrograph.
        创建三角形单位线。
        
        Parameters / 参数:
        -------------------
        Tp : float
            Time to peak (hours) / 峰现时间(小时)
        Tb : float
            Base time (hours) / 基流时间(小时)
        dt : float, optional
            Time step (hours) / 时间步长(小时)
            Default: 1.0
            
        Returns / 返回:
        ----------------
        np.ndarray : Unit hydrograph ordinates (1/hour) / 单位线纵坐标(1/小时)
        """
        # Number of ordinates / 纵坐标数量
        n = int(np.ceil(Tb / dt)) + 1
        t = np.arange(0, n * dt, dt)
        
        # Peak discharge for unit rainfall / 单位降雨的峰值流量
        # Area under triangle = 1 unit / 三角形下面积=1单位
        Qp = 2.0 / Tb
        
        # Triangular shape / 三角形形状
        UH = np.zeros(len(t))
        for i, ti in enumerate(t):
            if ti <= Tp:
                # Rising limb / 上升段
                UH[i] = Qp * (ti / Tp)
            elif ti <= Tb:
                # Falling limb / 下降段
                UH[i] = Qp * (Tb - ti) / (Tb - Tp)
            else:
                UH[i] = 0.0
        
        return UH
    
    @staticmethod
    def snyder_uh(A: float, L: float, Lca: float, Ct: float = 1.8,
                  Cp: float = 0.625, dt: float = 1.0) -> Tuple[np.ndarray, float, float]:
        """
        Create Snyder synthetic unit hydrograph.
        创建Snyder综合单位线。
        
        Parameters / 参数:
        -------------------
        A : float
            Catchment area (km²) / 流域面积(km²)
        L : float
            Length of main stream (km) / 主河道长度(km)
        Lca : float
            Distance along main stream to catchment centroid (km) / 
            沿主河道到流域重心的距离(km)
        Ct : float, optional
            Snyder coefficient (regional) / Snyder系数(区域性)
            Typical range / 典型范围: 1.8-2.2
            Default: 1.8
        Cp : float, optional
            Peak coefficient (regional) / 峰值系数(区域性)
            Typical range / 典型范围: 0.4-0.8
            Default: 0.625
        dt : float, optional
            Time step (hours) / 时间步长(小时)
            Default: 1.0
            
        Returns / 返回:
        ----------------
        Tuple containing / 包含的元组:
        - UH : np.ndarray - Unit hydrograph ordinates / 单位线纵坐标 (m³/s per mm)
        - Tp : float - Time to peak / 峰现时间 (hours)
        - Tb : float - Base time / 基流时间 (hours)
        """
        # Basin lag / 流域滞时
        tlag = Ct * (L * Lca) ** 0.3
        
        # Time to peak / 峰现时间
        Tp = tlag + dt / 2.0
        
        # Peak discharge per unit rainfall / 单位降雨的峰值流量
        # Qp in m³/s per mm of rainfall over area A
        Qp = (Cp * A) / Tp
        
        # Base time / 基流时间
        # Typically 3-5 times Tp / 通常为Tp的3-5倍
        Tb = 5.0 * Tp
        
        # Time array / 时间数组
        n = int(np.ceil(Tb / dt)) + 1
        t = np.arange(0, n * dt, dt)
        
        # Snyder UH shape / Snyder单位线形状
        # Uses empirical equations for rising and falling limbs
        # 使用上升段和下降段的经验方程
        UH = np.zeros(len(t))
        
        for i, ti in enumerate(t):
            if ti <= Tp:
                # Rising limb - parabolic / 上升段-抛物线
                UH[i] = Qp * (ti / Tp) ** 2.5
            else:
                # Falling limb - exponential decay / 下降段-指数衰减
                alpha = 0.5  # Recession coefficient / 消退系数
                UH[i] = Qp * np.exp(-alpha * (ti - Tp) / Tp)
        
        # Normalize to ensure unit volume / 归一化以确保单位体积
        # Convert to proper units / 转换到正确的单位
        volume = np.sum(UH) * dt * 3600  # m³ per mm
        expected_volume = A * 1000  # m³ per mm (A km² × 1000 m²/km² × 1 mm × 0.001 m/mm)
        UH = UH * expected_volume / volume
        
        return UH, Tp, Tb
    
    @staticmethod
    def dimensionless_uh(Tp: float, dt: float = 1.0) -> np.ndarray:
        """
        Create NRCS dimensionless unit hydrograph.
        创建NRCS无量纲单位线。
        
        This is a standard unit hydrograph shape recommended by NRCS.
        这是NRCS推荐的标准单位线形状。
        
        Parameters / 参数:
        -------------------
        Tp : float
            Time to peak (hours) / 峰现时间(小时)
        dt : float, optional
            Time step (hours) / 时间步长(小时)
            Default: 1.0
            
        Returns / 返回:
        ----------------
        np.ndarray : Unit hydrograph ordinates / 单位线纵坐标
        """
        # Dimensionless time ratios / 无量纲时间比
        # Based on SCS dimensionless UH / 基于SCS无量纲单位线
        t_ratios = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 
                            2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0])
        q_ratios = np.array([0, 0.10, 0.31, 0.66, 0.93, 1.00, 0.93, 0.78, 0.63, 0.51,
                            0.42, 0.35, 0.29, 0.24, 0.20, 0.17, 0.11, 0.07, 0.04, 0.01])
        
        # Convert to actual time / 转换为实际时间
        t_actual = t_ratios * Tp
        Tb = 5.0 * Tp  # Base time / 基流时间
        
        # Interpolate to specified time step / 插值到指定时间步长
        n = int(np.ceil(Tb / dt)) + 1
        t = np.arange(0, n * dt, dt)
        
        UH = np.interp(t, t_actual, q_ratios, left=0, right=0)
        
        # Peak discharge for unit rainfall / 单位降雨的峰值流量
        Qp = 1.0 / (0.5 * Tb)  # Approximate area normalization / 近似面积归一化
        UH = UH * Qp
        
        return UH


def convolve_rainfall_uh(Pe: np.ndarray, UH: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Convolve effective rainfall with unit hydrograph to produce direct runoff hydrograph.
    将有效降雨与单位线卷积以产生直接径流过程线。
    
    Parameters / 参数:
    -------------------
    Pe : np.ndarray
        Effective rainfall (mm/hour or mm per time step) / 
        有效降雨(mm/小时或每时间步mm)
    UH : np.ndarray
        Unit hydrograph ordinates (1/hour or 1/time step) / 
        单位线纵坐标(1/小时或1/时间步)
    dt : float, optional
        Time step (hours) / 时间步长(小时)
        Default: 1.0
        
    Returns / 返回:
    ----------------
    np.ndarray : Direct runoff hydrograph (mm/hour or mm per time step) / 
                 直接径流过程线(mm/小时或每时间步mm)
    """
    # Standard convolution / 标准卷积
    Q = np.convolve(Pe, UH * dt, mode='full')[:len(Pe)]
    return Q


class EventModel:
    """
    Complete event-based rainfall-runoff model using SCS-CN and Unit Hydrograph.
    使用SCS-CN和单位线的完整事件型降雨径流模型。
    
    This model combines the SCS Curve Number method for runoff estimation
    with Unit Hydrograph routing for hydrograph generation.
    本模型结合SCS曲线数法进行径流估算和单位线汇流进行过程线生成。
    
    Parameters / 参数:
    -------------------
    CN : float
        Curve Number / 曲线数 (0-100)
    AMC : Literal['I', 'II', 'III'], optional
        Antecedent Moisture Condition / 前期湿度条件
        Default: 'II'
    Tp : float, optional
        Time to peak for unit hydrograph (hours) / 单位线峰现时间(小时)
        Default: 2.0
    dt : float, optional
        Time step (hours) / 时间步长(小时)
        Default: 1.0
    uh_type : Literal['triangular', 'dimensionless'], optional
        Type of unit hydrograph / 单位线类型
        Default: 'triangular'
    """
    
    def __init__(self,
                 CN: float,
                 AMC: Literal['I', 'II', 'III'] = 'II',
                 Tp: float = 2.0,
                 dt: float = 1.0,
                 uh_type: Literal['triangular', 'dimensionless'] = 'triangular'):
        
        self.CN = CN
        self.AMC = AMC
        self.Tp = Tp
        self.dt = dt
        self.uh_type = uh_type
        
        # Initialize SCS-CN calculator / 初始化SCS-CN计算器
        self.scs = SCSCurveNumber(CN, AMC)
        
        # Create unit hydrograph / 创建单位线
        if uh_type == 'triangular':
            Tb = 2.67 * Tp  # Triangular UH base time / 三角形单位线基流时间
            self.UH = UnitHydrograph.triangular_uh(Tp, Tb, dt)
        elif uh_type == 'dimensionless':
            self.UH = UnitHydrograph.dimensionless_uh(Tp, dt)
        else:
            raise ValueError(f"uh_type must be 'triangular' or 'dimensionless', got {uh_type}")
    
    def run(self, P: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run event model for a rainfall time series.
        为降雨时间序列运行事件模型。
        
        Parameters / 参数:
        -------------------
        P : np.ndarray
            Rainfall time series (mm per time step) / 降雨时间序列(每时间步mm)
            
        Returns / 返回:
        ----------------
        dict : Dictionary containing / 包含的字典:
            - P : Rainfall / 降雨 (mm)
            - Q_direct : Direct runoff depth per time step / 每时间步直接径流深度 (mm)
            - Q_hydrograph : Direct runoff hydrograph / 直接径流过程线 (mm/hour)
            - Pe : Effective rainfall / 有效降雨 (mm)
            - cumulative_P : Cumulative rainfall / 累积降雨 (mm)
            - cumulative_Q : Cumulative runoff / 累积径流 (mm)
        """
        n_steps = len(P)
        
        # Calculate cumulative rainfall / 计算累积降雨
        P_cum = np.cumsum(P)
        
        # Calculate cumulative direct runoff using SCS-CN / 使用SCS-CN计算累积直接径流
        Q_direct_cum = np.array([self.scs.calculate_runoff(p) for p in P_cum])
        
        # Calculate incremental direct runoff / 计算增量直接径流
        Q_direct = np.diff(Q_direct_cum, prepend=0)
        
        # Effective rainfall (for UH convolution) / 有效降雨(用于单位线卷积)
        Pe = Q_direct.copy()
        
        # Convolve with unit hydrograph / 与单位线卷积
        Q_hydro = convolve_rainfall_uh(Pe, self.UH, self.dt)
        
        return {
            'P': P,                          # Rainfall / 降雨
            'Q_direct': Q_direct,            # Direct runoff increments / 直接径流增量
            'Q_hydrograph': Q_hydro,         # Routed hydrograph / 汇流后过程线
            'Pe': Pe,                        # Effective rainfall / 有效降雨
            'cumulative_P': P_cum,           # Cumulative rainfall / 累积降雨
            'cumulative_Q': Q_direct_cum,    # Cumulative runoff / 累积径流
            'UH': self.UH                    # Unit hydrograph / 单位线
        }


def create_event_plots(model: EventModel, results: Dict[str, np.ndarray], 
                       save_dir: str = "figures") -> None:
    """
    Create comprehensive visualization plots for event model results.
    为事件模型结果创建综合可视化图表。
    
    Parameters / 参数:
    -------------------
    model : EventModel
        The event model instance / 事件模型实例
    results : dict
        Model simulation results / 模型模拟结果
    save_dir : str, optional
        Directory to save figures / 保存图表的目录
    """
    # Ensure save directory exists / 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style / 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    
    n_steps = len(results['P'])
    t = np.arange(n_steps) * model.dt  # Time in hours / 时间(小时)
    
    # Figure 1: Complete event analysis / 完整事件分析
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle('SCS-CN + Unit Hydrograph Event Model Analysis\nSCS-CN + 单位线事件模型分析', 
                 fontsize=16, fontweight='bold')
    
    # Rainfall / 降雨
    axes[0].bar(t, results['P'], width=model.dt*0.8, color='steelblue', alpha=0.7, 
                label='Rainfall / 降雨')
    axes[0].set_ylabel('Rainfall (mm) / 降雨 (mm)', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(max(results['P']) * 1.2 if max(results['P']) > 0 else 1, 0)
    
    # Cumulative rainfall and runoff / 累积降雨和径流
    axes[1].plot(t, results['cumulative_P'], label='Cumulative Rainfall / 累积降雨', 
                 color='blue', linewidth=2.5)
    axes[1].plot(t, results['cumulative_Q'], label='Cumulative Direct Runoff / 累积直接径流', 
                 color='red', linewidth=2.5)
    axes[1].fill_between(t, results['cumulative_Q'], alpha=0.3, color='red')
    axes[1].set_ylabel('Cumulative Depth (mm) / 累积深度 (mm)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add text box with model parameters / 添加模型参数文本框
    info_text = f"CN={model.CN:.0f}, AMC={model.AMC}\n"
    info_text += f"S={model.scs.S:.1f} mm, Ia={model.scs.Ia:.1f} mm"
    axes[1].text(0.02, 0.98, info_text, transform=axes[1].transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 verticalalignment='top', fontsize=10, fontweight='bold')
    
    # Effective rainfall / 有效降雨
    axes[2].bar(t, results['Pe'], width=model.dt*0.8, color='green', alpha=0.7,
                label='Effective Rainfall / 有效降雨')
    axes[2].set_ylabel('Effective Rainfall (mm) / 有效降雨 (mm)', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Direct runoff hydrograph / 直接径流过程线
    axes[3].plot(t, results['Q_hydrograph'], color='red', linewidth=2.5,
                 label='Direct Runoff Hydrograph / 直接径流过程线')
    axes[3].fill_between(t, results['Q_hydrograph'], alpha=0.3, color='red')
    axes[3].set_ylabel('Discharge (mm/h) / 流量 (mm/h)', fontweight='bold')
    axes[3].set_xlabel('Time (hours) / 时间 (小时)', fontweight='bold')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Add peak discharge annotation / 添加峰值流量注释
    peak_Q = np.max(results['Q_hydrograph'])
    peak_t = t[np.argmax(results['Q_hydrograph'])]
    axes[3].annotate(f'Peak / 峰值: {peak_Q:.2f} mm/h\nat t={peak_t:.1f} h',
                     xy=(peak_t, peak_Q), xytext=(peak_t + 2, peak_Q * 0.8),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'event_model_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Unit Hydrograph / 单位线
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Unit Hydrograph / 单位线', fontsize=16, fontweight='bold')
    
    t_uh = np.arange(len(results['UH'])) * model.dt
    ax.plot(t_uh, results['UH'], color='blue', linewidth=2.5, marker='o', markersize=5)
    ax.fill_between(t_uh, results['UH'], alpha=0.3, color='blue')
    ax.set_xlabel('Time (hours) / 时间 (小时)', fontweight='bold')
    ax.set_ylabel('UH Ordinate (1/h) / 单位线纵坐标 (1/h)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add UH parameters / 添加单位线参数
    uh_text = f"Type / 类型: {model.uh_type}\n"
    uh_text += f"Tp = {model.Tp:.1f} h\n"
    uh_text += f"dt = {model.dt:.1f} h"
    ax.text(0.70, 0.95, uh_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='top', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'event_model_unit_hydrograph.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Event model visualization plots saved to '{save_dir}' directory")
    print(f"  ✓ 事件模型可视化图表已保存到'{save_dir}'目录")
    print("  Generated files / 生成的文件:")
    print("  - event_model_analysis.png")
    print("  - event_model_unit_hydrograph.png")


def main():
    """
    Example usage of SCS-CN + Unit Hydrograph event model.
    SCS-CN + 单位线事件模型的示例用法。
    
    This demonstration shows how to use the event model for analyzing
    a storm event, including runoff estimation and hydrograph generation.
    本演示展示如何使用事件模型分析暴雨事件，包括径流估算和过程线生成。
    """
    print("=" * 80)
    print("SCS-CN + Unit Hydrograph Event Model - Demonstration")
    print("SCS-CN + 单位线事件模型 - 演示")
    print("=" * 80)
    
    # Set random seed for reproducibility / 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # Create synthetic storm event / 创建合成暴雨事件
    # Storm duration: 12 hours / 暴雨持续时间: 12小时
    # Time step: 1 hour / 时间步长: 1小时
    
    n_hours = 30  # Total simulation time / 总模拟时间
    dt = 1.0  # Time step in hours / 时间步长(小时)
    
    # Create rainfall pattern / 创建降雨模式
    # Peak rainfall in the middle of the storm / 暴雨中部峰值降雨
    P = np.zeros(n_hours)
    
    # Storm event from hour 5 to hour 15 / 暴雨事件从第5小时到第15小时
    storm_start = 5
    storm_end = 15
    storm_duration = storm_end - storm_start
    
    # Create triangular rainfall pattern / 创建三角形降雨模式
    for i in range(storm_start, storm_end):
        if i < (storm_start + storm_end) / 2:
            # Rising intensity / 上升强度
            P[i] = 5 + 15 * (i - storm_start) / (storm_duration / 2)
        else:
            # Falling intensity / 下降强度
            P[i] = 20 - 15 * (i - (storm_start + storm_end) / 2) / (storm_duration / 2)
    
    # Add some variability / 添加一些变化
    for i in range(storm_start, storm_end):
        P[i] += np.random.uniform(-2, 2)
        P[i] = max(0, P[i])  # Ensure non-negative / 确保非负
    
    total_rainfall = np.sum(P)
    
    print("\nStorm Event Characteristics / 暴雨事件特征:")
    print(f"  Total duration / 总持续时间: {n_hours} hours / 小时")
    print(f"  Storm duration / 暴雨持续时间: {storm_duration} hours / 小时")
    print(f"  Time step / 时间步长: {dt} hour / 小时")
    print(f"  Total rainfall / 总降雨: {total_rainfall:.2f} mm")
    print(f"  Peak intensity / 峰值强度: {np.max(P):.2f} mm/h")
    print(f"  Average intensity (during storm) / 平均强度(暴雨期间): "
          f"{total_rainfall/storm_duration:.2f} mm/h")
    
    # Model parameters / 模型参数
    # Example: Agricultural watershed with moderate permeability
    # 示例: 中等渗透性的农业流域
    CN = 75  # Curve number / 曲线数
    AMC = 'II'  # Average antecedent moisture / 平均前期湿度
    Tp = 3.0  # Time to peak (hours) / 峰现时间(小时)
    
    print("\nModel Parameters / 模型参数:")
    print(f"  Curve Number (CN) / 曲线数: {CN}")
    print(f"  Antecedent Moisture Condition (AMC) / 前期湿度条件: {AMC}")
    print(f"  Time to Peak (Tp) / 峰现时间: {Tp} hours / 小时")
    print(f"  Unit Hydrograph Type / 单位线类型: Triangular / 三角形")
    
    # Initialize and run model / 初始化并运行模型
    print(f"\nRunning SCS-CN + UH event model... / 正在运行SCS-CN + UH事件模型...")
    model = EventModel(CN=CN, AMC=AMC, Tp=Tp, dt=dt, uh_type='triangular')
    results = model.run(P)
    
    # Display results / 显示结果
    print("\nSimulation Results / 模拟结果:")
    print(f"  SCS-CN Method / SCS-CN法:")
    print(f"    Maximum retention (S) / 最大持留量: {model.scs.S:.2f} mm")
    print(f"    Initial abstraction (Ia) / 初损: {model.scs.Ia:.2f} mm")
    print(f"    Adjusted CN for {AMC} / {AMC}条件下调整的CN: {model.scs.CN:.2f}")
    
    print(f"\n  Runoff Generation / 径流产生:")
    print(f"    Total rainfall / 总降雨: {total_rainfall:.2f} mm")
    print(f"    Total direct runoff / 总直接径流: {results['cumulative_Q'][-1]:.2f} mm")
    print(f"    Runoff coefficient / 径流系数: {results['cumulative_Q'][-1]/total_rainfall:.3f}")
    print(f"    Abstractions (infiltration + initial) / 损失(入渗+初损): "
          f"{total_rainfall - results['cumulative_Q'][-1]:.2f} mm")
    
    print(f"\n  Hydrograph Characteristics / 过程线特征:")
    peak_Q = np.max(results['Q_hydrograph'])
    peak_time = np.argmax(results['Q_hydrograph']) * dt
    print(f"    Peak discharge / 峰值流量: {peak_Q:.3f} mm/h")
    print(f"    Time to peak / 峰现时间: {peak_time:.1f} hours / 小时")
    print(f"    Peak occurs after storm center / 峰值出现在暴雨中心之后: "
          f"{peak_time - (storm_start + storm_end)/2:.1f} hours / 小时")
    
    # Generate comprehensive visualizations / 生成综合可视化
    print(f"\nGenerating comprehensive visualizations... / 正在生成综合可视化...")
    create_event_plots(model, results, save_dir="figures")
    
    # Demonstrate AMC sensitivity / 演示AMC敏感性
    print("\n" + "=" * 80)
    print("AMC Sensitivity Analysis / AMC敏感性分析")
    print("=" * 80)
    print("\nComparing runoff under different antecedent moisture conditions:")
    print("比较不同前期湿度条件下的径流:")
    print(f"\n{'AMC':<10} {'Adjusted CN':<15} {'Total Runoff (mm)':<20} {'Runoff Coef':<15}")
    print("-" * 60)
    
    for amc in ['I', 'II', 'III']:
        model_amc = EventModel(CN=CN, AMC=amc, Tp=Tp, dt=dt)
        results_amc = model_amc.run(P)
        total_runoff = results_amc['cumulative_Q'][-1]
        runoff_coef = total_runoff / total_rainfall
        print(f"{amc:<10} {model_amc.scs.CN:<15.2f} {total_runoff:<20.2f} {runoff_coef:<15.3f}")
    
    print("\nObservations / 观察:")
    print("  • AMC I (dry) / AMC I(干旱): Lower CN → Less runoff / 更低CN → 更少径流")
    print("  • AMC II (average) / AMC II(平均): Reference condition / 参考条件")
    print("  • AMC III (wet) / AMC III(湿润): Higher CN → More runoff / 更高CN → 更多径流")
    
    # Demonstrate CN sensitivity / 演示CN敏感性
    print("\n" + "=" * 80)
    print("Curve Number (CN) Sensitivity Analysis / 曲线数(CN)敏感性分析")
    print("=" * 80)
    print("\nComparing runoff for different land use / soil conditions:")
    print("比较不同土地利用/土壤条件下的径流:")
    print(f"\n{'CN':<10} {'Land Use Example':<30} {'Total Runoff (mm)':<20} {'Runoff Coef':<15}")
    print("-" * 75)
    
    cn_scenarios = [
        (55, "Forest, good cover / 森林,良好覆盖"),
        (70, "Agricultural, average / 农田,平均"),
        (85, "Residential, 38% impervious / 住宅,38%不透水"),
        (95, "Urban, 85% impervious / 城市,85%不透水")
    ]
    
    for cn, description in cn_scenarios:
        model_cn = EventModel(CN=cn, AMC='II', Tp=Tp, dt=dt)
        results_cn = model_cn.run(P)
        total_runoff = results_cn['cumulative_Q'][-1]
        runoff_coef = total_runoff / total_rainfall
        print(f"{cn:<10} {description:<30} {total_runoff:<20.2f} {runoff_coef:<15.3f}")
    
    print("\n" + "=" * 80)
    print("Event model simulation completed successfully!")
    print("事件模型模拟成功完成!")
    print("=" * 80)
    print("\nModel Features Demonstrated / 演示的模型特性:")
    print("  ✓ SCS Curve Number runoff estimation / SCS曲线数径流估算")
    print("  ✓ Unit hydrograph routing / 单位线汇流")
    print("  ✓ Antecedent moisture condition adjustments / 前期湿度条件调整")
    print("  ✓ Event-based hydrograph generation / 事件型过程线生成")
    print("  ✓ Suitable for storm event analysis / 适用于暴雨事件分析")
    print("\nPedagogical Value / 教学价值:")
    print("  • Demonstrates classic event-based modeling / 演示经典事件型建模")
    print("  • Shows impact of land use on runoff / 显示土地利用对径流的影响")
    print("  • Illustrates hydrograph timing concepts / 说明过程线时间概念")
    print("  • Easy to apply to design storms / 易于应用于设计暴雨")
    print("\nCheck the 'figures' directory for visualizations.")
    print("请查看'figures'目录中的可视化图表。")
    print("=" * 80)
    
    return results, model


if __name__ == "__main__":
    main()
