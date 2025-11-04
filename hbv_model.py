"""
HBV (Hydrologiska Byråns Vattenbalansavdelning) Hydrological Model
HBV (瑞典水文局水量平衡模型)

The HBV model is a conceptual rainfall-runoff model developed by the Swedish
Meteorological and Hydrological Institute (SMHI) in the 1970s. It has been widely
used for operational streamflow forecasting, water resources assessment, and
climate change impact studies worldwide.
HBV模型是瑞典气象水文研究所(SMHI)在20世纪70年代开发的概念性降雨径流模型。
它已被广泛应用于业务化径流预报、水资源评估和气候变化影响研究。

Mathematical Foundation / 数学基础:
===================================

The HBV model consists of several routines / HBV模型由以下几个模块组成:

1. Snow Routine / 积雪模块:
   Degree-day method for snow accumulation and melt / 采用度日法计算积雪累积和融化
   
   Snow accumulation / 积雪累积:
   If T < TT: SNOW = SNOW + P (precipitation falls as snow / 降水以雪的形式累积)
   
   Snowmelt / 融雪:
   If T > TT: MELT = CFMAX * (T - TT)  (degree-day factor / 度日因子法)
   
   Liquid water in snow / 雪中液态水:
   WMAX = CWH * SNOW  (water holding capacity / 持水能力)

2. Soil Moisture Routine / 土壤水模块:
   Accounts for actual evapotranspiration and contributes to runoff / 
   计算实际蒸散发并产生径流
   
   Actual ET / 实际蒸散发:
   EA = PET * min(SM / (LP * FC), 1.0)
   
   Recharge / 补给:
   RECHARGE = (SM / FC) ^ BETA * (P - EA)
   
   Where:
   - SM: Soil moisture content / 土壤含水量 (mm)
   - FC: Field capacity / 田间持水量 (mm)
   - LP: Limit for potential evapotranspiration / 蒸散发限制参数 (-)
   - BETA: Shape coefficient / 形状系数 (-)

3. Runoff Response Routine / 径流响应模块:
   Three reservoirs representing different runoff components / 
   三个水库代表不同径流成分
   
   Upper zone / 上层带:
   Q0 = K0 * max(SUZ - UZL, 0)  (fast runoff / 快速径流)
   Q1 = K1 * SUZ  (interflow / 壤中流)
   Percolation = PERC * min(SUZ / UZL, 1)
   
   Lower zone / 下层带:
   Q2 = K2 * SLZ  (baseflow / 基流)
   
   Total discharge / 总径流:
   Q = Q0 + Q1 + Q2

Parameters / 参数:
===================

Snow routine / 积雪模块:
- TT: Threshold temperature for snow/rain / 降雪临界温度 (°C, typically -1 to 2)
- CFMAX: Degree-day factor / 度日因子 (mm/°C/day, typically 2-4)
- CWH: Water holding capacity of snow / 雪持水能力 (-, typically 0.05-0.15)
- CFR: Refreezing coefficient / 再冻结系数 (-, typically 0.05)

Soil routine / 土壤模块:
- FC: Field capacity / 田间持水量 (mm, typically 100-400)
- LP: Limit for potential ET / 蒸散限制参数 (-, typically 0.3-0.9)
- BETA: Shape coefficient / 形状系数 (-, typically 1-6)

Response routine / 响应模块:
- PERC: Percolation / 渗漏 (mm/day, typically 0-3)
- UZL: Threshold for Q0 / Q0阈值 (mm, typically 10-50)
- K0: Recession coefficient for Q0 / Q0消退系数 (1/day, typically 0.1-0.5)
- K1: Recession coefficient for Q1 / Q1消退系数 (1/day, typically 0.05-0.2)
- K2: Recession coefficient for Q2 / Q2消退系数 (1/day, typically 0.001-0.05)
- MAXBAS: Routing parameter / 汇流参数 (days, typically 1-7)

References / 参考文献:
=======================
Bergström, S. (1992). The HBV model - its structure and applications. 
SMHI Reports RH No. 4, Norrköping.

Seibert, J., & Vis, M. J. P. (2012). Teaching hydrological modeling with a 
user-friendly catchment-runoff-model software package. Hydrology and Earth 
System Sciences, 16(9), 3315-3325.

Author: Implementation for HydroLearn教学库
Date: 2024
"""

import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import seaborn as sns


class HBVModel:
    """
    HBV (Hydrologiska Byråns Vattenbalansavdelning) hydrological model.
    HBV (瑞典水文局水量平衡模型)水文模型。
    
    This is a simplified implementation of HBV-96 suitable for teaching purposes.
    这是适用于教学目的的HBV-96简化实现。
    
    Parameters / 参数:
    -------------------
    TT : float, optional
        Threshold temperature for snow/rain (°C) / 降雪临界温度 (°C)
        Default: 0.0, Range: -2 to 2
        
    CFMAX : float, optional
        Degree-day factor (mm/°C/day) / 度日因子 (mm/°C/天)
        Default: 3.5, Range: 1 to 8
        
    CWH : float, optional
        Water holding capacity of snow (-) / 雪持水能力 (-)
        Default: 0.1, Range: 0.05 to 0.2
        
    CFR : float, optional
        Refreezing coefficient (-) / 再冻结系数 (-)
        Default: 0.05, Range: 0 to 0.1
        
    FC : float, optional
        Field capacity / maximum soil moisture (mm) / 田间持水量 (mm)
        Default: 200.0, Range: 50 to 500
        
    LP : float, optional
        Limit for potential evapotranspiration (-) / 蒸散限制参数 (-)
        Default: 0.7, Range: 0.3 to 1.0
        
    BETA : float, optional
        Shape coefficient for recharge (-) / 补给形状系数 (-)
        Default: 2.0, Range: 1 to 6
        
    PERC : float, optional
        Percolation from upper to lower zone (mm/day) / 渗漏 (mm/天)
        Default: 1.5, Range: 0 to 4
        
    UZL : float, optional
        Threshold for quick runoff (mm) / 快速径流阈值 (mm)
        Default: 20.0, Range: 0 to 50
        
    K0 : float, optional
        Recession coefficient for quick runoff (1/day) / 快速径流消退系数 (1/天)
        Default: 0.2, Range: 0.05 to 0.5
        
    K1 : float, optional
        Recession coefficient for interflow (1/day) / 壤中流消退系数 (1/天)
        Default: 0.1, Range: 0.01 to 0.3
        
    K2 : float, optional
        Recession coefficient for baseflow (1/day) / 基流消退系数 (1/天)
        Default: 0.05, Range: 0.001 to 0.1
        
    MAXBAS : float, optional
        Length of triangular weighting function (days) / 三角形权重函数长度 (天)
        Default: 2.5, Range: 1 to 7
    """
    
    def __init__(self,
                 TT: float = 0.0,
                 CFMAX: float = 3.5,
                 CWH: float = 0.1,
                 CFR: float = 0.05,
                 FC: float = 200.0,
                 LP: float = 0.7,
                 BETA: float = 2.0,
                 PERC: float = 1.5,
                 UZL: float = 20.0,
                 K0: float = 0.2,
                 K1: float = 0.1,
                 K2: float = 0.05,
                 MAXBAS: float = 2.5):
        
        # Snow routine parameters / 积雪模块参数
        self.TT = TT          # Threshold temperature / 临界温度
        self.CFMAX = CFMAX    # Degree-day factor / 度日因子
        self.CWH = CWH        # Water holding capacity / 持水能力
        self.CFR = CFR        # Refreezing coefficient / 再冻结系数
        
        # Soil routine parameters / 土壤模块参数
        self.FC = FC          # Field capacity / 田间持水量
        self.LP = LP          # Evapotranspiration limit / 蒸散限制
        self.BETA = BETA      # Shape coefficient / 形状系数
        
        # Response routine parameters / 响应模块参数
        self.PERC = PERC      # Percolation rate / 渗漏率
        self.UZL = UZL        # Upper zone threshold / 上层阈值
        self.K0 = K0          # Quick runoff recession / 快速径流消退
        self.K1 = K1          # Interflow recession / 壤中流消退
        self.K2 = K2          # Baseflow recession / 基流消退
        self.MAXBAS = MAXBAS  # Routing parameter / 汇流参数
        
        # State variables (initial conditions) / 状态变量(初始条件)
        self.SP = 0.0         # Snowpack / 积雪量 (mm)
        self.WC = 0.0         # Water content in snow / 雪中液态水 (mm)
        self.SM = FC * 0.5    # Soil moisture / 土壤水 (mm) - start at 50%
        self.SUZ = 10.0       # Upper zone storage / 上层蓄水 (mm)
        self.SLZ = 30.0       # Lower zone storage / 下层蓄水 (mm)
        
        # Triangular weighting function for routing / 汇流的三角形权重函数
        self._compute_routing_weights()
        
    def _compute_routing_weights(self) -> None:
        """
        Compute triangular weighting function for routing.
        计算用于汇流的三角形权重函数。
        
        The function has its peak at MAXBAS/2 and distributes runoff over
        MAXBAS time steps.
        该函数在MAXBAS/2处达到峰值，并将径流分配到MAXBAS个时间步。
        """
        n = int(np.ceil(self.MAXBAS))
        self.routing_weights = np.zeros(n)
        
        # Triangular shape / 三角形形状
        for i in range(n):
            if i < self.MAXBAS / 2:
                self.routing_weights[i] = i / (self.MAXBAS / 2)
            else:
                self.routing_weights[i] = (self.MAXBAS - i) / (self.MAXBAS / 2)
        
        # Normalize / 归一化
        self.routing_weights = self.routing_weights / np.sum(self.routing_weights)
        
        # Initialize routing queue / 初始化汇流队列
        self.routing_queue = np.zeros(n)
    
    def snow_routine(self, P: float, T: float) -> Tuple[float, float]:
        """
        Snow accumulation and melt routine using degree-day method.
        使用度日法进行积雪累积和融化计算。
        
        Parameters / 参数:
        -------------------
        P : float
            Precipitation (mm) / 降水量 (mm)
        T : float
            Air temperature (°C) / 气温 (°C)
            
        Returns / 返回:
        ----------------
        rain : float
            Liquid water input to soil (mm) / 进入土壤的液态水 (mm)
        snowmelt : float
            Snowmelt amount (mm) / 融雪量 (mm)
        """
        # Determine if precipitation is snow or rain / 判断降水是雪还是雨
        if T < self.TT:
            # Snow / 降雪
            snowfall = P
            rainfall = 0.0
        else:
            # Rain / 降雨
            snowfall = 0.0
            rainfall = P
        
        # Add snowfall to snowpack / 将降雪加到积雪中
        self.SP += snowfall
        
        # Calculate potential snowmelt / 计算潜在融雪量
        if T > self.TT and self.SP > 0:
            melt = self.CFMAX * (T - self.TT)
            melt = min(melt, self.SP)  # Cannot melt more than available / 不能超过可用积雪
        else:
            melt = 0.0
        
        # Update snowpack / 更新积雪量
        self.SP -= melt
        
        # Add melt to water content in snow / 将融雪加到雪中液态水
        self.WC += melt
        
        # Calculate refreezing / 计算再冻结
        if T < self.TT:
            refreeze = self.CFR * self.CFMAX * (self.TT - T)
            refreeze = min(refreeze, self.WC)
            self.WC -= refreeze
            self.SP += refreeze
        
        # Calculate water holding capacity / 计算持水能力
        water_capacity = self.CWH * self.SP
        
        # Excess water becomes liquid input / 超出持水能力的水成为液态输入
        if self.WC > water_capacity:
            liquid_water = self.WC - water_capacity
            self.WC = water_capacity
        else:
            liquid_water = 0.0
        
        # Total liquid input / 总液态输入
        rain = rainfall + liquid_water
        
        return rain, melt
    
    def soil_routine(self, rain: float, PET: float) -> Tuple[float, float]:
        """
        Soil moisture accounting and runoff generation.
        土壤水分核算和径流产生。
        
        Parameters / 参数:
        -------------------
        rain : float
            Liquid water input (mm) / 液态水输入 (mm)
        PET : float
            Potential evapotranspiration (mm) / 潜在蒸散发 (mm)
            
        Returns / 返回:
        ----------------
        EA : float
            Actual evapotranspiration (mm) / 实际蒸散发 (mm)
        recharge : float
            Recharge to upper zone (mm) / 上层补给 (mm)
        """
        # Calculate actual evapotranspiration / 计算实际蒸散发
        # EA decreases linearly from PET to 0 as SM decreases from LP*FC to 0
        # EA从PET线性减小到0，当SM从LP*FC减小到0时
        if self.SM / (self.LP * self.FC) < 1.0:
            EA = PET * (self.SM / (self.LP * self.FC))
        else:
            EA = PET
        
        EA = min(EA, self.SM)  # Cannot evaporate more than available / 不能超过可用水量
        
        # Update soil moisture / 更新土壤水分
        self.SM -= EA
        self.SM = max(0.0, self.SM)
        
        # Add rain to soil moisture / 将降雨加到土壤水分
        self.SM += rain
        
        # Calculate recharge to upper zone / 计算上层补给
        # Recharge is a non-linear function of soil moisture / 补给是土壤水分的非线性函数
        if self.SM > 0 and self.FC > 0:
            recharge = rain * (self.SM / self.FC) ** self.BETA
        else:
            recharge = 0.0
        
        # Limit recharge to available water / 限制补给不超过可用水量
        recharge = min(recharge, self.SM)
        
        # Update soil moisture after recharge / 补给后更新土壤水分
        self.SM -= recharge
        
        # Ensure soil moisture doesn't exceed field capacity / 确保土壤水分不超过田间持水量
        if self.SM > self.FC:
            excess = self.SM - self.FC
            recharge += excess
            self.SM = self.FC
        
        return EA, recharge
    
    def response_routine(self, recharge: float) -> Dict[str, float]:
        """
        Runoff response routine with three storage components.
        具有三个蓄水部分的径流响应模块。
        
        Parameters / 参数:
        -------------------
        recharge : float
            Recharge from soil routine (mm) / 来自土壤模块的补给 (mm)
            
        Returns / 返回:
        ----------------
        dict : Dictionary containing runoff components / 包含径流成分的字典
            - Q0: Quick runoff / 快速径流 (mm)
            - Q1: Interflow / 壤中流 (mm)
            - Q2: Baseflow / 基流 (mm)
            - Q_gen: Total generated runoff / 总产生径流 (mm)
        """
        # Add recharge to upper zone / 将补给加到上层
        self.SUZ += recharge
        
        # Quick runoff (Q0) - only when SUZ exceeds threshold UZL
        # 快速径流(Q0) - 仅当上层蓄水超过阈值UZL时
        if self.SUZ > self.UZL:
            Q0 = self.K0 * (self.SUZ - self.UZL)
            self.SUZ -= Q0
        else:
            Q0 = 0.0
        
        # Interflow (Q1) from upper zone / 上层的壤中流(Q1)
        Q1 = self.K1 * self.SUZ
        self.SUZ -= Q1
        self.SUZ = max(0.0, self.SUZ)
        
        # Percolation from upper to lower zone / 从上层到下层的渗漏
        if self.SUZ > 0:
            perc = min(self.PERC, self.SUZ)
            self.SUZ -= perc
            self.SLZ += perc
        else:
            perc = 0.0
        
        # Baseflow (Q2) from lower zone / 下层的基流(Q2)
        Q2 = self.K2 * self.SLZ
        self.SLZ -= Q2
        self.SLZ = max(0.0, self.SLZ)
        
        # Total generated runoff (before routing) / 总产生径流(汇流前)
        Q_gen = Q0 + Q1 + Q2
        
        return {
            'Q0': Q0,      # Quick runoff / 快速径流
            'Q1': Q1,      # Interflow / 壤中流
            'Q2': Q2,      # Baseflow / 基流
            'Q_gen': Q_gen # Generated runoff / 产生径流
        }
    
    def routing(self, Q_gen: float) -> float:
        """
        Apply triangular weighting function for routing.
        应用三角形权重函数进行汇流。
        
        Parameters / 参数:
        -------------------
        Q_gen : float
            Generated runoff (mm) / 产生的径流 (mm)
            
        Returns / 返回:
        ----------------
        Q : float
            Routed discharge (mm) / 汇流后的径流 (mm)
        """
        # Update routing queue / 更新汇流队列
        self.routing_queue = np.roll(self.routing_queue, 1)
        self.routing_queue[0] = Q_gen
        
        # Apply weighting function / 应用权重函数
        Q = np.sum(self.routing_queue * self.routing_weights)
        
        return Q
    
    def run_timestep(self, P: float, T: float, PET: float) -> Dict[str, float]:
        """
        Run one timestep of the HBV model.
        运行HBV模型的一个时间步。
        
        Parameters / 参数:
        -------------------
        P : float
            Precipitation (mm) / 降水量 (mm)
        T : float
            Air temperature (°C) / 气温 (°C)
        PET : float
            Potential evapotranspiration (mm) / 潜在蒸散发 (mm)
            
        Returns / 返回:
        ----------------
        dict : Model outputs / 模型输出
            - Q: Total discharge / 总径流 (mm)
            - Q0: Quick runoff / 快速径流 (mm)
            - Q1: Interflow / 壤中流 (mm)
            - Q2: Baseflow / 基流 (mm)
            - EA: Actual evapotranspiration / 实际蒸散发 (mm)
            - SP: Snowpack / 积雪 (mm)
            - WC: Water content in snow / 雪中液态水 (mm)
            - SM: Soil moisture / 土壤水 (mm)
            - SUZ: Upper zone storage / 上层蓄水 (mm)
            - SLZ: Lower zone storage / 下层蓄水 (mm)
        """
        # 1. Snow routine / 积雪模块
        rain, snowmelt = self.snow_routine(P, T)
        
        # 2. Soil routine / 土壤模块
        EA, recharge = self.soil_routine(rain, PET)
        
        # 3. Response routine / 响应模块
        runoff_components = self.response_routine(recharge)
        
        # 4. Routing / 汇流
        Q = self.routing(runoff_components['Q_gen'])
        
        return {
            'Q': Q,                      # Total discharge / 总径流
            'Q0': runoff_components['Q0'],  # Quick runoff / 快速径流
            'Q1': runoff_components['Q1'],  # Interflow / 壤中流
            'Q2': runoff_components['Q2'],  # Baseflow / 基流
            'EA': EA,                    # Actual ET / 实际蒸散发
            'SP': self.SP,               # Snowpack / 积雪
            'WC': self.WC,               # Water in snow / 雪中水
            'SM': self.SM,               # Soil moisture / 土壤水
            'SUZ': self.SUZ,             # Upper zone / 上层蓄水
            'SLZ': self.SLZ,             # Lower zone / 下层蓄水
            'melt': snowmelt,            # Snowmelt / 融雪量
            'recharge': recharge         # Recharge / 补给量
        }
    
    def run(self, P: np.ndarray, T: np.ndarray, PET: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run HBV model for multiple timesteps.
        运行多个时间步的HBV模型。
        
        Parameters / 参数:
        -------------------
        P : np.ndarray
            Precipitation time series (mm) / 降水时间序列 (mm)
        T : np.ndarray
            Temperature time series (°C) / 温度时间序列 (°C)
        PET : np.ndarray
            Potential evapotranspiration time series (mm) / 潜在蒸散发时间序列 (mm)
            
        Returns / 返回:
        ----------------
        dict : Model outputs as arrays / 模型输出数组
        """
        n_steps = len(P)
        
        # Initialize output arrays / 初始化输出数组
        Q = np.zeros(n_steps)
        Q0 = np.zeros(n_steps)
        Q1 = np.zeros(n_steps)
        Q2 = np.zeros(n_steps)
        EA = np.zeros(n_steps)
        SP = np.zeros(n_steps)
        WC = np.zeros(n_steps)
        SM = np.zeros(n_steps)
        SUZ = np.zeros(n_steps)
        SLZ = np.zeros(n_steps)
        melt = np.zeros(n_steps)
        recharge = np.zeros(n_steps)
        
        # Run model for each timestep / 对每个时间步运行模型
        for t in range(n_steps):
            result = self.run_timestep(P[t], T[t], PET[t])
            Q[t] = result['Q']
            Q0[t] = result['Q0']
            Q1[t] = result['Q1']
            Q2[t] = result['Q2']
            EA[t] = result['EA']
            SP[t] = result['SP']
            WC[t] = result['WC']
            SM[t] = result['SM']
            SUZ[t] = result['SUZ']
            SLZ[t] = result['SLZ']
            melt[t] = result['melt']
            recharge[t] = result['recharge']
        
        return {
            'Q': Q,           # Total discharge / 总径流
            'Q0': Q0,         # Quick runoff / 快速径流
            'Q1': Q1,         # Interflow / 壤中流
            'Q2': Q2,         # Baseflow / 基流
            'EA': EA,         # Actual ET / 实际蒸散发
            'SP': SP,         # Snowpack / 积雪
            'WC': WC,         # Water in snow / 雪中水
            'SM': SM,         # Soil moisture / 土壤水
            'SUZ': SUZ,       # Upper zone / 上层蓄水
            'SLZ': SLZ,       # Lower zone / 下层蓄水
            'melt': melt,     # Snowmelt / 融雪量
            'recharge': recharge  # Recharge / 补给量
        }


def create_hbv_plots(model: HBVModel, P: np.ndarray, T: np.ndarray, PET: np.ndarray, 
                     results: Dict[str, np.ndarray], save_dir: str = "figures") -> None:
    """
    Create comprehensive visualization plots for HBV model results.
    为HBV模型结果创建综合可视化图表。
    
    Parameters / 参数:
    -------------------
    model : HBVModel
        The HBV model instance / HBV模型实例
    P : np.ndarray
        Precipitation input / 降水输入
    T : np.ndarray
        Temperature input / 温度输入
    PET : np.ndarray
        Potential ET input / 潜在蒸散发输入
    results : dict
        Model simulation results / 模型模拟结果
    save_dir : str, optional
        Directory to save figures / 保存图表的目录
    """
    # Ensure save directory exists / 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style / 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create date index / 创建日期索引
    n_days = len(P)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Figure 1: Comprehensive time series / 综合时间序列
    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)
    fig.suptitle('HBV Model - Comprehensive Analysis / HBV模型-综合分析', 
                 fontsize=16, fontweight='bold')
    
    # Precipitation and temperature / 降水和温度
    ax1_twin = axes[0].twinx()
    axes[0].bar(dates, P, color='steelblue', alpha=0.7, width=1, label='Precipitation / 降水')
    ax1_twin.plot(dates, T, color='red', linewidth=2, label='Temperature / 气温')
    ax1_twin.axhline(y=model.TT, color='purple', linestyle='--', alpha=0.7, 
                     label=f'TT={model.TT}°C')
    axes[0].set_ylabel('Precipitation (mm) / 降水 (mm)', fontweight='bold')
    ax1_twin.set_ylabel('Temperature (°C) / 气温 (°C)', fontweight='bold', color='red')
    axes[0].legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()
    axes[0].set_ylim(max(P) * 1.1, 0)
    
    # Snow dynamics / 积雪动态
    axes[1].plot(dates, results['SP'], color='blue', linewidth=2, label='Snowpack / 积雪 (SP)')
    axes[1].plot(dates, results['WC'], color='cyan', linewidth=2, label='Water in Snow / 雪中水 (WC)')
    axes[1].plot(dates, results['melt'], color='red', linewidth=1.5, alpha=0.7, 
                 label='Snowmelt / 融雪')
    axes[1].fill_between(dates, results['SP'], alpha=0.2, color='blue')
    axes[1].set_ylabel('Snow Storage (mm) / 积雪 (mm)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Soil moisture / 土壤水分
    axes[2].plot(dates, results['SM'], color='brown', linewidth=2, label='Soil Moisture / 土壤水 (SM)')
    axes[2].axhline(y=model.FC, color='red', linestyle='--', alpha=0.7, 
                    label=f'FC={model.FC}mm')
    axes[2].axhline(y=model.FC * model.LP, color='orange', linestyle='--', alpha=0.7,
                    label=f'LP*FC={model.FC*model.LP:.1f}mm')
    axes[2].fill_between(dates, results['SM'], alpha=0.2, color='brown')
    axes[2].set_ylabel('Soil Moisture (mm) / 土壤水 (mm)', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Evapotranspiration / 蒸散发
    axes[3].plot(dates, PET, color='orange', linewidth=1.5, label='Potential ET / 潜在蒸散发', alpha=0.8)
    axes[3].plot(dates, results['EA'], color='green', linewidth=2, label='Actual ET / 实际蒸散发')
    axes[3].fill_between(dates, PET, alpha=0.2, color='orange')
    axes[3].fill_between(dates, results['EA'], alpha=0.3, color='green')
    axes[3].set_ylabel('Evapotranspiration (mm) / 蒸散发 (mm)', fontweight='bold')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Runoff components / 径流成分
    axes[4].plot(dates, results['Q'], color='blue', linewidth=2.5, label='Total Q / 总径流')
    axes[4].plot(dates, results['Q0'], color='red', linewidth=1.5, alpha=0.8, label='Q0 (Quick) / 快速径流')
    axes[4].plot(dates, results['Q1'], color='green', linewidth=1.5, alpha=0.8, label='Q1 (Interflow) / 壤中流')
    axes[4].plot(dates, results['Q2'], color='purple', linewidth=1.5, alpha=0.8, label='Q2 (Baseflow) / 基流')
    axes[4].set_ylabel('Discharge (mm) / 径流 (mm)', fontweight='bold')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    # Storage states / 蓄水状态
    axes[5].plot(dates, results['SUZ'], color='orange', linewidth=2, label='Upper Zone / 上层蓄水 (SUZ)')
    axes[5].plot(dates, results['SLZ'], color='purple', linewidth=2, label='Lower Zone / 下层蓄水 (SLZ)')
    axes[5].axhline(y=model.UZL, color='red', linestyle='--', alpha=0.7, 
                    label=f'UZL={model.UZL}mm')
    axes[5].fill_between(dates, results['SUZ'], alpha=0.2, color='orange')
    axes[5].fill_between(dates, results['SLZ'], alpha=0.2, color='purple')
    axes[5].set_ylabel('Storage (mm) / 蓄水 (mm)', fontweight='bold')
    axes[5].set_xlabel('Date / 日期', fontweight='bold')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    # Format x-axis / 格式化x轴
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hbv_comprehensive.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Water balance and performance / 水量平衡和性能
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('HBV Model - Water Balance & Performance / HBV模型-水量平衡和性能', 
                 fontsize=16, fontweight='bold')
    
    # Cumulative water balance / 累积水量平衡
    cum_P = np.cumsum(P)
    cum_PET = np.cumsum(PET)
    cum_EA = np.cumsum(results['EA'])
    cum_Q = np.cumsum(results['Q'])
    
    axes[0, 0].plot(dates, cum_P, label='Cumulative P / 累积降水', linewidth=2, color='blue')
    axes[0, 0].plot(dates, cum_PET, label='Cumulative PET / 累积潜在蒸散发', linewidth=2, color='orange')
    axes[0, 0].plot(dates, cum_EA, label='Cumulative EA / 累积实际蒸散发', linewidth=2, color='green')
    axes[0, 0].plot(dates, cum_Q, label='Cumulative Q / 累积径流', linewidth=2, color='red')
    axes[0, 0].set_title('Cumulative Water Balance / 累积水量平衡', fontweight='bold')
    axes[0, 0].set_ylabel('Cumulative Water (mm) / 累积水量 (mm)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Runoff components distribution / 径流成分分布
    total_q0 = np.sum(results['Q0'])
    total_q1 = np.sum(results['Q1'])
    total_q2 = np.sum(results['Q2'])
    
    labels = ['Q0 (Quick) / 快速径流', 'Q1 (Interflow) / 壤中流', 'Q2 (Baseflow) / 基流']
    sizes = [total_q0, total_q1, total_q2]
    colors = ['red', 'green', 'purple']
    
    axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Discharge Components / 径流成分分布', fontweight='bold')
    
    # Temperature vs snowmelt relationship / 温度与融雪关系
    axes[1, 0].scatter(T, results['melt'], alpha=0.6, c=results['SP'], cmap='Blues', s=20)
    axes[1, 0].set_xlabel('Temperature (°C) / 气温 (°C)', fontweight='bold')
    axes[1, 0].set_ylabel('Snowmelt (mm) / 融雪 (mm)', fontweight='bold')
    axes[1, 0].set_title('Temperature vs Snowmelt / 气温与融雪关系', fontweight='bold')
    axes[1, 0].axvline(x=model.TT, color='red', linestyle='--', alpha=0.7, label=f'TT={model.TT}°C')
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Snowpack (mm) / 积雪 (mm)', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Soil moisture vs ET efficiency / 土壤水分与蒸散发效率
    et_efficiency = results['EA'] / (PET + 1e-6)
    et_efficiency = np.clip(et_efficiency, 0, 1)
    
    axes[1, 1].scatter(results['SM'] / model.FC, et_efficiency, alpha=0.6, c=P, cmap='Blues', s=20)
    axes[1, 1].set_xlabel('Soil Moisture Ratio (SM/FC) / 土壤水分比', fontweight='bold')
    axes[1, 1].set_ylabel('ET Efficiency (EA/PET) / 蒸散发效率', fontweight='bold')
    axes[1, 1].set_title('Soil Moisture vs ET Efficiency / 土壤水分与蒸散发效率', fontweight='bold')
    axes[1, 1].axvline(x=model.LP, color='red', linestyle='--', alpha=0.7, label=f'LP={model.LP}')
    axes[1, 1].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Max Efficiency / 最大效率')
    cbar2 = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar2.set_label('Precipitation (mm) / 降水 (mm)', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hbv_water_balance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ HBV model visualization plots saved to '{save_dir}' directory")
    print(f"  ✓ HBV模型可视化图表已保存到'{save_dir}'目录")
    print("  Generated files / 生成的文件:")
    print("  - hbv_comprehensive.png")
    print("  - hbv_water_balance.png")


def main():
    """
    Example usage of HBV model with synthetic data.
    使用合成数据的HBV模型示例。
    
    This demonstration shows the complete workflow of using the HBV model
    including data preparation, model initialization, simulation, and
    visualization.
    本演示展示了使用HBV模型的完整工作流程，包括数据准备、模型初始化、模拟和可视化。
    """
    print("=" * 80)
    print("HBV (Hydrologiska Byråns Vattenbalansavdelning) Model - Demonstration")
    print("HBV (瑞典水文局水量平衡模型) - 演示")
    print("=" * 80)
    
    # Set random seed for reproducibility / 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # Generate synthetic data (2 years including winter season) / 生成合成数据(2年,包括冬季)
    n_days = 730  # 2 years / 2年
    
    # Create date array / 创建日期数组
    t = np.arange(n_days)
    
    # Generate temperature with seasonal cycle / 生成具有季节周期的温度
    # Average temperature varies from -5°C in winter to 20°C in summer
    # 平均温度在冬季-5°C到夏季20°C之间变化
    T_mean = 7.5
    T_amplitude = 12.5
    T = T_mean + T_amplitude * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 2, n_days)
    
    # Generate precipitation with seasonal pattern / 生成具有季节模式的降水
    # More precipitation in winter / 冬季降水较多
    seasonal_factor = 1.3 + 0.7 * np.sin(2 * np.pi * t / 365 + np.pi)
    P_base = np.random.gamma(1.5, 2.5, n_days) * seasonal_factor
    
    # Add dry periods / 添加干旱期
    dry_prob = 0.55 + 0.15 * np.sin(2 * np.pi * t / 365)
    P = np.where(np.random.rand(n_days) < dry_prob, 0, P_base)
    
    # Add some storm events / 添加一些暴雨事件
    storm_events = np.random.choice(n_days, size=10, replace=False)
    P[storm_events] = P[storm_events] + np.random.gamma(4, 5, 10)
    
    # Generate potential evapotranspiration (temperature-based) / 生成潜在蒸散发(基于温度)
    # PET is higher in summer when temperature is high / 夏季温度高时蒸散发较大
    PET = np.maximum(0.0, 0.3 * T + 1.0 + np.random.normal(0, 0.3, n_days))
    PET = np.maximum(PET, 0.1)  # Minimum PET / 最小蒸散发
    
    print("\nInput Data Summary / 输入数据摘要:")
    print(f"  Simulation period / 模拟期: {n_days} days / 天 ({n_days/365:.1f} years / 年)")
    print(f"  Total precipitation / 总降水: {np.sum(P):.2f} mm")
    print(f"  Average daily precipitation / 平均日降水: {np.mean(P):.2f} mm")
    print(f"  Maximum daily precipitation / 最大日降水: {np.max(P):.2f} mm")
    print(f"  Rainfall days / 降雨天数: {np.sum(P > 0.1)} ({np.sum(P > 0.1)/n_days*100:.1f}%)")
    print(f"  Average temperature / 平均温度: {np.mean(T):.2f} °C")
    print(f"  Min temperature / 最低温度: {np.min(T):.2f} °C")
    print(f"  Max temperature / 最高温度: {np.max(T):.2f} °C")
    print(f"  Days below 0°C / 0°C以下天数: {np.sum(T < 0)} ({np.sum(T < 0)/n_days*100:.1f}%)")
    print(f"  Total potential ET / 总潜在蒸散发: {np.sum(PET):.2f} mm")
    print(f"  Average daily PET / 平均日蒸散发: {np.mean(PET):.2f} mm")
    print(f"  P/PET ratio / 降水蒸散比: {np.sum(P)/np.sum(PET):.3f}")
    
    # Initialize HBV model with parameters suitable for temperate climate
    # 使用适合温带气候的参数初始化HBV模型
    model = HBVModel(
        TT=0.0,        # Threshold temperature / 临界温度
        CFMAX=3.5,     # Degree-day factor / 度日因子
        CWH=0.1,       # Water holding capacity / 持水能力
        CFR=0.05,      # Refreezing coefficient / 再冻结系数
        FC=200.0,      # Field capacity / 田间持水量
        LP=0.7,        # ET limit / 蒸散限制
        BETA=2.0,      # Shape coefficient / 形状系数
        PERC=1.5,      # Percolation / 渗漏
        UZL=20.0,      # Upper zone threshold / 上层阈值
        K0=0.2,        # Quick runoff recession / 快速径流消退
        K1=0.1,        # Interflow recession / 壤中流消退
        K2=0.05,       # Baseflow recession / 基流消退
        MAXBAS=2.5     # Routing parameter / 汇流参数
    )
    
    print("\nModel Parameters / 模型参数:")
    print("  Snow Routine / 积雪模块:")
    print(f"    TT (Threshold temperature / 临界温度): {model.TT} °C")
    print(f"    CFMAX (Degree-day factor / 度日因子): {model.CFMAX} mm/°C/day")
    print(f"    CWH (Water holding capacity / 持水能力): {model.CWH}")
    print(f"    CFR (Refreezing coefficient / 再冻结系数): {model.CFR}")
    print("  Soil Routine / 土壤模块:")
    print(f"    FC (Field capacity / 田间持水量): {model.FC} mm")
    print(f"    LP (ET limit / 蒸散限制): {model.LP}")
    print(f"    BETA (Shape coefficient / 形状系数): {model.BETA}")
    print("  Response Routine / 响应模块:")
    print(f"    PERC (Percolation / 渗漏): {model.PERC} mm/day")
    print(f"    UZL (Upper zone threshold / 上层阈值): {model.UZL} mm")
    print(f"    K0 (Quick runoff recession / 快速径流消退): {model.K0} 1/day")
    print(f"    K1 (Interflow recession / 壤中流消退): {model.K1} 1/day")
    print(f"    K2 (Baseflow recession / 基流消退): {model.K2} 1/day")
    print(f"    MAXBAS (Routing parameter / 汇流参数): {model.MAXBAS} days")
    
    # Run model simulation / 运行模型模拟
    print(f"\nRunning HBV model simulation... / 正在运行HBV模型模拟...")
    results = model.run(P, T, PET)
    
    # Calculate and display results / 计算并显示结果
    print("\nSimulation Results / 模拟结果:")
    print(f"  Water Balance / 水量平衡:")
    print(f"    Total precipitation / 总降水: {np.sum(P):.2f} mm")
    print(f"    Total actual ET / 总实际蒸散发: {np.sum(results['EA']):.2f} mm")
    print(f"    Total discharge / 总径流: {np.sum(results['Q']):.2f} mm")
    print(f"    Final snowpack / 最终积雪: {results['SP'][-1]:.2f} mm")
    print(f"    Final soil moisture / 最终土壤水: {results['SM'][-1]:.2f} mm")
    print(f"    Annual discharge / 年均径流: {np.sum(results['Q'])/2:.2f} mm/year")
    
    print(f"  Efficiency Indicators / 效率指标:")
    print(f"    Runoff coefficient / 径流系数: {np.sum(results['Q']) / np.sum(P):.3f}")
    print(f"    ET efficiency / 蒸散效率: {np.sum(results['EA']) / np.sum(PET):.3f}")
    
    print(f"  Flow Characteristics / 径流特征:")
    print(f"    Peak discharge / 峰值径流: {np.max(results['Q']):.2f} mm/day")
    print(f"    Mean discharge / 平均径流: {np.mean(results['Q']):.2f} mm/day")
    print(f"    Median discharge / 中位径流: {np.median(results['Q']):.2f} mm/day")
    
    # Runoff components analysis / 径流成分分析
    total_discharge = np.sum(results['Q'])
    q0_total = np.sum(results['Q0'])
    q1_total = np.sum(results['Q1'])
    q2_total = np.sum(results['Q2'])
    
    print(f"\nRunoff Components / 径流成分:")
    print(f"  Quick runoff (Q0) / 快速径流: {q0_total:.2f} mm ({q0_total/total_discharge*100:.1f}%)")
    print(f"  Interflow (Q1) / 壤中流: {q1_total:.2f} mm ({q1_total/total_discharge*100:.1f}%)")
    print(f"  Baseflow (Q2) / 基流: {q2_total:.2f} mm ({q2_total/total_discharge*100:.1f}%)")
    
    baseflow_index = q2_total / total_discharge
    print(f"  Baseflow index / 基流指数: {baseflow_index:.3f}")
    
    # Snow analysis / 积雪分析
    max_snowpack = np.max(results['SP'])
    total_snowmelt = np.sum(results['melt'])
    days_with_snow = np.sum(results['SP'] > 1.0)
    
    print(f"\nSnow Dynamics / 积雪动态:")
    print(f"  Maximum snowpack / 最大积雪: {max_snowpack:.2f} mm")
    print(f"  Total snowmelt / 总融雪量: {total_snowmelt:.2f} mm")
    print(f"  Days with snow cover / 有积雪天数: {days_with_snow} ({days_with_snow/n_days*100:.1f}%)")
    
    # Generate comprehensive visualizations / 生成综合可视化
    print(f"\nGenerating comprehensive visualizations... / 正在生成综合可视化...")
    create_hbv_plots(model, P, T, PET, results, save_dir="figures")
    
    # Display sample results / 显示样本结果
    print("\nDetailed Results for First 15 Days / 前15天的详细结果:")
    print("Day |  Date  |   P   |   T   |  PET  |   Q   |  SP   |  SM   |  EA   | Melt  |")
    print("----|--------|-------|-------|-------|-------|-------|-------|-------|-------|")
    
    start_date = datetime(2020, 1, 1)
    for i in range(15):
        date_str = (start_date + timedelta(days=i)).strftime("%m-%d")
        print(f"{i+1:3d} | {date_str} | {P[i]:5.2f} | {T[i]:5.1f} | {PET[i]:5.2f} | "
              f"{results['Q'][i]:5.2f} | {results['SP'][i]:5.2f} | {results['SM'][i]:5.2f} | "
              f"{results['EA'][i]:5.2f} | {results['melt'][i]:5.2f} |")
    
    # Water balance check / 水量平衡检查
    total_input = np.sum(P)
    total_output = np.sum(results['EA']) + np.sum(results['Q'])
    storage_change = (results['SP'][-1] + results['WC'][-1] + results['SM'][-1] + 
                     results['SUZ'][-1] + results['SLZ'][-1]) - \
                     (model.FC * 0.5 + 10.0 + 30.0)  # Initial storage estimate
    balance_error = total_input - total_output - storage_change
    
    print(f"\nWater Balance Check / 水量平衡检查:")
    print(f"  Total input (P) / 总输入(降水): {total_input:.2f} mm")
    print(f"  Total output (EA + Q) / 总输出(蒸散+径流): {total_output:.2f} mm")
    print(f"  Storage change / 蓄水变化: {storage_change:.2f} mm")
    print(f"  Balance error / 平衡误差: {balance_error:.2f} mm ({abs(balance_error)/total_input*100:.3f}%)")
    
    print("\n" + "=" * 80)
    print("HBV model simulation completed successfully! / HBV模型模拟成功完成!")
    print("=" * 80)
    print("\nModel Features Demonstrated / 演示的模型特性:")
    print("  ✓ Snow accumulation and melt / 积雪累积和融化")
    print("  ✓ Soil moisture accounting / 土壤水分核算")
    print("  ✓ Three-component runoff generation / 三成分径流产生")
    print("  ✓ Temperature-dependent processes / 温度相关过程")
    print("  ✓ Suitable for cold/temperate regions / 适用于寒冷/温带地区")
    print("\nCheck the 'figures' directory for visualizations.")
    print("请查看'figures'目录中的可视化图表。")
    print("=" * 80)
    
    return results, model


if __name__ == "__main__":
    main()
