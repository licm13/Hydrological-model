"""
Tank Model (タンクモデル) - Hydrological Rainfall-Runoff Model

The Tank model is a conceptual rainfall-runoff model developed by Sugawara (1961, 1995) 
in Japan. It represents the catchment as a series of vertically arranged tanks (reservoirs)
that simulate different runoff components (surface runoff, interflow, and baseflow).

Mathematical Foundation:
========================

The model consists of multiple tanks stacked vertically, where:
- Each tank has side outlets and a bottom outlet
- Side outlets represent different types of runoff (surface, interflow, baseflow)
- Bottom outlets represent percolation to lower tanks

For each tank i:

1. Storage Update:
   S_i(t+1) = S_i(t) + Input_i - ∑(Outflow_j) - Percolation_i
   
2. Side Outflow (j-th outlet):
   Q_j(t) = a_j * max(0, S_i(t) - h_j)
   
   Where:
   - Q_j: discharge from outlet j (mm/time)
   - a_j: discharge coefficient for outlet j (1/time)
   - S_i: storage in tank i (mm)
   - h_j: height threshold for outlet j (mm)

3. Bottom Outflow (Percolation):
   Perc_i(t) = b_i * S_i(t)
   
   Where:
   - Perc_i: percolation from tank i (mm/time)
   - b_i: percolation coefficient (1/time)

4. Total Discharge:
   Q_total(t) = ∑∑(Q_ij(t))  for all tanks and outlets

The number of tanks and outlets can vary:
- 1D: Single tank (simple)
- 2D: Two tanks (surface + baseflow)
- 3D: Three tanks (surface + interflow + baseflow)
- 4D: Four tanks (complete model with multiple runoff components)

Author: Sugawara (Original), Implementation by [Your Name]
Date: 2024
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats

# Configure matplotlib for Chinese font display / 配置matplotlib以显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'KaiTi', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display / 修复负号显示


class Tank:
    """
    Individual tank component for the Tank Model.
    
    Parameters:
    -----------
    side_outlets : list of tuples
        List of (coefficient, height_threshold) for side outlets
        [(a1, h1), (a2, h2), ...]
    bottom_coef : float
        Bottom percolation coefficient (b)
    initial_storage : float
        Initial storage in the tank (mm)
    """
    
    def __init__(self, 
                 side_outlets: List[Tuple[float, float]], 
                 bottom_coef: float = 0.0,
                 initial_storage: float = 0.0):
        self.side_outlets = side_outlets  # [(a1, h1), (a2, h2), ...]
        self.bottom_coef = bottom_coef    # b
        self.storage = initial_storage    # S
        
    def run_timestep(self, inflow: float) -> Tuple[float, List[float]]:
        """
        Run one timestep for this tank.
        
        Parameters:
        -----------
        inflow : float
            Input to the tank (mm)
            
        Returns:
        --------
        percolation : float
            Percolation to the next tank (mm)
        side_outflows : list
            Outflows from each side outlet (mm)
        """
        # Add inflow to storage
        self.storage += inflow
        
        # Calculate side outflows
        side_outflows = []
        for a, h in self.side_outlets:
            # Outflow = a * max(0, S - h)
            outflow = a * max(0, self.storage - h)
            side_outflows.append(outflow)
            self.storage -= outflow
            self.storage = max(0, self.storage)  # Ensure non-negative
        
        # Calculate bottom percolation
        percolation = self.bottom_coef * self.storage
        self.storage -= percolation
        self.storage = max(0, self.storage)  # Ensure non-negative
        
        return percolation, side_outflows


class TankModel:
    """
    Tank Model implementation with configurable number of tanks.
    
    This is a flexible implementation that supports 1D, 2D, 3D, and 4D configurations.
    
    Parameters:
    -----------
    tank_configs : list of dict
        Configuration for each tank from top to bottom
        Each dict should contain:
        - 'side_outlets': list of (coef, height) tuples
        - 'bottom_coef': percolation coefficient
        - 'initial_storage': initial storage (optional)
    """
    
    def __init__(self, tank_configs: List[Dict]):
        self.tanks = []
        for config in tank_configs:
            tank = Tank(
                side_outlets=config['side_outlets'],
                bottom_coef=config.get('bottom_coef', 0.0),
                initial_storage=config.get('initial_storage', 0.0)
            )
            self.tanks.append(tank)
        
        self.n_tanks = len(self.tanks)
        
    def run_timestep(self, precipitation: float, 
                     evapotranspiration: float = 0.0) -> Dict[str, float]:
        """
        Run one timestep of the tank model.
        
        Parameters:
        -----------
        precipitation : float
            Precipitation (mm)
        evapotranspiration : float
            Evapotranspiration (mm) - reduces top tank storage
            
        Returns:
        --------
        dict : Dictionary containing:
            - Q_total: Total discharge (mm)
            - Q_tanks: List of discharges from each tank
            - Q_outlets: List of all outlet discharges
            - storages: Current storage in each tank
        """
        # Apply ET to top tank (simple approach)
        net_input = max(0, precipitation - evapotranspiration)
        
        inflow = net_input
        total_discharge = 0.0
        all_outflows = []
        
        # Process each tank from top to bottom
        for i, tank in enumerate(self.tanks):
            percolation, side_outflows = tank.run_timestep(inflow)
            
            # Sum discharge from this tank
            tank_discharge = sum(side_outflows)
            total_discharge += tank_discharge
            all_outflows.extend(side_outflows)
            
            # Percolation becomes input to next tank
            inflow = percolation
        
        # Get current storages
        storages = [tank.storage for tank in self.tanks]
        
        return {
            'Q_total': total_discharge,
            'Q_outlets': all_outflows,
            'storages': storages,
            'percolation_loss': inflow  # Percolation from bottom tank
        }
    
    def run(self, P: np.ndarray, EP: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Run the model for multiple timesteps.
        
        Parameters:
        -----------
        P : np.ndarray
            Precipitation time series (mm)
        EP : np.ndarray, optional
            Evapotranspiration time series (mm)
            
        Returns:
        --------
        dict : Dictionary containing model outputs
        """
        n_steps = len(P)
        
        if EP is None:
            EP = np.zeros(n_steps)
        
        # Initialize output arrays
        Q_total = np.zeros(n_steps)
        storages = np.zeros((n_steps, self.n_tanks))
        flows = np.zeros((n_steps, self.n_tanks))  # Store flow from each tank
        
        # Run model
        for t in range(n_steps):
            result = self.run_timestep(P[t], EP[t])
            Q_total[t] = result['Q_total']
            storages[t, :] = result['storages']
            
            # Calculate flow from each tank
            inflow = max(0, P[t] - EP[t])
            tank_flows = []
            
            for i, tank in enumerate(self.tanks):
                # Calculate this tank's contribution to total flow
                tank_outflows = result['Q_outlets'][i*len(tank.side_outlets):(i+1)*len(tank.side_outlets)]
                tank_flow = sum(tank_outflows) if tank_outflows else 0
                tank_flows.append(tank_flow)
            
            flows[t, :] = tank_flows
        
        return {
            'Q': Q_total,
            'storages': storages,
            'flows': flows
        }


class TankModel1D(TankModel):
    """
    1D Tank Model - Single tank configuration
    
    Simple model with one tank representing the entire catchment.
    Suitable for simple runoff simulation.
    
    Parameters:
    -----------
    a1 : float
        Discharge coefficient for surface runoff (1/day, typical: 0.1-0.5)
    h1 : float
        Height threshold for surface runoff (mm, typical: 10-30)
    b1 : float
        Bottom percolation coefficient (1/day, typical: 0.01-0.1)
    """
    
    def __init__(self, a1: float = 0.3, h1: float = 20.0, b1: float = 0.05):
        config = [{
            'side_outlets': [(a1, h1)],
            'bottom_coef': b1,
            'initial_storage': 10.0
        }]
        super().__init__(config)
        self.a1 = a1
        self.h1 = h1
        self.b1 = b1


class TankModel2D(TankModel):
    """
    2D Tank Model - Two tanks configuration
    
    Two tanks representing surface and subsurface components.
    - Tank 1: Surface runoff (fast response)
    - Tank 2: Baseflow (slow response)
    
    Parameters:
    -----------
    a11 : float
        Discharge coefficient for surface runoff from Tank 1
    h11 : float
        Height threshold for surface runoff from Tank 1
    a21 : float
        Discharge coefficient for baseflow from Tank 2
    h21 : float
        Height threshold for baseflow from Tank 2
    b1 : float
        Percolation coefficient from Tank 1 to Tank 2
    b2 : float
        Bottom percolation coefficient from Tank 2
    """
    
    def __init__(self, 
                 a11: float = 0.3, h11: float = 20.0,
                 a21: float = 0.1, h21: float = 5.0,
                 b1: float = 0.1, b2: float = 0.01):
        configs = [
            {
                'side_outlets': [(a11, h11)],
                'bottom_coef': b1,
                'initial_storage': 15.0
            },
            {
                'side_outlets': [(a21, h21)],
                'bottom_coef': b2,
                'initial_storage': 30.0
            }
        ]
        super().__init__(configs)
        
        # Store parameters for reference
        self.a11 = a11
        self.h11 = h11
        self.a21 = a21
        self.h21 = h21
        self.b1 = b1
        self.b2 = b2


class TankModel3D(TankModel):
    """
    3D Tank Model - Three tanks configuration (Standard Configuration)
    
    Three tanks representing different runoff components:
    - Tank 1: Surface runoff (fast, direct runoff)
    - Tank 2: Interflow (intermediate response)
    - Tank 3: Baseflow (slow, groundwater)
    
    This is the most commonly used configuration.
    
    Parameters:
    -----------
    Tank 1 (Surface):
        a11, h11 : Surface runoff (fast)
        a12, h12 : Overflow
        b1 : Percolation to Tank 2
    
    Tank 2 (Interflow):
        a21, h21 : Interflow (medium)
        b2 : Percolation to Tank 3
    
    Tank 3 (Baseflow):
        a31, h31 : Baseflow (slow)
        b3 : Deep percolation loss
    """
    
    def __init__(self,
                 # Tank 1 - Surface
                 a11: float = 0.5, h11: float = 30.0,
                 a12: float = 0.8, h12: float = 50.0,
                 b1: float = 0.1,
                 # Tank 2 - Interflow
                 a21: float = 0.2, h21: float = 10.0,
                 b2: float = 0.05,
                 # Tank 3 - Baseflow
                 a31: float = 0.05, h31: float = 5.0,
                 b3: float = 0.001):
        
        configs = [
            # Tank 1: Surface runoff with two outlets
            {
                'side_outlets': [(a11, h11), (a12, h12)],
                'bottom_coef': b1,
                'initial_storage': 20.0
            },
            # Tank 2: Interflow
            {
                'side_outlets': [(a21, h21)],
                'bottom_coef': b2,
                'initial_storage': 30.0
            },
            # Tank 3: Baseflow
            {
                'side_outlets': [(a31, h31)],
                'bottom_coef': b3,
                'initial_storage': 50.0
            }
        ]
        super().__init__(configs)
        
        # Store parameters for reference
        self.a11 = a11
        self.h11 = h11
        self.a12 = a12
        self.h12 = h12
        self.b1 = b1
        self.a21 = a21
        self.h21 = h21
        self.b2 = b2
        self.a31 = a31
        self.h31 = h31
        self.b3 = b3


def create_tank_plots(model_1d, model_2d, model_3d, P, EP, results_1d, results_2d, results_3d, save_dir="figures"):
    """
    Create comprehensive visualization plots for Tank model results.
    
    Parameters:
    -----------
    model_1d, model_2d, model_3d : Tank models
        The model instances
    P, EP : np.ndarray
        Input precipitation and evapotranspiration
    results_1d, results_2d, results_3d : dict
        Model simulation results
    save_dir : str
        Directory to save figures
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Re-configure Chinese font after style setting / 样式设置后重新配置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong', 'KaiTi', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create date index
    n_days = len(P)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Figure 1: Tank Model Comparison
    fig, axes = plt.subplots(5, 1, figsize=(15, 16), sharex=True)
    fig.suptitle('Tank Models - Comprehensive Comparison', fontsize=16, fontweight='bold')
    
    # Precipitation (inverted)
    axes[0].bar(dates, P, color='steelblue', alpha=0.7, width=1)
    axes[0].set_ylabel('Precipitation\n(mm/day)', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(max(P) * 1.1, 0)
    
    # Evapotranspiration
    axes[1].plot(dates, EP, color='orange', linewidth=1.5, label='Potential ET')
    axes[1].fill_between(dates, EP, alpha=0.3, color='orange')
    axes[1].set_ylabel('Evapotranspiration\n(mm/day)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Discharge comparison
    axes[2].plot(dates, results_1d['Q'], color='red', linewidth=2, label='1D Tank', alpha=0.8)
    axes[2].plot(dates, results_2d['Q'], color='green', linewidth=2, label='2D Tank', alpha=0.8)
    axes[2].plot(dates, results_3d['Q'], color='blue', linewidth=2, label='3D Tank', alpha=0.8)
    axes[2].set_ylabel('Discharge\n(mm/day)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # 2D Tank storage components
    axes[3].plot(dates, results_2d['storages'][:, 0], color='red', linewidth=2, label='Tank 1 (Surface)')
    axes[3].plot(dates, results_2d['storages'][:, 1], color='blue', linewidth=2, label='Tank 2 (Baseflow)')
    axes[3].fill_between(dates, results_2d['storages'][:, 0], alpha=0.3, color='red')
    axes[3].fill_between(dates, results_2d['storages'][:, 1], alpha=0.3, color='blue')
    axes[3].set_ylabel('2D Tank Storage\n(mm)', fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # 3D Tank storage components
    axes[4].plot(dates, results_3d['storages'][:, 0], color='red', linewidth=2, label='Tank 1 (Surface)')
    axes[4].plot(dates, results_3d['storages'][:, 1], color='green', linewidth=2, label='Tank 2 (Interflow)')
    axes[4].plot(dates, results_3d['storages'][:, 2], color='blue', linewidth=2, label='Tank 3 (Baseflow)')
    axes[4].fill_between(dates, results_3d['storages'][:, 0], alpha=0.2, color='red')
    axes[4].fill_between(dates, results_3d['storages'][:, 1], alpha=0.2, color='green')
    axes[4].fill_between(dates, results_3d['storages'][:, 2], alpha=0.2, color='blue')
    axes[4].set_ylabel('3D Tank Storage\n(mm)', fontweight='bold')
    axes[4].set_xlabel('Date', fontweight='bold')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tank_models_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Flow components analysis for 3D Tank
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('3D Tank Model - Flow Components Analysis', fontsize=16, fontweight='bold')
    
    # Flow components (with safety check)
    if 'flows' in results_3d and results_3d['flows'].shape[1] >= 3:
        flow_1 = results_3d['flows'][:, 0]  # Surface flow
        flow_2 = results_3d['flows'][:, 1]  # Interflow
        flow_3 = results_3d['flows'][:, 2]  # Baseflow
        
        axes[0, 0].plot(dates, flow_1, label='Surface Flow', color='red', linewidth=2)
        axes[0, 0].plot(dates, flow_2, label='Interflow', color='green', linewidth=2)
        axes[0, 0].plot(dates, flow_3, label='Baseflow', color='blue', linewidth=2)
        axes[0, 0].set_title('Flow Components', fontweight='bold')
        axes[0, 0].set_ylabel('Flow (mm/day)', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Flow proportions (pie chart for total flows)
        total_flows = [np.sum(flow_1), np.sum(flow_2), np.sum(flow_3)]
        labels = ['Surface Flow', 'Interflow', 'Baseflow']
        colors = ['red', 'green', 'blue']
        
        axes[0, 1].pie(total_flows, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Total Flow Distribution', fontweight='bold')
    else:
        # Fallback plot if flows not available
        axes[0, 0].plot(dates, results_3d['Q'], label='Total Discharge', color='blue', linewidth=2)
        axes[0, 0].set_title('Total Discharge (Flow components not available)', fontweight='bold')
        axes[0, 0].set_ylabel('Flow (mm/day)', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].text(0.5, 0.5, 'Flow component\ndata not available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # Storage-discharge relationships
    total_storage = np.sum(results_3d['storages'], axis=1)
    axes[1, 0].scatter(total_storage, results_3d['Q'], alpha=0.6, c=P, cmap='Blues')
    axes[1, 0].set_xlabel('Total Storage (mm)', fontweight='bold')
    axes[1, 0].set_ylabel('Total Discharge (mm/day)', fontweight='bold')
    axes[1, 0].set_title('Storage-Discharge Relationship', fontweight='bold')
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Precipitation (mm/day)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precipitation vs discharge comparison for all models
    axes[1, 1].scatter(P, results_1d['Q'], alpha=0.5, label='1D Tank', color='red', s=20)
    axes[1, 1].scatter(P, results_2d['Q'], alpha=0.5, label='2D Tank', color='green', s=20)
    axes[1, 1].scatter(P, results_3d['Q'], alpha=0.5, label='3D Tank', color='blue', s=20)
    axes[1, 1].set_xlabel('Precipitation (mm/day)', fontweight='bold')
    axes[1, 1].set_ylabel('Discharge (mm/day)', fontweight='bold')
    axes[1, 1].set_title('P-Q Relationship Comparison', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tank_flow_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Statistical analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Tank Models - Statistical Analysis', fontsize=16, fontweight='bold')
    
    # Discharge distributions
    axes[0, 0].hist(results_1d['Q'], bins=30, alpha=0.7, label='1D Tank', color='red', density=True)
    axes[0, 0].hist(results_2d['Q'], bins=30, alpha=0.7, label='2D Tank', color='green', density=True)
    axes[0, 0].hist(results_3d['Q'], bins=30, alpha=0.7, label='3D Tank', color='blue', density=True)
    axes[0, 0].set_xlabel('Discharge (mm/day)', fontweight='bold')
    axes[0, 0].set_ylabel('Probability Density', fontweight='bold')
    axes[0, 0].set_title('Discharge Distributions', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Flow duration curves
    for i, (result, label, color) in enumerate([(results_1d, '1D Tank', 'red'),
                                               (results_2d, '2D Tank', 'green'),
                                               (results_3d, '3D Tank', 'blue')]):
        q_sorted = np.sort(result['Q'])[::-1]
        exceedance = np.arange(1, len(q_sorted) + 1) / len(q_sorted) * 100
        axes[0, 1].semilogy(exceedance, q_sorted, color=color, linewidth=2, label=label)
    
    axes[0, 1].set_xlabel('Exceedance Probability (%)', fontweight='bold')
    axes[0, 1].set_ylabel('Discharge (mm/day)', fontweight='bold')
    axes[0, 1].set_title('Flow Duration Curves', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Model performance comparison
    models = ['1D Tank', '2D Tank', '3D Tank']
    results_list = [results_1d, results_2d, results_3d]
    
    total_q = [np.sum(r['Q']) for r in results_list]
    runoff_coef = [np.sum(r['Q']) / np.sum(P) for r in results_list]
    peak_q = [np.max(r['Q']) for r in results_list]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = axes[1, 0].bar(x - width, total_q, width, label='Total Discharge (mm)', color='skyblue')
    bars2 = axes[1, 0].bar(x, [rc * 1000 for rc in runoff_coef], width, label='Runoff Coef × 1000', color='lightgreen')
    bars3 = axes[1, 0].bar(x + width, [pq * 100 for pq in peak_q], width, label='Peak Q × 100', color='salmon')
    
    axes[1, 0].set_xlabel('Model Type', fontweight='bold')
    axes[1, 0].set_ylabel('Value', fontweight='bold')
    axes[1, 0].set_title('Model Performance Metrics', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Tank storage capacity utilization (3D model)
    tank_capacities = [50, 30, 20]  # Assumed capacities
    avg_storages = [np.mean(results_3d['storages'][:, i]) for i in range(3)]
    utilizations = [avg/cap*100 for avg, cap in zip(avg_storages, tank_capacities)]
    
    bars = axes[1, 1].bar(['Tank 1', 'Tank 2', 'Tank 3'], utilizations, 
                         color=['red', 'green', 'blue'], alpha=0.7)
    axes[1, 1].set_ylabel('Capacity Utilization (%)', fontweight='bold')
    axes[1, 1].set_title('Storage Capacity Utilization (3D Model)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, util in zip(bars, utilizations):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{util:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tank_statistical_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Tank model visualization plots saved to '{save_dir}' directory")
    print("  Generated files:")
    print("  - tank_models_comparison.png")
    print("  - tank_flow_analysis.png")
    print("  - tank_statistical_analysis.png")


def main():
    """
    Example usage of Tank models with randomly generated data.
    """
    print("=" * 80)
    print("Tank Model - Hydrological Model Examples")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic input data (365 days)
    n_days = 365
    
    # Precipitation: realistic pattern with dry and wet periods
    P = np.random.gamma(2, 5, n_days)
    P[np.random.rand(n_days) > 0.4] = 0  # 60% dry days
    
    # Potential evapotranspiration: seasonal variation
    t = np.arange(n_days)
    EP = 3.0 + 2.0 * np.sin(2 * np.pi * t / 365)
    
    print("\nInput Data Summary:")
    print(f"  Simulation period: {n_days} days")
    print(f"  Total precipitation: {np.sum(P):.2f} mm")
    print(f"  Average daily precipitation: {np.mean(P):.2f} mm")
    print(f"  Total ET: {np.sum(EP):.2f} mm")
    
    # ========================================================================
    # 1D Tank Model Example
    # ========================================================================
    print("\n" + "=" * 80)
    print("1D Tank Model (Single Tank)")
    print("=" * 80)
    
    model_1d = TankModel1D(a1=0.3, h1=20.0, b1=0.05)
    results_1d = model_1d.run(P, EP)
    
    print("\nModel Parameters:")
    print(f"  a1 (discharge coef): {model_1d.a1}")
    print(f"  h1 (height threshold): {model_1d.h1} mm")
    print(f"  b1 (percolation coef): {model_1d.b1}")
    
    print("\nResults:")
    print(f"  Total discharge: {np.sum(results_1d['Q']):.2f} mm")
    print(f"  Runoff coefficient: {np.sum(results_1d['Q']) / np.sum(P):.3f}")
    print(f"  Peak discharge: {np.max(results_1d['Q']):.2f} mm/day")
    print(f"  Average storage: {np.mean(results_1d['storages']):.2f} mm")
    
    # ========================================================================
    # 2D Tank Model Example
    # ========================================================================
    print("\n" + "=" * 80)
    print("2D Tank Model (Two Tanks - Surface + Baseflow)")
    print("=" * 80)
    
    model_2d = TankModel2D(
        a11=0.3, h11=20.0,
        a21=0.1, h21=5.0,
        b1=0.1, b2=0.01
    )
    results_2d = model_2d.run(P, EP)
    
    print("\nModel Parameters:")
    print("  Tank 1 (Surface):")
    print(f"    a11: 0.3, h11: 20.0 mm")
    print(f"    b1: 0.1")
    print("  Tank 2 (Baseflow):")
    print(f"    a21: 0.1, h21: 5.0 mm")
    print(f"    b2: 0.01")
    
    print("\nResults:")
    print(f"  Total discharge: {np.sum(results_2d['Q']):.2f} mm")
    print(f"  Runoff coefficient: {np.sum(results_2d['Q']) / np.sum(P):.3f}")
    print(f"  Peak discharge: {np.max(results_2d['Q']):.2f} mm/day")
    print(f"  Average Tank 1 storage: {np.mean(results_2d['storages'][:, 0]):.2f} mm")
    print(f"  Average Tank 2 storage: {np.mean(results_2d['storages'][:, 1]):.2f} mm")
    
    # ========================================================================
    # 3D Tank Model Example
    # ========================================================================
    print("\n" + "=" * 80)
    print("3D Tank Model (Three Tanks - Surface + Interflow + Baseflow)")
    print("=" * 80)
    
    model_3d = TankModel3D(
        # Tank 1
        a11=0.5, h11=30.0,
        a12=0.8, h12=50.0,
        b1=0.1,
        # Tank 2
        a21=0.2, h21=10.0,
        b2=0.05,
        # Tank 3
        a31=0.05, h31=5.0,
        b3=0.001
    )
    results_3d = model_3d.run(P, EP)
    
    print("\nModel Parameters:")
    print("  Tank 1 (Surface):")
    print(f"    Outlet 1: a11=0.5, h11=30.0 mm")
    print(f"    Outlet 2: a12=0.8, h12=50.0 mm")
    print(f"    Bottom: b1=0.1")
    print("  Tank 2 (Interflow):")
    print(f"    Outlet 1: a21=0.2, h21=10.0 mm")
    print(f"    Bottom: b2=0.05")
    print("  Tank 3 (Baseflow):")
    print(f"    Outlet 1: a31=0.05, h31=5.0 mm")
    print(f"    Bottom: b3=0.001")
    
    print("\nResults:")
    print(f"  Total discharge: {np.sum(results_3d['Q']):.2f} mm")
    print(f"  Runoff coefficient: {np.sum(results_3d['Q']) / np.sum(P):.3f}")
    print(f"  Peak discharge: {np.max(results_3d['Q']):.2f} mm/day")
    print(f"  Average Tank 1 storage: {np.mean(results_3d['storages'][:, 0]):.2f} mm")
    print(f"  Average Tank 2 storage: {np.mean(results_3d['storages'][:, 1]):.2f} mm")
    print(f"  Average Tank 3 storage: {np.mean(results_3d['storages'][:, 2]):.2f} mm")
    
    # ========================================================================
    # Comparison
    # ========================================================================
def main():
    """
    Enhanced demonstration of Tank Model variants with visualization.
    """
    print("=" * 80)
    print("Tank Model (タンクモデル) - Enhanced Demonstration")
    print("Comprehensive Rainfall-Runoff Model with Multiple Configurations")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate more realistic synthetic data (2 years)
    n_days = 730
    
    # More complex precipitation pattern
    t = np.arange(n_days)
    
    # Seasonal precipitation with wet/dry cycles
    seasonal_factor = 1.2 + 0.8 * np.sin(2 * np.pi * t / 365 + np.pi/3)
    P_base = np.random.gamma(1.8, 3, n_days) * seasonal_factor
    
    # Add dry periods (more realistic)
    dry_prob = 0.65 + 0.15 * np.sin(2 * np.pi * t / 365)
    P = np.where(np.random.rand(n_days) < dry_prob, 0, P_base)
    
    # Add extreme events
    extreme_events = np.random.choice(n_days, size=8, replace=False)
    P[extreme_events] = P[extreme_events] + np.random.gamma(4, 8, 8)
    
    # Enhanced evapotranspiration with realistic seasonal cycle
    EP_mean = 3.5
    EP_amplitude = 2.0
    EP = EP_mean + EP_amplitude * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.2, n_days)
    EP = np.maximum(EP, 0.5)
    
    print("\nEnhanced Input Data Summary:")
    print(f"  Simulation period: {n_days} days ({n_days/365:.1f} years)")
    print(f"  Total precipitation: {np.sum(P):.2f} mm")
    print(f"  Average daily precipitation: {np.mean(P):.2f} mm")
    print(f"  Maximum daily precipitation: {np.max(P):.2f} mm")
    print(f"  Rainfall days: {np.sum(P > 0.1)} ({np.sum(P > 0.1)/n_days*100:.1f}%)")
    print(f"  Total potential ET: {np.sum(EP):.2f} mm")
    print(f"  Average daily ET: {np.mean(EP):.2f} mm")
    print(f"  P/ET ratio: {np.sum(P)/np.sum(EP):.3f}")
    
    # ========================================================================
    # 1D Tank Model Example
    # ========================================================================
    print("\n" + "=" * 80)
    print("1D Tank Model (Single Tank) - Enhanced Configuration")
    print("=" * 80)
    
    model_1d = TankModel1D(a1=0.35, h1=25.0, b1=0.08)
    results_1d = model_1d.run(P, EP)
    
    print("\nModel Parameters:")
    print(f"  a1 (discharge coef): {model_1d.a1}")
    print(f"  h1 (height threshold): {model_1d.h1} mm")
    print(f"  b1 (percolation coef): {model_1d.b1}")
    
    print("\nResults:")
    print(f"  Total discharge: {np.sum(results_1d['Q']):.2f} mm")
    print(f"  Annual discharge: {np.sum(results_1d['Q'])/2:.2f} mm/year")
    print(f"  Runoff coefficient: {np.sum(results_1d['Q']) / np.sum(P):.3f}")
    print(f"  Peak discharge: {np.max(results_1d['Q']):.2f} mm/day")
    print(f"  Mean discharge: {np.mean(results_1d['Q']):.2f} mm/day")
    print(f"  Average storage: {np.mean(results_1d['storages']):.2f} mm")
    print(f"  Storage coefficient of variation: {np.std(results_1d['storages'])/np.mean(results_1d['storages']):.3f}")
    
    # ========================================================================
    # 2D Tank Model Example
    # ========================================================================
    print("\n" + "=" * 80)
    print("2D Tank Model (Two Tanks - Surface + Baseflow) - Enhanced Configuration")
    print("=" * 80)
    
    model_2d = TankModel2D(
        a11=0.4, h11=25.0,
        a21=0.15, h21=8.0,
        b1=0.12, b2=0.02
    )
    results_2d = model_2d.run(P, EP)
    
    print("\nModel Parameters:")
    print("  Tank 1 (Surface):")
    print(f"    a11: {model_2d.a11}, h11: {model_2d.h11} mm")
    print(f"    b1: {model_2d.b1}")
    print("  Tank 2 (Baseflow):")
    print(f"    a21: {model_2d.a21}, h21: {model_2d.h21} mm")
    print(f"    b2: {model_2d.b2}")
    
    print("\nResults:")
    print(f"  Total discharge: {np.sum(results_2d['Q']):.2f} mm")
    print(f"  Annual discharge: {np.sum(results_2d['Q'])/2:.2f} mm/year")
    print(f"  Runoff coefficient: {np.sum(results_2d['Q']) / np.sum(P):.3f}")
    print(f"  Peak discharge: {np.max(results_2d['Q']):.2f} mm/day")
    print(f"  Mean discharge: {np.mean(results_2d['Q']):.2f} mm/day")
    print(f"  Average Tank 1 storage: {np.mean(results_2d['storages'][:, 0]):.2f} mm")
    print(f"  Average Tank 2 storage: {np.mean(results_2d['storages'][:, 1]):.2f} mm")
    
    # Calculate flow components for 2D model
    if len(results_2d['flows'][0]) >= 2:
        surface_flow = np.mean(results_2d['flows'][:, 0])
        baseflow = np.mean(results_2d['flows'][:, 1]) 
        print(f"  Average surface flow: {surface_flow:.2f} mm/day ({surface_flow/np.mean(results_2d['Q'])*100:.1f}%)")
        print(f"  Average baseflow: {baseflow:.2f} mm/day ({baseflow/np.mean(results_2d['Q'])*100:.1f}%)")
    else:
        print(f"  Flow component details not available")
    
    # ========================================================================
    # 3D Tank Model Example
    # ========================================================================
    print("\n" + "=" * 80)
    print("3D Tank Model (Three Tanks - Surface + Interflow + Baseflow) - Enhanced Configuration")
    print("=" * 80)
    
    model_3d = TankModel3D(
        # Tank 1 - Surface runoff with two outlets
        a11=0.6, h11=35.0,
        a12=0.9, h12=55.0,
        b1=0.15,
        # Tank 2 - Interflow
        a21=0.25, h21=12.0,
        b2=0.08,
        # Tank 3 - Baseflow
        a31=0.08, h31=6.0,
        b3=0.005
    )
    results_3d = model_3d.run(P, EP)
    
    print("\nModel Parameters:")
    print("  Tank 1 (Surface):")
    print(f"    Outlet 1: a11={model_3d.a11}, h11={model_3d.h11} mm")
    print(f"    Outlet 2: a12={model_3d.a12}, h12={model_3d.h12} mm")
    print(f"    Bottom: b1={model_3d.b1}")
    print("  Tank 2 (Interflow):")
    print(f"    Outlet 1: a21={model_3d.a21}, h21={model_3d.h21} mm")
    print(f"    Bottom: b2={model_3d.b2}")
    print("  Tank 3 (Baseflow):")
    print(f"    Outlet 1: a31={model_3d.a31}, h31={model_3d.h31} mm")
    print(f"    Bottom: b3={model_3d.b3}")
    
    print("\nResults:")
    print(f"  Total discharge: {np.sum(results_3d['Q']):.2f} mm")
    print(f"  Annual discharge: {np.sum(results_3d['Q'])/2:.2f} mm/year")
    print(f"  Runoff coefficient: {np.sum(results_3d['Q']) / np.sum(P):.3f}")
    print(f"  Peak discharge: {np.max(results_3d['Q']):.2f} mm/day")
    print(f"  Mean discharge: {np.mean(results_3d['Q']):.2f} mm/day")
    print(f"  Average Tank 1 storage: {np.mean(results_3d['storages'][:, 0]):.2f} mm")
    print(f"  Average Tank 2 storage: {np.mean(results_3d['storages'][:, 1]):.2f} mm")
    print(f"  Average Tank 3 storage: {np.mean(results_3d['storages'][:, 2]):.2f} mm")
    
    # Calculate flow components for 3D model
    if len(results_3d['flows'][0]) >= 3:
        surface_flow = np.mean(results_3d['flows'][:, 0])
        interflow = np.mean(results_3d['flows'][:, 1])
        baseflow = np.mean(results_3d['flows'][:, 2])
        total_flow = surface_flow + interflow + baseflow
        
        print(f"  Flow components:")
        print(f"    Surface flow: {surface_flow:.2f} mm/day ({surface_flow/total_flow*100:.1f}%)")
        print(f"    Interflow: {interflow:.2f} mm/day ({interflow/total_flow*100:.1f}%)")
        print(f"    Baseflow: {baseflow:.2f} mm/day ({baseflow/total_flow*100:.1f}%)")
    else:
        print(f"  Flow component details not available")
    
    # ========================================================================
    # Enhanced Comparison and Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("Comprehensive Model Comparison and Analysis")
    print("=" * 80)
    
    models_data = [
        ("1D Tank", results_1d, 'Single storage unit'),
        ("2D Tank", results_2d, 'Surface + Baseflow'),
        ("3D Tank", results_3d, 'Surface + Interflow + Baseflow')
    ]
    
    print(f"\n{'Model':<12} {'Total Q':<12} {'Runoff':<12} {'Peak Q':<12} {'Mean Q':<12} {'Q95':<12} {'Q5':<12}")
    print(f"{'Type':<12} {'(mm)':<12} {'Coef':<12} {'(mm/d)':<12} {'(mm/d)':<12} {'(mm/d)':<12} {'(mm/d)':<12}")
    print("-" * 84)
    
    for name, result, desc in models_data:
        total_q = np.sum(result['Q'])
        runoff_coef = total_q / np.sum(P)
        peak_q = np.max(result['Q'])
        mean_q = np.mean(result['Q'])
        q95 = np.percentile(result['Q'], 5)  # Low flows
        q5 = np.percentile(result['Q'], 95)  # High flows
        
        print(f"{name:<12} {total_q:<12.2f} {runoff_coef:<12.3f} {peak_q:<12.2f} "
              f"{mean_q:<12.2f} {q95:<12.3f} {q5:<12.2f}")
    
    # Water balance analysis
    print(f"\nWater Balance Analysis:")
    for name, result, desc in models_data:
        total_input = np.sum(P)
        total_output = np.sum(result['Q']) + np.sum(EP)
        if 'storages' in result:
            if result['storages'].ndim == 1:
                storage_change = result['storages'][-1] - result['storages'][0]
            else:
                storage_change = np.sum(result['storages'][-1]) - np.sum(result['storages'][0])
        else:
            storage_change = 0
        
        balance_error = total_input - total_output - storage_change
        print(f"  {name}: Balance error = {balance_error:.2f} mm ({abs(balance_error)/total_input*100:.2f}%)")
    
    # Create comprehensive visualizations
    print(f"\nGenerating comprehensive visualizations...")
    create_tank_plots(model_1d, model_2d, model_3d, P, EP, results_1d, results_2d, results_3d, save_dir="figures")
    
    # Display enhanced sample results
    print("\nDetailed Results for First 15 Days (3D Tank Model):")
    if 'flows' in results_3d and results_3d['flows'].shape[1] >= 3:
        print("Day |   Date   |   P    |   EP   |   Q    | Tank1  | Tank2  | Tank3  | Surface | Inter | Base |")
        print("----|----------|--------|--------|--------|--------|--------|--------|---------|-------|------|")
        
        start_date = datetime(2020, 1, 1)
        for i in range(15):
            date_str = (start_date + timedelta(days=i)).strftime("%m-%d")
            print(f"{i+1:3d} | {date_str} | {P[i]:6.2f} | {EP[i]:6.2f} | {results_3d['Q'][i]:6.2f} | "
                  f"{results_3d['storages'][i,0]:6.2f} | {results_3d['storages'][i,1]:6.2f} | "
                  f"{results_3d['storages'][i,2]:6.2f} | {results_3d['flows'][i,0]:7.3f} | "
                  f"{results_3d['flows'][i,1]:5.3f} | {results_3d['flows'][i,2]:4.3f} |")
    else:
        print("Day |   Date   |   P    |   EP   |   Q    | Tank1  | Tank2  | Tank3  |")
        print("----|----------|--------|--------|--------|--------|--------|--------|")
        
        start_date = datetime(2020, 1, 1)
        for i in range(15):
            date_str = (start_date + timedelta(days=i)).strftime("%m-%d")
            print(f"{i+1:3d} | {date_str} | {P[i]:6.2f} | {EP[i]:6.2f} | {results_3d['Q'][i]:6.2f} | "
                  f"{results_3d['storages'][i,0]:6.2f} | {results_3d['storages'][i,1]:6.2f} | "
                  f"{results_3d['storages'][i,2]:6.2f} |")
    
    # Performance statistics
    print(f"\nHydrological Performance Indicators:")
    print(f"  Model complexity vs efficiency:")
    
    for name, result, desc in models_data:
        # Calculate some performance metrics
        q_std = np.std(result['Q'])
        q_cv = q_std / np.mean(result['Q'])
        peak_ratio = np.max(result['Q']) / np.mean(result['Q'])
        
        print(f"    {name}: CV = {q_cv:.3f}, Peak/Mean = {peak_ratio:.2f}")
    
    print("\n" + "=" * 80)
    print("Enhanced Tank Model simulation completed successfully!")
    print("Check the 'figures' directory for comprehensive visualizations.")
    print("Models demonstrate different runoff generation mechanisms:")
    print("  - 1D: Simple lumped response")
    print("  - 2D: Surface/subsurface separation") 
    print("  - 3D: Complete runoff component separation")
    print("=" * 80)
    
    return results_1d, results_2d, results_3d


if __name__ == "__main__":
    main()
