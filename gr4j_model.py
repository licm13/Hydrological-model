"""
GR4J (Modèle du Génie Rural à 4 paramètres Journalier)
Rural Engineering 4-Parameter Daily Model

GR4J is a lumped, conceptual rainfall-runoff model developed by Perrin et al. (2003)
at INRAE (formerly Cemagref), France. It is widely used for daily streamflow simulation
and forecasting.

Mathematical Foundation:
========================

The model operates in several steps:

1. Production Store (Soil Moisture Accounting):
   
   Net rainfall: Pn = max(0, P - E)
   Net evapotranspiration: En = max(0, E - P)
   
   Reservoir level S: 
   - If Pn > 0:
     Ps = (X1 * (1 - (S/X1)^2) * tanh(Pn/X1)) / (1 + (S/X1) * tanh(Pn/X1))
     S = S + Ps
   
   - If En > 0:
     Es = (S * (2 - S/X1) * tanh(En/X1)) / (1 + (1 - S/X1) * tanh(En/X1))
     S = S - Es
   
   Percolation: Perc = S * (1 - (1 + (4/9 * S/X1)^4)^(-1/4))
   S = S - Perc

2. Unit Hydrograph Split:
   
   Total water to route: Pr = Pn - Ps + Perc
   
   90% routed through UH1 (unit hydrograph 1)
   10% routed through UH2 (unit hydrograph 2)

3. Routing Store:
   
   Input to routing: Q9 (from UH1)
   
   If Q9 > 0:
     Qr = (X3 * (1 - (R/X3)^2) * tanh(Q9/X3)) / (1 + (R/X3) * tanh(Q9/X3))
     R = R + Qr
   
   Outflow: Qd = R * (1 - (1 + (X2/R)^4)^(-1/4))
   R = R - Qd

4. Groundwater Exchange:
   
   F = X2 * (R / X3)^(7/2)

5. Final Discharge:
   
   Q1 (from UH2) + max(0, Qd + F)

Parameters:
-----------
X1 : Maximum capacity of production store (mm, typically 100-1200)
X2 : Groundwater exchange coefficient (mm, typically -5 to 5)
X3 : Maximum capacity of routing store (mm, typically 20-300)
X4 : Time base of unit hydrograph (days, typically 1.0-4.0)

References:
-----------
Perrin, C., Michel, C., & Andréassian, V. (2003). 
Improvement of a parsimonious model for streamflow simulation. 
Journal of Hydrology, 279(1-4), 275-289.

Author: Perrin et al. (Original), Implementation by [Your Name]
Date: 2024
"""

import numpy as np
from typing import Tuple, Dict
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


class GR4J:
    """
    GR4J hydrological model implementation.
    
    Parameters:
    -----------
    X1 : float
        Maximum capacity of production store (mm, default: 350)
    X2 : float
        Groundwater exchange coefficient (mm, default: 0.0)
    X3 : float
        Maximum capacity of routing store (mm, default: 90)
    X4 : float
        Time base of unit hydrograph (days, default: 1.7)
    """
    
    def __init__(self, X1: float = 350.0, X2: float = 0.0, 
                 X3: float = 90.0, X4: float = 1.7):
        # Model parameters
        self.X1 = X1  # Production store capacity
        self.X2 = X2  # Water exchange coefficient
        self.X3 = X3  # Routing store capacity
        self.X4 = X4  # Unit hydrograph time base
        
        # State variables
        self.S = X1 * 0.5   # Production store level (initial: 50%)
        self.R = X3 * 0.3   # Routing store level (initial: 30%)
        
        # Unit hydrographs
        self._compute_unit_hydrographs()
        
        # UH ordinates storage
        self.UH1_queue = np.zeros(len(self.UH1))
        self.UH2_queue = np.zeros(len(self.UH2))
        
    def _compute_unit_hydrographs(self):
        """
        Compute unit hydrograph ordinates (SH1 and SH2).
        
        UH1: Fast response unit hydrograph (time base = X4)
        UH2: Slow response unit hydrograph (time base = 2*X4)
        """
        # UH1 (fast component) - time base X4
        n1 = int(np.ceil(self.X4))
        SH1 = np.zeros(n1)
        
        for t in range(1, n1 + 1):
            if t <= self.X4:
                SH1[t-1] = (t / self.X4) ** 2.5
            else:
                SH1[t-1] = 1.0
        
        # Convert S-curve to unit hydrograph
        self.UH1 = np.zeros(n1)
        self.UH1[0] = SH1[0]
        for t in range(1, n1):
            self.UH1[t] = SH1[t] - SH1[t-1]
        
        # UH2 (slow component) - time base 2*X4
        n2 = int(np.ceil(2 * self.X4))
        SH2 = np.zeros(n2)
        
        for t in range(1, n2 + 1):
            if t <= self.X4:
                SH2[t-1] = 0.5 * (t / self.X4) ** 2.5
            elif t <= 2 * self.X4:
                SH2[t-1] = 1.0 - 0.5 * (2 - t / self.X4) ** 2.5
            else:
                SH2[t-1] = 1.0
        
        # Convert S-curve to unit hydrograph
        self.UH2 = np.zeros(n2)
        self.UH2[0] = SH2[0]
        for t in range(1, n2):
            self.UH2[t] = SH2[t] - SH2[t-1]
    
    def _production_store(self, P: float, E: float) -> Tuple[float, float]:
        """
        Production store (soil moisture accounting).
        
        Parameters:
        -----------
        P : float
            Precipitation (mm)
        E : float
            Potential evapotranspiration (mm)
            
        Returns:
        --------
        Pn : float
            Net rainfall (mm)
        Perc : float
            Percolation (mm)
        """
        # Net precipitation and evapotranspiration
        if P >= E:
            Pn = P - E
            En = 0.0
        else:
            Pn = 0.0
            En = E - P
        
        # Production store update
        if Pn > 0:
            # Add water to store
            S_norm = self.S / self.X1
            if Pn > 0.01:  # Avoid numerical issues
                part1 = self.X1 * (1 - S_norm ** 2) * np.tanh(Pn / self.X1)
                part2 = 1 + S_norm * np.tanh(Pn / self.X1)
                if abs(part2) > 0.001:  # Safety check for division by zero
                    Ps = part1 / part2
                else:
                    Ps = 0.0
            else:
                Ps = 0.0
            
            self.S = self.S + Ps
        else:
            Ps = 0.0
        
        if En > 0:
            # Remove water from store
            S_norm = self.S / self.X1
            if En > 0.01:  # Avoid numerical issues
                part1 = self.S * (2 - S_norm) * np.tanh(En / self.X1)
                part2 = 1 + (1 - S_norm) * np.tanh(En / self.X1)
                Es = part1 / part2
            else:
                Es = 0.0
            
            self.S = max(0, self.S - Es)
        
        # Percolation from production store
        S_norm = self.S / self.X1
        if S_norm > 0:
            Perc = self.S * (1 - (1 + (4 * S_norm / 9) ** 4) ** (-0.25))
        else:
            Perc = 0.0
        
        self.S = max(0, self.S - Perc)
        
        # Effective rainfall for routing
        Pr = Pn - Ps + Perc
        
        return Pr, Ps
    
    def _routing(self, Q9: float, Q1: float) -> float:
        """
        Routing store and groundwater exchange.
        
        Parameters:
        -----------
        Q9 : float
            Input from UH1 (90% of effective rainfall)
        Q1 : float
            Input from UH2 (10% of effective rainfall)
            
        Returns:
        --------
        Q : float
            Total discharge (mm)
        """
        # Routing store update
        if Q9 > 0:
            R_norm = self.R / self.X3
            if Q9 > 0.01:
                part1 = self.X3 * (1 - R_norm ** 2) * np.tanh(Q9 / self.X3)
                part2 = 1 + R_norm * np.tanh(Q9 / self.X3)
                Qr = part1 / part2
            else:
                Qr = 0.0
            
            self.R = self.R + Qr
        
        # Outflow from routing store
        R_norm = self.R / self.X3
        if R_norm > 0 and self.R > 0.01:  # Add safety check for division by zero
            Qd = self.R * (1 - (1 + (self.X2 / self.R) ** 4) ** (-0.25))
        else:
            Qd = 0.0
        
        self.R = max(0, self.R - Qd)
        
        # Groundwater exchange
        R_norm = self.R / self.X3
        F = self.X2 * (R_norm ** 3.5)
        
        # Total discharge
        Q = max(0, Qd + F) + Q1
        
        return Q
    
    def run_timestep(self, P: float, E: float) -> Dict[str, float]:
        """
        Run one timestep of GR4J model.
        
        Parameters:
        -----------
        P : float
            Precipitation (mm)
        E : float
            Potential evapotranspiration (mm)
            
        Returns:
        --------
        dict : Model outputs
        """
        # 1. Production store
        Pr, Ps = self._production_store(P, E)
        
        # 2. Split effective rainfall
        # 90% goes through UH1, 10% through UH2
        Pr_90 = 0.9 * Pr
        Pr_10 = 0.1 * Pr
        
        # 3. Unit hydrograph convolution
        # Update queues
        self.UH1_queue = np.roll(self.UH1_queue, 1)
        self.UH1_queue[0] = Pr_90
        
        self.UH2_queue = np.roll(self.UH2_queue, 1)
        self.UH2_queue[0] = Pr_10
        
        # Calculate outputs
        Q9 = np.sum(self.UH1_queue * self.UH1)
        Q1 = np.sum(self.UH2_queue * self.UH2)
        
        # 4. Routing
        Q = self._routing(Q9, Q1)
        
        return {
            'Q': Q,
            'S': self.S,
            'R': self.R,
            'Pr': Pr
        }
    
    def run(self, P: np.ndarray, E: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run GR4J model for multiple timesteps.
        
        Parameters:
        -----------
        P : np.ndarray
            Precipitation time series (mm/day)
        E : np.ndarray
            Potential evapotranspiration time series (mm/day)
            
        Returns:
        --------
        dict : Model outputs
        """
        n_steps = len(P)
        
        Q = np.zeros(n_steps)
        S = np.zeros(n_steps)
        R = np.zeros(n_steps)
        
        for t in range(n_steps):
            result = self.run_timestep(P[t], E[t])
            Q[t] = result['Q']
            S[t] = result['S']
            R[t] = result['R']
        
        return {
            'Q': Q,
            'S': S,
            'R': R
        }

    def calibrate(self, P: np.ndarray, E: np.ndarray, Q_obs: np.ndarray, 
                  n_iterations: int = 1000) -> Dict[str, float]:
        """
        Simple calibration using random search.
        
        Parameters:
        -----------
        P, E : np.ndarray
            Input precipitation and ET
        Q_obs : np.ndarray
            Observed discharge for calibration
        n_iterations : int
            Number of calibration iterations
            
        Returns:
        --------
        dict : Best parameters found
        """
        best_nse = -np.inf
        best_params = None
        
        # Parameter bounds
        X1_bounds = (100, 1200)
        X2_bounds = (-5, 5)
        X3_bounds = (20, 300)
        X4_bounds = (1.0, 4.0)
        
        for i in range(n_iterations):
            # Random parameter sampling
            X1 = np.random.uniform(*X1_bounds)
            X2 = np.random.uniform(*X2_bounds)
            X3 = np.random.uniform(*X3_bounds)
            X4 = np.random.uniform(*X4_bounds)
            
            # Test parameters
            test_model = GR4J(X1, X2, X3, X4)
            results = test_model.run(P, E)
            
            # Calculate Nash-Sutcliffe Efficiency
            nse = self._calculate_nse(Q_obs, results['Q'])
            
            if nse > best_nse:
                best_nse = nse
                best_params = {'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'NSE': nse}
        
        return best_params
    
    def _calculate_nse(self, obs: np.ndarray, sim: np.ndarray) -> float:
        """Calculate Nash-Sutcliffe Efficiency."""
        if len(obs) != len(sim):
            return -np.inf
        
        # Remove NaN values
        valid_idx = ~(np.isnan(obs) | np.isnan(sim))
        obs_clean = obs[valid_idx]
        sim_clean = sim[valid_idx]
        
        if len(obs_clean) == 0:
            return -np.inf
        
        numerator = np.sum((obs_clean - sim_clean) ** 2)
        denominator = np.sum((obs_clean - np.mean(obs_clean)) ** 2)
        
        if denominator == 0:
            return -np.inf
        
        return 1 - numerator / denominator


def create_comprehensive_plots(model, P, E, results, save_dir="figures"):
    """
    Create comprehensive visualization plots for GR4J model results.
    
    Parameters:
    -----------
    model : GR4J
        The model instance
    P, E : np.ndarray
        Input precipitation and evapotranspiration
    results : dict
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
    
    # Figure 1: Time series overview
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('GR4J Model - Comprehensive Time Series Analysis', fontsize=16, fontweight='bold')
    
    # Precipitation (inverted)
    axes[0].bar(dates, P, color='steelblue', alpha=0.7, width=1)
    axes[0].set_ylabel('Precipitation\n(mm/day)', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(max(P) * 1.1, 0)
    
    # Evapotranspiration
    axes[1].plot(dates, E, color='orange', linewidth=1.5, label='Potential ET')
    axes[1].fill_between(dates, E, alpha=0.3, color='orange')
    axes[1].set_ylabel('Evapotranspiration\n(mm/day)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Discharge
    axes[2].plot(dates, results['Q'], color='red', linewidth=2, label='Simulated Discharge')
    axes[2].fill_between(dates, results['Q'], alpha=0.3, color='red')
    axes[2].set_ylabel('Discharge\n(mm/day)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Storage states
    axes[3].plot(dates, results['S'], color='green', linewidth=2, label=f'Production Store (S)')
    axes[3].plot(dates, results['R'], color='purple', linewidth=2, label=f'Routing Store (R)')
    axes[3].fill_between(dates, results['S'], alpha=0.2, color='green')
    axes[3].fill_between(dates, results['R'], alpha=0.2, color='purple')
    axes[3].set_ylabel('Storage\n(mm)', fontweight='bold')
    axes[3].set_xlabel('Date', fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gr4j_timeseries_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Water balance analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GR4J Model - Water Balance Analysis', fontsize=16, fontweight='bold')
    
    # Cumulative water balance
    cum_P = np.cumsum(P)
    cum_E = np.cumsum(E)
    cum_Q = np.cumsum(results['Q'])
    
    axes[0, 0].plot(dates, cum_P, label='Cumulative Precipitation', linewidth=2, color='blue')
    axes[0, 0].plot(dates, cum_E, label='Cumulative ET', linewidth=2, color='orange')
    axes[0, 0].plot(dates, cum_Q, label='Cumulative Discharge', linewidth=2, color='red')
    axes[0, 0].set_title('Cumulative Water Balance', fontweight='bold')
    axes[0, 0].set_ylabel('Cumulative Water (mm)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Monthly statistics
    monthly_P = []
    monthly_E = []
    monthly_Q = []
    months = []
    
    for month in range(1, 13):
        mask = [d.month == month for d in dates]
        if any(mask):
            monthly_P.append(np.mean(P[mask]))
            monthly_E.append(np.mean(E[mask]))
            monthly_Q.append(np.mean(results['Q'][mask]))
            months.append(month)
    
    x = np.arange(len(months))
    width = 0.25
    
    axes[0, 1].bar(x - width, monthly_P, width, label='Precipitation', color='blue', alpha=0.7)
    axes[0, 1].bar(x, monthly_E, width, label='Evapotranspiration', color='orange', alpha=0.7)
    axes[0, 1].bar(x + width, monthly_Q, width, label='Discharge', color='red', alpha=0.7)
    axes[0, 1].set_title('Monthly Average Water Fluxes', fontweight='bold')
    axes[0, 1].set_ylabel('Water Flux (mm/day)', fontweight='bold')
    axes[0, 1].set_xlabel('Month', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'M{m}' for m in months])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Storage dynamics
    axes[1, 0].scatter(results['S'], results['R'], c=results['Q'], cmap='viridis', alpha=0.6)
    axes[1, 0].set_xlabel('Production Store (mm)', fontweight='bold')
    axes[1, 0].set_ylabel('Routing Store (mm)', fontweight='bold')
    axes[1, 0].set_title('Storage State Space (colored by discharge)', fontweight='bold')
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Discharge (mm/day)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Runoff coefficient analysis
    runoff_coef = results['Q'] / (P + 1e-6)  # Add small value to avoid division by zero
    runoff_coef[P < 0.1] = np.nan  # Exclude very small precipitation events
    
    axes[1, 1].scatter(P, runoff_coef, alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('Precipitation (mm/day)', fontweight='bold')
    axes[1, 1].set_ylabel('Runoff Coefficient (-)', fontweight='bold')
    axes[1, 1].set_title('Runoff Coefficient vs Precipitation', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gr4j_water_balance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Model structure and parameters
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GR4J Model - Structure and Parameters Analysis', fontsize=16, fontweight='bold')
    
    # Unit hydrographs
    uh1_time = np.arange(len(model.UH1))
    uh2_time = np.arange(len(model.UH2))
    
    axes[0, 0].bar(uh1_time, model.UH1, alpha=0.7, color='blue', label=f'UH1 (X4={model.X4:.2f})')
    axes[0, 0].bar(uh2_time, model.UH2, alpha=0.7, color='red', label=f'UH2 (2×X4={2*model.X4:.2f})')
    axes[0, 0].set_title('Unit Hydrographs', fontweight='bold')
    axes[0, 0].set_xlabel('Time Steps', fontweight='bold')
    axes[0, 0].set_ylabel('Ordinate (-)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Parameter sensitivity (simplified)
    param_names = ['X1', 'X2', 'X3', 'X4']
    param_values = [model.X1, model.X2, model.X3, model.X4]
    param_colors = ['green', 'blue', 'purple', 'orange']
    
    bars = axes[0, 1].bar(param_names, param_values, color=param_colors, alpha=0.7)
    axes[0, 1].set_title('Model Parameters', fontweight='bold')
    axes[0, 1].set_ylabel('Parameter Value', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, param_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Flow duration curve
    q_sorted = np.sort(results['Q'])[::-1]
    exceedance = np.arange(1, len(q_sorted) + 1) / len(q_sorted) * 100
    
    axes[1, 0].semilogy(exceedance, q_sorted, color='red', linewidth=2)
    axes[1, 0].set_xlabel('Exceedance Probability (%)', fontweight='bold')
    axes[1, 0].set_ylabel('Discharge (mm/day)', fontweight='bold')
    axes[1, 0].set_title('Flow Duration Curve', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add flow statistics
    q_stats = {
        'Q95': np.percentile(results['Q'], 5),
        'Q50': np.percentile(results['Q'], 50),
        'Q10': np.percentile(results['Q'], 90)
    }
    
    for stat_name, stat_value in q_stats.items():
        axes[1, 0].axhline(y=stat_value, linestyle='--', alpha=0.7, 
                          label=f'{stat_name}: {stat_value:.3f} mm/day')
    axes[1, 0].legend()
    
    # Storage-discharge relationship
    axes[1, 1].scatter(results['S'] + results['R'], results['Q'], alpha=0.6, color='green')
    axes[1, 1].set_xlabel('Total Storage (S + R) (mm)', fontweight='bold')
    axes[1, 1].set_ylabel('Discharge (mm/day)', fontweight='bold')
    axes[1, 1].set_title('Total Storage vs Discharge', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    total_storage = results['S'] + results['R']
    correlation, p_value = pearsonr(total_storage, results['Q'])
    axes[1, 1].text(0.05, 0.95, f'r = {correlation:.3f}\np = {p_value:.3e}', 
                   transform=axes[1, 1].transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                   verticalalignment='top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gr4j_model_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Statistical analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GR4J Model - Statistical Analysis', fontsize=16, fontweight='bold')
    
    # Discharge histogram with statistics
    axes[0, 0].hist(results['Q'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(results['Q']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(results["Q"]):.3f}')
    axes[0, 0].axvline(np.median(results['Q']), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(results["Q"]):.3f}')
    axes[0, 0].set_xlabel('Discharge (mm/day)', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Discharge Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q plot for normality check
    stats.probplot(results['Q'], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precipitation vs discharge scatter
    axes[1, 0].scatter(P, results['Q'], alpha=0.6, color='purple')
    
    # Add trend line
    valid_idx = (P > 0) & (results['Q'] > 0)
    if np.sum(valid_idx) > 5:
        z = np.polyfit(P[valid_idx], results['Q'][valid_idx], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(P[valid_idx], p(P[valid_idx]), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr, p_val = pearsonr(P[valid_idx], results['Q'][valid_idx])
        axes[1, 0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3e}', 
                       transform=axes[1, 0].transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                       verticalalignment='top', fontweight='bold')
    
    axes[1, 0].set_xlabel('Precipitation (mm/day)', fontweight='bold')
    axes[1, 0].set_ylabel('Discharge (mm/day)', fontweight='bold')
    axes[1, 0].set_title('Precipitation vs Discharge', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Storage dynamics over time (rolling statistics)
    window = 30  # 30-day rolling window
    if len(results['S']) >= window:
        s_rolling = np.convolve(results['S'], np.ones(window)/window, mode='valid')
        r_rolling = np.convolve(results['R'], np.ones(window)/window, mode='valid')
        dates_rolling = dates[window-1:]
        
        axes[1, 1].plot(dates_rolling, s_rolling, label='Production Store (30-day avg)', linewidth=2, color='green')
        axes[1, 1].plot(dates_rolling, r_rolling, label='Routing Store (30-day avg)', linewidth=2, color='purple')
        axes[1, 1].fill_between(dates_rolling, s_rolling, alpha=0.3, color='green')
        axes[1, 1].fill_between(dates_rolling, r_rolling, alpha=0.3, color='purple')
        
        # Format x-axis
        axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[1, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
    
    axes[1, 1].set_xlabel('Date', fontweight='bold')
    axes[1, 1].set_ylabel('Storage (mm)', fontweight='bold')
    axes[1, 1].set_title('Storage Dynamics (Rolling Average)', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gr4j_statistical_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
def main():
    """
    Enhanced example usage of GR4J model with comprehensive analysis and visualization.
    """
    print("=" * 80)
    print("GR4J (Modèle du Génie Rural à 4 paramètres Journalier)")
    print("Rural Engineering 4-Parameter Daily Model - Enhanced Version")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate more realistic synthetic data (3 years for better analysis)
    n_days = 1095  # 3 years
    
    # Create more realistic precipitation pattern
    # Seasonal and wet/dry periods
    t = np.arange(n_days)
    
    # Base precipitation with seasonal cycle
    seasonal_factor = 1.5 + 0.8 * np.sin(2 * np.pi * t / 365 + np.pi)  # Winter wet season
    
    # Generate precipitation with gamma distribution
    P_base = np.random.gamma(1.5, 4, n_days) * seasonal_factor
    
    # Add dry periods (60% chance of no rain)
    dry_prob = 0.6 + 0.2 * np.sin(2 * np.pi * t / 365)  # More dry days in summer
    P = np.where(np.random.rand(n_days) < dry_prob, 0, P_base)
    
    # Add some extreme events
    extreme_events = np.random.choice(n_days, size=10, replace=False)
    P[extreme_events] = P[extreme_events] + np.random.gamma(5, 10, 10)
    
    # Realistic evapotranspiration with seasonal cycle
    E_mean = 4.0
    E_amplitude = 2.5
    E = E_mean + E_amplitude * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.3, n_days)
    E = np.maximum(E, 0.5)  # Minimum ET
    
    # Add temperature effect (higher ET in summer)
    temp_effect = 1 + 0.5 * np.sin(2 * np.pi * t / 365)
    E = E * temp_effect
    
    print("\nEnhanced Input Data Summary:")
    print(f"  Simulation period: {n_days} days ({n_days/365:.1f} years)")
    print(f"  Total precipitation: {np.sum(P):.2f} mm")
    print(f"  Average daily precipitation: {np.mean(P):.2f} mm")
    print(f"  Maximum daily precipitation: {np.max(P):.2f} mm")
    print(f"  Rainfall days: {np.sum(P > 0.1)} ({np.sum(P > 0.1)/n_days*100:.1f}%)")
    print(f"  Total potential ET: {np.sum(E):.2f} mm")
    print(f"  Average daily ET: {np.mean(E):.2f} mm")
    print(f"  P/ET ratio: {np.sum(P)/np.sum(E):.3f}")
    
    # Initialize model with realistic parameters for temperate climate
    model = GR4J(X1=400.0, X2=1.2, X3=85.0, X4=2.1)
    
    print("\nModel Parameters:")
    print(f"  X1 (Production capacity): {model.X1:.1f} mm")
    print(f"  X2 (Water exchange): {model.X2:.2f} mm")
    print(f"  X3 (Routing capacity): {model.X3:.1f} mm")
    print(f"  X4 (Unit hydrograph time): {model.X4:.2f} days")
    
    # Run model simulation
    print("\nRunning enhanced simulation...")
    results = model.run(P, E)
    
    # Calculate comprehensive statistics
    print("\nComprehensive Simulation Results:")
    print(f"  Total discharge: {np.sum(results['Q']):.2f} mm")
    print(f"  Annual discharge: {np.sum(results['Q'])/3:.2f} mm/year")
    print(f"  Runoff coefficient: {np.sum(results['Q']) / np.sum(P):.3f}")
    print(f"  Peak discharge: {np.max(results['Q']):.2f} mm/day")
    print(f"  Mean discharge: {np.mean(results['Q']):.2f} mm/day")
    print(f"  Median discharge: {np.median(results['Q']):.2f} mm/day")
    print(f"  Q95 (low flows): {np.percentile(results['Q'], 5):.3f} mm/day")
    print(f"  Q10 (high flows): {np.percentile(results['Q'], 90):.2f} mm/day")
    
    # Storage statistics
    print(f"\nStorage Statistics:")
    print(f"  Average production store: {np.mean(results['S']):.2f} mm ({np.mean(results['S'])/model.X1*100:.1f}% capacity)")
    print(f"  Maximum production store: {np.max(results['S']):.2f} mm ({np.max(results['S'])/model.X1*100:.1f}% capacity)")
    print(f"  Average routing store: {np.mean(results['R']):.2f} mm ({np.mean(results['R'])/model.X3*100:.1f}% capacity)")
    print(f"  Maximum routing store: {np.max(results['R']):.2f} mm ({np.max(results['R'])/model.X3*100:.1f}% capacity)")
    
    # Water balance check
    total_input = np.sum(P)
    total_output = np.sum(results['Q']) + np.sum(E)
    storage_change = (results['S'][-1] + results['R'][-1]) - (results['S'][0] + results['R'][0])
    balance_error = total_input - total_output - storage_change
    
    print(f"\nWater Balance Check:")
    print(f"  Total input (P): {total_input:.2f} mm")
    print(f"  Total output (Q + E): {total_output:.2f} mm")
    print(f"  Storage change: {storage_change:.2f} mm")
    print(f"  Balance error: {balance_error:.2f} mm ({abs(balance_error)/total_input*100:.3f}%)")
    
    # Create comprehensive visualizations
    print(f"\nGenerating comprehensive visualizations...")
    create_comprehensive_plots(model, P, E, results, save_dir="figures")
    
    # Display sample results with enhanced formatting
    print("\nDetailed Results for First 15 Days:")
    print("Day |   Date   |   P    |   E    |   Q    |   S    |   R    | Runoff Coef |")
    print("----|----------|--------|--------|--------|--------|--------|-------------|")
    
    start_date = datetime(2020, 1, 1)
    for i in range(15):
        date_str = (start_date + timedelta(days=i)).strftime("%m-%d")
        runoff_coef = results['Q'][i] / P[i] if P[i] > 0.1 else 0
        print(f"{i+1:3d} | {date_str} | {P[i]:6.2f} | {E[i]:6.2f} | {results['Q'][i]:6.2f} | "
              f"{results['S'][i]:6.2f} | {results['R'][i]:6.2f} | {runoff_coef:10.3f} |")
    
    # Demonstrate calibration capability
    print(f"\nDemonstrating calibration capability...")
    print("  (Using simulated 'observed' data with noise)")
    
    # Create synthetic observed data (simulated discharge + noise)
    Q_obs = results['Q'] + np.random.normal(0, 0.1 * np.std(results['Q']), len(results['Q']))
    Q_obs = np.maximum(Q_obs, 0)  # Ensure non-negative
    
    # Run simple calibration
    best_params = model.calibrate(P, E, Q_obs, n_iterations=500)
    
    print(f"  Calibration Results:")
    print(f"    Best X1: {best_params['X1']:.1f} mm (original: {model.X1:.1f})")
    print(f"    Best X2: {best_params['X2']:.2f} mm (original: {model.X2:.2f})")
    print(f"    Best X3: {best_params['X3']:.1f} mm (original: {model.X3:.1f})")
    print(f"    Best X4: {best_params['X4']:.2f} days (original: {model.X4:.2f})")
    print(f"    Nash-Sutcliffe Efficiency: {best_params['NSE']:.3f}")
    
    print("\n" + "=" * 80)
    print("Enhanced GR4J simulation completed successfully!")
    print("Check the 'figures' directory for comprehensive visualizations.")
    print("=" * 80)
    
    return results, model


if __name__ == "__main__":
    main()
