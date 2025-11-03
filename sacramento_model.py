"""
SACRAMENTO Soil Moisture Accounting Model (SAC-SMA)

The Sacramento model is a continuous soil moisture accounting model developed by
the National Weather Service (NWS) for river forecasting in the United States.
It was originally developed by Burnash et al. (1973) for the Sacramento River basin.

Mathematical Foundation:
========================

The model divides the soil profile into upper and lower zones:

1. Upper Zone:
   - Upper Zone Tension Water (UZTWC): supplies evapotranspiration
   - Upper Zone Free Water (UZFWC): generates surface and interflow runoff
   
2. Lower Zone:
   - Lower Zone Tension Water (LZTWC): supplies evapotranspiration
   - Lower Zone Primary Free Water (LZFPC): generates primary baseflow
   - Lower Zone Supplementary Free Water (LZFSC): generates supplementary baseflow

Water Balance Equations:
------------------------

1. Evapotranspiration:
   - From UZTWC: E1 = min(UZTWC, PET * UZTWC/UZTWM)
   - From UZFWC: E2 = min(UZFWC, (PET - E1) * UZFWC/UZFWM)
   - From LZTWC: E3 = min(LZTWC, (PET - E1 - E2) * LZTWC/LZTWM)
   - From LZFPC: E4 = min(LZFPC, (PET - E1 - E2 - E3) * LZFPC/(LZFPM+LZFSM))
   - From LZFSC: E5 = min(LZFSC, (PET - E1 - E2 - E3 - E4) * LZFSC/(LZFPM+LZFSM))

2. Runoff Generation:
   Surface runoff: ROIMP = P * PCTIM (impervious area)
   
   Interflow: RSSUR = UZFWC * UZK if UZFWC > 0
   
   Primary baseflow: RBFP = LZFPC * LZPK if LZFPC > 0
   
   Supplementary baseflow: RBFS = LZFSC * LZSK if LZFSC > 0

3. Percolation:
   PERC = LZFPM * (LZFPC/LZFPM) * (1 + ZPERC * (ADIMC/UZTWM + LZTWC/LZTWM))

Parameters:
-----------
UZTWM : Upper zone tension water maximum (mm)
UZFWM : Upper zone free water maximum (mm)
LZTWM : Lower zone tension water maximum (mm)
LZFPM : Lower zone primary free water maximum (mm)
LZFSM : Lower zone supplementary free water maximum (mm)
UZK   : Upper zone free water lateral depletion rate (1/day)
LZPK  : Lower zone primary free water depletion rate (1/day)
LZSK  : Lower zone supplementary free water depletion rate (1/day)
ZPERC : Maximum percolation rate coefficient
REXP  : Exponent of the percolation equation
PCTIM : Fraction of impervious area
ADIMP : Additional impervious area
PFREE : Fraction of water percolating directly to lower zone free water

References:
-----------
Burnash, R.J.C., Ferral, R.L., & McGuire, R.A. (1973). 
A generalized streamflow simulation system: Conceptual models for digital computers. 
US Department of Commerce, National Weather Service, and State of California.

Author: Burnash et al. (Original), Implementation by [Your Name]
Date: 2024
"""

import numpy as np
from typing import Dict
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats


class SacramentoModel:
    """
    Sacramento Soil Moisture Accounting (SAC-SMA) model implementation.
    
    This is a simplified version of the full SAC-SMA model.
    
    Parameters:
    -----------
    UZTWM : float
        Upper zone tension water capacity (mm, default: 80)
    UZFWM : float
        Upper zone free water capacity (mm, default: 40)
    LZTWM : float
        Lower zone tension water capacity (mm, default: 150)
    LZFPM : float
        Lower zone primary free water capacity (mm, default: 100)
    LZFSM : float
        Lower zone supplementary free water capacity (mm, default: 50)
    UZK : float
        Upper zone depletion rate (1/day, default: 0.3)
    LZPK : float
        Lower zone primary depletion rate (1/day, default: 0.01)
    LZSK : float
        Lower zone supplementary depletion rate (1/day, default: 0.05)
    ZPERC : float
        Maximum percolation rate (default: 40)
    REXP : float
        Exponent of percolation equation (default: 2.0)
    PCTIM : float
        Impervious fraction (0-1, default: 0.01)
    ADIMP : float
        Additional impervious area (0-1, default: 0.0)
    PFREE : float
        Fraction percolating directly to lower zone (0-1, default: 0.1)
    """
    
    def __init__(self,
                 UZTWM: float = 80.0,
                 UZFWM: float = 40.0,
                 LZTWM: float = 150.0,
                 LZFPM: float = 100.0,
                 LZFSM: float = 50.0,
                 UZK: float = 0.3,
                 LZPK: float = 0.01,
                 LZSK: float = 0.05,
                 ZPERC: float = 40.0,
                 REXP: float = 2.0,
                 PCTIM: float = 0.01,
                 ADIMP: float = 0.0,
                 PFREE: float = 0.1):
        
        # Model parameters
        self.UZTWM = UZTWM  # Upper zone tension water max
        self.UZFWM = UZFWM  # Upper zone free water max
        self.LZTWM = LZTWM  # Lower zone tension water max
        self.LZFPM = LZFPM  # Lower zone primary free water max
        self.LZFSM = LZFSM  # Lower zone supplementary free water max
        self.UZK = UZK      # Upper zone depletion
        self.LZPK = LZPK    # Lower zone primary depletion
        self.LZSK = LZSK    # Lower zone supplementary depletion
        self.ZPERC = ZPERC  # Percolation rate
        self.REXP = REXP    # Percolation exponent
        self.PCTIM = PCTIM  # Impervious area
        self.ADIMP = ADIMP  # Additional impervious
        self.PFREE = PFREE  # Direct percolation fraction
        
        # State variables (initial conditions - 60% filled)
        self.UZTWC = UZTWM * 0.6  # Upper zone tension water content
        self.UZFWC = UZFWM * 0.6  # Upper zone free water content
        self.LZTWC = LZTWM * 0.6  # Lower zone tension water content
        self.LZFPC = LZFPM * 0.6  # Lower zone primary free water content
        self.LZFSC = LZFSM * 0.6  # Lower zone supplementary free water content
        
    def evapotranspiration(self, PET: float) -> float:
        """
        Calculate actual evapotranspiration from all zones.
        
        Parameters:
        -----------
        PET : float
            Potential evapotranspiration (mm)
            
        Returns:
        --------
        E : float
            Actual evapotranspiration (mm)
        """
        E_total = 0.0
        PET_remaining = PET
        
        # E1: From upper zone tension water
        if self.UZTWM > 0:
            E1 = min(self.UZTWC, PET_remaining * self.UZTWC / self.UZTWM)
            self.UZTWC -= E1
            E_total += E1
            PET_remaining -= E1
        
        # E2: From upper zone free water
        if self.UZFWM > 0 and PET_remaining > 0:
            E2 = min(self.UZFWC, PET_remaining * self.UZFWC / self.UZFWM)
            self.UZFWC -= E2
            E_total += E2
            PET_remaining -= E2
        
        # E3: From lower zone tension water
        if self.LZTWM > 0 and PET_remaining > 0:
            E3 = min(self.LZTWC, PET_remaining * self.LZTWC / self.LZTWM)
            self.LZTWC -= E3
            E_total += E3
            PET_remaining -= E3
        
        # E4 & E5: From lower zone free water (proportional)
        if PET_remaining > 0:
            LZFWM_total = self.LZFPM + self.LZFSM
            if LZFWM_total > 0:
                E4 = min(self.LZFPC, PET_remaining * self.LZFPC / LZFWM_total)
                self.LZFPC -= E4
                E_total += E4
                PET_remaining -= E4
                
                if PET_remaining > 0:
                    E5 = min(self.LZFSC, PET_remaining * self.LZFSC / LZFWM_total)
                    self.LZFSC -= E5
                    E_total += E5
        
        return E_total
    
    def infiltration(self, P: float) -> float:
        """
        Calculate infiltration to upper zone.
        
        Parameters:
        -----------
        P : float
            Precipitation (mm)
            
        Returns:
        --------
        surface_runoff : float
            Surface runoff from impervious area (mm)
        """
        # Impervious area runoff
        surface_runoff = P * self.PCTIM
        
        # Infiltration to pervious area
        P_pervious = P * (1 - self.PCTIM)
        
        # Fill upper zone tension water first
        space_UZTW = max(0, self.UZTWM - self.UZTWC)
        to_UZTW = min(P_pervious, space_UZTW)
        self.UZTWC += to_UZTW
        P_pervious -= to_UZTW
        
        # Remaining goes to upper zone free water
        if P_pervious > 0:
            space_UZFW = max(0, self.UZFWM - self.UZFWC)
            to_UZFW = min(P_pervious, space_UZFW)
            self.UZFWC += to_UZFW
            P_pervious -= to_UZFW
            
            # Excess becomes surface runoff
            if P_pervious > 0:
                surface_runoff += P_pervious
        
        return surface_runoff
    
    def percolation(self) -> float:
        """
        Calculate percolation from upper to lower zone.
        
        Returns:
        --------
        perc : float
            Percolation amount (mm)
        """
        # Calculate deficit in lower zone
        deficit_lower = max(0, (self.LZTWM - self.LZTWC) + 
                           (self.LZFPM - self.LZFPC) + 
                           (self.LZFSM - self.LZFSC))
        
        if deficit_lower <= 0:
            return 0.0
        
        # Calculate percolation rate based on upper zone contents
        if self.UZTWM > 0:
            ratio = (self.UZTWC / self.UZTWM) + (self.UZFWC / self.UZFWM) * 0.5
        else:
            ratio = 0.0
        
        # Percolation equation
        if ratio > 0:
            perc = self.ZPERC * (ratio ** self.REXP)
        else:
            perc = 0.0
        
        # Limit percolation to available water and deficit
        perc = min(perc, self.UZFWC + self.UZTWC * 0.5)
        perc = min(perc, deficit_lower)
        
        # Remove from upper zone (preferentially from free water)
        if perc > 0:
            from_UZFW = min(perc, self.UZFWC)
            self.UZFWC -= from_UZFW
            perc_remaining = perc - from_UZFW
            
            if perc_remaining > 0:
                from_UZTW = min(perc_remaining, self.UZTWC)
                self.UZTWC -= from_UZTW
        
        # Add to lower zone
        # Direct percolation to free water
        to_free = perc * self.PFREE
        
        # Split free water between primary and supplementary
        if self.LZFPM + self.LZFSM > 0:
            ratio_primary = self.LZFPM / (self.LZFPM + self.LZFSM)
            to_LZFP = to_free * ratio_primary
            to_LZFS = to_free * (1 - ratio_primary)
            
            self.LZFPC = min(self.LZFPM, self.LZFPC + to_LZFP)
            self.LZFSC = min(self.LZFSM, self.LZFSC + to_LZFS)
        
        # Remaining to tension water
        to_tension = perc * (1 - self.PFREE)
        self.LZTWC = min(self.LZTWM, self.LZTWC + to_tension)
        
        return perc
    
    def generate_runoff(self) -> Dict[str, float]:
        """
        Generate runoff components.
        
        Returns:
        --------
        dict : Runoff components
        """
        # Interflow from upper zone free water
        interflow = self.UZFWC * self.UZK
        self.UZFWC -= interflow
        self.UZFWC = max(0, self.UZFWC)
        
        # Primary baseflow from lower zone primary free water
        baseflow_p = self.LZFPC * self.LZPK
        self.LZFPC -= baseflow_p
        self.LZFPC = max(0, self.LZFPC)
        
        # Supplementary baseflow from lower zone supplementary free water
        baseflow_s = self.LZFSC * self.LZSK
        self.LZFSC -= baseflow_s
        self.LZFSC = max(0, self.LZFSC)
        
        return {
            'interflow': interflow,
            'baseflow_primary': baseflow_p,
            'baseflow_supplementary': baseflow_s
        }
    
    def run_timestep(self, P: float, PET: float) -> Dict[str, float]:
        """
        Run one timestep of the Sacramento model.
        
        Parameters:
        -----------
        P : float
            Precipitation (mm)
        PET : float
            Potential evapotranspiration (mm)
            
        Returns:
        --------
        dict : Model outputs
        """
        # 1. Evapotranspiration
        E = self.evapotranspiration(PET)
        
        # 2. Infiltration and surface runoff
        surface_runoff = self.infiltration(P)
        
        # 3. Percolation
        perc = self.percolation()
        
        # 4. Generate runoff
        runoff_components = self.generate_runoff()
        
        # Total discharge
        Q = (surface_runoff + 
             runoff_components['interflow'] + 
             runoff_components['baseflow_primary'] + 
             runoff_components['baseflow_supplementary'])
        
        return {
            'Q': Q,
            'surface': surface_runoff,
            'interflow': runoff_components['interflow'],
            'baseflow_p': runoff_components['baseflow_primary'],
            'baseflow_s': runoff_components['baseflow_supplementary'],
            'E': E,
            'UZTWC': self.UZTWC,
            'UZFWC': self.UZFWC,
            'LZTWC': self.LZTWC,
            'LZFPC': self.LZFPC,
            'LZFSC': self.LZFSC
        }
    
    def run(self, P: np.ndarray, PET: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run Sacramento model for multiple timesteps.
        
        Parameters:
        -----------
        P : np.ndarray
            Precipitation time series (mm)
        PET : np.ndarray
            Potential evapotranspiration time series (mm)
            
        Returns:
        --------
        dict : Model outputs
        """
        n_steps = len(P)
        
        # Initialize output arrays
        Q = np.zeros(n_steps)
        surface = np.zeros(n_steps)
        interflow = np.zeros(n_steps)
        baseflow_p = np.zeros(n_steps)
        baseflow_s = np.zeros(n_steps)
        E = np.zeros(n_steps)
        UZTWC = np.zeros(n_steps)
        UZFWC = np.zeros(n_steps)
        LZTWC = np.zeros(n_steps)
        LZFPC = np.zeros(n_steps)
        LZFSC = np.zeros(n_steps)
        
        for t in range(n_steps):
            result = self.run_timestep(P[t], PET[t])
            Q[t] = result['Q']
            surface[t] = result['surface']
            interflow[t] = result['interflow']
            baseflow_p[t] = result['baseflow_p']
            baseflow_s[t] = result['baseflow_s']
            E[t] = result['E']
            UZTWC[t] = result['UZTWC']
            UZFWC[t] = result['UZFWC']
            LZTWC[t] = result['LZTWC']
            LZFPC[t] = result['LZFPC']
            LZFSC[t] = result['LZFSC']
        
        return {
            'Q': Q,
            'surface': surface,
            'interflow': interflow,
            'baseflow_primary': baseflow_p,
            'baseflow_supplementary': baseflow_s,
            'E': E,
            'UZTWC': UZTWC,
            'UZFWC': UZFWC,
            'LZTWC': LZTWC,
            'LZFPC': LZFPC,
            'LZFSC': LZFSC
        }


def main():
    """
    Example usage of Sacramento model.
    """
    print("=" * 80)
    print("SACRAMENTO Soil Moisture Accounting Model (SAC-SMA)")
    print("=" * 80)
    
    # Set random seed
    np.random.seed(42)
    
    # Generate synthetic data (365 days)
    n_days = 365
    
    # Precipitation
    P = np.random.gamma(2, 5, n_days)
    P[np.random.rand(n_days) > 0.4] = 0
    
    # Evapotranspiration
    t = np.arange(n_days)
    PET = 3.0 + 2.0 * np.sin(2 * np.pi * t / 365)
    
    print("\nInput Data Summary:")
    print(f"  Simulation period: {n_days} days")
    print(f"  Total precipitation: {np.sum(P):.2f} mm")
def create_sacramento_plots(model, P, PET, results, save_dir="figures"):
    """
    Create comprehensive visualization plots for Sacramento model results.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create date index
    n_days = len(P)
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Figure 1: Sacramento Model Comprehensive Analysis
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Sacramento (SAC-SMA) Model - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # Precipitation (inverted)
    axes[0].bar(dates, P, color='steelblue', alpha=0.7, width=1)
    axes[0].set_ylabel('Precipitation\n(mm/day)', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(max(P) * 1.1, 0)
    
    # Discharge components
    axes[1].plot(dates, results['Q'], color='blue', linewidth=2, label='Total Discharge')
    axes[1].plot(dates, results['surface'], color='red', linewidth=1.5, label='Surface Runoff', alpha=0.8)
    axes[1].plot(dates, results['interflow'], color='green', linewidth=1.5, label='Interflow', alpha=0.8)
    axes[1].plot(dates, results['baseflow_primary'], color='purple', linewidth=1.5, label='Primary Baseflow', alpha=0.8)
    axes[1].plot(dates, results['baseflow_supplementary'], color='orange', linewidth=1.5, label='Supp. Baseflow', alpha=0.8)
    axes[1].set_ylabel('Discharge\n(mm/day)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Soil moisture - upper zone
    axes[2].plot(dates, results['UZTWC'], color='brown', linewidth=2, label='Upper Zone Tension Water')
    axes[2].plot(dates, results['UZFWC'], color='green', linewidth=2, label='Upper Zone Free Water')
    axes[2].axhline(y=model.UZTWM, color='red', linestyle='--', alpha=0.7, label=f'UZTWM={model.UZTWM}mm')
    axes[2].axhline(y=model.UZFWM, color='green', linestyle='--', alpha=0.7, label=f'UZFWM={model.UZFWM}mm')
    axes[2].fill_between(dates, results['UZTWC'], alpha=0.2, color='brown')
    axes[2].set_ylabel('Upper Zone\n(mm)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Soil moisture - lower zone
    axes[3].plot(dates, results['LZTWC'], color='blue', linewidth=2, label='Lower Zone Tension Water')
    axes[3].plot(dates, results['LZFPC'], color='purple', linewidth=2, label='Lower Zone Primary Free Water')
    axes[3].plot(dates, results['LZFSC'], color='orange', linewidth=2, label='Lower Zone Supp. Free Water')
    axes[3].axhline(y=model.LZTWM, color='blue', linestyle='--', alpha=0.7, label=f'LZTWM={model.LZTWM}mm')
    axes[3].fill_between(dates, results['LZTWC'], alpha=0.2, color='blue')
    axes[3].set_ylabel('Lower Zone\n(mm)', fontweight='bold')
    axes[3].set_xlabel('Date', fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sacramento_comprehensive.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Sacramento model visualization plot saved to '{save_dir}' directory")
    print("  Generated files:")
    print("  - sacramento_comprehensive.png")


def main():
    """
    Enhanced demonstration of Sacramento Soil Moisture Accounting Model.
    """
    print("=" * 80)
    print("Sacramento Soil Moisture Accounting Model (SAC-SMA) - Enhanced Demonstration")
    print("Continuous Soil Moisture Accounting for River Forecasting")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate enhanced synthetic data (2 years)
    n_days = 730
    
    # More realistic precipitation pattern
    t = np.arange(n_days)
    
    # Seasonal precipitation with winter wet season
    seasonal_factor = 1.5 + 0.9 * np.sin(2 * np.pi * t / 365 + np.pi)
    P_base = np.random.gamma(1.8, 3, n_days) * seasonal_factor
    
    # Add dry periods (50% chance of no rain)
    dry_prob = 0.5 + 0.2 * np.sin(2 * np.pi * t / 365)
    P = np.where(np.random.rand(n_days) < dry_prob, 0, P_base)
    
    # Add storm events
    storm_events = np.random.choice(n_days, size=15, replace=False)
    P[storm_events] = P[storm_events] + np.random.gamma(4, 8, 15)
    
    # Enhanced potential evapotranspiration
    PET_mean = 3.8
    PET_amplitude = 2.2
    PET = PET_mean + PET_amplitude * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.2, n_days)
    PET = np.maximum(PET, 0.5)
    
    print("\nEnhanced Input Data Summary:")
    print(f"  Simulation period: {n_days} days ({n_days/365:.1f} years)")
    print(f"  Total precipitation: {np.sum(P):.2f} mm")
    print(f"  Average daily precipitation: {np.mean(P):.2f} mm")
    print(f"  Maximum daily precipitation: {np.max(P):.2f} mm")
    print(f"  Rainfall days: {np.sum(P > 0.1)} ({np.sum(P > 0.1)/n_days*100:.1f}%)")
    print(f"  Total potential ET: {np.sum(PET):.2f} mm")
    print(f"  Average daily PET: {np.mean(PET):.2f} mm")
    print(f"  P/PET ratio: {np.sum(P)/np.sum(PET):.3f}")
    
    # Initialize enhanced model parameters
    model = SacramentoModel(
        UZTWM=90.0, UZFWM=45.0,          # Enhanced upper zone capacities
        LZTWM=180.0, LZFPM=120.0, LZFSM=60.0,  # Enhanced lower zone capacities
        UZK=0.35, LZPK=0.015, LZSK=0.08,      # Enhanced depletion rates
        ZPERC=50.0, REXP=2.5,             # Enhanced percolation parameters
        PCTIM=0.02, ADIMP=0.01, PFREE=0.15    # Enhanced surface characteristics
    )
    
    print("\nEnhanced Model Parameters:")
    print("  Upper Zone Configuration:")
    print(f"    UZTWM (Upper zone tension water max): {model.UZTWM} mm")
    print(f"    UZFWM (Upper zone free water max): {model.UZFWM} mm")
    print(f"    UZK (Upper zone depletion rate): {model.UZK} /day")
    print("  Lower Zone Configuration:")
    print(f"    LZTWM (Lower zone tension water max): {model.LZTWM} mm")
    print(f"    LZFPM (Lower zone primary free water max): {model.LZFPM} mm")
    print(f"    LZFSM (Lower zone supplementary free water max): {model.LZFSM} mm")
    print(f"    LZPK (Lower zone primary depletion rate): {model.LZPK} /day")
    print(f"    LZSK (Lower zone supplementary depletion rate): {model.LZSK} /day")
    print("  Percolation and Surface:")
    print(f"    ZPERC (Maximum percolation rate): {model.ZPERC}")
    print(f"    REXP (Percolation equation exponent): {model.REXP}")
    print(f"    PCTIM (Impervious area fraction): {model.PCTIM}")
    print(f"    PFREE (Direct percolation fraction): {model.PFREE}")
    
    # Run enhanced simulation
    print("\nRunning enhanced Sacramento simulation...")
    results = model.run(P, PET)
    
    # Comprehensive results analysis
    print("\nComprehensive Simulation Results:")
    print(f"  Water Balance:")
    print(f"    Total precipitation: {np.sum(P):.2f} mm")
    print(f"    Total discharge: {np.sum(results['Q']):.2f} mm")
    print(f"    Annual discharge: {np.sum(results['Q'])/2:.2f} mm/year")
    print(f"    Total ET: {np.sum(results['E']):.2f} mm")
    print(f"    Runoff coefficient: {np.sum(results['Q']) / np.sum(P):.3f}")
    print(f"    ET efficiency: {np.sum(results['E']) / np.sum(PET):.3f}")
    
    print(f"  Flow Characteristics:")
    print(f"    Peak discharge: {np.max(results['Q']):.2f} mm/day")
    print(f"    Mean discharge: {np.mean(results['Q']):.2f} mm/day")
    print(f"    Median discharge: {np.median(results['Q']):.2f} mm/day")
    print(f"    Flow variability (CV): {np.std(results['Q'])/np.mean(results['Q']):.3f}")
    
    # Comprehensive runoff components analysis
    total_discharge = np.sum(results['Q'])
    surface_total = np.sum(results['surface'])
    interflow_total = np.sum(results['interflow'])
    baseflow_primary_total = np.sum(results['baseflow_primary'])
    baseflow_supp_total = np.sum(results['baseflow_supplementary'])
    
    print(f"\nDetailed Runoff Components Analysis:")
    print(f"  Surface runoff: {surface_total:.2f} mm ({surface_total/total_discharge*100:.1f}%)")
    print(f"  Interflow: {interflow_total:.2f} mm ({interflow_total/total_discharge*100:.1f}%)")
    print(f"  Primary baseflow: {baseflow_primary_total:.2f} mm ({baseflow_primary_total/total_discharge*100:.1f}%)")
    print(f"  Supplementary baseflow: {baseflow_supp_total:.2f} mm ({baseflow_supp_total/total_discharge*100:.1f}%)")
    
    total_baseflow = baseflow_primary_total + baseflow_supp_total
    quick_flow = surface_total + interflow_total
    
    print(f"\nFlow Regime Analysis:")
    print(f"  Total baseflow: {total_baseflow:.2f} mm ({total_baseflow/total_discharge*100:.1f}%)")
    print(f"  Total quickflow: {quick_flow:.2f} mm ({quick_flow/total_discharge*100:.1f}%)")
    print(f"  Baseflow index: {total_baseflow/total_discharge:.3f}")
    print(f"  Flow regime: {'Baseflow dominated' if total_baseflow > quick_flow else 'Quickflow dominated'}")
    
    # Storage analysis
    print(f"\nSoil Moisture Storage Analysis:")
    avg_uztwc = np.mean(results['UZTWC'])
    avg_lztwc = np.mean(results['LZTWC'])
    total_avg_storage = avg_uztwc + avg_lztwc
    total_capacity = model.UZTWM + model.LZTWM
    
    print(f"  Average upper zone tension water: {avg_uztwc:.1f} mm ({avg_uztwc/model.UZTWM*100:.1f}% of capacity)")
    print(f"  Average lower zone tension water: {avg_lztwc:.1f} mm ({avg_lztwc/model.LZTWM*100:.1f}% of capacity)")
    print(f"  Total average tension water storage: {total_avg_storage:.1f} mm ({total_avg_storage/total_capacity*100:.1f}% of capacity)")
    
    # Generate comprehensive visualizations
    print(f"\nGenerating comprehensive visualizations...")
    create_sacramento_plots(model, P, PET, results, save_dir="figures")
    
    # Enhanced sample output
    print("\nDetailed Results for First 15 Days:")
    print("Day |   Date   |   P    |  PET   |   E    |   Q    | UZTWC  | UZFWC  | LZTWC  | Surf | Inter| Base |")
    print("----|----------|--------|--------|--------|--------|--------|--------|--------|------|------|------|")
    
    start_date = datetime(2020, 1, 1)
    for i in range(15):
        date_str = (start_date + timedelta(days=i)).strftime("%m-%d")
        total_base = results['baseflow_primary'][i] + results['baseflow_supplementary'][i]
        print(f"{i+1:3d} | {date_str} | {P[i]:6.2f} | {PET[i]:6.2f} | {results['E'][i]:6.2f} | "
              f"{results['Q'][i]:6.2f} | {results['UZTWC'][i]:6.2f} | {results['UZFWC'][i]:6.2f} | "
              f"{results['LZTWC'][i]:6.2f} | {results['surface'][i]:4.3f} | "
              f"{results['interflow'][i]:5.3f} | {total_base:4.3f} |")
    
    # Water balance check
    total_input = np.sum(P)
    total_output = np.sum(results['E']) + np.sum(results['Q'])
    initial_storage = (model.UZTWM + model.LZTWM) * 0.5  # Assume 50% initial
    final_storage = results['UZTWC'][-1] + results['UZFWC'][-1] + results['LZTWC'][-1] + results['LZFPC'][-1] + results['LZFSC'][-1]
    storage_change = final_storage - initial_storage
    balance_error = total_input - total_output - storage_change
    
    print(f"\nWater Balance Check:")
    print(f"  Total input (P): {total_input:.2f} mm")
    print(f"  Total output (E + Q): {total_output:.2f} mm")
    print(f"  Storage change: {storage_change:.2f} mm")
    print(f"  Balance error: {balance_error:.2f} mm ({abs(balance_error)/total_input*100:.3f}%)")
    
    print("\n" + "=" * 80)
    print("Enhanced Sacramento (SAC-SMA) simulation completed successfully!")
    print("This model demonstrates:")
    print("  - Continuous soil moisture accounting")
    print("  - Five-storage structure (upper and lower zones)")
    print("  - Four runoff components (surface, interflow, two baseflow types)")
    print("  - Suitable for operational river forecasting")
    print("Check the 'figures' directory for comprehensive visualizations.")
    print("=" * 80)
    
    return results, model


if __name__ == "__main__":
    main()
