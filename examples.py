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

# Import all models
from xinanjiang_model import XinanjiangModel
from tank_model import TankModel1D, TankModel2D, TankModel3D
from gr4j_model import GR4J
from sacramento_model import SacramentoModel


def generate_synthetic_data(n_days: int = 365, seed: int = 42):
    """
    Generate synthetic hydrological data for testing.
    
    This function creates realistic precipitation and evapotranspiration
    time series that can be used as input for hydrological models.
    
    Parameters:
    -----------
    n_days : int
        Number of days to simulate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Dictionary containing P (precipitation) and ET (evapotranspiration)
    """
    np.random.seed(seed)
    
    # Precipitation: Gamma distribution with intermittent rain
    # Typical pattern: some days with heavy rain, many dry days
    P = np.random.gamma(2, 5, n_days)
    dry_days = np.random.rand(n_days) > 0.4  # 60% of days are dry
    P[dry_days] = 0
    
    # Potential Evapotranspiration: Seasonal sinusoidal pattern
    # Higher in summer, lower in winter
    t = np.arange(n_days)
    ET = 3.0 + 2.0 * np.sin(2 * np.pi * t / 365 - np.pi / 2)  # Peak in day 91 (summer)
    ET = np.maximum(ET, 0.5)  # Minimum 0.5 mm/day
    
    return {
        'P': P,
        'ET': ET,
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
    """
    print("=" * 80)
    print("Comparing All Hydrological Models")
    print("=" * 80)
    
    # Generate common input data
    data = generate_synthetic_data(n_days=365, seed=42)
    P = data['P']
    ET = data['ET']
    
    print(f"\nInput Data: {data['days']} days")
    print(f"Total Precipitation: {np.sum(P):.2f} mm")
    print(f"Total Potential ET: {np.sum(ET):.2f} mm")
    print(f"Average Daily P: {np.mean(P):.2f} mm")
    print(f"Average Daily ET: {np.mean(ET):.2f} mm")
    
    results = {}

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
    
    # 1. Xinanjiang Model
    print("\n" + "-" * 80)
    print("1. Running Xinanjiang Model...")
    xaj = XinanjiangModel()
    results['Xinanjiang'] = xaj.run(P, ET)
    rng = np.random.default_rng(123)
    observed_Q = np.clip(
        results['Xinanjiang']['Q'] + rng.normal(0.0, 0.5, size=len(P)),
        a_min=0.0,
        a_max=None
    )
    print(f"   Total discharge: {np.sum(results['Xinanjiang']['Q']):.2f} mm")
    print(f"   Runoff coefficient: {np.sum(results['Xinanjiang']['Q'])/np.sum(P):.3f}")
    
    # 2. Tank Model 1D
    print("\n" + "-" * 80)
    print("2. Running Tank Model 1D...")
    tank1d = TankModel1D()
    results['Tank_1D'] = tank1d.run(P, ET)
    print(f"   Total discharge: {np.sum(results['Tank_1D']['Q']):.2f} mm")
    print(f"   Runoff coefficient: {np.sum(results['Tank_1D']['Q'])/np.sum(P):.3f}")
    
    # 3. Tank Model 2D
    print("\n" + "-" * 80)
    print("3. Running Tank Model 2D...")
    tank2d = TankModel2D()
    results['Tank_2D'] = tank2d.run(P, ET)
    print(f"   Total discharge: {np.sum(results['Tank_2D']['Q']):.2f} mm")
    print(f"   Runoff coefficient: {np.sum(results['Tank_2D']['Q'])/np.sum(P):.3f}")
    
    # 4. Tank Model 3D
    print("\n" + "-" * 80)
    print("4. Running Tank Model 3D...")
    tank3d = TankModel3D()
    results['Tank_3D'] = tank3d.run(P, ET)
    print(f"   Total discharge: {np.sum(results['Tank_3D']['Q']):.2f} mm")
    print(f"   Runoff coefficient: {np.sum(results['Tank_3D']['Q'])/np.sum(P):.3f}")
    
    # 5. GR4J Model
    print("\n" + "-" * 80)
    print("5. Running GR4J Model...")
    gr4j = GR4J()
    results['GR4J'] = gr4j.run(P, ET)
    print(f"   Total discharge: {np.sum(results['GR4J']['Q']):.2f} mm")
    print(f"   Runoff coefficient: {np.sum(results['GR4J']['Q'])/np.sum(P):.3f}")
    
    # 6. Sacramento Model
    print("\n" + "-" * 80)
    print("6. Running Sacramento Model...")
    sac = SacramentoModel()
    results['Sacramento'] = sac.run(P, ET)
    print(f"   Total discharge: {np.sum(results['Sacramento']['Q']):.2f} mm")
    print(f"   Runoff coefficient: {np.sum(results['Sacramento']['Q'])/np.sum(P):.3f}")
    
    # Comparison Table
    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print("=" * 80)
    print("\n{:<20} {:<15} {:<15} {:<15} {:<15} {:<12} {:<12} {:<12}".format(
        "Model", "Total Q (mm)", "Runoff Coef", "Peak Q (mm)", "Mean Q (mm)",
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

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

    days = np.arange(len(P))
    ax1.bar(days, P, color='royalblue', alpha=0.8, label='Precipitation')
    ax1.set_ylabel('Precipitation (mm/day)')
    ax1.invert_yaxis()
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right')

    ax2.plot(days, observed_Q, label='Synthetic Observed Flow', color='black', linestyle='--')
    for model_name, result in results.items():
        ax2.plot(days, result['Q'], label=model_name)

    ax2.set_ylabel('Discharge (mm/day)')
    ax2.set_xlabel('Days')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right')

    fig.suptitle('Comparison of Hydrological Models')
    plt.tight_layout()
    plt.show()

    return results


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
    
    input("\nPress Enter to continue to model comparison...")
    
    # Example 2: Compare all models
    compare_all_models()
    
    input("\nPress Enter to continue to sensitivity analysis...")
    
    # Example 3: Sensitivity analysis
    sensitivity_analysis_example()
    
    input("\nPress Enter to continue to storm event simulation...")
    
    # Example 4: Storm event
    storm_event_example()
    
    input("\nPress Enter to continue to seasonal analysis...")
    
    # Example 5: Seasonal patterns
    seasonal_pattern_example()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print("\nFor more information, see individual model files:")
    print("  - xinanjiang_model.py")
    print("  - tank_model.py")
    print("  - gr4j_model.py")
    print("  - sacramento_model.py")
    print("\n")


if __name__ == "__main__":
    main()
