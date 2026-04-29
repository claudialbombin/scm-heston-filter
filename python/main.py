"""
Particle Filter for Stochastic Volatility — Main Entry Point
==============================================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/scm-heston-filter

This is the main entry point that orchestrates the entire project:
1. Generate synthetic Heston data (known ground truth).
2. Run bootstrap particle filter to estimate latent volatility.
3. Compare estimates against true values.
4. Visualize results (true vs estimated, uncertainty bands).
5. Perform convergence analysis (RMSE vs number of particles).

RUNNING THE FULL PIPELINE
---------------------------
    cd python/
    python main.py

This will:
- Simulate 1 year of daily Heston data (252 steps).
- Run the particle filter with N=500 particles.
- Generate plots showing true vs estimated volatility.
- Compute RMSE between estimate and ground truth.
- Save results to ../data/ and figures to ../results/figures/.

CONFIGURATION
--------------
All parameters are in the config dictionary at the bottom.
Modify them to experiment with different scenarios.

EXPECTED OUTPUT
-----------------
The filter should track true volatility with some lag (inherent
to filtering) and uncertainty bands that widen during volatile
periods and narrow during calm periods.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from heston_simulator import HestonSimulator, HestonParams
from particle_filter import BootstrapParticleFilter, FilterConfig
from utils import (
    Timer,
    ensure_dir,
    save_results_csv,
    compute_rmse,
    plot_filter_results,
    plot_convergence_analysis,
)
from convergence_analysis import ConvergenceAnalyzer


# ============================================================================
# CONFIGURATION — Modify these parameters to experiment
# ============================================================================

config = {
    # ------------------------------------------------------------------
    # Heston Model Parameters (Data Generation)
    # ------------------------------------------------------------------
    "mu": 0.05,          # Annual expected return (5%)
    "kappa": 2.0,        # Mean reversion speed
    "theta": 0.04,       # Long-run variance (σ_long = 20%)
    "xi": 0.3,           # Volatility of variance
    "rho": -0.7,         # Price-variance correlation (leverage effect)
    
    # ------------------------------------------------------------------
    # Simulation Settings
    # ------------------------------------------------------------------
    "T": 1.0,            # Time horizon (years)
    "S0": 100.0,         # Initial price
    "v0": 0.04,          # Initial variance (default: theta)
    "dt": 1/252,         # Time step (daily)
    "base_seed": 42,     # Random seed for reproducibility
    
    # ------------------------------------------------------------------
    # Particle Filter Settings
    # ------------------------------------------------------------------
    "N_particles": 500,  # Number of particles
    "resample_threshold": 0.5,  # Resample when ESS < 50% of N
    
    # ------------------------------------------------------------------
    # Convergence Analysis
    # ------------------------------------------------------------------
    "convergence_N_values": [50, 100, 200, 500, 1000, 2000],
    "convergence_n_trials": 5,  # Monte Carlo trials per N
}


# ============================================================================
# SECTION 1: Generate Synthetic Data
# ============================================================================

def run_data_generation():
    """
    Generate synthetic Heston data with known ground truth.
    
    Returns
    -------
    t : ndarray
        Time grid
    S : ndarray
        Asset prices (observable)
    v_true : ndarray
        True variance (latent state — what we want to recover)
    """
    print("\n" + "="*60)
    print("SECTION 1: SYNTHETIC DATA GENERATION")
    print("="*60)
    
    # Create simulator with specified parameters
    params = HestonParams(
        mu=config["mu"],
        kappa=config["kappa"],
        theta=config["theta"],
        xi=config["xi"],
        rho=config["rho"]
    )
    
    simulator = HestonSimulator(params=params, dt=config["dt"])
    print(f"\n{simulator.get_parameter_summary()}")
    
    # Generate data
    print(f"\nSimulating {config['T']:.1f} years of daily data...")
    
    with Timer("Data Generation"):
        t, S, v_true = simulator.simulate(
            T=config["T"],
            S0=config["S0"],
            v0=config["v0"],
            random_seed=config["base_seed"]
        )
    
    n_steps = len(t) - 1
    print(f"  Generated {n_steps} time steps")
    print(f"  Price range: [${S.min():.2f}, ${S.max():.2f}]")
    print(f"  Variance range: [{v_true.min():.6f}, {v_true.max():.6f}]")
    print(f"  Mean variance: {v_true.mean():.6f} "
          f"(theoretical: {params.theta:.6f})")
    
    # Save data
    save_results_csv(
        "heston_synthetic_data.csv",
        ["time", "price", "true_variance"],
        [[t[i], S[i], v_true[i]] for i in range(len(t))],
        base_dir="../data"
    )
    
    return t, S, v_true


# ============================================================================
# SECTION 2: Run Particle Filter
# ============================================================================

def run_particle_filter(t, S, v_true):
    """
    Run bootstrap particle filter on synthetic data.
    
    Parameters
    ----------
    t : ndarray
        Time grid
    S : ndarray
        Asset prices
    v_true : ndarray
        True variance (for validation)
    
    Returns
    -------
    v_est : ndarray
        Estimated variance (filter mean)
    v_std : ndarray
        Estimation uncertainty (filter std)
    ess : ndarray
        Effective sample size over time
    """
    print("\n" + "="*60)
    print("SECTION 2: PARTICLE FILTER ESTIMATION")
    print("="*60)
    
    # Configure and initialize filter
    filter_config = FilterConfig(
        N_particles=config["N_particles"],
        dt=config["dt"],
        mu=config["mu"],
        kappa=config["kappa"],
        theta=config["theta"],
        xi=config["xi"],
        resample_threshold=config["resample_threshold"]
    )
    
    pf = BootstrapParticleFilter(config=filter_config)
    
    print(f"\n  Number of particles: {config['N_particles']}")
    print(f"  Resample threshold: {config['resample_threshold']:.1%} of N")
    print()
    
    # Run filter from prices
    with Timer("Particle Filter"):
        v_est, v_std, ess = pf.filter_from_prices(
            S, v0=config["v0"], verbose=True
        )
    
    # Compute error metrics
    v_true_aligned = v_true[1:]  # Align: filter estimates at t correspond to true v at t-1
    rmse = compute_rmse(v_est, v_true_aligned)
    
    print(f"\n  Final RMSE: {rmse:.6f}")
    print(f"  Mean ESS: {ess.mean():.1f} / {config['N_particles']}")
    print(f"  Mean uncertainty: ±{v_std.mean():.6f}")
    
    # Save filter results
    save_results_csv(
        "filter_results.csv",
        ["time", "true_variance", "estimated_variance", "uncertainty", "ess"],
        [[t[i+1], v_true[i+1], v_est[i], v_std[i], ess[i]] 
         for i in range(len(v_est))],
        base_dir="../data"
    )
    
    return v_est, v_std, ess


# ============================================================================
# SECTION 3: Visualization
# ============================================================================

def run_visualization(t, S, v_true, v_est, v_std):
    """
    Generate and save visualization plots.
    
    Parameters
    ----------
    t : ndarray
        Time grid (253 elements: t_0 to t_252)
    S : ndarray
        Asset prices (253 elements)
    v_true : ndarray
        True variance (253 elements)
    v_est : ndarray
        Estimated variance (251 elements, starts at t_2)
    v_std : ndarray
        Estimation uncertainty (251 elements)
    """
    print("\n" + "="*60)
    print("SECTION 3: VISUALIZATION")
    print("="*60)
    
    ensure_dir("../results/figures")
    
    # Main filter results plot
    print("\n  Generating filter results plot...")
    plot_filter_results(
        time_grid=t,           # Full time grid (253 points)
        true_variance=v_true,  # Full true variance (253 points)
        estimated_variance=v_est,  # Filter estimates (251 points)
        uncertainty_band=v_std,    # Filter uncertainty (251 points)
        prices=S,              # Full price series (253 points)
        save_path="../results/figures/filter_results.png"
    )
    
    print("  Visualization complete.")


# ============================================================================
# SECTION 4: Convergence Analysis
# ============================================================================

def run_convergence_analysis():
    """
    Analyze how RMSE scales with number of particles.
    
    Validates the theoretical O(1/√N) convergence rate.
    """
    print("\n" + "="*60)
    print("SECTION 4: CONVERGENCE ANALYSIS")
    print("="*60)
    
    # Set up simulator matching config
    params = HestonParams(
        mu=config["mu"],
        kappa=config["kappa"],
        theta=config["theta"],
        xi=config["xi"],
        rho=config["rho"]
    )
    simulator = HestonSimulator(params=params, dt=config["dt"])
    
    # Run convergence analysis
    analyzer = ConvergenceAnalyzer(
        simulator=simulator,
        n_trials=config["convergence_n_trials"],
        T=config["T"]
    )
    
    N_values, rmse_mean, rmse_std = analyzer.analyze(
        N_values=config["convergence_N_values"],
        base_seed=config["base_seed"]
    )
    
    # Estimate empirical convergence rate
    alpha = analyzer.estimate_convergence_rate(N_values, rmse_mean)
    print(f"  Empirical convergence rate: α = {alpha:.3f}")
    print(f"  Theoretical rate: α = 0.500 (1/√N)")
    
    # Save convergence results
    save_results_csv(
        "convergence_results.csv",
        ["N_particles", "rmse_mean", "rmse_std"],
        [[N_values[i], rmse_mean[i], rmse_std[i]] 
         for i in range(len(N_values))],
        base_dir="../data"
    )
    
    # Plot convergence
    ensure_dir("../results/figures")
    print("\n  Generating convergence plot...")
    plot_convergence_analysis(
        N_values=N_values,
        rmse_values=rmse_mean,
        theoretical_rate=True,
        save_path="../results/figures/convergence_analysis.png"
    )
    
    return N_values, rmse_mean, rmse_std


# ============================================================================
# SECTION 5: Volatility Smile Analysis
# ============================================================================

def run_volatility_smile_analysis(v0_current: float = None):
    """
    Generate the implied volatility smile using Heston model.
    
    This demonstrates WHY the Heston model matters:
    - Black-Scholes predicts flat volatility across strikes
    - Heston naturally produces the smile/skew observed in markets
    - The filtered variance v_t can be used as input
    
    Parameters
    ----------
    v0_current : float, optional
        Current variance estimate from particle filter
    """
    print("\n" + "="*60)
    print("SECTION 5: VOLATILITY SMILE ANALYSIS")
    print("="*60)
    
    # Use filtered variance if available, otherwise use config default
    if v0_current is None:
        v0_current = config["v0"]
    
    print(f"\n  Using current variance estimate: v0 = {v0_current:.6f}")
    print(f"  Equivalent spot volatility: {np.sqrt(v0_current)*100:.2f}%")
    
    # Import smile generator
    from implied_vol_smile import (
        SmileConfig,
        VolatilitySmileGenerator,
        plot_volatility_smile,
        plot_parameter_study,
    )
    
    # Configure smile generation
    smile_config = SmileConfig(
        S0=config["S0"],
        r=0.03,
        T=1.0,
        kappa=config["kappa"],
        theta=config["theta"],
        xi=config["xi"],
        rho=config["rho"],
        v0=v0_current,
        moneyness_range=(0.5, 1.5),
        n_strikes=50
    )
    
    generator = VolatilitySmileGenerator(smile_config)
    
    # Generate base smile
    print("\n  Generating base volatility smile...")
    base_smile = generator.generate_single_smile(
        label=f"Heston (v0={v0_current:.4f})"
    )
    
    # Black-Scholes equivalent
    bs_vol = np.sqrt(smile_config.theta)
    
    # Plot
    ensure_dir("../results/figures")
    plot_volatility_smile(
        smiles=[base_smile],
        title="Heston Implied Volatility Smile",
        black_scholes_vol=bs_vol,
        save_path="../results/figures/volatility_smile.png"
    )
    
    # Parameter sensitivity: rho effect on skew
    print("\n  Studying correlation (ρ) effect on skew...")
    rho_smiles = generator.generate_parameter_study(
        'rho', [-0.9, -0.7, -0.5, -0.3, 0.0]
    )
    plot_parameter_study(
        rho_smiles, 'rho',
        title='Effect of Correlation on Volatility Smile',
        save_path="../results/figures/smile_rho_study.png"
    )
    
    # Parameter sensitivity: xi effect on convexity
    print("\n  Studying vol-of-vol (ξ) effect on convexity...")
    xi_smiles = generator.generate_parameter_study(
        'xi', [0.1, 0.3, 0.5, 0.7]
    )
    plot_parameter_study(
        xi_smiles, 'xi',
        title='Effect of Vol-of-Vol on Volatility Smile',
        save_path="../results/figures/smile_xi_study.png"
    )
    
    print("  Smile analysis complete.")
    return base_smile


# ============================================================================
# MAIN — Run everything
# ============================================================================

def main():
    """
    Execute the complete particle filter pipeline.
    
    Pipeline steps:
    1. Generate synthetic Heston data
    2. Run bootstrap particle filter
    3. Visualize results
    4. Analyze convergence properties
    """
    print("="*60)
    print("PARTICLE FILTER FOR STOCHASTIC VOLATILITY ESTIMATION")
    print("="*60)
    print("Author: [Tu Nombre]")
    print("Model: Heston Stochastic Volatility")
    print("Method: Bootstrap Particle Filter (Sequential Monte Carlo)")
    print("="*60)
    
    # Create output directories
    ensure_dir("../data")
    ensure_dir("../results/figures")
    ensure_dir("../results/metrics")
    
    # ------------------------------------------------------------------
    # Run pipeline with overall timing
    # ------------------------------------------------------------------
    with Timer("Total Pipeline Runtime"):
        
        # Section 1: Generate synthetic data
        t, S, v_true = run_data_generation()
        
        # Section 2: Run particle filter
        v_est, v_std, ess = run_particle_filter(t, S, v_true)
        
        # Section 3: Visualization
        run_visualization(t, S, v_true, v_est, v_std)
        
        # Section 4: Convergence analysis
        run_convergence_analysis()

        # Section 5: Volatility smile analysis (NUEVO)
        # Use the mean of filtered variance as current spot variance
        v0_filtered = v_est.mean()
        run_volatility_smile_analysis(v0_current=v0_filtered)
        
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\n  Output files:")
    print("    ../data/heston_synthetic_data.csv")
    print("    ../data/filter_results.csv")
    print("    ../data/convergence_results.csv")
    print("    ../results/figures/filter_results.png")
    print("    ../results/figures/convergence_analysis.png")
    print("    ../results/figures/volatility_smile.png")


if __name__ == "__main__":
    main()