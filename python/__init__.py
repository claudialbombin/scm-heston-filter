"""
Sequential Monte Carlo Heston Filter — Python Implementation
==============================================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/scm-heston-filter

A bootstrap particle filter for real-time stochastic volatility estimation
in the Heston model. This package provides:

1. Heston model simulation (synthetic data generation)
2. Bootstrap particle filter (SMC estimation)
3. Convergence analysis (error vs number of particles)
4. Visualization and metrics utilities

Modules:
    - heston_simulator: Synthetic data from Heston model
    - particle_filter: Bootstrap particle filter
    - utils: Helpers, visualization, metrics
    - convergence_analysis: Error vs √N study
    - main: Entry point orchestrating the full pipeline

USAGE:
    from python import HestonSimulator, BootstrapParticleFilter
    
    sim = HestonSimulator()
    t, S, v_true = sim.simulate(T=1.0)
    
    pf = BootstrapParticleFilter(N_particles=500)
    v_est, v_std, ess = pf.filter_from_prices(S)
"""

__version__ = "1.0.0"
__author__ = "[Tu Nombre]"

from .heston_simulator import HestonSimulator, HestonParams
from .particle_filter import BootstrapParticleFilter, FilterConfig
from .utils import (
    Timer,
    ensure_dir,
    save_results_csv,
    compute_rmse,
    plot_filter_results,
    plot_convergence_analysis,
)
from .convergence_analysis import ConvergenceAnalyzer

__all__ = [
    # Core classes
    "HestonSimulator",
    "HestonParams",
    "BootstrapParticleFilter",
    "FilterConfig",
    "ConvergenceAnalyzer",
    # Utilities
    "Timer",
    "ensure_dir",
    "save_results_csv",
    "compute_rmse",
    "plot_filter_results",
    "plot_convergence_analysis",
]