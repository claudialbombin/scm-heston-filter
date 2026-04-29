"""
Utility Functions for Visualization, Timing, and I/O
======================================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/scm-heston-filter

Helper functions used across the project for:
- Timing code execution (performance profiling)
- File I/O operations (saving/loading data)
- Computing error metrics (RMSE, MAE)
- Visualization (filter results, convergence plots)

All functions are designed to be re-usable and independent
of the specific models used in the main pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter
from typing import Tuple, List, Optional
import csv


# ============================================================================
# TIMING UTILITIES
# ============================================================================

class Timer:
    """
    Context manager for timing code execution.
    
    Usage
    -----
    with Timer("My operation"):
        do_something_expensive()
    
    Output
    ------
    [My operation] completed in 2.34 seconds
    """
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = perf_counter() - self.start_time
        print(f"[{self.name}] completed in {elapsed:.2f} seconds")


# ============================================================================
# FILE SYSTEM UTILITIES
# ============================================================================

def ensure_dir(path: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Parameters
    ----------
    path : str
        Directory path to create
    
    Returns
    -------
    Path
        Path object for created directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_results_csv(
    filename: str,
    headers: List[str],
    rows: List[List],
    base_dir: str = "../data"
) -> None:
    """
    Save results to CSV file.
    
    Parameters
    ----------
    filename : str
        Output filename (e.g., 'convergence_results.csv')
    headers : list of str
        Column headers
    rows : list of lists
        Data rows
    base_dir : str
        Output directory
    """
    ensure_dir(base_dir)
    filepath = Path(base_dir) / filename
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"Results saved to {filepath}")


# ============================================================================
# METRICS
# ============================================================================

def compute_rmse(estimated: np.ndarray, true: np.ndarray) -> float:
    """
    Compute Root Mean Square Error.
    
    RMSE = √( (1/n) Σ(est_i - true_i)² )
    
    Parameters
    ----------
    estimated : ndarray
        Estimated values
    true : ndarray
        Ground truth values
    
    Returns
    -------
    float
        RMSE value
    """
    return np.sqrt(np.mean((estimated - true) ** 2))


def compute_mae(estimated: np.ndarray, true: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    Parameters
    ----------
    estimated : ndarray
        Estimated values
    true : ndarray
        Ground truth values
    
    Returns
    -------
    float
        MAE value
    """
    return np.mean(np.abs(estimated - true))


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_filter_results(
    time_grid: np.ndarray,
    true_variance: np.ndarray,
    estimated_variance: np.ndarray,
    uncertainty_band: np.ndarray,
    prices: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Plot particle filter results: true vs estimated volatility.
    
    Creates a multi-panel figure showing:
    - Top: Asset price
    - Middle: True vs estimated volatility with uncertainty bands
    - Bottom: Absolute estimation error
    
    ARRAY ALIGNMENT LOGIC
    ----------------------
    The filter produces estimates for time steps 1..T (after observing returns).
    We align by taking min_length across all arrays to avoid shape mismatches.
    
    Parameters
    ----------
    time_grid : ndarray (N+1,)
        Time points [t_0, t_1, ..., t_N]
    true_variance : ndarray (N,)
        True variance aligned with filter estimates
    estimated_variance : ndarray (M,)
        Filter estimates where M = N or N-1
    uncertainty_band : ndarray (M,)
        Uncertainty (same length as estimates)
    prices : ndarray (N+1,), optional
        Asset prices for top panel
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure dimensions
    """
    # ================================================================
    # Align all arrays to same minimum length
    # ================================================================
    min_len = min(
        len(estimated_variance),
        len(uncertainty_band),
        len(true_variance) - 1  # true_variance has one extra element
    )
    
    # Truncate all arrays to min_len for plotting
    t_plot = time_grid[1:1+min_len]
    true_plot = true_variance[1:1+min_len]
    est_plot = estimated_variance[:min_len]
    std_plot = uncertainty_band[:min_len]
    
    # Debug info (uncomment to check shapes)
    # print(f"Plotting {min_len} points")
    # print(f"t_plot: {t_plot.shape}, true: {true_plot.shape}, est: {est_plot.shape}")
    
    # ================================================================
    # Create figure with 3 panels
    # ================================================================
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # ----------------------------------------------------------------
    # Top panel: Asset price
    # ----------------------------------------------------------------
    if prices is not None:
        ax1 = axes[0]
        # Plot full price series (can be longer than estimates)
        ax1.plot(time_grid, prices, 'b-', linewidth=0.8, alpha=0.8)
        # Add vertical line showing where estimation starts
        ax1.axvline(x=time_grid[1], color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Asset Price ($)', fontsize=12)
        ax1.set_title('Particle Filter: Stochastic Volatility Estimation', 
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(['Simulated Price', 'Filter Start'], loc='upper left', fontsize=9)
    
    # ----------------------------------------------------------------
    # Middle panel: True vs Estimated Volatility
    # ----------------------------------------------------------------
    ax2 = axes[1]
    
    # Convert variance to volatility (%) for interpretability
    true_vol = np.sqrt(np.maximum(true_plot, 1e-10)) * 100
    est_vol = np.sqrt(np.maximum(est_plot, 1e-10)) * 100
    std_vol = np.sqrt(np.maximum(std_plot, 1e-10)) * 100
    
    # Plot true volatility (thick black line behind)
    ax2.plot(t_plot, true_vol, 'k-', 
             linewidth=1.5, label='True Volatility', alpha=0.7, zorder=1)
    
    # Plot estimated volatility (red line on top)
    ax2.plot(t_plot, est_vol, 'r-', 
             linewidth=1.5, label='Estimated Volatility', alpha=0.9, zorder=2)
    
    # 95% credible interval (±2 standard deviations)
    upper_bound = est_vol + 2 * std_vol
    lower_bound = np.maximum(est_vol - 2 * std_vol, 0)
    ax2.fill_between(t_plot, lower_bound, upper_bound,
                     color='red', alpha=0.15, zorder=0,
                     label='95% Credible Interval')
    
    ax2.set_ylabel('Volatility (%)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ----------------------------------------------------------------
    # Bottom panel: Absolute estimation error
    # ----------------------------------------------------------------
    ax3 = axes[2]
    abs_error = np.abs(est_vol - true_vol)
    
    # Area plot for error magnitude
    ax3.fill_between(t_plot, 0, abs_error, color='orange', alpha=0.3)
    ax3.plot(t_plot, abs_error, 'orange', linewidth=0.8)
    
    # Add mean error line
    mean_error = np.mean(abs_error)
    ax3.axhline(y=mean_error, color='darkorange', linestyle='--', 
                linewidth=1, alpha=0.7, label=f'Mean Error: {mean_error:.2f}%')
    
    ax3.set_xlabel('Time (years)', fontsize=12)
    ax3.set_ylabel('Absolute Error (%)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_convergence_analysis(
    N_values: List[int],
    rmse_values: List[float],
    theoretical_rate: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot convergence analysis: RMSE vs number of particles.
    
    Demonstrates the Monte Carlo convergence rate O(1/√N).
    
    Parameters
    ----------
    N_values : list of int
        Number of particles tested
    rmse_values : list of float
        RMSE for each N
    theoretical_rate : bool
        Overlay theoretical 1/√N curve
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure dimensions
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot empirical RMSE
    ax.loglog(N_values, rmse_values, 'bo-', 
              linewidth=2, markersize=8, 
              label='Empirical RMSE')
    
    # Overlay theoretical rate if requested
    if theoretical_rate and len(N_values) > 0:
        # Scale to match at first point
        scale = rmse_values[0] * np.sqrt(N_values[0])
        theory = scale / np.sqrt(np.array(N_values))
        ax.loglog(N_values, theory, 'r--', 
                  linewidth=2, alpha=0.7,
                  label=r'Theory: $\propto 1/\sqrt{N}$')
    
    ax.set_xlabel('Number of Particles (N)', fontsize=12)
    ax.set_ylabel('RMSE (variance units)', fontsize=12)
    ax.set_title('Convergence Analysis: Error vs Number of Particles',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()