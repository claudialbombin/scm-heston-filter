"""
Convergence Analysis for Particle Filter
==========================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/scm-heston-filter

Studies the relationship between estimation error and number of
particles, demonstrating the theoretical O(1/√N) Monte Carlo rate.

THEORETICAL BACKGROUND
------------------------
For a bootstrap particle filter approximating the posterior
distribution p(v_t | S_{1:t}):
    RMSE(N) ≈ C / √N

where C depends on:
- Model parameters (κ, θ, ξ)
- Observation noise level
- Time horizon
- Resampling scheme

This module empirically validates this relationship by running
the filter multiple times with different particle counts and
measuring RMSE against known ground truth.
"""

import numpy as np
from typing import List, Tuple, Optional
from heston_simulator import HestonSimulator
from particle_filter import BootstrapParticleFilter, FilterConfig
from utils import compute_rmse, Timer


class ConvergenceAnalyzer:
    """
    Analyze particle filter convergence properties.
    
    Runs repeated experiments to measure how estimation error
    decreases as the number of particles increases.
    
    Attributes
    ----------
    simulator : HestonSimulator
        Generates ground truth data
    config : FilterConfig
        Base filter configuration
    n_trials : int
        Number of Monte Carlo trials per N value
    
    Example
    -------
    >>> analyzer = ConvergenceAnalyzer(n_trials=10)
    >>> N_values, rmse = analyzer.analyze(
    ...     N_values=[100, 200, 500, 1000, 2000]
    ... )
    """
    
    def __init__(
        self,
        simulator: Optional[HestonSimulator] = None,
        n_trials: int = 5,
        T: float = 1.0
    ):
        """
        Initialize convergence analyzer.
        
        Parameters
        ----------
        simulator : HestonSimulator, optional
            Simulator instance (creates default if None)
        n_trials : int
            Independent trials per N value
        T : float
            Simulation horizon in years
        """
        self.simulator = simulator or HestonSimulator(dt=1/252)
        self.n_trials = n_trials
        self.T = T
    
    def single_experiment(
        self,
        N_particles: int,
        random_seed: int
    ) -> float:
        """
        Run one filter experiment and return RMSE.
        
        Parameters
        ----------
        N_particles : int
            Number of particles to use
        random_seed : int
            Seed for reproducibility
        
        Returns
        -------
        rmse : float
            RMSE between estimated and true variance
        """
        # Generate ground truth
        t, S, v_true = self.simulator.simulate(
            T=self.T, S0=100, random_seed=random_seed
        )
        log_returns = np.diff(np.log(S))
        v_true_filter = v_true[1:]  # Align with returns
        
        # Run particle filter
        config = FilterConfig(N_particles=N_particles)
        pf = BootstrapParticleFilter(config=config)
        
        v_est, _, _ = pf.filter_from_returns(
            log_returns, v0=v_true[0], verbose=False
        )
        
        # Compute error
        return compute_rmse(v_est, v_true_filter)
    
    def analyze(
        self,
        N_values: List[int],
        base_seed: int = 42
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Run convergence analysis across multiple N values.
        
        Parameters
        ----------
        N_values : list of int
            Particle counts to test
        base_seed : int
            Base random seed
        
        Returns
        -------
        N_values : list of int
            Particle counts
        mean_rmse : list of float
            Mean RMSE across trials
        std_rmse : list of float
            Standard deviation of RMSE across trials
        """
        mean_rmse_list = []
        std_rmse_list = []
        
        print("="*60)
        print("CONVERGENCE ANALYSIS")
        print("="*60)
        print(f"  Trials per N: {self.n_trials}")
        print(f"  N values: {N_values}")
        print()
        
        for N in N_values:
            rmse_trials = []
            
            for trial in range(self.n_trials):
                seed = base_seed + trial
                rmse = self.single_experiment(N, seed)
                rmse_trials.append(rmse)
            
            mean_rmse = np.mean(rmse_trials)
            std_rmse = np.std(rmse_trials)
            
            mean_rmse_list.append(mean_rmse)
            std_rmse_list.append(std_rmse)
            
            print(f"  N={N:5d}: RMSE={mean_rmse:.6f} ± {std_rmse:.6f}")
        
        print()
        return N_values, mean_rmse_list, std_rmse_list
    
    def estimate_convergence_rate(
        self,
        N_values: List[int],
        rmse_values: List[float]
    ) -> float:
        """
        Estimate empirical convergence rate α where RMSE ∝ N^{-α}.
        
        Fits log(RMSE) = C - α·log(N) using linear regression.
        Theoretical rate is α = 0.5 (i.e., 1/√N).
        
        Parameters
        ----------
        N_values : list of int
            Particle counts
        rmse_values : list of float
            Corresponding RMSE values
        
        Returns
        -------
        alpha : float
            Estimated convergence rate
        """
        log_N = np.log(N_values)
        log_rmse = np.log(rmse_values)
        
        # Linear regression: log(RMSE) = C - α log(N)
        coeffs = np.polyfit(log_N, log_rmse, 1)
        alpha = -coeffs[0]
        
        return alpha