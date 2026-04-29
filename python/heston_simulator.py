"""
Heston Stochastic Volatility Model Simulator
==============================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/scm-heston-filter

Implements the numerical simulation of the Heston stochastic volatility
model using the Euler-Maruyama scheme with full truncation correction.

THEORETICAL FRAMEWORK
-----------------------
The Heston model (Heston, 1993) describes the joint evolution of an
asset price S_t and its instantaneous variance v_t:

    dS_t = μ S_t dt + √(v_t) S_t dW_1(t)
    dv_t = κ(θ - v_t) dt + ξ √(v_t) dW_2(t)
    
    E[dW_1 dW_2] = ρ dt

KEY FEATURES OF THE HESTON MODEL
----------------------------------
1. Stochastic volatility: Variance evolves randomly (not constant)
2. Mean reversion: Variance reverts to long-run level θ at speed κ
3. Leverage effect: ρ < 0 captures the empirical fact that
   volatility increases when prices drop
4. Closed-form option pricing possible (via characteristic function)

WHY WE SIMULATE FROM THIS MODEL
---------------------------------
In this project, we use the Heston model as the TRUE data-generating
process. We then try to RECOVER the latent variance v_t from observed
prices S_t using a particle filter. This is a common setup in:
- Algorithmic trading (real-time vol estimation)
- Risk management (VaR with stochastic vol)
- Market making (vol regime detection)

THE FULL TRUNCATION SCHEME
-----------------------------
The Euler-Maruyama discretization:

    v_{t+Δt} = v_t + κ(θ - v_t)Δt + ξ√(v_t)√(Δt) ε

can produce v_{t+Δt} < 0 when v_t is small and ε is negative.
The "full truncation" fix (Lord et al., 2010):

    v_{t+Δt} = max(v_t + κ(θ - v_t)Δt + ξ√(max(v_t,0))√(Δt) ε, 0)

This is:
- Simple to implement
- Empirically accurate for equity parameters
- Widely used in industry

REFERENCES
-----------
- Heston, S. L. (1993). "A Closed-Form Solution for Options with
  Stochastic Volatility"
- Lord, R., Koekkoek, R., & Van Dijk, D. (2010). "A comparison of
  biased simulation schemes for stochastic volatility models"
- Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class HestonParams:
    """
    Heston model parameters with built-in validation.
    
    This immutable container ensures all parameters are economically
    meaningful before simulation begins. Catches errors early.
    
    Parameters
    ----------
    mu : float
        Annual expected return (drift under P-measure)
        Default: 0.05 (5% per year)
        Typical range: [-0.10, 0.20] for equities
    
    kappa : float
        Speed of variance mean reversion
        Default: 2.0 (half-life ≈ 0.35 years)
        Must be > 0 for stationarity
        Typical range: [0.5, 5.0]
    
    theta : float
        Long-run mean variance (NOT volatility!)
        Default: 0.04 (implies σ_long = 20%)
        Must be > 0
        Typical range: [0.01, 0.25] (σ: 10% to 50%)
    
    xi : float
        Volatility of variance ("vol-of-vol")
        Default: 0.3
        Must be > 0
        Typical range: [0.1, 1.0]
    
    rho : float
        Correlation between price and variance Brownian motions
        Default: -0.7 (strong leverage effect)
        Must be in [-1, 1]
        Typical range: [-0.9, -0.3] for equities
    
    Notes
    -----
    The Feller condition (2κθ > ξ²) ensures v_t > 0 almost surely
    in the continuous-time model. With our parameters:
        2κθ = 2 × 2.0 × 0.04 = 0.16
        ξ² = 0.3² = 0.09
        0.16 > 0.09 → Condition holds ✓
    
    When the Feller condition is violated (common in FX markets),
    the full truncation scheme becomes essential.
    """
    mu: float = 0.05
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if self.xi <= 0:
            raise ValueError(f"xi must be positive, got {self.xi}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
    
    @property
    def feller_condition(self) -> bool:
        """Check Feller condition: 2κθ > ξ²"""
        return 2 * self.kappa * self.theta > self.xi ** 2
    
    @property
    def long_vol(self) -> float:
        """Long-run volatility (sqrt of theta)"""
        return np.sqrt(self.theta)


# ============================================================================
# MAIN SIMULATOR CLASS
# ============================================================================

class HestonSimulator:
    """
    Simulate asset prices and variances from the Heston model.
    
    This simulator generates synthetic data where:
    - Prices S_t are OBSERVABLE (what you see in the market)
    - Variances v_t are LATENT (hidden state to estimate)
    
    The generated data serves as "ground truth" for validating
    the particle filter. You know the true variances (since you
    simulated them), so you can measure estimation accuracy.
    
    Attributes
    ----------
    params : HestonParams
        Model parameters (mu, kappa, theta, xi, rho)
    dt : float
        Time step in years (1/252 = daily, 1/12 = monthly)
    
    Methods
    -------
    simulate(T, S0, v0, seed)
        Generate one price/variance path
    
    Example
    -------
    >>> sim = HestonSimulator(dt=1/252)
    >>> t, S, v = sim.simulate(T=1.0, S0=100)
    >>> print(f"Prices: {S[:5]}")  # First 5 days
    >>> print(f"Variances: {v[:5]}")
    """
    
    def __init__(
        self,
        params: Optional[HestonParams] = None,
        dt: float = 1/252
    ):
        """
        Initialize simulator with model parameters.
        
        Parameters
        ----------
        params : HestonParams, optional
            Model parameters. Uses defaults if None.
        dt : float
            Time step in years (e.g., 1/252 for daily)
        """
        self.params = params if params is not None else HestonParams()
        self.dt = dt
        
        # Diagnostics
        if not self.params.feller_condition:
            print("⚠ Warning: Feller condition violated — full truncation active")
    
    def simulate(
        self,
        T: float = 1.0,
        S0: float = 100.0,
        v0: Optional[float] = None,
        random_seed: Optional[int] = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a single path from the Heston model.
        
        Algorithm (Euler-Maruyama with full truncation)
        ------------------------------------------------
        For each time step i = 0, ..., N-1:
        
        1. Generate correlated Gaussian shocks:
           ε₁, ε₂ ~ N(0,1) independent
           ΔW₁ = ε₁ √Δt
           ΔW₂ = (ρ ε₁ + √(1-ρ²) ε₂) √Δt
        
        2. Update variance:
           v_{i+1} = v_i + κ(θ-v_i)Δt + ξ √(max(v_i,0)) ΔW₂
           v_{i+1} ← max(v_{i+1}, 1e-10)  [full truncation]
        
        3. Update price:
           S_{i+1} = S_i · exp( μΔt + √(max(v_i,0)) ΔW₁ )
        
        Why full truncation instead of absorption/reflection?
        -------------------------------------------------------
        - Absorption (v←0 if v<0): Creates bias at zero, distorts mean
        - Reflection (v←|v|): Changes path distribution significantly
        - Full truncation (v←max(v,0)): Minimal bias, best empirical fit
          for equity parameters (Lord et al., 2010)
        
        Parameters
        ----------
        T : float
            Time horizon in years
        S0 : float
            Initial asset price
        v0 : float, optional
            Initial variance (default: theta)
        random_seed : int, optional
            Seed for reproducible simulation
        
        Returns
        -------
        time_grid : ndarray of shape (N+1,)
            Time points [0, dt, 2dt, ..., T]
        prices : ndarray of shape (N+1,)
            Asset prices S_0, S_dt, ..., S_T
        variances : ndarray of shape (N+1,)
            True variances v_0, v_dt, ..., v_T
        """
        # Set initial variance to long-run mean if not specified
        if v0 is None:
            v0 = self.params.theta
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Number of steps and time grid
        N = int(T / self.dt)
        time_grid = np.linspace(0, T, N + 1)
        
        # Initialize arrays
        S = np.zeros(N + 1)
        v = np.zeros(N + 1)
        S[0] = S0
        v[0] = v0
        
        # ================================================================
        # Generate all random numbers upfront (vectorized — much faster)
        # ================================================================
        # Independent standard normal shocks
        eps1 = np.random.standard_normal(N)  # Price shocks
        eps2 = np.random.standard_normal(N)  # Variance shocks
        
        # Correlated Brownian increments via Cholesky decomposition
        # dW1 = ε1 · √Δt
        # dW2 = (ρ·ε1 + √(1-ρ²)·ε2) · √Δt
        dw1 = eps1 * np.sqrt(self.dt)
        dw2 = (self.params.rho * eps1 + 
               np.sqrt(1 - self.params.rho**2) * eps2) * np.sqrt(self.dt)
        
        # ================================================================
        # Time evolution loop (Euler-Maruyama)
        # ================================================================
        for i in range(N):
            # Update variance with full truncation
            v_pred = (v[i] + 
                     self.params.kappa * (self.params.theta - v[i]) * self.dt +
                     self.params.xi * np.sqrt(max(v[i], 0)) * dw2[i])
            v[i + 1] = max(v_pred, 1e-10)
            
            # Update price using variance at time i
            S[i + 1] = S[i] * np.exp(
                self.params.mu * self.dt +
                np.sqrt(max(v[i], 0)) * dw1[i]
            )
        
        return time_grid, S, v
    
    def get_parameter_summary(self) -> str:
        """
        Generate formatted parameter summary for logging.
        
        Returns
        -------
        str
            Multiline string with parameter values and derived quantities
        """
        p = self.params
        return (
            f"HestonSimulator(dt={self.dt:.4f}, "
            f"mu={p.mu}, kappa={p.kappa}, theta={p.theta}, "
            f"xi={p.xi}, rho={p.rho})\n"
            f"  Feller condition: {p.feller_condition}\n"
            f"  Long-run vol: {p.long_vol:.4f} ({p.long_vol*100:.1f}%)"
        )


# ============================================================================
# DEMO FUNCTION
# ============================================================================

def demo_simulation():
    """
    Quick demonstration of the Heston simulator.
    
    Generates one year of daily data and prints summary statistics.
    Useful for verifying installation and understanding the model.
    """
    print("\n" + "="*60)
    print("HESTON SIMULATOR — DEMONSTRATION")
    print("="*60 + "\n")
    
    # Create simulator with default parameters
    sim = HestonSimulator(dt=1/252)
    print(sim.get_parameter_summary() + "\n")
    
    # Simulate one year of daily data
    t, S, v = sim.simulate(T=1.0, S0=100, random_seed=42)
    
    # Compute and display statistics
    log_returns = np.diff(np.log(S))
    print(f"Simulation Results ({len(t)-1} steps):")
    print(f"  Initial price:    ${S[0]:.2f}")
    print(f"  Final price:      ${S[-1]:.2f}")
    print(f"  Annualized vol:   {np.std(log_returns)*np.sqrt(252):.4f}")
    print(f"  Mean variance:    {np.mean(v):.6f}")
    print(f"  Theoretical mean: {sim.params.theta:.6f}")
    
    return t, S, v


if __name__ == "__main__":
    demo_simulation()