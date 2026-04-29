"""
Bootstrap Particle Filter for Stochastic Volatility Estimation
================================================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/scm-heston-filter

Implements Sequential Monte Carlo (SMC) for tracking latent variance
in the Heston model from observed asset prices only.

WHAT IS A PARTICLE FILTER?
----------------------------
A particle filter approximates the posterior distribution p(v_t | S_{1:t})
using a set of N weighted samples (particles):

    {v_t^(i), w_t^(i)}_{i=1}^N

where:
- v_t^(i): particle i's estimate of volatility at time t
- w_t^(i): importance weight (how well particle i explains observations)

THREE STEPS OF THE BOOTSTRAP FILTER
-------------------------------------
1. PREDICT: Propagate particles using state dynamics
   v_t^(i) ~ p(v_t | v_{t-1}^(i))  [Heston variance SDE]

2. UPDATE: Weight particles by observation likelihood
   w_t^(i) ∝ p(S_t | v_t^(i))  [log-normal likelihood]

3. RESAMPLE: Eliminate low-weight particles, duplicate high-weight ones
   v_t^(i) ← v_t^(j) with P(j) = w_t^(j)

WHY THIS IS POWERFUL FOR FINANCE
-----------------------------------
- Online estimation: updates as each new price arrives
- Full distribution: provides uncertainty bands, not just point estimates
- Non-parametric: no assumption about posterior shape
- Robust: handles regime switches, jumps, and fat tails
- Used at: Jane Street, Citadel, Two Sigma for vol estimation

CONVERGENCE PROPERTIES
------------------------
As N → ∞, the particle approximation converges to the true posterior
at rate O(1/√N) under mild regularity conditions (Del Moral, 2004).

REFERENCES
-----------
- Gordon, Salmond & Smith (1993): Original bootstrap filter
- Doucet, de Freitas & Gordon (2001): SMC methods in practice
- Javaheri (2011): Inside Volatility Arbitrage (finance applications)
- Del Moral (2004): Feynman-Kac formulae (convergence theory)
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================================
# CONFIGURATION DATA CLASS
# ============================================================================

@dataclass
class FilterConfig:
    """
    Configuration for the bootstrap particle filter.
    
    Parameters
    ----------
    N_particles : int
        Number of particles (trading accuracy vs speed)
        Default: 500
        Practical range: [100, 10000]
        Note: RMSE ∝ 1/√N, so 4× particles halves the error
    
    dt : float
        Time step in years (must match simulation)
        Default: 1/252 (daily)
    
    mu : float
        Expected return (drift for log-return distribution)
        Default: 0.05
    
    kappa : float
        Variance mean reversion speed
        Default: 2.0
    
    theta : float
        Long-run mean variance
        Default: 0.04
    
    xi : float
        Volatility of variance
        Default: 0.3
    
    rho : float
        Correlation (available for extensions, not used in basic filter)
        Default: -0.7
    
    resample_threshold : float
        Fraction of N below which we resample
        Default: 0.5 (resample when ESS < N/2)
    """
    N_particles: int = 500
    dt: float = 1/252
    mu: float = 0.05
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    resample_threshold: float = 0.5


# ============================================================================
# PARTICLE FILTER CLASS
# ============================================================================

class BootstrapParticleFilter:
    """
    Bootstrap particle filter for Heston variance estimation.
    
    Maintains a cloud of particles representing possible variance values
    and updates them sequentially as new price observations arrive.
    
    Key Design Decisions
    --------------------
    1. Bootstrap proposal (not auxiliary): Simpler, more robust
    2. Systematic resampling (not multinomial): Lower variance
    3. Log-normal likelihood: Matches Heston's conditional distribution
    4. Full truncation in prediction: Ensures variance positivity
    
    Attributes
    ----------
    config : FilterConfig
        Filter configuration
    N : int
        Number of particles
    particles : ndarray (N,)
        Current variance particles
    weights : ndarray (N,)
        Current normalized weights
    history_particles : list
        History of particle clouds for analysis
    
    Example
    -------
    >>> pf = BootstrapParticleFilter(N_particles=500)
    >>> pf.initialize_particles(v0=0.04)
    >>> for price in observed_prices:
    ...     pf.predict()
    ...     pf.update(log_return)
    ...     if pf.needs_resampling():
    ...         pf.resample()
    ...     mean, std, ess = pf.estimate()
    """
    
    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        N_particles: Optional[int] = None
    ):
        """
        Initialize the particle filter.
        
        Parameters
        ----------
        config : FilterConfig, optional
            Full configuration object
        N_particles : int, optional
            Shorthand to override number of particles
        """
        if config is None:
            config = FilterConfig()
        if N_particles is not None:
            config.N_particles = N_particles
        
        self.config = config
        self.N = config.N_particles
        
        # Particle storage
        self.particles = None
        self.weights = None
        
        # Track history for analysis
        self.estimated_means = []
        self.estimated_stds = []
        self.effective_sample_sizes = []
    
    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------
    
    def initialize_particles(
        self,
        v0: Optional[float] = None,
        init_dispersion: float = 0.5
    ) -> np.ndarray:
        """
        Initialize particle cloud from prior distribution.
        
        Strategy: Sample from log-normal distribution centered at
        long-run mean θ with controlled dispersion.
        
        Why log-normal?
        ---------------
        Variance must be positive. A log-normal prior:
        - Ensures v > 0 always
        - Produces realistic right-skewed distribution
        - Allows easy control via dispersion parameter
        
        Parameters
        ----------
        v0 : float, optional
            Initial variance guess (default: theta)
        init_dispersion : float
            Controls initial uncertainty (0.1 = tight, 1.0 = diffuse)
        
        Returns
        -------
        particles : ndarray (N,)
            Initial particle values
        """
        if v0 is None:
            v0 = self.config.theta
        
        # Compute log-normal parameters
        sigma_log = init_dispersion
        mu_log = np.log(v0) - 0.5 * sigma_log**2
        
        # Sample particles
        self.particles = np.random.lognormal(
            mean=mu_log,
            sigma=sigma_log,
            size=self.N
        )
        
        # Equal initial weights
        self.weights = np.ones(self.N) / self.N
        
        return self.particles
    
    # ------------------------------------------------------------------
    # PREDICTION STEP
    # ------------------------------------------------------------------
    
    def predict(self) -> np.ndarray:
        """
        Propagate particles according to Heston variance dynamics.
        
        Euler-Maruyama discretization:
            v_{k}^{(i)} = v_{k-1}^{(i)} + κ(θ - v_{k-1}^{(i)})Δt
                         + ξ √(v_{k-1}^{(i)}) √Δt ε^{(i)}
        
        where ε^{(i)} ~ N(0,1) are independent for each particle.
        
        The noise terms ε^{(i)} model our uncertainty about how
        volatility evolves. Without them, all particles would
        deterministically converge to θ (sample impoverishment).
        
        Returns
        -------
        particles : ndarray (N,)
            Propagated particles
        """
        # Generate independent shocks for each particle
        shocks = np.random.standard_normal(self.N)
        
        # Compute drift: κ(θ - v)Δt
        drift = (self.config.kappa * 
                (self.config.theta - self.particles) * 
                self.config.dt)
        
        # Compute diffusion: ξ √(v) √(Δt) ε
        vol_terms = np.sqrt(np.maximum(self.particles, 0))
        diffusion = (self.config.xi * vol_terms * 
                    np.sqrt(self.config.dt) * shocks)
        
        # Update particles
        self.particles = self.particles + drift + diffusion
        
        # Full truncation: ensure variance ≥ 0
        self.particles = np.maximum(self.particles, 1e-10)
        
        return self.particles
    
    # ------------------------------------------------------------------
    # UPDATE STEP (WEIGHTING)
    # ------------------------------------------------------------------
    
    def update(self, log_return: float) -> np.ndarray:
        """
        Compute importance weights from observation likelihood.
        
        The observation model (log-return given variance):
            r_k = log(S_k / S_{k-1}) | v_{k-1} ~ N(μΔt, v_{k-1}Δt)
        
        Each particle's weight is proportional to the likelihood:
            w_k^{(i)} ∝ exp(-0.5 * (r_k - μΔt)² / (v_{k-1}^{(i)}Δt))
                   / √(2π v_{k-1}^{(i)}Δt)
        
        Implementation uses log-space for numerical stability
        (avoids underflow when multiplying many small numbers).
        
        Parameters
        ----------
        log_return : float
            Observed log return r_k = log(S_k / S_{k-1})
        
        Returns
        -------
        weights : ndarray (N,)
            Normalized importance weights
        """
        # Mean return (drift component)
        mean_return = self.config.mu * self.config.dt
        
        # Standard deviation for each particle: √(v^i * dt)
        std_particles = np.sqrt(
            np.maximum(self.particles, 1e-10) * self.config.dt
        )
        
        # Log-likelihood for each particle
        # log p(r_k | v^i) = -0.5 log(2π) - log(σ^i) - 0.5 ((r - μdt)/σ^i)^2
        log_weights = (
            -0.5 * np.log(2 * np.pi)
            - np.log(std_particles)
            - 0.5 * ((log_return - mean_return) / std_particles) ** 2
        )
        
        # Normalize using log-sum-exp trick for stability
        log_weights -= np.max(log_weights)  # Prevent overflow
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        
        self.weights = weights
        return weights
    
    # ------------------------------------------------------------------
    # RESAMPLING STEP
    # ------------------------------------------------------------------
    
    def needs_resampling(self) -> bool:
        """
        Check if resampling is necessary.
        
        Criterion: Effective Sample Size (ESS) < threshold × N
        
        ESS = 1 / Σ(w_i²) measures particle diversity:
        - ESS = N: Perfect diversity (all weights equal)
        - ESS = 1: Complete collapse (one particle dominates)
        - ESS < N/2: Standard threshold for resampling
        
        Returns
        -------
        bool
            True if resampling should be performed
        """
        ess = 1.0 / np.sum(self.weights ** 2)
        return ess < (self.config.resample_threshold * self.N)
    
    def systematic_resample(self) -> np.ndarray:
        """
        Perform systematic resampling of particles.
        
        Algorithm
        ---------
        1. Draw u ~ Uniform(0, 1/N)
        2. Create N ordered points: u, u+1/N, ..., u+(N-1)/N
        3. For each point, find which particle's CDF it falls in
        4. Copy selected particles, reset weights to 1/N
        
        Systematic resampling has lower variance than multinomial
        and guarantees that particles with weight > 1/N are selected
        at least floor(N*w_i) times.
        
        Returns
        -------
        indices : ndarray (N,)
            Indices of resampled particles
        """
        N = self.N
        
        # Generate systematic points
        u0 = np.random.uniform(0, 1/N)
        systematic_points = u0 + np.arange(N) / N
        
        # Find which particles to keep
        cumulative_weights = np.cumsum(self.weights)
        indices = np.searchsorted(cumulative_weights, systematic_points)
        
        # Apply resampling
        self.particles = self.particles[indices]
        self.weights = np.ones(N) / N
        
        return indices
    
    # ------------------------------------------------------------------
    # ESTIMATION
    # ------------------------------------------------------------------
    
    def estimate(self) -> Tuple[float, float, float]:
        """
        Compute point estimate and uncertainty from particle cloud.
        
        Returns
        -------
        mean : float
            Weighted mean variance estimate
        std : float
            Weighted standard deviation (uncertainty)
        ess : float
            Effective sample size (quality indicator)
        """
        # Weighted mean
        mean = np.average(self.particles, weights=self.weights)
        
        # Weighted variance
        variance = np.average(
            (self.particles - mean) ** 2,
            weights=self.weights
        )
        std = np.sqrt(variance)
        
        # Effective sample size
        ess = 1.0 / np.sum(self.weights ** 2)
        
        # Store in history
        self.estimated_means.append(mean)
        self.estimated_stds.append(std)
        self.effective_sample_sizes.append(ess)
        
        return mean, std, ess
    
    # ------------------------------------------------------------------
    # FULL FILTERING LOOP
    # ------------------------------------------------------------------
    
    def filter_from_returns(
        self,
        log_returns: np.ndarray,
        v0: Optional[float] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the complete particle filter on a sequence of log returns.
        
        This is the MAIN method. It orchestrates predict → update →
        (resample) → estimate for each time step.
        
        Parameters
        ----------
        log_returns : ndarray (T,)
            Sequence of log returns r_1, ..., r_T
        v0 : float, optional
            Initial variance (default: theta)
        verbose : bool
            Print progress at each step
        
        Returns
        -------
        estimated_means : ndarray (T,)
            Filtered variance estimates
        estimated_stds : ndarray (T,)
            Uncertainty around estimates
        ess_values : ndarray (T,)
            Effective sample sizes across time
        """
        T = len(log_returns)
        
        # Reset histories
        self.estimated_means = []
        self.estimated_stds = []
        self.effective_sample_sizes = []
        
        # Initialize particles
        self.initialize_particles(v0=v0, init_dispersion=0.3)
        
        if verbose:
            print(f"Running particle filter with N={self.N} particles...")
        
        # Sequential processing
        for t in range(T):
            # Step 1: Predict forward
            self.predict()
            
            # Step 2: Weight by observation
            self.update(log_returns[t])
            
            # Step 3: Resample if needed
            if self.needs_resampling():
                self.systematic_resample()
            
            # Step 4: Record estimate
            self.estimate()
            
            if verbose and (t + 1) % 50 == 0:
                mean, std, ess = (self.estimated_means[-1],
                                 self.estimated_stds[-1],
                                 self.effective_sample_sizes[-1])
                print(f"  Step {t+1}/{T}: v_est={mean:.6f} ± {std:.6f}, "
                      f"ESS={ess:.1f}")
        
        if verbose:
            print("Filtering complete.\n")
        
        return (np.array(self.estimated_means),
                np.array(self.estimated_stds),
                np.array(self.effective_sample_sizes))
    
    def filter_from_prices(
        self,
        prices: np.ndarray,
        v0: Optional[float] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run filter directly from price series (computes log returns).
        
        Convenience wrapper around filter_from_returns.
        
        Parameters
        ----------
        prices : ndarray (T,)
            Asset prices S_0, S_1, ..., S_T
        v0 : float, optional
            Initial variance
        verbose : bool
            Print progress
        
        Returns
        -------
        Same as filter_from_returns
        """
        # Compute log returns: r_k = log(S_k / S_{k-1})
        log_returns = np.diff(np.log(prices))
        
        return self.filter_from_returns(log_returns, v0=v0, verbose=verbose)