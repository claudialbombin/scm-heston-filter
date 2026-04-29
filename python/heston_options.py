"""
Heston Model Option Pricing & Implied Volatility
==================================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/scm-heston-filter

Implements European option pricing under the Heston stochastic
volatility model using the characteristic function approach and
computes implied volatilities to generate the famous "volatility smile."

THEORETICAL BACKGROUND
-----------------------
Heston (1993) derived a closed-form solution for European option
prices using the characteristic function of the log-spot price.

The price of a European call with strike K and maturity T is:

    C(S, K, T) = S·P₁ - K·e^{-rT}·P₂

where P₁ and P₂ are risk-neutral probabilities computed via
Fourier inversion of the characteristic function:

    Pⱼ = 1/2 + 1/π ∫₀^∞ Re[ e^{-iu log(K)} · φ(u - i·aⱼ) / (iu) ] du

    φ(u) = E[e^{iu log(S_T)}]  (characteristic function)

    a₁ = 1, a₂ = 0

HESTON CHARACTERISTIC FUNCTION
--------------------------------
    φ(u) = exp{ C(u, T) + D(u, T)·v₀ + iu·log(S₀) + iu·rT }

where:
    C(u, τ) = κθ/ξ² · [ (κ - ρξiu - d)·τ - 2·log((1 - g·e^{-dτ})/(1 - g)) ]
    D(u, τ) = (κ - ρξiu - d)/ξ² · (1 - e^{-dτ})/(1 - g·e^{-dτ})
    
    d = √((ρξiu - κ)² + ξ²(iu + u²))
    g = (κ - ρξiu - d)/(κ - ρξiu + d)

WHY THIS IS IMPORTANT
----------------------
1. The volatility smile is THE defining feature of options markets
2. Black-Scholes assumes flat vol → cannot explain the smile
3. Heston naturally generates smile through:
   - ρ < 0: negative skew (equity index "leverage effect")
   - ξ > 0: curvature (vol-of-vol controls smile convexity)
   - κ: speed of mean reversion (controls term structure)

CONNECTING TO THE PARTICLE FILTER
------------------------------------
The particle filter estimates the instantaneous variance v_t.
This v_t can be used as input to price options and generate
the implied volatility smile at any point in time.

REFERENCES
-----------
- Heston, S. L. (1993). "A Closed-Form Solution for Options with 
  Stochastic Volatility with Applications to Bond and Currency Options"
- Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
- Carr, P. & Madan, D. (1999). "Option Valuation Using the Fast Fourier Transform"
- Albrecher, H. et al. (2007). "The Little Heston Trap"
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class OptionContract:
    """
    European option contract specification.
    
    Parameters
    ----------
    S0 : float
        Current spot price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate (continuously compounded)
    option_type : str
        'call' or 'put'
    """
    S0: float
    K: float
    T: float
    r: float
    option_type: str = 'call'
    
    def __post_init__(self):
        if self.option_type not in ['call', 'put']:
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type}")
    
    @property
    def moneyness(self) -> float:
        """Log-moneyness: log(K/S0)"""
        return np.log(self.K / self.S0)
    
    @property
    def is_itm(self) -> bool:
        """Check if option is in-the-money"""
        if self.option_type == 'call':
            return self.S0 > self.K
        return self.S0 < self.K


# ============================================================================
# HESTON CHARACTERISTIC FUNCTION
# ============================================================================

class HestonCharacteristicFunction:
    """
    Heston model characteristic function.
    
    The characteristic function φ(u) = E[e^{iu log(S_T)}] is the
    Fourier transform of the log-return distribution. It's the
    key ingredient for option pricing via Fourier inversion.
    
    WHY CHARACTERISTIC FUNCTION?
    -----------------------------
    Unlike Black-Scholes where returns are normal, Heston returns
    have no closed-form density. But the characteristic function
    IS available in closed form, allowing us to price options
    using numerical integration (quadrature).
    
    Parameters
    ----------
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Volatility of variance
    rho : float
        Correlation
    v0 : float
        Current (spot) variance
    """
    
    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        v0: float = 0.04
    ):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
    
    def _compute_d(self, u: complex) -> complex:
        """
        Compute the discriminant d(u).
        
        d = √((ρξiu - κ)² + ξ²(iu + u²))
        
        The branch cut is handled carefully to ensure continuity
        (this is the "Heston trap" — wrong branch leads to
        incorrect prices for long maturities).
        """
        return np.sqrt(
            (self.rho * self.xi * 1j * u - self.kappa)**2 +
            self.xi**2 * (1j * u + u**2)
        )
    
    def _compute_g(self, u: complex, d: complex) -> complex:
        """Compute the g coefficient."""
        numerator = self.kappa - self.rho * self.xi * 1j * u - d
        denominator = self.kappa - self.rho * self.xi * 1j * u + d
        return numerator / denominator
    
    def _compute_D(self, u: complex, tau: float) -> complex:
        """Compute D(u, τ) coefficient."""
        d = self._compute_d(u)
        g = self._compute_g(u, d)
        
        exp_term = np.exp(-d * tau)
        numerator = (1 - exp_term)
        denominator = (1 - g * exp_term)
        
        return (self.kappa - self.rho * self.xi * 1j * u - d) / self.xi**2 * (numerator / denominator)
    
    def _compute_C(self, u: complex, tau: float) -> complex:
        """Compute C(u, τ) coefficient."""
        d = self._compute_d(u)
        g = self._compute_g(u, d)
        
        term1 = (self.kappa - self.rho * self.xi * 1j * u - d) * tau
        term2 = -2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g))
        
        return self.kappa * self.theta / self.xi**2 * (term1 + term2)
    
    def evaluate(
        self,
        u: complex,
        S0: float,
        r: float,
        tau: float
    ) -> complex:
        """
        Evaluate the characteristic function φ(u).
        
        φ(u) = exp{ C(u,τ) + D(u,τ)·v₀ + iu·log(S₀) + iu·r·τ }
        
        Parameters
        ----------
        u : complex
            Fourier variable
        S0 : float
            Current spot price
        r : float
            Risk-free rate
        tau : float
            Time to maturity
            
        Returns
        -------
        complex
            Characteristic function value
        """
        C = self._compute_C(u, tau)
        D = self._compute_D(u, tau)
        
        phi = np.exp(
            C +
            D * self.v0 +
            1j * u * (np.log(S0) + r * tau)
        )
        
        return phi


# ============================================================================
# HESTON OPTION PRICER
# ============================================================================

class HestonOptionPricer:
    """
    European option pricing under the Heston model.
    
    Computes call/put prices via numerical integration of the
    characteristic function (Fourier inversion).
    
    The pricing formula:
        C = S₀·P₁ - K·e^{-rT}·P₂
        
        P₁ = 1/2 + 1/π ∫₀^∞ Re[ e^{-iu log(K)} · φ(u - i) / (iu·φ(-i)) ] du
        P₂ = 1/2 + 1/π ∫₀^∞ Re[ e^{-iu log(K)} · φ(u) / (iu) ] du
    
    NUMERICAL INTEGRATION
    ----------------------
    We use scipy.integrate.quad with adaptive quadrature.
    Integration limit: [0, 100] is typically sufficient.
    For extreme parameters, increase to [0, 500].
    
    PUT-CALL PARITY
    ----------------
    Put prices are obtained via put-call parity:
        P = C - S₀ + K·e^{-rT}
    
    This avoids numerical issues with the put formula.
    
    Parameters
    ----------
    char_func : HestonCharacteristicFunction
        Characteristic function for the model
    integration_limit : float
        Upper limit for numerical integration
    """
    
    def __init__(
        self,
        char_func: HestonCharacteristicFunction,
        integration_limit: float = 100.0
    ):
        self.char_func = char_func
        self.integration_limit = integration_limit
    
    def _integrand_P1(self, u: float, S0: float, K: float, r: float, tau: float) -> float:
        """
        Integrand for P₁ probability.
        
        P₁ = 1/2 + 1/π ∫₀^∞ Re[ e^{-iu log(K)} · φ(u - i) / (iu·φ(-i)) ] du
        """
        phi_u_minus_i = self.char_func.evaluate(u - 1j, S0, r, tau)
        phi_minus_i = self.char_func.evaluate(-1j, S0, r, tau)
        
        numerator = np.exp(-1j * u * np.log(K)) * phi_u_minus_i
        denominator = 1j * u * phi_minus_i
        
        return np.real(numerator / denominator)
    
    def _integrand_P2(self, u: float, S0: float, K: float, r: float, tau: float) -> float:
        """
        Integrand for P₂ probability.
        
        P₂ = 1/2 + 1/π ∫₀^∞ Re[ e^{-iu log(K)} · φ(u) / (iu) ] du
        """
        phi_u = self.char_func.evaluate(u, S0, r, tau)
        numerator = np.exp(-1j * u * np.log(K)) * phi_u
        
        return np.real(numerator / (1j * u))
    
    def price_call(
        self,
        S0: float,
        K: float,
        r: float,
        tau: float
    ) -> float:
        """
        Price a European call option under Heston.
        
        Parameters
        ----------
        S0 : float
            Current spot price
        K : float
            Strike price
        r : float
            Risk-free rate
        tau : float
            Time to maturity
            
        Returns
        -------
        float
            Call option price
        """
        # Numerical integration for P₁ and P₂
        P1_integral, _ = quad(
            self._integrand_P1,
            0, self.integration_limit,
            args=(S0, K, r, tau),
            limit=200,
            epsabs=1e-8,
            epsrel=1e-8
        )
        
        P2_integral, _ = quad(
            self._integrand_P2,
            0, self.integration_limit,
            args=(S0, K, r, tau),
            limit=200,
            epsabs=1e-8,
            epsrel=1e-8
        )
        
        P1 = 0.5 + P1_integral / np.pi
        P2 = 0.5 + P2_integral / np.pi
        
        call_price = S0 * P1 - K * np.exp(-r * tau) * P2
        
        return max(call_price, 0.0)  # Ensure non-negative
    
    def price_put(
        self,
        S0: float,
        K: float,
        r: float,
        tau: float
    ) -> float:
        """
        Price a European put via put-call parity.
        
        P = C - S₀ + K·e^{-rT}
        """
        call_price = self.price_call(S0, K, r, tau)
        put_price = call_price - S0 + K * np.exp(-r * tau)
        return max(put_price, 0.0)
    
    def price(
        self,
        contract: OptionContract
    ) -> float:
        """
        Price an option contract.
        
        Parameters
        ----------
        contract : OptionContract
            Option specification
            
        Returns
        -------
        float
            Option price
        """
        if contract.option_type == 'call':
            return self.price_call(
                contract.S0, contract.K, contract.r, contract.T
            )
        return self.price_put(
            contract.S0, contract.K, contract.r, contract.T
        )


# ============================================================================
# IMPLIED VOLATILITY CALCULATION
# ============================================================================

class ImpliedVolatilityCalculator:
    """
    Compute Black-Scholes implied volatility.
    
    Implied volatility is the σ that, when plugged into the
    Black-Scholes formula, gives the observed market price.
    
    It's found by numerically solving:
        C_BS(σ) - C_market = 0
    
    using Brent's method (robust root-finding).
    
    INTERPRETATION
    ---------------
    - IV > realized vol: Options are expensive (risk premium)
    - IV smile: IV varies with strike → Black-Scholes is misspecified
    - IV skew: Downside puts more expensive (crash protection demand)
    """
    
    @staticmethod
    def black_scholes_price(
        S0: float, K: float, T: float, r: float, sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Black-Scholes European option price.
        
        C = S₀·N(d₁) - K·e^{-rT}·N(d₂)
        P = K·e^{-rT}·N(-d₂) - S₀·N(-d₁)
        
        d₁ = [log(S₀/K) + (r + σ²/2)T] / (σ√T)
        d₂ = d₁ - σ√T
        """
        if T <= 0 or sigma <= 0:
            return max(S0 - K, 0) if option_type == 'call' else max(K - S0, 0)
        
        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    @staticmethod
    def compute_iv(
        market_price: float,
        S0: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call'
    ) -> float:
        """
        Compute implied volatility via root-finding.
        
        Uses Brent's method to solve:
            BS_price(σ) - market_price = 0
        
        Parameters
        ----------
        market_price : float
            Observed option price
        S0 : float
            Spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
            
        Returns
        -------
        float
            Implied volatility (decimal, not percentage)
        """
        # Check if option has zero intrinsic value
        intrinsic = max(S0 - K, 0) if option_type == 'call' else max(K - S0, 0)
        if market_price <= intrinsic:
            return 0.0
        
        # Objective function: BS_price - market_price
        def objective(sigma):
            return (ImpliedVolatilityCalculator.black_scholes_price(
                S0, K, T, r, sigma, option_type
            ) - market_price)
        
        try:
            iv = brentq(objective, 1e-10, 5.0, xtol=1e-8, rtol=1e-8)
            return iv
        except ValueError:
            return np.nan
    
    @staticmethod
    def compute_iv_from_heston(
        heston_price: float,
        contract: OptionContract
    ) -> float:
        """
        Compute IV from a Heston model price.
        
        This is the key function for generating the volatility smile:
        1. Price an option under Heston → produces price
        2. Invert Black-Scholes → find σ that matches that price
        3. Plot σ vs K → the volatility smile emerges!
        
        Parameters
        ----------
        heston_price : float
            Option price from Heston model
        contract : OptionContract
            Option specification
            
        Returns
        -------
        float
            Implied volatility
        """
        return ImpliedVolatilityCalculator.compute_iv(
            heston_price,
            contract.S0,
            contract.K,
            contract.T,
            contract.r,
            contract.option_type
        )