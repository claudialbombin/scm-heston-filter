"""
Heston Implied Volatility Smile Generator & Visualizer
========================================================
Author: Claudia Maria Lopez Bombin
License: MIT
Repository: github.com/claudialbombin/scm-heston-filter

Generates and visualizes the famous "volatility smile/skew" that
emerges naturally from the Heston stochastic volatility model.

WHAT IS THE VOLATILITY SMILE?
-------------------------------
In Black-Scholes, volatility is constant → all options on the same
underlying should have the SAME implied volatility regardless of
strike. Reality: ITM/OTM options have HIGHER implied vol than ATM
options → "the smile."

In equity markets, the smile is asymmetric (a "skew" or "smirk"):
- OTM puts (low strikes): HIGH implied vol → crash protection demand
- ATM options: LOWER implied vol  
- OTM calls (high strikes): Slightly higher than ATM

WHY DOES HESTON GENERATE THE SMILE?
-------------------------------------
The Heston model naturally produces the smile through:
1. ρ < 0 (negative correlation): Creates skew
   - When price drops, volatility spikes → OTM puts more valuable
   
2. ξ > 0 (vol-of-vol): Creates curvature  
   - Random volatility creates fat tails in the return distribution
   - OTM options more likely to finish ITM → higher prices → higher IV

3. κ (mean reversion): Controls term structure
   - Fast mean reversion → smile flattens with maturity
   - Slow mean reversion → smile persists

CONNECTING TO THE PARTICLE FILTER
------------------------------------
The particle filter provides real-time estimates of v_t (spot variance).
We can:
1. Use filtered v_t as input to Heston option pricing
2. Generate the implied volatility smile at each time step
3. Track how the smile evolves with market conditions
4. Identify mispriced options relative to the model

REFERENCES
-----------
- Derman, E. & Miller, M. (2016). "The Volatility Smile"
- Gatheral, J. (2006). "The Volatility Surface"
- Cont, R. & da Fonseca, J. (2002). "Dynamics of implied volatility surfaces"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass

from heston_options import (
    HestonCharacteristicFunction,
    HestonOptionPricer,
    ImpliedVolatilityCalculator,
    OptionContract,
)


# ============================================================================
# SMILE GENERATOR
# ============================================================================

@dataclass
class SmileConfig:
    """
    Configuration for volatility smile generation.
    
    Parameters
    ----------
    S0 : float
        Current spot price
    r : float
        Risk-free rate
    T : float
        Time to maturity
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Vol-of-vol
    rho : float
        Correlation
    v0 : float
        Current (spot) variance
    moneyness_range : tuple
        (min, max) moneyness as fraction of spot
    n_strikes : int
        Number of strike points
    """
    S0: float = 100.0
    r: float = 0.03
    T: float = 1.0
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    v0: float = 0.04
    moneyness_range: Tuple[float, float] = (0.5, 1.5)
    n_strikes: int = 50


class VolatilitySmileGenerator:
    """
    Generate Heston implied volatility smiles.
    
    Creates the full volatility smile/skew by:
    1. Computing Heston option prices for a range of strikes
    2. Inverting each price to get Black-Scholes implied vol
    3. Plotting IV vs strike/moneyness
    
    Example
    -------
    >>> config = SmileConfig()
    >>> generator = VolatilitySmileGenerator(config)
    >>> smiles = generator.generate_smiles()
    >>> generator.plot_smiles(smiles)
    """
    
    def __init__(self, config: SmileConfig):
        """
        Initialize the smile generator.
        
        Parameters
        ----------
        config : SmileConfig
            Configuration for smile generation
        """
        self.config = config
        self._setup_pricer()
    
    def _setup_pricer(self):
        """Initialize the Heston pricer with current parameters."""
        char_func = HestonCharacteristicFunction(
            kappa=self.config.kappa,
            theta=self.config.theta,
            xi=self.config.xi,
            rho=self.config.rho,
            v0=self.config.v0
        )
        self.pricer = HestonOptionPricer(char_func)
        self.iv_calculator = ImpliedVolatilityCalculator()
    
    def generate_strikes(self) -> np.ndarray:
        """
        Generate strike prices for the smile.
        
        Creates n_strikes points between moneyness_range[0]*S0
        and moneyness_range[1]*S0.
        """
        min_K = self.config.moneyness_range[0] * self.config.S0
        max_K = self.config.moneyness_range[1] * self.config.S0
        
        return np.linspace(min_K, max_K, self.config.n_strikes)
    
    def generate_single_smile(
        self,
        v0: Optional[float] = None,
        label: str = "Heston Smile"
    ) -> dict:
        """
        Generate one volatility smile.
        
        Parameters
        ----------
        v0 : float, optional
            Current variance (overrides config)
        label : str
            Label for this smile
            
        Returns
        -------
        dict with keys:
            - strikes: array of strikes
            - moneyness: log(K/S0)
            - prices: Heston option prices
            - implied_vols: Black-Scholes implied volatilities
            - label: smile label
        """
        if v0 is not None:
            self.config.v0 = v0
            self._setup_pricer()  # Rebuild pricer with new v0
        
        strikes = self.generate_strikes()
        moneyness = np.log(strikes / self.config.S0)
        prices = np.zeros(self.config.n_strikes)
        implied_vols = np.zeros(self.config.n_strikes)
        
        config_ref = self.config
        print(f"  Generating smile: {label} (S0={config_ref.S0}, T={config_ref.T}, v0={config_ref.v0:.4f})")
        
        for i, K in enumerate(strikes):
            # Create option contract
            contract = OptionContract(
                S0=config_ref.S0,
                K=K,
                T=config_ref.T,
                r=config_ref.r,
                option_type='call'
            )
            
            # Price under Heston
            heston_price = self.pricer.price(contract)
            prices[i] = heston_price
            
            # Compute implied volatility
            iv = self.iv_calculator.compute_iv_from_heston(
                heston_price, contract
            )
            implied_vols[i] = iv
        
        return {
            'strikes': strikes,
            'moneyness': moneyness,
            'prices': prices,
            'implied_vols': implied_vols * 100,  # Convert to percentage
            'label': label
        }
    
    def generate_parameter_study(
        self,
        parameter_name: str,
        parameter_values: List[float]
    ) -> List[dict]:
        """
        Generate smiles for different parameter values.
        
        Study how individual parameters affect the smile shape.
        
        Parameters
        ----------
        parameter_name : str
            'rho', 'xi', 'kappa', or 'theta'
        parameter_values : list of float
            Values to test
            
        Returns
        -------
        list of dict
            Smile data for each parameter value
        """
        smiles = []
        original_value = getattr(self.config, parameter_name)
        
        for value in parameter_values:
            # Update parameter
            setattr(self.config, parameter_name, value)
            self._setup_pricer()
            
            # Generate smile
            label = f"{parameter_name} = {value}"
            smile = self.generate_single_smile(label=label)
            smiles.append(smile)
        
        # Restore original value
        setattr(self.config, parameter_name, original_value)
        self._setup_pricer()
        
        return smiles


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_volatility_smile(
    smiles: List[dict],
    title: str = "Heston Implied Volatility Smile",
    black_scholes_vol: Optional[float] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot volatility smiles/skews.
    
    Parameters
    ----------
    smiles : list of dict
        Smile data from generate_single_smile
    title : str
        Plot title
    black_scholes_vol : float, optional
        BS constant vol for comparison
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure dimensions
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ----------------------------------------------------------------
    # Left panel: IV vs Strike
    # ----------------------------------------------------------------
    ax1 = axes[0]
    
    for smile in smiles:
        valid_idx = ~np.isnan(smile['implied_vols'])
        ax1.plot(
            smile['strikes'][valid_idx],
            smile['implied_vols'][valid_idx],
            'o-', linewidth=2, markersize=4,
            label=smile['label'], alpha=0.8
        )
    
    # Black-Scholes flat vol
    if black_scholes_vol is not None:
        bs_vol_pct = black_scholes_vol * 100
        ax1.axhline(y=bs_vol_pct, color='black', linestyle='--',
                   linewidth=1.5, alpha=0.7,
                   label=f'Black-Scholes (σ={bs_vol_pct:.1f}%)')
    
    # Mark ATM strike
    if smiles:
        atm_strike = smiles[0]['strikes'][len(smiles[0]['strikes'])//2]
        ax1.axvline(x=atm_strike, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Strike Price ($)', fontsize=12)
    ax1.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax1.set_title(f'{title}\n(IV vs Strike)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ----------------------------------------------------------------
    # Right panel: IV vs Log-Moneyness
    # ----------------------------------------------------------------
    ax2 = axes[1]
    
    for smile in smiles:
        valid_idx = ~np.isnan(smile['implied_vols'])
        ax2.plot(
            smile['moneyness'][valid_idx],
            smile['implied_vols'][valid_idx],
            'o-', linewidth=2, markersize=4,
            label=smile['label'], alpha=0.8
        )
    
    if black_scholes_vol is not None:
        bs_vol_pct = black_scholes_vol * 100
        ax2.axhline(y=bs_vol_pct, color='black', linestyle='--',
                   linewidth=1.5, alpha=0.7,
                   label=f'Black-Scholes (σ={bs_vol_pct:.1f}%)')
    
    # Mark ATM
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Log-Moneyness log(K/S)', fontsize=12)
    ax2.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax2.set_title('IV vs Log-Moneyness', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_parameter_study(
    smiles: List[dict],
    parameter_name: str,
    title: str = "Parameter Sensitivity Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Visualize how smile changes with parameter values.
    
    Parameters
    ----------
    smiles : list of dict
        Smiles for different parameter values
    parameter_name : str
        Name of parameter varied
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure dimensions
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(smiles)))
    
    for i, smile in enumerate(smiles):
        valid_idx = ~np.isnan(smile['implied_vols'])
        ax.plot(
            smile['moneyness'][valid_idx],
            smile['implied_vols'][valid_idx],
            '-', linewidth=2.5, color=colors[i],
            label=smile['label'], alpha=0.9
        )
    
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='ATM')
    
    ax.set_xlabel('Log-Moneyness log(K/S)', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax.set_title(f'{title}\nParameter: {parameter_name}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_basic_smile():
    """Demonstrate basic volatility smile generation."""
    print("\n" + "="*60)
    print("VOLATILITY SMILE DEMO")
    print("="*60)
    
    config = SmileConfig(
        S0=100, r=0.03, T=1.0,
        kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04
    )
    
    generator = VolatilitySmileGenerator(config)
    smile = generator.generate_single_smile(
        label="Heston Base Case"
    )
    
    # Black-Scholes equivalent vol (sqrt of current variance)
    bs_vol = np.sqrt(config.v0)
    
    plot_volatility_smile(
        smiles=[smile],
        title="Heston Implied Volatility Smile",
        black_scholes_vol=bs_vol,
        save_path="../results/figures/volatility_smile_basic.png"
    )
    
    return smile


def demo_parameter_study():
    """
    Study how smile changes with key parameters.
    
    Demonstrates:
    1. ρ effect: negative correlation creates skew
    2. ξ effect: higher vol-of-vol creates more convex smile
    3. v0 effect: higher spot vol raises overall level
    """
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY STUDY")
    print("="*60)
    
    config = SmileConfig(
        S0=100, r=0.03, T=1.0,
        kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04
    )
    
    generator = VolatilitySmileGenerator(config)
    
    # Study 1: Vary ρ (correlation)
    print("\n1. Correlation (ρ) effect on skew:")
    rho_smiles = generator.generate_parameter_study(
        'rho', [-0.9, -0.7, -0.5, -0.3, 0.0]
    )
    plot_parameter_study(
        rho_smiles, 'rho',
        title='Effect of Correlation (ρ) on Volatility Smile',
        save_path="../results/figures/smile_rho_study.png"
    )
    
    # Study 2: Vary ξ (vol-of-vol)
    print("\n2. Vol-of-Vol (ξ) effect on smile convexity:")
    xi_smiles = generator.generate_parameter_study(
        'xi', [0.1, 0.3, 0.5, 0.7, 1.0]
    )
    plot_parameter_study(
        xi_smiles, 'xi',
        title='Effect of Vol-of-Vol (ξ) on Volatility Smile',
        save_path="../results/figures/smile_xi_study.png"
    )
    
    # Study 3: Vary v0 (spot variance)
    print("\n3. Spot Variance (v0) effect on smile level:")
    v0_smiles = generator.generate_parameter_study(
        'v0', [0.01, 0.04, 0.09, 0.16]
    )
    plot_parameter_study(
        v0_smiles, 'v0',
        title='Effect of Spot Variance (v0) on Volatility Smile',
        save_path="../results/figures/smile_v0_study.png"
    )


if __name__ == "__main__":
    demo_basic_smile()
    demo_parameter_study()