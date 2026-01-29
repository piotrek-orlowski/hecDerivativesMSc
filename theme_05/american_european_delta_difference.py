"""
American vs European Option Delta Difference

This script computes and plots the difference between American and European
option deltas for at-the-money calls and puts across different volatility levels.

American option deltas are computed using binomial trees with sufficient steps
for convergence. European option deltas are computed using Black-Scholes formulas.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
from pathlib import Path

# Add bintree module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "bintree"))
from forward_binomial_tree import ForwardBinomialTree, call_payoff, put_payoff


# =============================================================================
# Black-Scholes Functions
# =============================================================================

def bs_d1(S, K, r, delta, sigma, T):
    """Compute d1 for Black-Scholes formula."""
    return (np.log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_call_delta(S, K, r, delta, sigma, T):
    """Black-Scholes delta for a European call option."""
    d1 = bs_d1(S, K, r, delta, sigma, T)
    return np.exp(-delta * T) * norm.cdf(d1)


def bs_put_delta(S, K, r, delta, sigma, T):
    """Black-Scholes delta for a European put option."""
    d1 = bs_d1(S, K, r, delta, sigma, T)
    return -np.exp(-delta * T) * norm.cdf(-d1)


def bs_call_price(S, K, r, delta, sigma, T):
    """Black-Scholes price for a European call option."""
    d1 = bs_d1(S, K, r, delta, sigma, T)
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-delta * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, r, delta, sigma, T):
    """Black-Scholes price for a European put option."""
    d1 = bs_d1(S, K, r, delta, sigma, T)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-delta * T) * norm.cdf(-d1)


# =============================================================================
# American Option Delta via Binomial Tree
# =============================================================================

def american_option_delta(S0, K, r, div_yld, sigma, T, N_periods, option_type='call'):
    """
    Compute American option delta using binomial tree.

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    div_yld : float
        Dividend yield
    sigma : float
        Volatility
    T : float
        Time to maturity
    N_periods : int
        Number of periods in binomial tree
    option_type : str
        'call' or 'put'

    Returns
    -------
    tuple
        (price, delta)
    """
    tree = ForwardBinomialTree(
        S0=S0,
        T_maturity=T,
        N_periods=N_periods,
        sigma=sigma,
        rf=r,
        div_yld=div_yld
    )

    if option_type == 'call':
        payoff = call_payoff(K)
    else:
        payoff = put_payoff(K)

    price = tree.price_american(payoff)
    delta = tree.tree[0][0].delta

    return price, delta


# =============================================================================
# Main Script
# =============================================================================

def main():
    # Model parameters
    K = 100.0           # Strike price (fixed)
    r = 0.05            # Risk-free rate
    div_yld = 0.05      # Dividend yield (positive to make early exercise relevant)
    T = 0.25            # 3-month maturity
    N_periods = 500     # Number of periods (sufficient for convergence)

    # Volatility levels
    volatilities = [0.15, 0.30, 0.50]
    vol_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green

    # Range of spot prices (moneyness from 0.7 to 1.3)
    S_values = np.linspace(70, 130, 61)

    # Store results
    results = {sigma: {'S': S_values,
                       'call_delta_diff': [],
                       'put_delta_diff': []}
               for sigma in volatilities}

    print("Computing American vs European delta differences...")
    print(f"Parameters: K={K}, r={r:.1%}, δ={div_yld:.1%}, T={T} years, N={N_periods} periods")
    print("-" * 70)

    for sigma in volatilities:
        print(f"\nVolatility σ = {sigma:.0%}:")

        call_delta_diffs = []
        put_delta_diffs = []

        for i, S in enumerate(S_values):
            # European deltas (Black-Scholes)
            euro_call_delta = bs_call_delta(S, K, r, div_yld, sigma, T)
            euro_put_delta = bs_put_delta(S, K, r, div_yld, sigma, T)

            # American deltas (Binomial Tree)
            _, amer_call_delta = american_option_delta(S, K, r, div_yld, sigma, T, N_periods, 'call')
            _, amer_put_delta = american_option_delta(S, K, r, div_yld, sigma, T, N_periods, 'put')

            # Delta differences
            call_diff = amer_call_delta - euro_call_delta
            put_diff = amer_put_delta - euro_put_delta

            call_delta_diffs.append(call_diff)
            put_delta_diffs.append(put_diff)

            # Progress indicator
            if (i + 1) % 20 == 0 or i == len(S_values) - 1:
                print(f"  Processed {i + 1}/{len(S_values)} spot prices")

        results[sigma]['call_delta_diff'] = np.array(call_delta_diffs)
        results[sigma]['put_delta_diff'] = np.array(put_delta_diffs)

    # ==========================================================================
    # Create the plot
    # ==========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Call Delta Difference
    ax1 = axes[0]
    for sigma, color in zip(volatilities, vol_colors):
        ax1.plot(S_values / K, results[sigma]['call_delta_diff'],
                 color=color, linewidth=2, label=f'σ = {sigma:.0%}')

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.axvline(x=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax1.set_xlabel('Moneyness (S/K)', fontsize=12)
    ax1.set_ylabel('Delta Difference (American − European)', fontsize=12)
    ax1.set_title('Call Options', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.7, 1.3)

    # Plot Put Delta Difference
    ax2 = axes[1]
    for sigma, color in zip(volatilities, vol_colors):
        ax2.plot(S_values / K, results[sigma]['put_delta_diff'],
                 color=color, linewidth=2, label=f'σ = {sigma:.0%}')

    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.axvline(x=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax2.set_xlabel('Moneyness (S/K)', fontsize=12)
    ax2.set_ylabel('Delta Difference (American − European)', fontsize=12)
    ax2.set_title('Put Options', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.7, 1.3)

    # Main title
    fig.suptitle(
        f'American vs European Option Delta Difference\n'
        f'K = {K}, r = {r:.1%}, δ = {div_yld:.1%}, T = {T*12:.0f} months, '
        f'Binomial tree: {N_periods} steps',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'american_european_delta_diff.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n" + "=" * 70)
    print(f"Figure saved to: {output_path}")

    # Also display
    plt.show()

    # Print summary statistics at ATM
    print("\n" + "=" * 70)
    print("Delta differences at the money (S = K = 100):")
    print("-" * 70)
    atm_idx = len(S_values) // 2  # Middle index (S = 100)
    for sigma in volatilities:
        call_diff = results[sigma]['call_delta_diff'][atm_idx]
        put_diff = results[sigma]['put_delta_diff'][atm_idx]
        print(f"σ = {sigma:.0%}:  Call Δ diff = {call_diff:+.6f},  Put Δ diff = {put_diff:+.6f}")


if __name__ == "__main__":
    main()
