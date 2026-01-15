"""
Quick Start Guide for Forward Binomial Tree

This script shows the most common use cases in a concise format.
"""

from forward_binomial_tree import ForwardBinomialTree, call_payoff, put_payoff
import matplotlib.pyplot as plt


def main():
    # Setup parameters
    S0 = 100.0          # Initial stock price
    K = 100.0           # Strike price
    T = 1.0             # Time to maturity (years)
    N = 50              # Number of periods
    sigma = 0.3         # Volatility
    rf = 0.05           # Risk-free rate
    div_yld = 0.02      # Dividend yield

    # Create the tree
    tree = ForwardBinomialTree(
        S0=S0,
        T_maturity=T,
        N_periods=N,
        sigma=sigma,
        rf=rf,
        div_yld=div_yld
    )

    # Price options
    print("Option Prices:")
    print("-" * 50)

    # European Call
    european_call = tree.price_european(call_payoff(K))
    print(f"European Call: ${european_call:.4f}")

    # American Put (need new tree instance for different option type)
    tree_put = ForwardBinomialTree(S0=S0, T_maturity=T, N_periods=N,
                                   sigma=sigma, rf=rf, div_yld=div_yld)
    american_put = tree_put.price_american(put_payoff(K))
    print(f"American Put:  ${american_put:.4f}")

    # Show initial replicating portfolio for the call
    print(f"\nReplicating Portfolio (Call at t=0):")
    print(f"  Delta: {tree.tree[0][0].delta:.6f} shares")
    print(f"  Bond:  ${tree.tree[0][0].bond:.4f}")

    # Display tree parameters with formulas (optional - comment out for cleaner output)
    # tree.print_parameters()

    # Display key parameters
    print(f"\nKey Tree Parameters:")
    print(f"  Up factor (u):   {tree.up_factor:.6f}")
    print(f"  Down factor (d): {tree.down_factor:.6f}")
    print(f"  p*:              {tree.pstar:.6f}")

    # Visualize the put option tree with exercise boundary
    print("\nGenerating visualization...")

    # Price European put for comparison
    tree_euro_put = ForwardBinomialTree(S0=S0, T_maturity=T, N_periods=N,
                                        sigma=sigma, rf=rf, div_yld=div_yld)
    european_put = tree_euro_put.price_european(put_payoff(K))

    tree_put.plot_tree(
        show_stock_prices=False,
        show_derivative_values=True,
        show_exercise_boundary=True,
        show_continuation_values=True,
        payoff_description=f"max({K} - S, 0)",
        option_type="American",
        european_value=european_put,
        max_periods_display=15
    )
    plt.savefig('quickstart_tree.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'quickstart_tree.png'")

    # Custom payoff example (Bull Call Spread: long K1 call, short K2 call)
    print(f"\nCustom Payoff - Bull Call Spread:")
    K1, K2 = 95.0, 105.0

    def bull_call_spread(S):
        """Bull call spread payoff: max(S-K1, 0) - max(S-K2, 0)"""
        return max(S - K1, 0) - max(S - K2, 0)

    tree_spread = ForwardBinomialTree(S0=S0, T_maturity=T, N_periods=N,
                                      sigma=sigma, rf=rf, div_yld=div_yld)
    spread_value = tree_spread.price_european(bull_call_spread)
    print(f"  Long ${K1} Call, Short ${K2} Call")
    print(f"  Spread Value: ${spread_value:.4f}")


if __name__ == "__main__":
    main()
