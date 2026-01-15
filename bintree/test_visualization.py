"""
Test visualization improvements
"""

from forward_binomial_tree import ForwardBinomialTree, put_payoff
import matplotlib.pyplot as plt


# Create a small tree for clear visualization
S0 = 100.0
K = 100.0
T = 0.5
N = 5
sigma = 0.3
rf = 0.05
div_yld = 0.02

# Price American Put
tree = ForwardBinomialTree(
    S0=S0,
    T_maturity=T,
    N_periods=N,
    sigma=sigma,
    rf=rf,
    div_yld=div_yld
)
american_value = tree.price_american(put_payoff(K))

# Price European Put for comparison
tree_euro = ForwardBinomialTree(
    S0=S0,
    T_maturity=T,
    N_periods=N,
    sigma=sigma,
    rf=rf,
    div_yld=div_yld
)
european_value = tree_euro.price_european(put_payoff(K))

print(f"American Put: ${american_value:.4f}")
print(f"European Put: ${european_value:.4f}")
print(f"Early Exercise Premium: ${american_value - european_value:.4f}")

# Visualize with all new features
tree.plot_tree(
    show_stock_prices=True,
    show_derivative_values=True,
    show_exercise_boundary=True,
    show_continuation_values=True,
    payoff_description=f"max({K} - S, 0)",
    option_type="American",
    european_value=european_value,
    max_periods_display=N
)
plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to 'test_visualization.png'")
print("\nVisualization features:")
print("✓ High stock prices at TOP, low at BOTTOM")
print("✓ Payoff formula displayed in info box")
print("✓ Initial replicating portfolio (Δ and B) displayed")
print("✓ Early exercise premium displayed")
print("✓ Continuation values (C=) shown at exercise nodes (red)")
