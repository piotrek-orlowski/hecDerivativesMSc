"""
Test the print_parameters() method
"""

from forward_binomial_tree import ForwardBinomialTree

# Create a tree with typical parameters
tree = ForwardBinomialTree(
    S0=100.0,
    T_maturity=1.0,
    N_periods=50,
    sigma=0.3,
    rf=0.05,
    div_yld=0.02
)

# Print all parameters with formulas
tree.print_parameters()
