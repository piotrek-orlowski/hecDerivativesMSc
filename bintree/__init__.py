"""
Binomial Tree Module for Derivatives Pricing

This package provides tools for pricing options using forward binomial trees.
"""

from .forward_binomial_tree import (
    ForwardBinomialTree,
    TreeNode,
    call_payoff,
    put_payoff,
    digital_call_payoff,
    digital_put_payoff,
    FastAmericanPricer,
    implied_volatility_american,
)

__all__ = [
    'ForwardBinomialTree',
    'TreeNode',
    'call_payoff',
    'put_payoff',
    'digital_call_payoff',
    'digital_put_payoff',
    'FastAmericanPricer',
    'implied_volatility_american',
]
