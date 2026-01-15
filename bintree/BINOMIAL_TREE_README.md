# Forward Binomial Tree - Complete Guide

A comprehensive Python implementation of forward binomial trees for pricing European, American, and Bermudan options.

## Features

✓ **Forward tree formulation** with proper forward ratio calculation
✓ **Multiple option types**: European, American, and Bermudan
✓ **Flexible payoff functions** for custom derivatives
✓ **Complete replicating portfolio** calculations (delta and bond)
✓ **Node probability** calculations
✓ **Detailed parameter inspection** with formulas
✓ **Rich visualization** with exercise boundaries
✓ **Fast performance**: ~850K nodes/sec for pricing

## Quick Start

```python
from forward_binomial_tree import ForwardBinomialTree, call_payoff, put_payoff

# Create tree
tree = ForwardBinomialTree(
    S0=100.0,           # Initial stock price
    T_maturity=1.0,     # Time to maturity (years)
    N_periods=50,       # Number of periods
    sigma=0.3,          # Volatility
    rf=0.05,            # Risk-free rate
    div_yld=0.02        # Dividend yield
)

# Price European call
call_value = tree.price_european(call_payoff(100))
print(f"European Call: ${call_value:.4f}")

# Price American put
tree_put = ForwardBinomialTree(S0=100, T_maturity=1.0, N_periods=50,
                               sigma=0.3, rf=0.05, div_yld=0.02)
put_value = tree_put.price_american(put_payoff(100))
print(f"American Put: ${put_value:.4f}")
```

## Core Methods

### Parameter Inspection

```python
# Print all parameters with formulas
tree.print_parameters()
```

This displays:
- Input parameters (S0, T, N, σ, rf, div_yld)
- Calculated parameters with formulas (h, fwd_ratio, u, d, p*, discount_factor)
- Stock price dynamics examples
- Replicating portfolio formulas
- Risk-neutral valuation formula

### Pricing Methods

```python
# European option
value = tree.price_european(payoff_func)

# American option (early exercise after period 0)
value = tree.price_american(payoff_func)

# Bermudan option (exercise at specified periods)
value = tree.price_bermudan(payoff_func, exercise_periods=[3, 6, 9, 12])
```

### Analysis Methods

```python
# Get exercise boundary for American/Bermudan
boundary = tree.get_exercise_boundary()

# Print tree values at specific periods
tree.print_tree(
    periods=[0, 5, 10],
    show_exercise=True,
    show_probability=True
)

# Access individual nodes
node = tree.tree[period][index]
print(f"Stock: ${node.stock_price:.2f}")
print(f"Value: ${node.derivative_value:.2f}")
print(f"Delta: {node.delta:.6f}")
print(f"Bond: ${node.bond:.2f}")
```

### Visualization

```python
import matplotlib.pyplot as plt

# Visualize with all features
tree.plot_tree(
    show_stock_prices=True,
    show_derivative_values=True,
    show_exercise_boundary=True,
    show_continuation_values=True,
    payoff_description="max(100 - S, 0)",
    option_type="American",
    european_value=euro_value,
    max_periods_display=15
)
plt.savefig('tree.png', dpi=150)
```

## Built-in Payoff Functions

```python
from forward_binomial_tree import (
    call_payoff,
    put_payoff,
    digital_call_payoff,
    digital_put_payoff
)

# Standard options
call = call_payoff(strike=100)
put = put_payoff(strike=100)

# Digital/binary options
digital_call = digital_call_payoff(strike=100, payout=10)
digital_put = digital_put_payoff(strike=100, payout=10)
```

## Custom Payoffs

```python
# Bull call spread
def bull_call_spread(S):
    return max(S - 95, 0) - max(S - 105, 0)

tree = ForwardBinomialTree(S0=100, T_maturity=1.0, N_periods=50,
                          sigma=0.3, rf=0.05, div_yld=0.02)
value = tree.price_european(bull_call_spread)

# Butterfly spread
def butterfly(S):
    return max(S - 90, 0) - 2*max(S - 100, 0) + max(S - 110, 0)

# Custom barrier (check at terminal nodes)
def up_and_out_call(S, barrier=120, strike=100):
    if S >= barrier:
        return 0.0
    return max(S - strike, 0)
```

## Tree Structure

The tree stores nodes in a dictionary: `tree[period][index]`
- `period`: Time step (0 to N_periods)
- `index`: Price level (0 = highest price, period = lowest price)

Each node contains:
- `stock_price`: Stock price at this node
- `derivative_value`: Option value
- `delta`: Shares in replicating portfolio
- `bond`: Bond amount in replicating portfolio
- `probability`: Probability of reaching this node
- `exercise_value`: Immediate exercise value (American/Bermudan)
- `continuation_value`: Value from holding (American/Bermudan)
- `exercised`: Whether optimal to exercise (American/Bermudan)

## Key Formulas

### Tree Parameters
```
h = T_maturity / N_periods
fwd_ratio = exp((rf - div_yld) × h)
u = fwd_ratio × exp(σ × √h)
d = fwd_ratio × exp(-σ × √h)
p* = (fwd_ratio - d) / (u - d)
```

### Stock Price Dynamics
```
S(period, index) = S0 × u^(period - index) × d^index
```

### Replicating Portfolio
```
Δ = exp(-div_yld × h) × (V_u - V_d) / (S_u - S_d)
B = exp(-rf × h) × (u × V_d - d × V_u) / (u - d)
V = Δ × S + B
```

### Risk-Neutral Valuation
```
V = exp(-rf × h) × [p* × V_u + (1 - p*) × V_d]
```

For American/Bermudan:
```
V = max(Exercise Value, Continuation Value)
```

## Performance

Optimized with lazy probability calculation:

**500 periods (125,751 nodes):**
- Tree construction: 0.09 seconds
- American option pricing: 0.16 seconds (804,810 nodes/sec)
- Total: ~0.25 seconds

**Key optimizations:**
- **Lazy probability calculation:** Probabilities only computed when explicitly requested
- **Factorial lookup table:** Precomputed factorials 0-5000 (~16 MB) for instant binomial coefficients
- **Backward induction:** No probability calculations needed
- **Perfect for implied volatility:** American option pricing is extremely fast

## Example Files

- `quickstart_binomial_tree.py` - Simple examples for common use cases
- `example_forward_binomial_tree.py` - Comprehensive examples (6 scenarios)
- `test_print_parameters.py` - Parameter inspection demo
- `test_visualization.py` - Visualization features demo
- `test_performance.py` - Performance benchmarks (1000 periods)

## Documentation

- `VISUALIZATION_GUIDE.md` - Complete visualization reference
- `BINOMIAL_TREE_README.md` - This file

## Requirements

```
numpy>=1.24.0
matplotlib>=3.7.0
```

## Tips

1. **For large trees** (N > 50): Set `show_stock_prices=False` in visualization
2. **Reuse trees**: If pricing multiple derivatives with same parameters, reuse the tree
3. **Convergence**: American options typically converge with N=100-200 periods
4. **Custom payoffs**: Can be any callable that takes stock price and returns payoff
5. **Parameter inspection**: Use `print_parameters()` to verify formulas

## License

Free to use for educational and research purposes.
