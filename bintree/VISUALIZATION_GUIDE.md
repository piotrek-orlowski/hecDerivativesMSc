# Forward Binomial Tree - Visualization and Analysis Guide

## Overview
The Forward Binomial Tree module provides comprehensive tools for analyzing and visualizing binomial trees, including parameter inspection and tree visualization with automatic annotation of key information.

## Key Methods

### 1. `print_parameters()` - Display All Tree Parameters
This method prints all input and calculated parameters with their formulas, providing a complete reference for understanding the tree construction.

**Usage:**
```python
tree = ForwardBinomialTree(S0=100, T_maturity=1.0, N_periods=50,
                          sigma=0.3, rf=0.05, div_yld=0.02)
tree.print_parameters()
```

**Output includes:**
- **Input Parameters**: S0, T_maturity, N_periods, σ, rf, div_yld
- **Calculated Parameters** with formulas:
  - `h_period = T_maturity / N_periods`
  - `fwd_ratio = exp((rf - div_yld) × h)`
  - `u = fwd_ratio × exp(σ × √h)` (up factor)
  - `d = fwd_ratio × exp(-σ × √h)` (down factor)
  - `p* = (fwd_ratio - d) / (u - d)` (risk-neutral probability)
  - `discount_factor = exp(-rf × h)`
- **Stock Price Dynamics**: Example of one-period price movements
- **Replicating Portfolio Formulas**: Delta and Bond calculations
- **Risk-Neutral Valuation**: Backward induction formula

### 2. `plot_tree()` - Visualize the Binomial Tree
See details below in the "Tree Visualization Features" section.

## Tree Visualization Features

### 1. Stock Price Orientation
- **High prices at TOP** of the tree
- **Low prices at BOTTOM** of the tree
- More intuitive visualization matching typical financial charts

### 2. Payoff Formula Display
- Automatically shown in info box at bottom-left
- Example: `Payoff: max(100 - S, 0)` for a put option
- Customizable via `payoff_description` parameter

### 3. Initial Replicating Portfolio
- Displayed in info box for all option types
- Shows:
  - **Δ (Delta)**: Number of shares of underlying
  - **B (Bond)**: Amount invested in risk-free bond
  - **V (Value)**: Total derivative value

### 4. Early Exercise Premium
- For American and Bermudan options only
- Automatically calculated as: `American - European`
- Displayed in info box
- Shows value of early exercise flexibility

### 5. Continuation Values at Exercise Nodes
- Shown on exercise nodes (red nodes) with American/Bermudan options
- Format: `C=XX.XX` below the option value
- Helps visualize the exercise decision:
  - **Exercise if**: Payoff > Continuation Value
  - **Hold if**: Continuation Value > Payoff

## Usage Examples

### European Option
```python
tree = ForwardBinomialTree(S0=100, T_maturity=1.0, N_periods=50,
                          sigma=0.3, rf=0.05, div_yld=0.02)
call_value = tree.price_european(call_payoff(100))

tree.plot_tree(
    show_stock_prices=True,
    show_derivative_values=True,
    payoff_description="max(S - 100, 0)",
    option_type="European"
)
plt.savefig('european_call.png')
```

### American Option
```python
# Price both European and American
tree_euro = ForwardBinomialTree(S0=100, T_maturity=1.0, N_periods=50,
                                sigma=0.3, rf=0.05, div_yld=0.02)
euro_value = tree_euro.price_european(put_payoff(100))

tree_american = ForwardBinomialTree(S0=100, T_maturity=1.0, N_periods=50,
                                   sigma=0.3, rf=0.05, div_yld=0.02)
american_value = tree_american.price_american(put_payoff(100))

tree_american.plot_tree(
    show_stock_prices=True,
    show_derivative_values=True,
    show_exercise_boundary=True,
    show_continuation_values=True,
    payoff_description="max(100 - S, 0)",
    option_type="American",
    european_value=euro_value,  # For early exercise premium calculation
    max_periods_display=15
)
plt.savefig('american_put.png')
```

### Bermudan Option
```python
tree_euro = ForwardBinomialTree(S0=100, T_maturity=1.0, N_periods=12,
                                sigma=0.25, rf=0.04, div_yld=0.01)
euro_value = tree_euro.price_european(put_payoff(100))

tree_bermudan = ForwardBinomialTree(S0=100, T_maturity=1.0, N_periods=12,
                                   sigma=0.25, rf=0.04, div_yld=0.01)
bermudan_value = tree_bermudan.price_bermudan(put_payoff(100),
                                              exercise_periods=[3, 6, 9, 12])

tree_bermudan.plot_tree(
    show_stock_prices=False,  # Cleaner for larger trees
    show_derivative_values=True,
    show_exercise_boundary=True,
    show_continuation_values=True,
    payoff_description="max(100 - S, 0)",
    option_type="Bermudan",
    european_value=euro_value,
    max_periods_display=12
)
plt.savefig('bermudan_put.png')
```

## Parameters Reference

### plot_tree() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_stock_prices` | bool | True | Display stock price at each node |
| `show_derivative_values` | bool | True | Display derivative value at each node |
| `show_exercise_boundary` | bool | False | Highlight exercise nodes in red |
| `show_continuation_values` | bool | True | Show continuation values at exercise nodes |
| `payoff_description` | str | None | Description of payoff (e.g., "max(K - S, 0)") |
| `option_type` | str | None | "European", "American", or "Bermudan" |
| `european_value` | float | None | European value for premium calculation |
| `max_periods_display` | int | min(N, 15) | Max periods to show (for readability) |
| `figsize` | tuple | (14, 10) | Figure size in inches |

## Visual Elements

### Node Colors
- **Light Blue**: Hold nodes (continuation value > exercise value)
- **Light Coral** (Red border): Exercise nodes (exercise value ≥ continuation value)

### Info Box (Bottom Left)
Contains:
1. Payoff formula (if provided)
2. Initial replicating portfolio (Δ, B, V)
3. Early exercise premium (for American/Bermudan)

### Legend (Top Right)
- Shows node color meanings
- Explains continuation value notation

## Tips

1. **For large trees** (N > 20): Set `show_stock_prices=False` for cleaner visualization
2. **Max periods**: Use `max_periods_display` to limit display to first N periods
3. **High DPI**: Use `plt.savefig(..., dpi=200)` for publication-quality images
4. **Custom payoffs**: Always provide `payoff_description` for clarity

## Example Output Interpretation

Looking at a node:
```
S=85.23
V=14.76
C=14.50
```

This means:
- Stock price at this node: $85.23
- Option value: $14.76
- Continuation value: $14.50
- Since node is red: Exercise value ($14.76) > Continuation ($14.50), so optimal to exercise

## See Also
- `example_forward_binomial_tree.py`: Comprehensive examples
- `quickstart_binomial_tree.py`: Quick reference
- `test_visualization.py`: Visualization feature demonstration
