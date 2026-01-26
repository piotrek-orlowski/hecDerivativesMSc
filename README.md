# HEC Derivatives - Python Scripts

Python scripts and notebooks for derivatives and financial analysis courses.

## Project Structure

```
.
├── bintree/                    # Binomial tree pricing module
│   ├── forward_binomial_tree.py    # Main module
│   ├── example_forward_binomial_tree.py
│   ├── quickstart_binomial_tree.py
│   ├── BINOMIAL_TREE_README.md
│   └── VISUALIZATION_GUIDE.md
├── theme_01/                   # Theme 1: Forwards and Probability
│   ├── forward_pricing_problems.ipynb
│   ├── forward_pricing_problems_fr.ipynb
│   ├── probability_problem_generator.ipynb
│   ├── probability_problem_generator_fr.ipynb
│   └── ...
├── theme_02/                   # Theme 2: Option Pricing with Binomial Trees
│   ├── market-option-prices-binomial-model.ipynb
│   └── binomial_tree_practice_app.py
├── theme_03/                   # Theme 3: Lognormal Price Model
│   ├── normal-distributions.ipynb
│   ├── normal-distributions-fr.ipynb
│   ├── convexity-adjustment.ipynb
│   └── convexity-adjustment-fr.ipynb
├── theme_05/                   # Theme 5: Black-Scholes and Option Greeks
│   └── option_greeks_app.py
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Locked dependency versions
├── README.md                   # This file (English)
└── README_fr.md                # French version
```

## Setup with uv

`uv` is a fast Python package manager that handles dependencies automatically.

### Install uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew (macOS)
brew install uv

# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip (all platforms)
pip install uv
```

## Running Python Scripts

Use `uv run` to execute any Python script. Dependencies are installed automatically.

```bash
# Run a script from the bintree folder
uv run bintree/example_forward_binomial_tree.py

# Run a script from theme_01
uv run theme_01/02-scrape-sp500-returns.py
```

## Running Jupyter Notebooks

### Option 1: Start Jupyter Lab (Recommended)

```bash
uv run jupyter lab
```

This opens Jupyter Lab in your browser. Navigate to the desired notebook file.

### Option 2: Start Classic Jupyter Notebook

```bash
uv run jupyter notebook
```

### Option 3: Open a Specific Notebook

```bash
# Open a specific notebook directly
uv run jupyter notebook theme_02/market-option-prices-binomial-model.ipynb
```

## Running Python Interactively

```bash
# Start an interactive Python session with all dependencies
uv run python

# Or start IPython for a better interactive experience
uv run ipython
```

## Managing Dependencies

```bash
# Sync/install all dependencies
uv sync

# Add a new package
uv add package-name

# Update all packages
uv sync --upgrade
```

## Course Content

### Theme 1: Forwards and Probability (`theme_01/`)

- **forward_pricing_problems.ipynb** - Forward pricing for stocks, commodities, and currencies
- **probability_problem_generator.ipynb** - Probability problems with expectations and variance
- French versions available (`_fr.ipynb`)

### Theme 2: Option Pricing (`theme_02/`)

- **market-option-prices-binomial-model.ipynb** - Download real market option data, price with binomial trees, calculate implied volatility, analyze the volatility smile
- **binomial_tree_practice_app.py** - Interactive Bokeh app for practicing binomial tree calculations

To run the practice app:
```bash
uv run bokeh serve --show theme_02/binomial_tree_practice_app.py
```

### Theme 3: Lognormal Price Model (`theme_03/`)

- **normal-distributions.ipynb** - Introduction to normal distributions: simulating random samples, computing sample statistics (mean, variance), using scipy.stats for PDF/CDF calculations, comparing analytical vs simulation-based probabilities
- **convexity-adjustment.ipynb** - Understanding the convexity adjustment in the lognormal price model: Jensen's inequality, why we need the $-\frac{1}{2}\sigma^2$ term, stock price simulation with and without the adjustment, visualizing the lognormal distribution (mode, median, mean)
- French versions available (`_fr.ipynb`)

### Theme 5: Black-Scholes and Option Greeks (`theme_05/`)

- **option_greeks_app.py** - Interactive Bokeh app for visualizing option Greeks (Delta, Gamma, Vega) as functions of strike or maturity for European options using Black-Scholes formulas

To run the Greeks visualization app:
```bash
uv run bokeh serve --show theme_05/option_greeks_app.py
```

### Binomial Tree Module (`bintree/`)

A complete implementation of forward binomial trees for:
- European, American, and Bermudan option pricing
- Replicating portfolio calculation
- Tree visualization
- Risk-neutral probability analysis

See `bintree/BINOMIAL_TREE_README.md` for detailed documentation.

## Quick Start Example

```bash
# 1. Clone or download this repository
# 2. Navigate to the folder
cd python-scripts

# 3. Run an example
uv run bintree/quickstart_binomial_tree.py

# 4. Start Jupyter to explore notebooks
uv run jupyter lab
```

## Requirements

- Python 3.9 or higher
- uv (installed as shown above)

All other dependencies are managed automatically by uv.
