"""
Forward Binomial Tree for Derivatives Pricing

This module implements a forward binomial tree model for pricing European,
American, and Bermudan options, along with tools for analyzing the replicating
portfolio and visualizing the tree structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass, field
from math import factorial

# Precompute factorial lookup table for fast binomial coefficient calculation
# This table speeds up probability calculations when needed
_FACTORIAL_CACHE_SIZE = 5001  # Support up to 5000 periods
_FACTORIAL_LOOKUP = [factorial(n) for n in range(_FACTORIAL_CACHE_SIZE)]


@dataclass
class TreeNode:
    """Represents a single node in the binomial tree"""
    period: int
    index: int  # 0 is lowest price, period is highest price at this time
    stock_price: float
    derivative_value: float = 0.0
    delta: float = 0.0
    bond: float = 0.0
    probability: float = 0.0
    exercise_value: Optional[float] = None
    continuation_value: Optional[float] = None
    exercised: bool = False


@dataclass
class ForwardBinomialTree:
    """
    Forward binomial tree for pricing derivatives.

    Parameters:
    -----------
    S0 : float
        Initial stock price
    T_maturity : float
        Time to maturity in years
    N_periods : int
        Number of periods in the tree
    sigma : float
        Volatility (annualized)
    rf : float
        Risk-free rate (continuously compounded)
    div_yld : float
        Dividend yield or lease rate (continuously compounded)
    """
    S0: float
    T_maturity: float
    N_periods: int
    sigma: float
    rf: float
    div_yld: float = 0.0

    # Calculated parameters
    h_period: float = field(init=False)
    fwd_ratio: float = field(init=False)
    up_factor: float = field(init=False)
    down_factor: float = field(init=False)
    pstar: float = field(init=False)
    discount_factor: float = field(init=False)

    # Tree structure: tree[period][index] -> TreeNode
    tree: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Calculate derived parameters"""
        self.h_period = self.T_maturity / self.N_periods
        self.fwd_ratio = np.exp((self.rf - self.div_yld) * self.h_period)
        self.up_factor = self.fwd_ratio * np.exp(self.sigma * np.sqrt(self.h_period))
        self.down_factor = self.fwd_ratio * np.exp(-self.sigma * np.sqrt(self.h_period))
        self.pstar = (self.fwd_ratio - self.down_factor) / (self.up_factor - self.down_factor)
        self.discount_factor = np.exp(-self.rf * self.h_period)

        # Initialize tree structure
        self._build_stock_prices()

    def _build_stock_prices(self):
        """Build the stock price tree (probabilities computed lazily)"""
        for period in range(self.N_periods + 1):
            self.tree[period] = {}
            for index in range(period + 1):
                # Number of up moves: (period - index), down moves: index
                n_up = period - index
                n_down = index
                stock_price = self.S0 * (self.up_factor ** n_up) * (self.down_factor ** n_down)

                self.tree[period][index] = TreeNode(
                    period=period,
                    index=index,
                    stock_price=stock_price,
                    probability=0.0  # Computed lazily when needed
                )

    @staticmethod
    def _binomial_coefficient(n: int, k: int) -> float:
        """
        Calculate binomial coefficient C(n, k).

        Uses precomputed factorial lookup table for n < 5001.
        Falls back to direct factorial calculation for larger n.
        """
        if k < 0 or k > n:
            return 0.0
        if k == 0 or k == n:
            return 1.0

        # Use lookup table if available (much faster)
        if n < _FACTORIAL_CACHE_SIZE:
            return _FACTORIAL_LOOKUP[n] / (_FACTORIAL_LOOKUP[k] * _FACTORIAL_LOOKUP[n - k])

        # Fallback for very large trees (n >= 5001)
        return factorial(n) / (factorial(k) * factorial(n - k))

    def calculate_probabilities(self):
        """
        Calculate the probability of reaching each node.

        This is computed lazily and only needed for:
        - Risk-neutral expectation pricing (terminal nodes only)
        - Probability analysis
        - print_tree() with show_probability=True

        For backward induction pricing, probabilities are NOT needed.

        Uses precomputed factorial lookup table for trees with N ≤ 5000 periods
        (~16 MB memory, instant lookup). Falls back to factorial() for larger N.
        """
        for period in range(self.N_periods + 1):
            for index in range(period + 1):
                node = self.tree[period][index]

                # Number of up moves: (period - index), down moves: index
                n_up = period - index
                n_down = index

                # Binomial probability: C(n,k) * p^k * (1-p)^(n-k)
                prob = self._binomial_coefficient(period, n_up) * \
                       (self.pstar ** n_up) * ((1 - self.pstar) ** n_down)

                node.probability = prob

    def price_european(
        self,
        payoff_func: Callable[[float], float]
    ) -> float:
        """
        Price a European option using backward induction.

        Parameters:
        -----------
        payoff_func : Callable[[float], float]
            Function that takes stock price and returns option payoff

        Returns:
        --------
        float : Option value at t=0
        """
        # Set terminal payoffs
        for index in range(self.N_periods + 1):
            node = self.tree[self.N_periods][index]
            node.derivative_value = payoff_func(node.stock_price)

        # Backward induction
        for period in range(self.N_periods - 1, -1, -1):
            for index in range(period + 1):
                self._calculate_node_value(period, index, payoff_func,
                                          exercise_allowed=False)

        return self.tree[0][0].derivative_value

    def price_american(
        self,
        payoff_func: Callable[[float], float]
    ) -> float:
        """
        Price an American option with early exercise allowed after period 0.

        Parameters:
        -----------
        payoff_func : Callable[[float], float]
            Function that takes stock price and returns option payoff

        Returns:
        --------
        float : Option value at t=0
        """
        # Set terminal payoffs
        for index in range(self.N_periods + 1):
            node = self.tree[self.N_periods][index]
            node.derivative_value = payoff_func(node.stock_price)
            node.exercise_value = node.derivative_value
            node.continuation_value = node.derivative_value

        # Backward induction with early exercise check
        for period in range(self.N_periods - 1, -1, -1):
            for index in range(period + 1):
                # Exercise allowed after period 0
                exercise_allowed = (period > 0)
                self._calculate_node_value(period, index, payoff_func,
                                          exercise_allowed=exercise_allowed)

        return self.tree[0][0].derivative_value

    def price_bermudan(
        self,
        payoff_func: Callable[[float], float],
        exercise_periods: List[int]
    ) -> float:
        """
        Price a Bermudan option with exercise allowed only at specified periods.

        Parameters:
        -----------
        payoff_func : Callable[[float], float]
            Function that takes stock price and returns option payoff
        exercise_periods : List[int]
            List of periods where exercise is allowed (should not include 0)

        Returns:
        --------
        float : Option value at t=0
        """
        # Set terminal payoffs
        for index in range(self.N_periods + 1):
            node = self.tree[self.N_periods][index]
            node.derivative_value = payoff_func(node.stock_price)
            node.exercise_value = node.derivative_value
            node.continuation_value = node.derivative_value

        # Backward induction with conditional early exercise
        for period in range(self.N_periods - 1, -1, -1):
            for index in range(period + 1):
                # Exercise allowed only at specified periods (not at period 0)
                exercise_allowed = (period in exercise_periods and period > 0)
                self._calculate_node_value(period, index, payoff_func,
                                          exercise_allowed=exercise_allowed)

        return self.tree[0][0].derivative_value

    def _calculate_node_value(
        self,
        period: int,
        index: int,
        payoff_func: Callable[[float], float],
        exercise_allowed: bool
    ):
        """Calculate derivative value and replicating portfolio at a node"""
        node = self.tree[period][index]

        # Get values at next period
        node_up = self.tree[period + 1][index]  # Same index = up move
        node_down = self.tree[period + 1][index + 1]  # Index+1 = down move

        V_u = node_up.derivative_value
        V_d = node_down.derivative_value
        S = node.stock_price
        S_u = node_up.stock_price
        S_d = node_down.stock_price

        # Calculate replicating portfolio
        # Delta: shares of stock
        if S_u != S_d:
            node.delta = np.exp(-self.div_yld * self.h_period) * \
                        (V_u - V_d) / (S_u - S_d)
        else:
            node.delta = 0.0

        # Bond: amount in risk-free bond
        if self.up_factor != self.down_factor:
            node.bond = np.exp(-self.rf * self.h_period) * \
                       (self.up_factor * V_d - self.down_factor * V_u) / \
                       (self.up_factor - self.down_factor)
        else:
            node.bond = 0.0

        # Continuation value (risk-neutral expectation)
        continuation_value = self.discount_factor * \
                           (self.pstar * V_u + (1 - self.pstar) * V_d)

        # Exercise value
        exercise_value = payoff_func(S)

        # Store values for analysis
        node.continuation_value = continuation_value
        node.exercise_value = exercise_value

        # Determine derivative value
        if exercise_allowed and exercise_value > continuation_value:
            node.derivative_value = exercise_value
            node.exercised = True
        else:
            node.derivative_value = continuation_value
            node.exercised = False

    def print_tree(
        self,
        periods: Optional[List[int]] = None,
        show_stock: bool = True,
        show_value: bool = True,
        show_delta: bool = True,
        show_bond: bool = True,
        show_probability: bool = False,
        show_exercise: bool = False
    ):
        """
        Print tree values at specified periods.

        Parameters:
        -----------
        periods : List[int], optional
            Periods to display. If None, shows all periods.
        show_stock : bool
            Show stock prices
        show_value : bool
            Show derivative values
        show_delta : bool
            Show delta (shares of stock in replicating portfolio)
        show_bond : bool
            Show bond amount in replicating portfolio
        show_probability : bool
            Show probability of reaching each node (will compute if not already done)
        show_exercise : bool
            Show exercise information (for American/Bermudan options)
        """
        if periods is None:
            periods = list(range(self.N_periods + 1))

        # Calculate probabilities if requested
        if show_probability:
            self.calculate_probabilities()

        for period in periods:
            if period not in self.tree:
                continue

            print(f"\n{'='*80}")
            print(f"Period {period} (t = {period * self.h_period:.4f} years)")
            print(f"{'='*80}")

            for index in range(period + 1):
                node = self.tree[period][index]
                print(f"\nNode [{period}, {index}]:")

                if show_stock:
                    print(f"  Stock Price:       ${node.stock_price:,.4f}")

                if show_value:
                    print(f"  Derivative Value:  ${node.derivative_value:,.4f}")

                if show_delta:
                    print(f"  Delta (Δ):         {node.delta:,.6f}")

                if show_bond:
                    print(f"  Bond (B):          ${node.bond:,.4f}")

                if show_probability:
                    print(f"  Probability:       {node.probability:.6f}")

                if show_exercise and node.exercise_value is not None:
                    print(f"  Exercise Value:    ${node.exercise_value:,.4f}")
                    print(f"  Continuation Val:  ${node.continuation_value:,.4f}")
                    print(f"  Exercised:         {'Yes' if node.exercised else 'No'}")

    def get_exercise_boundary(self) -> List[Tuple[int, float]]:
        """
        Get the optimal exercise boundary for American/Bermudan options.

        Returns:
        --------
        List[Tuple[int, float]]
            List of (period, critical_stock_price) tuples where exercise begins
        """
        boundary = []

        for period in range(1, self.N_periods + 1):
            # Find the critical stock price where exercise switches on
            critical_prices = []
            for index in range(period + 1):
                node = self.tree[period][index]
                if node.exercised:
                    critical_prices.append(node.stock_price)

            if critical_prices:
                # For calls: exercise at high prices (take max)
                # For puts: exercise at low prices (take min)
                # We'll record both the min and max transition points
                boundary.append((period, min(critical_prices), max(critical_prices)))

        return boundary

    def plot_tree(
        self,
        show_stock_prices: bool = True,
        show_derivative_values: bool = True,
        show_exercise_boundary: bool = False,
        show_continuation_values: bool = True,
        payoff_description: Optional[str] = None,
        option_type: Optional[str] = None,
        european_value: Optional[float] = None,
        max_periods_display: Optional[int] = None,
        figsize: Tuple[int, int] = (16, 9)
    ):
        """
        Visualize the binomial tree.

        Parameters:
        -----------
        show_stock_prices : bool
            Display stock prices on nodes
        show_derivative_values : bool
            Display derivative values on nodes
        show_exercise_boundary : bool
            Highlight exercise boundary for American/Bermudan options
        show_continuation_values : bool
            Show continuation values at exercise nodes (for American/Bermudan)
        payoff_description : str, optional
            Description of the payoff function (e.g., "Call(K=100)")
        option_type : str, optional
            Type of option: 'European', 'American', or 'Bermudan'
        european_value : float, optional
            European option value for comparison (to show early exercise premium)
        max_periods_display : int, optional
            Maximum number of periods to display (for readability)
        figsize : Tuple[int, int]
            Figure size
        """
        if max_periods_display is None:
            max_periods_display = min(self.N_periods, 15)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot nodes
        for period in range(max_periods_display + 1):
            for index in range(period + 1):
                node = self.tree[period][index]

                # Position: x=period, y=-index (inverted so high prices at top)
                x = period
                y = -index + period / 2  # Inverted: index 0 (highest price) at top

                # Node color
                if show_exercise_boundary and node.exercised:
                    color = 'lightcoral'
                    edgecolor = 'red'
                    linewidth = 2
                else:
                    color = 'lightblue'
                    edgecolor = 'black'
                    linewidth = 1

                # Draw node
                circle = plt.Circle((x, y), 0.3, facecolor=color,
                                   edgecolor=edgecolor, linewidth=linewidth,
                                   zorder=2)
                ax.add_patch(circle)

                # Label
                label_parts = []
                if show_stock_prices:
                    label_parts.append(f"S={node.stock_price:.2f}")
                if show_derivative_values:
                    label_parts.append(f"V={node.derivative_value:.2f}")

                # Show continuation value at exercise nodes
                if (show_exercise_boundary and show_continuation_values and
                    node.exercised and node.continuation_value is not None):
                    label_parts.append(f"C={node.continuation_value:.2f}")

                label = '\n'.join(label_parts)
                ax.text(x, y, label, ha='center', va='center',
                       fontsize=8, zorder=3)

                # Draw edges to next period
                if period < max_periods_display:
                    # Up edge
                    x_up = period + 1
                    y_up = -index + (period + 1) / 2
                    ax.plot([x, x_up], [y, y_up], 'k-', alpha=0.3, zorder=1)

                    # Down edge
                    x_down = period + 1
                    y_down = -(index + 1) + (period + 1) / 2
                    ax.plot([x, x_down], [y, y_down], 'k-', alpha=0.3, zorder=1)

        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Underlying Price', fontsize=12)

        # Build title
        title = 'Forward Binomial Tree'
        if option_type:
            title += f' - {option_type} Option'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.grid(True, alpha=0.3)

        # Set y-axis ticks to show up/down move notation
        # Use nodes from the final displayed period
        final_period = max_periods_display
        y_positions = []
        move_labels = []
        stock_prices = []

        for index in range(final_period + 1):
            node = self.tree[final_period][index]
            y_pos = -index + final_period / 2
            y_positions.append(y_pos)
            stock_prices.append(node.stock_price)

            # Calculate number of up and down moves
            n_up = final_period - index
            n_down = index
            move_labels.append(f"{n_up}u/{n_down}d")

        # Set yticks with up/down notation and prices
        # Show every other tick if there are too many to avoid crowding
        if len(y_positions) > 10:
            step = 2
        else:
            step = 1

        ax.set_yticks(y_positions[::step])
        # Create labels showing both moves and prices
        tick_labels = [f"{move_labels[i]} (${stock_prices[i]:.2f})"
                      for i in range(0, len(move_labels), step)]
        ax.set_yticklabels(tick_labels, fontsize=8)

        # Create info text box
        info_lines = []

        # Payoff description
        if payoff_description:
            info_lines.append(f"Payoff: {payoff_description}")

        # Initial replicating portfolio
        initial_node = self.tree[0][0]
        info_lines.append(f"\nInitial Portfolio (t=0):")
        info_lines.append(f"  Δ = {initial_node.delta:.6f}")
        info_lines.append(f"  B = ${initial_node.bond:.4f}")
        info_lines.append(f"  V = ${initial_node.derivative_value:.4f}")

        # Early exercise premium
        if option_type in ['American', 'Bermudan'] and european_value is not None:
            premium = initial_node.derivative_value - european_value
            info_lines.append(f"\nEarly Exercise Premium:")
            info_lines.append(f"  {option_type} - European = ${premium:.4f}")

        info_text = '\n'.join(info_lines)

        # Position text box at bottom left
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               family='monospace')

        # Legend
        if show_exercise_boundary:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', edgecolor='black', label='Hold'),
                Patch(facecolor='lightcoral', edgecolor='red', label='Exercise')
            ]
            if show_continuation_values:
                legend_elements.append(
                    Patch(facecolor='white', edgecolor='white',
                          label='C = Continuation Value')
                )
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        plt.tight_layout()
        return fig, ax

    def print_parameters(self):
        """
        Print all tree parameters with their formulas.
        Shows both input parameters and calculated parameters.
        """
        print("\n" + "="*80)
        print("FORWARD BINOMIAL TREE PARAMETERS")
        print("="*80)

        # Input Parameters
        print("\nINPUT PARAMETERS:")
        print("-" * 80)
        print(f"  S0              Initial stock price                    = {self.S0:.6f}")
        print(f"  T_maturity      Time to maturity (years)               = {self.T_maturity:.6f}")
        print(f"  N_periods       Number of periods                      = {self.N_periods}")
        print(f"  σ (sigma)       Volatility (annualized)                = {self.sigma:.6f}")
        print(f"  rf              Risk-free rate (cont. compounded)      = {self.rf:.6f}")
        print(f"  div_yld         Dividend yield (cont. compounded)      = {self.div_yld:.6f}")

        # Calculated Parameters
        print("\nCALCULATED PARAMETERS:")
        print("-" * 80)

        print(f"\n  h_period        Period length (years)")
        print(f"                  Formula: h = T_maturity / N_periods")
        print(f"                         = {self.T_maturity} / {self.N_periods}")
        print(f"                         = {self.h_period:.6f}")

        print(f"\n  fwd_ratio       Forward price ratio")
        print(f"                  Formula: fwd_ratio = exp((rf - div_yld) × h)")
        print(f"                         = exp(({self.rf} - {self.div_yld}) × {self.h_period:.6f})")
        print(f"                         = exp({(self.rf - self.div_yld) * self.h_period:.6f})")
        print(f"                         = {self.fwd_ratio:.6f}")

        print(f"\n  u (up_factor)   Multiplicative factor for UP moves")
        print(f"                  Formula: u = fwd_ratio × exp(σ × √h)")
        print(f"                         = {self.fwd_ratio:.6f} × exp({self.sigma} × √{self.h_period:.6f})")
        print(f"                         = {self.fwd_ratio:.6f} × exp({self.sigma * np.sqrt(self.h_period):.6f})")
        print(f"                         = {self.up_factor:.6f}")

        print(f"\n  d (down_factor) Multiplicative factor for DOWN moves")
        print(f"                  Formula: d = fwd_ratio × exp(-σ × √h)")
        print(f"                         = {self.fwd_ratio:.6f} × exp(-{self.sigma} × √{self.h_period:.6f})")
        print(f"                         = {self.fwd_ratio:.6f} × exp({-self.sigma * np.sqrt(self.h_period):.6f})")
        print(f"                         = {self.down_factor:.6f}")

        print(f"\n  p* (pstar)      Risk-neutral probability of UP move")
        print(f"                  Formula: p* = (fwd_ratio - d) / (u - d)")
        print(f"                         = ({self.fwd_ratio:.6f} - {self.down_factor:.6f}) / ({self.up_factor:.6f} - {self.down_factor:.6f})")
        print(f"                         = {self.fwd_ratio - self.down_factor:.6f} / {self.up_factor - self.down_factor:.6f}")
        print(f"                         = {self.pstar:.6f}")

        print(f"\n  discount_factor Single-period discount factor")
        print(f"                  Formula: discount = exp(-rf × h)")
        print(f"                         = exp(-{self.rf} × {self.h_period:.6f})")
        print(f"                         = exp({-self.rf * self.h_period:.6f})")
        print(f"                         = {self.discount_factor:.6f}")

        # Stock price dynamics example
        print(f"\nSTOCK PRICE DYNAMICS:")
        print("-" * 80)
        print(f"  Starting from S0 = ${self.S0:.2f}:")
        print(f"    After UP move:   S × u = ${self.S0:.2f} × {self.up_factor:.6f} = ${self.S0 * self.up_factor:.2f}")
        print(f"    After DOWN move: S × d = ${self.S0:.2f} × {self.down_factor:.6f} = ${self.S0 * self.down_factor:.2f}")

        # Replicating portfolio formulas
        print(f"\nREPLICATING PORTFOLIO FORMULAS:")
        print("-" * 80)
        print(f"  At each node (except terminal), calculate:")
        print(f"")
        print(f"  Δ (delta)       Number of shares of underlying")
        print(f"                  Formula: Δ = exp(-div_yld × h) × (V_u - V_d) / (S_u - S_d)")
        print(f"                  where V_u, V_d are successor option values")
        print(f"                        S_u, S_d are successor stock prices")
        print(f"")
        print(f"  B (bond)        Amount invested in risk-free bond")
        print(f"                  Formula: B = exp(-rf × h) × (u × V_d - d × V_u) / (u - d)")
        print(f"")
        print(f"  Option Value    Value equals replicating portfolio")
        print(f"                  Formula: V = Δ × S + B")

        # Risk-neutral valuation
        print(f"\nRISK-NEUTRAL VALUATION:")
        print("-" * 80)
        print(f"  At each node (except terminal), option value is:")
        print(f"")
        print(f"  Formula: V = exp(-rf × h) × [p* × V_u + (1 - p*) × V_d]")
        print(f"         = {self.discount_factor:.6f} × [{self.pstar:.6f} × V_u + {1-self.pstar:.6f} × V_d]")
        print(f"")
        print(f"  For American/Bermudan options:")
        print(f"         V = max(Exercise Value, Continuation Value)")

        print("\n" + "="*80)

    def get_summary_stats(self) -> dict:
        """Get summary statistics of the tree"""
        stats = {
            'S0': self.S0,
            'T_maturity': self.T_maturity,
            'N_periods': self.N_periods,
            'h_period': self.h_period,
            'sigma': self.sigma,
            'rf': self.rf,
            'div_yld': self.div_yld,
            'fwd_ratio': self.fwd_ratio,
            'up_factor': self.up_factor,
            'down_factor': self.down_factor,
            'pstar': self.pstar,
            'discount_factor': self.discount_factor,
            'initial_value': self.tree[0][0].derivative_value,
            'initial_delta': self.tree[0][0].delta,
            'initial_bond': self.tree[0][0].bond,
        }
        return stats


# Convenience payoff functions
def call_payoff(strike: float) -> Callable[[float], float]:
    """Create a call option payoff function"""
    return lambda S: max(S - strike, 0.0)


def put_payoff(strike: float) -> Callable[[float], float]:
    """Create a put option payoff function"""
    return lambda S: max(strike - S, 0.0)


def digital_call_payoff(strike: float, payout: float = 1.0) -> Callable[[float], float]:
    """Create a digital/binary call option payoff function"""
    return lambda S: payout if S >= strike else 0.0


def digital_put_payoff(strike: float, payout: float = 1.0) -> Callable[[float], float]:
    """Create a digital/binary put option payoff function"""
    return lambda S: payout if S <= strike else 0.0
