"""
Example usage of the Forward Binomial Tree module.

This script demonstrates:
1. European call and put option pricing
2. American call and put option pricing
3. Bermudan option pricing
4. Tree visualization with exercise boundaries
5. Detailed tree analysis
6. Convergence analysis
7. American digital call with early exercise
"""

import numpy as np
import matplotlib.pyplot as plt
from forward_binomial_tree import (
    ForwardBinomialTree,
    call_payoff,
    put_payoff,
    digital_call_payoff
)


def example_1_european_call():
    """Example 1: Price a European call option"""
    print("\n" + "="*80)
    print("EXAMPLE 1: European Call Option")
    print("="*80)

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    N = 5
    sigma = 0.3
    rf = 0.05
    div_yld = 0.02

    # Create tree
    tree = ForwardBinomialTree(
        S0=S0,
        T_maturity=T,
        N_periods=N,
        sigma=sigma,
        rf=rf,
        div_yld=div_yld
    )

    # Print all parameters with formulas
    tree.print_parameters()

    # Price European call
    call_value = tree.price_european(call_payoff(K))

    print(f"\nEuropean Call Option")
    print(f"Strike: ${K}")
    print(f"Value:  ${call_value:.4f}")

    # Print tree at initial and final periods
    print(f"\nInitial and Final Period Details:")
    tree.print_tree(
        periods=[0, N],
        show_exercise=False,
        show_probability=True
    )

    # Visualize
    fig, ax = tree.plot_tree(
        show_stock_prices=True,
        show_derivative_values=True,
        show_exercise_boundary=False,
        payoff_description=f"max(S - {K}, 0)",
        option_type="European"
    )
    plt.savefig('european_call_tree.png', dpi=150, bbox_inches='tight')
    print(f"\nTree visualization saved to 'european_call_tree.png'")
    plt.close()


def example_2_american_put():
    """Example 2: Price an American put option"""
    print("\n" + "="*80)
    print("EXAMPLE 2: American Put Option")
    print("="*80)

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    N = 4
    sigma = 0.3
    rf = 0.05
    div_yld = 0.03

    # Create tree
    tree = ForwardBinomialTree(
        S0=S0,
        T_maturity=T,
        N_periods=N,
        sigma=sigma,
        rf=rf,
        div_yld=div_yld
    )

    # Price American put
    american_value = tree.price_american(put_payoff(K))

    # Price European put for comparison
    tree_european = ForwardBinomialTree(
        S0=S0,
        T_maturity=T,
        N_periods=N,
        sigma=sigma,
        rf=rf,
        div_yld=div_yld
    )
    european_value = tree_european.price_european(put_payoff(K))

    print(f"\nPut Option Comparison (Strike = ${K})")
    print(f"European Put Value: ${european_value:.4f}")
    print(f"American Put Value: ${american_value:.4f}")
    print(f"Early Exercise Premium: ${american_value - european_value:.4f}")

    # Show exercise boundary
    boundary = tree.get_exercise_boundary()
    print(f"\nOptimal Exercise Boundary:")
    print(f"Period | Min Exercise Price | Max Exercise Price")
    print(f"-------|-------------------|-------------------")
    for period, min_price, max_price in boundary:
        print(f"  {period:2d}   | ${min_price:10.2f}        | ${max_price:10.2f}")

    # Print detailed information for middle periods
    print(f"\nDetailed Analysis at Selected Periods:")
    tree.print_tree(
        periods=[0, N//2, N],
        show_exercise=True,
        show_probability=True
    )

    # Visualize with exercise boundary
    fig, ax = tree.plot_tree(
        show_stock_prices=True,
        show_derivative_values=True,
        show_exercise_boundary=True,
        show_continuation_values=True,
        payoff_description=f"max({K} - S, 0)",
        option_type="American",
        european_value=european_value,
        max_periods_display=min(N, 10)
    )
    plt.savefig('american_put_tree.png', dpi=150, bbox_inches='tight')
    print(f"\nTree visualization saved to 'american_put_tree.png'")
    plt.close()


def example_3_bermudan_put():
    """Example 3: Price a Bermudan put option"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Bermudan Put Option")
    print("="*80)

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    N = 12  # Monthly periods
    sigma = 0.25
    rf = 0.04
    div_yld = 0.01

    # Create tree
    tree = ForwardBinomialTree(
        S0=S0,
        T_maturity=T,
        N_periods=N,
        sigma=sigma,
        rf=rf,
        div_yld=div_yld
    )

    # Exercise allowed only at periods 3, 6, 9, 12 (quarterly)
    exercise_periods = [3, 6, 9, 12]

    # Price Bermudan put
    bermudan_value = tree.price_bermudan(put_payoff(K), exercise_periods)

    # Price European and American for comparison
    tree_european = ForwardBinomialTree(
        S0=S0, T_maturity=T, N_periods=N,
        sigma=sigma, rf=rf, div_yld=div_yld
    )
    european_value = tree_european.price_european(put_payoff(K))

    tree_american = ForwardBinomialTree(
        S0=S0, T_maturity=T, N_periods=N,
        sigma=sigma, rf=rf, div_yld=div_yld
    )
    american_value = tree_american.price_american(put_payoff(K))

    print(f"\nPut Option Comparison (Strike = ${K})")
    print(f"European Put Value:  ${european_value:.4f}")
    print(f"Bermudan Put Value:  ${bermudan_value:.4f}")
    print(f"American Put Value:  ${american_value:.4f}")
    print(f"\nExercise Periods: {exercise_periods}")

    # Show when exercise occurs
    boundary = tree.get_exercise_boundary()
    print(f"\nExercise Boundary (at allowed periods):")
    print(f"Period | Min Exercise Price | Max Exercise Price")
    print(f"-------|-------------------|-------------------")
    for period, min_price, max_price in boundary:
        if period in exercise_periods:
            print(f"  {period:2d}   | ${min_price:10.2f}        | ${max_price:10.2f}")

    # Visualize
    fig, ax = tree.plot_tree(
        show_stock_prices=False,
        show_derivative_values=True,
        show_exercise_boundary=True,
        show_continuation_values=True,
        payoff_description=f"max({K} - S, 0)",
        option_type="Bermudan",
        european_value=european_value,
        max_periods_display=N
    )
    plt.savefig('bermudan_put_tree.png', dpi=150, bbox_inches='tight')
    print(f"\nTree visualization saved to 'bermudan_put_tree.png'")
    plt.close()


def example_4_replicating_portfolio():
    """Example 4: Analyze the replicating portfolio"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Replicating Portfolio Analysis")
    print("="*80)

    # Parameters
    S0 = 50.0
    K = 50.0
    T = 0.5
    N = 3
    sigma = 0.4
    rf = 0.06
    div_yld = 0.0

    # Create tree
    tree = ForwardBinomialTree(
        S0=S0,
        T_maturity=T,
        N_periods=N,
        sigma=sigma,
        rf=rf,
        div_yld=div_yld
    )

    # Price European call
    call_value = tree.price_european(call_payoff(K))

    print(f"\nEuropean Call Option (Strike = ${K})")
    print(f"Option Value: ${call_value:.4f}")

    # Detailed replicating portfolio at t=0
    node = tree.tree[0][0]
    print(f"\nReplicating Portfolio at t=0:")
    print(f"  Delta (Δ):         {node.delta:.6f} shares")
    print(f"  Bond (B):          ${node.bond:.4f}")
    print(f"  Stock Price:       ${node.stock_price:.2f}")
    print(f"  Portfolio Value:   ${node.delta * node.stock_price + node.bond:.4f}")
    print(f"  Option Value:      ${node.derivative_value:.4f}")

    # Show complete tree details
    print(f"\nComplete Tree Structure:")
    tree.print_tree(
        show_stock=True,
        show_value=True,
        show_delta=True,
        show_bond=True,
        show_probability=True
    )

    # Verify replication at each node (excluding maturity)
    print(f"\nVerifying Replication (Portfolio Value = Option Value):")
    print(f"Note: Replication applies only before maturity. At maturity, the option value IS the payoff.")
    print(f"Period | Node | Portfolio Value | Option Value | Match?")
    print(f"-------|------|-----------------|--------------|-------")
    payoff = call_payoff(K)
    for period in range(N):  # Exclude maturity period N
        for index in range(period + 1):
            node = tree.tree[period][index]
            portfolio_value = node.delta * node.stock_price + node.bond
            match = "✓" if abs(portfolio_value - node.derivative_value) < 0.01 else "✗"
            print(f"  {period:2d}   | {index:2d}   | ${portfolio_value:10.4f}     | "
                  f"${node.derivative_value:9.4f}    | {match}")

    # Verify payoffs at maturity
    print(f"\nVerifying Payoffs at Maturity (Period {N}):")
    print(f"Node | Stock Price | Option Value | Payoff = max(S-K,0) | Match?")
    print(f"-----|-------------|--------------|---------------------|-------")
    for index in range(N + 1):
        node = tree.tree[N][index]
        expected_payoff = payoff(node.stock_price)
        match = "✓" if abs(node.derivative_value - expected_payoff) < 0.01 else "✗"
        print(f" {index:2d}  | ${node.stock_price:8.2f}    | ${node.derivative_value:9.4f}     | "
              f"${expected_payoff:10.4f}          | {match}")

    # Verify that replicating portfolios from period N-1 replicate terminal payoffs
    print(f"\nVerifying Replication from Period {N-1} to Maturity:")
    print(f"Each portfolio at period {N-1} should replicate both successor payoffs.")
    print(f"Prev | Succ | Prev Portfolio @ t={N-1}   | Portfolio Value @ t={N} | Terminal Payoff | Match?")
    print(f"Node | Node | (Δ, B)                     | Δ*S + B*exp(r*h)        |                 |")
    print(f"-----|------|----------------------------|-------------------------|-----------------|-------")

    for prev_idx in range(N):
        prev_node = tree.tree[N-1][prev_idx]

        # Check UP successor (same index)
        up_node = tree.tree[N][prev_idx]
        portfolio_value_up = (prev_node.delta * up_node.stock_price +
                             prev_node.bond * np.exp(rf * tree.h_period))
        match_up = "✓" if abs(portfolio_value_up - up_node.derivative_value) < 0.01 else "✗"
        print(f" [{N-1},{prev_idx}] | [{N},{prev_idx}] | Δ={prev_node.delta:.4f}, B=${prev_node.bond:.2f} | "
              f"${portfolio_value_up:10.4f}          | ${up_node.derivative_value:10.4f}      | {match_up}")

        # Check DOWN successor (index + 1)
        down_node = tree.tree[N][prev_idx + 1]
        portfolio_value_down = (prev_node.delta * down_node.stock_price +
                               prev_node.bond * np.exp(rf * tree.h_period))
        match_down = "✓" if abs(portfolio_value_down - down_node.derivative_value) < 0.01 else "✗"
        print(f" [{N-1},{prev_idx}] | [{N},{prev_idx+1}] | Δ={prev_node.delta:.4f}, B=${prev_node.bond:.2f} | "
              f"${portfolio_value_down:10.4f}          | ${down_node.derivative_value:10.4f}      | {match_down}")


def example_5_custom_payoff():
    """Example 5: Custom payoff function (digital option)"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Payoff - Digital Call Option")
    print("="*80)

    # Parameters
    S0 = 100.0
    K = 100.0
    payout = 10.0
    T = 0.25
    N = 10
    sigma = 0.2
    rf = 0.03
    div_yld = 0.0

    # Create tree
    tree = ForwardBinomialTree(
        S0=S0,
        T_maturity=T,
        N_periods=N,
        sigma=sigma,
        rf=rf,
        div_yld=div_yld
    )

    # Price digital call
    digital_value = tree.price_european(digital_call_payoff(K, payout))

    print(f"\nDigital Call Option")
    print(f"Strike:  ${K}")
    print(f"Payout:  ${payout} if S >= K at maturity")
    print(f"Value:   ${digital_value:.4f}")

    # Show terminal payoffs
    print(f"\nTerminal Payoffs at Maturity (Period {N}):")
    print(f"Node | Stock Price | Payoff | Probability")
    print(f"-----|-------------|--------|------------")
    for index in range(N + 1):
        node = tree.tree[N][index]
        print(f" {index:2d}  | ${node.stock_price:8.2f}    | "
              f"${node.derivative_value:5.2f}  | {node.probability:.6f}")


def example_6_convergence_test():
    """Example 6: Test convergence as N increases"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Convergence Test")
    print("="*80)

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    sigma = 0.3
    rf = 0.05
    div_yld = 0.02

    print(f"\nAmerican Put Option - Convergence Analysis")
    print(f"N Periods | Option Value | Δ from Previous")
    print(f"----------|--------------|----------------")

    N_values = [5, 10, 20, 50, 100, 200]
    prev_value = None

    for N in N_values:
        tree = ForwardBinomialTree(
            S0=S0, T_maturity=T, N_periods=N,
            sigma=sigma, rf=rf, div_yld=div_yld
        )
        value = tree.price_american(put_payoff(K))

        if prev_value is not None:
            delta = value - prev_value
            print(f"  {N:4d}    | ${value:9.6f}   | ${delta:+9.6f}")
        else:
            print(f"  {N:4d}    | ${value:9.6f}   |     ---")

        prev_value = value


def example_7_american_digital_call():
    """Example 7: American digital call with early exercise analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 7: American Digital Call Option")
    print("="*80)

    # Educational Introduction
    print("\nDigital (Binary) Call Option:")
    print("- Pays fixed amount if S >= K at exercise")
    print("- Unlike vanilla calls, payoff is BOUNDED (fixed amount)")
    print("- Early exercise optimal as soon as option is ATM/ITM--no upside to waiting!")

    # Parameters chosen to demonstrate early exercise
    S0 = 100.0
    K = 105.0        # Lower strike → more likely to be ITM
    payout = 100.0  # Substantial fixed payout
    T = 1.0
    N = 4           # Enough periods to show clear boundary
    sigma = 0.25
    rf = 0.04
    div_yld = 0.08  # HIGH dividend yield → incentive for early exercise

    print(f"\nParameters:")
    print(f"  Initial Stock Price: ${S0}")
    print(f"  Strike Price:        ${K}")
    print(f"  Digital Payout:      ${payout} (if S >= K)")
    print(f"  Time to Maturity:    {T} year")
    print(f"  Periods:             {N}")
    print(f"  Volatility:          {sigma*100}%")
    print(f"  Risk-free Rate:      {rf*100}%")
    print(f"  Dividend Yield:      {div_yld*100}% (HIGH → favors early exercise)")

    # Create tree for American option
    tree_american = ForwardBinomialTree(
        S0=S0, T_maturity=T, N_periods=N,
        sigma=sigma, rf=rf, div_yld=div_yld
    )
    american_value = tree_american.price_american(
        digital_call_payoff(K, payout)
    )

    # Create separate tree for European comparison
    tree_european = ForwardBinomialTree(
        S0=S0, T_maturity=T, N_periods=N,
        sigma=sigma, rf=rf, div_yld=div_yld
    )
    european_value = tree_european.price_european(
        digital_call_payoff(K, payout)
    )

    # Comparison Table
    print(f"\n{'='*80}")
    print("OPTION VALUATION COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Option Type':<30} {'Value':<15} {'Notes'}")
    print(f"{'-'*80}")
    print(f"{'European Digital Call':<30} ${european_value:<14.4f} {'Must hold to maturity'}")
    print(f"{'American Digital Call':<30} ${american_value:<14.4f} {'Can exercise early'}")
    print(f"{'-'*80}")
    print(f"{'Early Exercise Premium':<30} ${american_value - european_value:<14.4f} "
          f"{'(American - European)'}")
    print(f"{'Premium as % of European':<30} {100*(american_value-european_value)/european_value:<14.2f}%")

    # Economic interpretation
    print(f"\n{'='*80}")
    print("ECONOMIC INTERPRETATION")
    print(f"{'='*80}")
    if american_value > european_value + 0.01:  # Meaningful premium
        print("\nThe American option is MORE VALUABLE than European:")
        print(f"  → Early exercise premium: ${american_value - european_value:.4f}")
        print(f"  → Early exercise is OPTIMAL in some states")
        print(f"\nWhy early exercise makes sense for digital calls:")
        print(f"  1. BOUNDED PAYOFF: Payout is fixed at ${payout}, no upside from waiting")
        print(f"  2. DIVIDEND DRAG: {div_yld*100}% dividend yield reduces forward prices")
        print(f"  3. LOCK IN GAINS: When deep ITM, lock in ${payout} rather than risk falling below ${K}")
        print(f"  4. TIME VALUE: ${payout} today worth more than ${payout} later (reinvest at {rf*100}%)")
    else:
        print("\nThe American option has MINIMAL early exercise value:")
        print(f"  → Premium is only ${american_value - european_value:.4f}")
        print(f"  → Early exercise rarely optimal with these parameters")

    # Exercise Boundary Analysis
    boundary = tree_american.get_exercise_boundary()
    print(f"\n{'='*80}")
    print("OPTIMAL EXERCISE BOUNDARY")
    print(f"{'='*80}")
    print("\nExercise occurs when stock price is HIGH ENOUGH and it's optimal to lock in payout.")
    print(f"\n{'Period':<8} {'Time (years)':<15} {'Min Exercise $':<18} {'Max Exercise $':<18} {'# Nodes Exercised'}")
    print(f"{'-'*80}")

    for period, min_price, max_price in boundary:
        time_years = period * tree_american.h_period
        # Count how many nodes exercised at this period
        num_exercised = sum(1 for idx in range(period + 1)
                          if tree_american.tree[period][idx].exercised)
        print(f"{period:<8} {time_years:<15.4f} ${min_price:<17.2f} ${max_price:<17.2f} {num_exercised}/{period+1}")

    if not boundary:
        print("No early exercise occurs - European and American values are equal.")

    # Detailed Node Analysis at Key Periods
    print(f"\n{'='*80}")
    print("DETAILED NODE ANALYSIS")
    print(f"{'='*80}")
    print("\nAt selected periods, compare Exercise vs Continuation values:")

    # Analyze middle period in detail
    analysis_periods = [0, N//2, N]
    for period in analysis_periods:
        if period == 0:
            print(f"\n--- Period {period} (Initial, t=0) ---")
            print("Exercise NOT allowed at t=0 (standard convention)")
        elif period == N:
            print(f"\n--- Period {period} (Maturity, t={tree_american.h_period * period:.4f}) ---")
            print("At maturity: Option value = Payoff (no choice)")
        else:
            print(f"\n--- Period {period} (Middle, t={tree_american.h_period * period:.4f}) ---")

        print(f"\n{'Node':<6} {'Stock $':<12} {'Exercise $':<14} {'Continue $':<14} {'Value $':<12} {'Decision'}")
        print(f"{'-'*80}")

        for index in range(min(period + 1, 5)):  # Show first 5 nodes max
            node = tree_american.tree[period][index]
            ex_val = f"${node.exercise_value:.2f}" if node.exercise_value is not None else "N/A"
            cont_val = f"${node.continuation_value:.2f}" if node.continuation_value is not None else "N/A"
            decision = "EXERCISE" if node.exercised else "Hold"

            print(f"[{period},{index}] {node.stock_price:<11.2f} {ex_val:<14} {cont_val:<14} "
                  f"${node.derivative_value:<11.2f} {decision}")

    # Show a few detailed examples of WHY exercise happens
    print(f"\n{'='*80}")
    print("WHY EARLY EXERCISE OCCURS - SPECIFIC EXAMPLES")
    print(f"{'='*80}")

    # Find an exercised node to explain
    exercised_example_found = False
    for period in range(1, N):
        for index in range(period + 1):
            node = tree_american.tree[period][index]
            if node.exercised and node.stock_price > K:
                print(f"\nExample: Node [{period}, {index}] at t={period * tree_american.h_period:.4f} years")
                print(f"  Stock Price:        ${node.stock_price:.2f}")
                print(f"  Strike:             ${K:.2f}")
                print(f"  Status:             IN THE MONEY (S > K)")
                print(f"  Exercise Value:     ${node.exercise_value:.4f} (get ${payout} immediately)")
                print(f"  Continuation Value: ${node.continuation_value:.4f} (expected value of holding)")
                print(f"  Decision:           EXERCISE (exercise > continuation)")
                print(f"\n  Interpretation:")
                print(f"    → Exercising locks in ${payout} payout NOW")
                print(f"    → Present value of ${payout} today > expected value of waiting")
                print(f"    → Avoids risk of stock falling below ${K} before maturity")
                print(f"    → High dividend yield ({div_yld*100}%) creates downward drift in forward prices")
                exercised_example_found = True
                break
        if exercised_example_found:
            break

    if not exercised_example_found:
        print("\nNo early exercise occurs with these parameters.")
        print("Consider increasing dividend yield or choosing lower strike for more ITM scenarios.")

    # Visualization
    print(f"\n{'='*80}")
    print("TREE VISUALIZATION")
    print(f"{'='*80}")

    fig, ax = tree_american.plot_tree(
        show_stock_prices=True,
        show_derivative_values=True,
        show_exercise_boundary=True,
        show_continuation_values=True,
        payoff_description=f"${payout} if S >= ${K}",
        option_type="American Digital Call",
        european_value=european_value,
        max_periods_display=min(N, 10)
    )

    filename = 'american_digital_call_tree.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nTree visualization saved to '{filename}'")
    print("  → Red nodes = Early exercise optimal")
    print("  → Blue nodes = Hold (continuation optimal)")
    print("  → Numbers show: S=Stock price, V=Option value, C=Continuation value")
    plt.close()

    # Compare with vanilla American call
    print(f"\n{'='*80}")
    print("COMPARISON: DIGITAL vs VANILLA AMERICAN CALLS")
    print(f"{'='*80}")

    print("\nKey Differences:")
    print(f"\n  {'Feature':<35} {'Digital Call':<25} {'Vanilla Call'}")
    print(f"  {'-'*80}")
    print(f"  {'Payoff if ITM':<35} {'Fixed: $' + str(payout):<25} {'Unlimited: S - K'}")
    print(f"  {'Early Exercise':<35} {'Often optimal (high div)':<25} {'Rarely optimal*'}")
    print(f"  {'Upside from waiting':<35} {'None (bounded)':<25} {'Yes (unbounded)'}")
    print(f"  {'Main risk of waiting':<35} {'Fall below strike':<25} {'Dividend loss only'}")
    print(f"  {'When to exercise':<35} {'Lock in fixed payout':<25} {'Usually at maturity*'}")

    print("\n  * For vanilla calls: early exercise only optimal just before dividends")
    print("    For digital calls: early exercise rational when deep ITM with high dividends")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nAmerican digital call value: ${american_value:.4f}")
    print(f"European digital call value:  ${european_value:.4f}")
    print(f"Early exercise premium:       ${american_value - european_value:.4f}")

    if american_value > european_value + 0.01:
        print(f"\nConclusion: Early exercise IS valuable for this digital call")
        print(f"The bounded payoff and high dividend yield make it rational to")
        print(f"exercise early in some states to lock in the ${payout} payout.")
    else:
        print(f"\nConclusion: Early exercise has minimal value")
        print(f"Consider parameters with higher dividend yield or lower strike")
        print(f"to see more pronounced early exercise behavior.")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("FORWARD BINOMIAL TREE - COMPREHENSIVE EXAMPLES")
    print("="*80)

    example_1_european_call()
    example_2_american_put()
    example_3_bermudan_put()
    example_4_replicating_portfolio()
    example_5_custom_payoff()
    example_6_convergence_test()
    example_7_american_digital_call()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
