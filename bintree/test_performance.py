"""
Performance test for Forward Binomial Tree with 1000 periods
"""

import time
from forward_binomial_tree import ForwardBinomialTree, call_payoff, put_payoff


def test_performance():
    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    N = 1000  # 1000 periods
    sigma = 0.3
    rf = 0.05
    div_yld = 0.02

    print("="*70)
    print("PERFORMANCE TEST: Forward Binomial Tree with 1000 Periods")
    print("="*70)
    print(f"\nParameters:")
    print(f"  S0 = ${S0}")
    print(f"  Strike = ${K}")
    print(f"  T = {T} year")
    print(f"  N = {N} periods")
    print(f"  σ = {sigma}")
    print(f"  rf = {rf}")
    print(f"  div_yld = {div_yld}")

    # Test 1: Tree construction
    print(f"\n{'='*70}")
    print("Test 1: Tree Construction")
    print("="*70)

    start = time.time()
    tree = ForwardBinomialTree(
        S0=S0,
        T_maturity=T,
        N_periods=N,
        sigma=sigma,
        rf=rf,
        div_yld=div_yld
    )
    end = time.time()

    construction_time = end - start
    num_nodes = sum(i + 1 for i in range(N + 1))

    print(f"Total nodes in tree: {num_nodes:,}")
    print(f"Construction time: {construction_time:.4f} seconds")

    # Test 2: European Call
    print(f"\n{'='*70}")
    print("Test 2: European Call Option Pricing")
    print("="*70)

    start = time.time()
    euro_call_value = tree.price_european(call_payoff(K))
    end = time.time()

    euro_call_time = end - start
    print(f"European Call Value: ${euro_call_value:.6f}")
    print(f"Pricing time: {euro_call_time:.4f} seconds")
    print(f"Total time (construction + pricing): {construction_time + euro_call_time:.4f} seconds")

    # Test 3: American Put (requires new tree)
    print(f"\n{'='*70}")
    print("Test 3: American Put Option Pricing")
    print("="*70)

    start = time.time()
    tree_put = ForwardBinomialTree(
        S0=S0,
        T_maturity=T,
        N_periods=N,
        sigma=sigma,
        rf=rf,
        div_yld=div_yld
    )
    construction_time_2 = time.time() - start

    start = time.time()
    american_put_value = tree_put.price_american(put_payoff(K))
    end = time.time()

    american_put_time = end - start
    print(f"American Put Value: ${american_put_value:.6f}")
    print(f"Construction time: {construction_time_2:.4f} seconds")
    print(f"Pricing time: {american_put_time:.4f} seconds")
    print(f"Total time: {construction_time_2 + american_put_time:.4f} seconds")

    # Test 4: European Put (for comparison with American)
    print(f"\n{'='*70}")
    print("Test 4: European Put Option Pricing")
    print("="*70)

    start = time.time()
    tree_euro_put = ForwardBinomialTree(
        S0=S0,
        T_maturity=T,
        N_periods=N,
        sigma=sigma,
        rf=rf,
        div_yld=div_yld
    )
    construction_time_3 = time.time() - start

    start = time.time()
    euro_put_value = tree_euro_put.price_european(put_payoff(K))
    end = time.time()

    euro_put_time = end - start
    print(f"European Put Value: ${euro_put_value:.6f}")
    print(f"Construction time: {construction_time_3:.4f} seconds")
    print(f"Pricing time: {euro_put_time:.4f} seconds")
    print(f"Total time: {construction_time_3 + euro_put_time:.4f} seconds")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"Number of nodes: {num_nodes:,}")
    print(f"\nAverage construction time: {(construction_time + construction_time_2 + construction_time_3)/3:.4f} seconds")
    print(f"Average pricing time (European): {(euro_call_time + euro_put_time)/2:.4f} seconds")
    print(f"American pricing time: {american_put_time:.4f} seconds")
    print(f"\nEuropean Put vs American Put:")
    print(f"  European: ${euro_put_value:.6f}")
    print(f"  American: ${american_put_value:.6f}")
    print(f"  Early Exercise Premium: ${american_put_value - euro_put_value:.6f}")

    print(f"\nNodes per second (construction): {num_nodes/construction_time:,.0f}")
    print(f"Nodes per second (European pricing): {num_nodes/euro_call_time:,.0f}")
    print(f"Nodes per second (American pricing): {num_nodes/american_put_time:,.0f}")


if __name__ == "__main__":
    test_performance()
