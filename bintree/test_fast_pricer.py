"""
Benchmark: FastAmericanPricer vs ForwardBinomialTree

Compare performance and accuracy of the vectorized fast pricer against
the original tree-based implementation.
"""

import time
import numpy as np
from forward_binomial_tree import (
    ForwardBinomialTree,
    FastAmericanPricer,
    put_payoff,
    call_payoff,
    implied_volatility_american,
)


def benchmark_single_price(N: int = 200, n_trials: int = 10):
    """Compare single option pricing speed."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Single Option Pricing (N={N} periods, {n_trials} trials)")
    print(f"{'='*70}")

    # Parameters
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.30

    # -------------------------------------------------------------------------
    # Original ForwardBinomialTree
    # -------------------------------------------------------------------------
    times_original = []
    for _ in range(n_trials):
        start = time.perf_counter()
        tree = ForwardBinomialTree(
            S0=S0, T_maturity=T, N_periods=N,
            sigma=sigma, rf=r, div_yld=q
        )
        price_original = tree.price_american(put_payoff(K))
        times_original.append(time.perf_counter() - start)

    avg_original = np.mean(times_original) * 1000  # ms

    # -------------------------------------------------------------------------
    # FastAmericanPricer
    # -------------------------------------------------------------------------
    times_fast = []
    for _ in range(n_trials):
        start = time.perf_counter()
        pricer = FastAmericanPricer(S0=S0, K=K, T_maturity=T, rf=r, div_yld=q, N_periods=N, is_call=False)
        price_fast = pricer.price(sigma)
        times_fast.append(time.perf_counter() - start)

    avg_fast = np.mean(times_fast) * 1000  # ms

    # -------------------------------------------------------------------------
    # FastAmericanPricer - pricing only (reuse pricer)
    # -------------------------------------------------------------------------
    pricer = FastAmericanPricer(S0=S0, K=K, T_maturity=T, rf=r, div_yld=q, N_periods=N, is_call=False)
    times_price_only = []
    for _ in range(n_trials):
        start = time.perf_counter()
        price_fast2 = pricer.price(sigma)
        times_price_only.append(time.perf_counter() - start)

    avg_price_only = np.mean(times_price_only) * 1000  # ms

    # Results
    print(f"\nAmerican Put: S0={S0}, K={K}, T={T}, r={r}, q={q}, σ={sigma}")
    print(f"\n{'Method':<40} {'Time (ms)':<12} {'Price':<12} {'Speedup':<10}")
    print("-" * 70)
    print(f"{'ForwardBinomialTree (original)':<40} {avg_original:<12.3f} ${price_original:<11.6f} {'1.0x':<10}")
    print(f"{'FastAmericanPricer (with init)':<40} {avg_fast:<12.3f} ${price_fast:<11.6f} {avg_original/avg_fast:<10.1f}x")
    print(f"{'FastAmericanPricer (price only)':<40} {avg_price_only:<12.3f} ${price_fast2:<11.6f} {avg_original/avg_price_only:<10.1f}x")

    # Verify accuracy
    print(f"\nPrice difference: ${abs(price_original - price_fast):.10f}")


def benchmark_implied_vol(N: int = 200):
    """Benchmark implied volatility calculation."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Implied Volatility Calculation (N={N} periods)")
    print(f"{'='*70}")

    # Parameters
    S0, K, T, r, q = 100.0, 100.0, 0.5, 0.05, 0.03
    true_sigma = 0.25

    # Generate "market price" using high-precision tree
    pricer = FastAmericanPricer(S0=S0, K=K, T_maturity=T, rf=r, div_yld=q, N_periods=500, is_call=False)
    market_price = pricer.price(true_sigma)

    print(f"\nAmerican Put: S0={S0}, K={K}, T={T}, r={r}, q={q}")
    print(f"True σ = {true_sigma}, Market price = ${market_price:.6f}")

    # -------------------------------------------------------------------------
    # Implied vol using FastAmericanPricer
    # -------------------------------------------------------------------------
    start = time.perf_counter()
    impl_vol, iterations = implied_volatility_american(
        market_price=market_price,
        S0=S0, K=K, T_maturity=T, rf=r, div_yld=q,
        is_call=False, N_periods=N
    )
    time_fast = (time.perf_counter() - start) * 1000

    print(f"\n{'Method':<40} {'Time (ms)':<12} {'Impl Vol':<12} {'Iterations':<10}")
    print("-" * 70)
    print(f"{'FastAmericanPricer + Newton-Raphson':<40} {time_fast:<12.2f} {impl_vol:<12.6f} {iterations:<10}")
    print(f"\nImplied vol error: {abs(impl_vol - true_sigma):.8f}")


def benchmark_multiple_prices(N: int = 200, n_options: int = 100):
    """Benchmark pricing multiple options (typical for vol surface)."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Price {n_options} Options (N={N} periods each)")
    print(f"{'='*70}")

    # Parameters
    S0, T, r, q, sigma = 100.0, 0.5, 0.05, 0.02, 0.30
    strikes = np.linspace(80, 120, n_options)

    # -------------------------------------------------------------------------
    # Original ForwardBinomialTree
    # -------------------------------------------------------------------------
    start = time.perf_counter()
    prices_original = []
    for K in strikes:
        tree = ForwardBinomialTree(
            S0=S0, T_maturity=T, N_periods=N,
            sigma=sigma, rf=r, div_yld=q
        )
        prices_original.append(tree.price_american(put_payoff(K)))
    time_original = (time.perf_counter() - start) * 1000

    # -------------------------------------------------------------------------
    # FastAmericanPricer
    # -------------------------------------------------------------------------
    start = time.perf_counter()
    prices_fast = []
    for K in strikes:
        pricer = FastAmericanPricer(S0=S0, K=K, T_maturity=T, rf=r, div_yld=q, N_periods=N, is_call=False)
        prices_fast.append(pricer.price(sigma))
    time_fast = (time.perf_counter() - start) * 1000

    # Results
    print(f"\n{n_options} American Puts with varying strikes")
    print(f"\n{'Method':<40} {'Total Time (ms)':<15} {'Per Option (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    print(f"{'ForwardBinomialTree':<40} {time_original:<15.1f} {time_original/n_options:<15.3f} {'1.0x':<10}")
    print(f"{'FastAmericanPricer':<40} {time_fast:<15.1f} {time_fast/n_options:<15.3f} {time_original/time_fast:<10.1f}x")

    # Verify accuracy
    max_diff = max(abs(p1 - p2) for p1, p2 in zip(prices_original, prices_fast))
    print(f"\nMax price difference: ${max_diff:.10f}")


def test_accuracy():
    """Test accuracy across different scenarios."""
    print(f"\n{'='*70}")
    print("ACCURACY TEST: Compare prices across scenarios")
    print(f"{'='*70}")

    test_cases = [
        # (S0, K, T, r, q, sigma, is_call, description)
        (100, 100, 1.0, 0.05, 0.02, 0.30, False, "ATM Put"),
        (100, 100, 1.0, 0.05, 0.02, 0.30, True, "ATM Call"),
        (100, 90, 0.5, 0.05, 0.00, 0.20, False, "OTM Put, no div"),
        (100, 110, 0.5, 0.05, 0.00, 0.20, True, "OTM Call, no div"),
        (100, 100, 0.25, 0.08, 0.05, 0.40, False, "High vol Put"),
        (100, 80, 1.0, 0.03, 0.06, 0.25, False, "ITM Put, high div"),
    ]

    N = 300
    print(f"\nUsing N={N} periods")
    print(f"\n{'Description':<25} {'Original':<12} {'Fast':<12} {'Diff':<15}")
    print("-" * 65)

    for S0, K, T, r, q, sigma, is_call, desc in test_cases:
        # Original
        tree = ForwardBinomialTree(S0=S0, T_maturity=T, N_periods=N,
                                   sigma=sigma, rf=r, div_yld=q)
        payoff = call_payoff(K) if is_call else put_payoff(K)
        price_orig = tree.price_american(payoff)

        # Fast
        pricer = FastAmericanPricer(S0=S0, K=K, T_maturity=T, rf=r, div_yld=q, N_periods=N, is_call=is_call)
        price_fast = pricer.price(sigma)

        diff = abs(price_orig - price_fast)
        print(f"{desc:<25} ${price_orig:<11.6f} ${price_fast:<11.6f} {diff:<15.2e}")


def test_delta():
    """Test delta calculation."""
    print(f"\n{'='*70}")
    print("DELTA TEST: Compare delta calculations")
    print(f"{'='*70}")

    S0, K, T, r, q, sigma, N = 100, 100, 1.0, 0.05, 0.02, 0.30, 200

    # Original
    tree = ForwardBinomialTree(S0=S0, T_maturity=T, N_periods=N,
                               sigma=sigma, rf=r, div_yld=q)
    tree.price_american(put_payoff(K))
    delta_orig = tree.tree[0][0].delta

    # Fast
    pricer = FastAmericanPricer(S0=S0, K=K, T_maturity=T, rf=r, div_yld=q, N_periods=N, is_call=False)
    price_fast, delta_fast = pricer.price_and_delta(sigma)

    print(f"\nAmerican Put: S0={S0}, K={K}, T={T}, r={r}, q={q}, σ={sigma}")
    print(f"\n{'Method':<30} {'Delta':<15}")
    print("-" * 45)
    print(f"{'ForwardBinomialTree':<30} {delta_orig:<15.8f}")
    print(f"{'FastAmericanPricer':<30} {delta_fast:<15.8f}")
    print(f"\nDelta difference: {abs(delta_orig - delta_fast):.2e}")


if __name__ == "__main__":
    test_accuracy()
    test_delta()
    benchmark_single_price(N=200, n_trials=10)
    benchmark_single_price(N=500, n_trials=5)
    benchmark_multiple_prices(N=200, n_options=100)
    benchmark_implied_vol(N=200)
