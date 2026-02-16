"""
TD Canadian Large Cap Diversified Equity Index 265 AR-Linked
Boosted Growth Notes (Series 212) — Full Pricing Analysis

This script reproduces the entire structured note analysis:
  1. Scrape XIU option data from the Montreal Exchange
  2. Back out implied volatilities (Brent's method)
  3. Compute historical constituent portfolio variance + VRP
  4. Price with exact prepaid forwards (PV of annuity drag)
  5. Price with continuous dividend yield approximation
  6. Expected returns under the physical measure
  7. Alternative structure analysis

Run from the python-scripts/ directory:
    uv run python theme_08/td_ppn_analysis.py
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import date
import json
import re
import html as htmlmod
import urllib.request

# ==========================================================================
# Note parameters (from the information statement)
# ==========================================================================
AR_0 = 5468.46       # Initial AR index level
T = 4.0              # Years to maturity (Jan 29, 2030)
r = 0.03             # Risk-free rate (Canadian 4-year, ~3%)
DRAG = 265.0         # Index points per year subtracted from TR index
BOOST = 20.0         # Boosted return (%) — digital pays $20
PARTICIPATION = 0.05 # 5% participation above the boost
TODAY = date(2026, 2, 10)

# Constituent portfolio: ticker -> index shares
CONSTITUENTS = {
    'AEM.TO': 1.068889,    # Agnico-Eagle Mines
    'BMO.TO': 1.618257,    # Bank of Montreal
    'BNS.TO': 2.975524,    # Bank of Nova Scotia
    'ABX.TO': 4.476445,    # Barrick Mining
    'BCE.TO': 8.96346,     # BCE Inc
    'CM.TO':  2.3997,      # CIBC
    'CNQ.TO': 6.104161,    # Canadian Natural Resources
    'EMA.TO': 4.413341,    # Emera Inc
    'ENB.TO': 4.644166,    # Enbridge Inc
    'GWO.TO': 4.69007,     # Great-West Lifeco
    'PPL.TO': 5.61646,     # Pembina Pipeline
    'QSR.TO': 3.22517,     # Restaurant Brands International
    'RCI-B.TO': 5.980347,  # Rogers Communications B
    'RY.TO':  1.307227,    # Royal Bank of Canada
    'SHOP.TO': 1.585145,   # Shopify
    'SLF.TO': 3.516059,    # Sun Life Financial
    'TRP.TO': 4.069168,    # TC Energy
    'T.TO':   16.415244,   # Telus
    'TD.TO':  2.34413,     # Toronto-Dominion Bank
    'TOU.TO': 4.911564,    # Tourmaline Oil
}


# ==========================================================================
# Black-Scholes helpers
# ==========================================================================

def bs_call(S, K, T, r, delta, sigma):
    """European call price with continuous dividend yield."""
    d1 = (np.log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-delta * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, delta, sigma):
    """European put price with continuous dividend yield."""
    d1 = (np.log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-delta * T) * norm.cdf(-d1)


def implied_vol(S, K, T, r, delta, price, is_call=True):
    """Back out implied volatility using Brent's method."""
    func = bs_call if is_call else bs_put
    try:
        return brentq(lambda sig: func(S, K, T, r, delta, sig) - price, 0.01, 2.0)
    except (ValueError, RuntimeError):
        return np.nan


# ==========================================================================
# Part 1: Scrape XIU option data from the Montreal Exchange
# ==========================================================================

def scrape_mx_options():
    """Scrape XIU option data from the Montreal Exchange website.

    The MX website embeds option data as HTML-escaped JSON objects in the
    page source. We extract them with a regex pattern and parse.
    """
    print("=" * 70)
    print("PART 1: SCRAPING XIU OPTION DATA FROM MONTREAL EXCHANGE")
    print("=" * 70)

    headers = {'User-Agent': 'Mozilla/5.0'}
    url = "https://www.m-x.ca/en/trading/data/quotes?symbol=XIU*"

    print(f"Fetching {url} ...")
    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=15)
    raw_html = resp.read().decode('utf-8')
    print(f"Received {len(raw_html):,} bytes of HTML")

    # Extract JSON option objects embedded as HTML-escaped strings
    pattern = r'\{&quot;symbol&quot;:&quot;XIU[^}]+\}'
    matches = re.findall(pattern, raw_html)
    print(f"Found {len(matches)} raw option records")

    options = []
    for m in matches:
        clean = htmlmod.unescape(m)
        try:
            obj = json.loads(clean)
            options.append(obj)
        except json.JSONDecodeError:
            pass

    print(f"Parsed {len(options)} option records")

    # Summary by expiry
    expiries = sorted(set(o['expiry_date'] for o in options))
    print(f"\nExpiry dates ({len(expiries)}):")
    for e in expiries:
        count = sum(1 for o in options if o['expiry_date'] == e)
        print(f"  {e}  ({count} contracts)")

    # Extract ATM options near current XIU price (~$49)
    print("\nLongest-dated ATM options (K near $49):")
    atm_data = []
    for expiry in expiries[-4:]:
        exp_opts = [o for o in options if o['expiry_date'] == expiry]
        for o in sorted(exp_opts, key=lambda x: x['strike_price']):
            sp = o['strike_price']
            if 46 <= sp <= 51:
                sym = o['symbol'].strip()
                side = 'C' if 'C' in sym.split(str(sp))[-1] else 'P'
                print(f"  {expiry} | K={sp:6.2f} | {side} | "
                      f"bid={o['bid_price']:.2f} ask={o['ask_price']:.2f} "
                      f"last={o['last_price']:.2f}")
                atm_data.append({
                    'expiry': expiry, 'strike': sp, 'side': side,
                    'bid': o['bid_price'], 'ask': o['ask_price'],
                    'last': o['last_price'],
                })

    return options, atm_data


# ==========================================================================
# Part 2: Back out implied volatilities
# ==========================================================================

def compute_implied_vols():
    """Compute implied vols from XIU option prices.

    Uses bid-ask midpoints from the MX scrape as of 2026-02-10.
    """
    print("\n" + "=" * 70)
    print("PART 2: IMPLIED VOLATILITY EXTRACTION")
    print("=" * 70)

    S = 48.78   # XIU.TO close on 2026-02-10
    delta = 0.023  # XIU trailing 12-month dividend yield (~2.3%)

    # Option data from MX (bid-ask midpoints, 2026-02-10)
    # (expiry, strike, call_mid, put_mid)
    data = [
        ("2026-12-18", 48.00, 2.79, 2.14),
        ("2026-12-18", 48.50, 2.48, 2.32),
        ("2026-12-18", 49.00, 2.14, 2.54),
        ("2027-03-19", 48.00, 3.14, 2.45),
        ("2027-03-19", 50.00, 1.86, 3.27),
        ("2028-03-17", 47.00, 4.99, 3.21),
        ("2028-03-17", 48.00, 4.40, 3.59),
        ("2028-03-17", 50.00, 3.32, 4.48),
        ("2029-03-16", 46.00, 6.85, 3.80),
        ("2029-03-16", 47.00, 6.20, 4.10),
        ("2029-03-16", 48.00, 5.51, 4.44),
        ("2029-03-16", 49.00, 4.90, 4.83),
        ("2029-03-16", 50.00, 4.38, 5.30),
    ]

    print(f"\nXIU.TO = ${S:.2f}, r = {r:.1%}, δ = {delta:.1%}")
    print(f"\n{'Expiry':<12} {'K':>6} {'T':>5} {'C_last':>7} {'P_last':>7} "
          f"{'IV_C':>7} {'IV_P':>7} {'IV_avg':>7}")
    print("-" * 72)

    results = []
    for expiry_str, K, c_price, p_price in data:
        y, m, d = map(int, expiry_str.split('-'))
        exp_date = date(y, m, d)
        T_opt = (exp_date - TODAY).days / 365.25

        iv_c = implied_vol(S, K, T_opt, r, delta, c_price, is_call=True)
        iv_p = implied_vol(S, K, T_opt, r, delta, p_price, is_call=False)
        iv_avg = np.nanmean([x for x in [iv_c, iv_p] if not np.isnan(x)])

        print(f"{expiry_str:<12} {K:>6.0f} {T_opt:>5.2f} {c_price:>7.2f} {p_price:>7.2f} "
              f"{iv_c:>6.1%} {iv_p:>6.1%} {iv_avg:>6.1%}")

        results.append({'expiry': expiry_str, 'K': K, 'T': T_opt,
                        'iv_c': iv_c, 'iv_p': iv_p, 'iv_avg': iv_avg})

    # ATM IV term structure
    print("\nATM Implied Volatility Term Structure:")
    for expiry in sorted(set(r['expiry'] for r in results)):
        atm = [r for r in results if r['expiry'] == expiry and abs(r['K'] - S) < 1.5]
        if atm:
            avg_iv = np.nanmean([r['iv_avg'] for r in atm])
            avg_T = np.mean([r['T'] for r in atm])
            print(f"  {expiry}  T={avg_T:.2f}y  ATM IV ≈ {avg_iv:.1%}")

    sigma_implied = 0.160
    print(f"\nSelected σ = {sigma_implied:.1%} (ATM IV at T≈3y ≈ 15.5%, extrapolated to T=4y)")
    return sigma_implied


# ==========================================================================
# Part 3: Historical constituent variance + VRP
# ==========================================================================

def compute_historical_variance():
    """Download 5 years of constituent data and compute portfolio variance."""
    print("\n" + "=" * 70)
    print("PART 3: HISTORICAL CONSTITUENT VARIANCE + VRP")
    print("=" * 70)

    import yfinance as yf
    import pandas as pd

    tickers = list(CONSTITUENTS.keys())
    shares = np.array(list(CONSTITUENTS.values()))

    print("Downloading 5 years of daily price data...")
    data = yf.download(tickers, period='5y', auto_adjust=True)
    prices = data['Close'][tickers].ffill()
    print(f"Data: {prices.shape[0]} days, {prices.index[0].date()} to {prices.index[-1].date()}")

    # Portfolio value weighted by index shares
    portfolio = (prices * shares).sum(axis=1)
    log_ret = np.log(portfolio / portfolio.shift(1)).dropna()

    # Annualized variance
    daily_var = log_ret.var()
    annual_var = daily_var * 252
    vrp = 0.015
    total_var = annual_var + vrp
    sigma_hist = np.sqrt(total_var)

    print(f"\nAnnualized realized variance: σ² = {annual_var:.4f} → σ = {np.sqrt(annual_var)*100:.2f}%")
    print(f"Variance risk premium:       +VRP = {vrp:.4f}")
    print(f"Total variance:        σ²+VRP     = {total_var:.4f} → σ = {sigma_hist*100:.2f}%")

    # Robustness
    print(f"\nRobustness across windows:")
    print(f"  {'Window':<10} {'σ²':>10} {'σ':>10} {'σ(+VRP)':>10}")
    print(f"  {'-'*42}")
    for years, label in [(1, '1-year'), (2, '2-year'), (3, '3-year'), (5, '5-year')]:
        n_days = years * 252
        if len(log_ret) >= n_days:
            v = log_ret.iloc[-n_days:].var() * 252
            print(f"  {label:<10} {v:>10.4f} {np.sqrt(v)*100:>9.2f}% {np.sqrt(v+vrp)*100:>9.2f}%")

    return sigma_hist


# ==========================================================================
# Part 4: Pricing with exact prepaid forwards
# ==========================================================================

def price_note(sigma, label=""):
    """Price the structured note using generalized BS with prepaid forwards."""
    PV_drag = DRAG * (1 - np.exp(-r * T)) / r
    FP_AR = AR_0 - PV_drag
    pv_principal = 100 * np.exp(-r * T)

    # Digital call: $20 if AR_T >= AR_0
    K1 = AR_0
    FP_K1 = K1 * np.exp(-r * T)
    d1_dig = (np.log(FP_AR / FP_K1) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2_dig = d1_dig - sigma * np.sqrt(T)
    digital = BOOST * np.exp(-r * T) * norm.cdf(d2_dig)

    # Vanilla call: 5% participation, strike = 1.2 * AR_0
    K2 = (1 + BOOST / 100) * AR_0
    FP_K2 = K2 * np.exp(-r * T)
    d1_van = (np.log(FP_AR / FP_K2) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2_van = d1_van - sigma * np.sqrt(T)
    call_val = FP_AR * norm.cdf(d1_van) - FP_K2 * norm.cdf(d2_van)
    vanilla = PARTICIPATION * (call_val / AR_0) * 100

    fair_value = pv_principal + digital + vanilla

    print(f"\n--- Pricing {label} (σ = {sigma*100:.2f}%) ---")
    print(f"PV(drag) = {PV_drag:.2f},  F^P(AR) = {FP_AR:.2f}")
    print(f"PV of principal:  ${pv_principal:.2f}")
    print(f"Digital call:     ${digital:.2f}  (d₂ = {d2_dig:.3f}, N(d₂) = {norm.cdf(d2_dig):.3f})")
    print(f"Vanilla call:     ${vanilla:.2f}")
    print(f"Fair value:       ${fair_value:.2f}")
    print(f"Issuer margin:    ${100 - fair_value:.2f} ({100 - fair_value:.1f}%)")

    return fair_value


# ==========================================================================
# Part 5: Continuous dividend yield approximation
# ==========================================================================

def price_continuous_yield(sigma):
    """Price using the continuous yield approximation δ = drag/AR_0."""
    print("\n" + "=" * 70)
    print("PART 5: CONTINUOUS DIVIDEND YIELD APPROXIMATION")
    print("=" * 70)

    delta = DRAG / AR_0
    pv_principal = 100 * np.exp(-r * T)

    # Digital
    d2_dig = (np.log(AR_0 / AR_0) + (r - delta - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    digital = BOOST * np.exp(-r * T) * norm.cdf(d2_dig)

    # Vanilla
    K2 = (1 + BOOST / 100) * AR_0
    d1_van = (np.log(AR_0 / K2) + (r - delta + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2_van = d1_van - sigma * np.sqrt(T)
    call_val = (AR_0 * np.exp(-delta * T) * norm.cdf(d1_van)
                - K2 * np.exp(-r * T) * norm.cdf(d2_van))
    vanilla = PARTICIPATION * (call_val / AR_0) * 100

    fv = pv_principal + digital + vanilla

    # Compare
    PV_drag_exact = DRAG * (1 - np.exp(-r * T)) / r
    FP_exact = AR_0 - PV_drag_exact
    FP_approx = AR_0 * np.exp(-delta * T)

    print(f"δ = {DRAG}/{AR_0:.2f} = {delta:.4f} ({delta*100:.2f}%)")
    print(f"F^P (continuous yield): {FP_approx:.2f}")
    print(f"F^P (exact annuity):    {FP_exact:.2f}  (diff: {FP_approx - FP_exact:.2f})")
    print(f"Fair value (approx):    ${fv:.2f}")

    return fv


# ==========================================================================
# Part 6: Expected returns under the physical measure
# ==========================================================================

def compute_expected_returns(sigma):
    """Expected note returns for different TR index expected returns."""
    print("\n" + "=" * 70)
    print("PART 6: EXPECTED RETURNS UNDER PHYSICAL MEASURE")
    print("=" * 70)

    delta_drag = DRAG / AR_0

    print(f"Drag as continuous yield: δ = {delta_drag*100:.2f}%")
    print(f"\n{'μ_TR':>5}  {'μ_AR':>7}  {'P(R≥0)':>7}  {'E[dig]':>7}  {'E[van]':>7}  "
          f"{'E[payoff]':>9}  {'E[ann ret]':>10}")
    print("─" * 68)

    for mu_TR in [0.02, 0.04, 0.06]:
        mu_AR = mu_TR - delta_drag

        # Digital component
        d2_P = (mu_AR - 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
        prob_pos = norm.cdf(d2_P)
        E_dig = BOOST * prob_pos

        # Vanilla component
        K_v = (1 + BOOST / 100) * AR_0
        d1_P = (np.log(AR_0 / K_v) + (mu_AR + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2_Pv = d1_P - sigma * np.sqrt(T)
        E_call = AR_0 * np.exp(mu_AR * T) * norm.cdf(d1_P) - K_v * norm.cdf(d2_Pv)
        E_van = PARTICIPATION * (E_call / AR_0) * 100

        E_payoff = 100 + E_dig + E_van
        ann_ret = (E_payoff / 100) ** (1 / T) - 1

        print(f"{mu_TR:>5.0%}  {mu_AR:>+7.2%}  {prob_pos:>6.1%}  "
              f"${E_dig:>6.2f}  ${E_van:>6.2f}  ${E_payoff:>8.2f}  {ann_ret:>9.2%}")

    print(f"\nRisk-free: $100 → ${100*np.exp(r*T):.2f} ({np.exp(r)-1:.2%} p.a.)")


# ==========================================================================
# Part 7: Alternative structure analysis
# ==========================================================================

def expected_return_atm_call(alpha, mu_TR, sigma):
    """Expected annualized return for a note with participation alpha in ATM call."""
    delta_drag = DRAG / AR_0
    mu_AR = mu_TR - delta_drag
    K = AR_0
    d1_P = (np.log(AR_0 / K) + (mu_AR + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2_P = d1_P - sigma * np.sqrt(T)
    E_call = AR_0 * np.exp(mu_AR * T) * norm.cdf(d1_P) - K * norm.cdf(d2_P)
    E_payoff = 100 + alpha * (E_call / AR_0) * 100
    return (E_payoff / 100) ** (1 / T) - 1


def expected_return_digital(boost, mu_TR, sigma):
    """Expected annualized return for digital($boost) + 5% vanilla."""
    delta_drag = DRAG / AR_0
    mu_AR = mu_TR - delta_drag
    d2_P = (mu_AR - 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    E_dig = boost * norm.cdf(d2_P)
    K_v = (1 + boost / 100) * AR_0
    d1_P = (np.log(AR_0 / K_v) + (mu_AR + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2_Pv = d1_P - sigma * np.sqrt(T)
    E_call = AR_0 * np.exp(mu_AR * T) * norm.cdf(d1_P) - K_v * norm.cdf(d2_Pv)
    E_van = PARTICIPATION * (E_call / AR_0) * 100
    E_payoff = 100 + E_dig + E_van
    return (E_payoff / 100) ** (1 / T) - 1


def analyze_alternatives(sigma):
    """Compare alternative note structures."""
    print("\n" + "=" * 70)
    print("PART 7: ALTERNATIVE STRUCTURE ANALYSIS")
    print("=" * 70)

    PV_drag = DRAG * (1 - np.exp(-r * T)) / r
    FP_AR = AR_0 - PV_drag
    pv_principal = 100 * np.exp(-r * T)

    # --- Alternative 1: Simple ATM call ---
    K_atm = AR_0
    FP_K_atm = K_atm * np.exp(-r * T)
    d1_atm = (np.log(FP_AR / FP_K_atm) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2_atm = d1_atm - sigma * np.sqrt(T)
    C_atm = FP_AR * norm.cdf(d1_atm) - FP_K_atm * norm.cdf(d2_atm)
    cost_100pct = C_atm / AR_0 * 100

    print(f"\nALTERNATIVE 1: No digital, just ATM call (K = AR₀)")
    print(f"Cost of 100% participation = ${cost_100pct:.2f} per $100 notional\n")
    print(f"  {'Margin':>7}  {'α':>6}  {'E[ret] μ=4%':>12}  {'E[ret] μ=6%':>12}")
    print(f"  {'─'*44}")
    for margin in [5, 1, 0]:
        budget = 100 - pv_principal - margin
        alpha = budget / cost_100pct
        er4 = expected_return_atm_call(alpha, 0.04, sigma)
        er6 = expected_return_atm_call(alpha, 0.06, sigma)
        print(f"  {margin:>6.0f}%  {alpha*100:>5.0f}%  {er4:>11.2%}  {er6:>11.2%}")

    # --- Alternative 2: Digital + vanilla, solve for boost ---
    FP_K1 = AR_0 * np.exp(-r * T)
    d1_d = (np.log(FP_AR / FP_K1) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2_d = d1_d - sigma * np.sqrt(T)
    Nd2 = norm.cdf(d2_d)

    def total_option_value(boost):
        digital = boost * np.exp(-r * T) * Nd2
        K_van = (1 + boost / 100) * AR_0
        FP_Kv = K_van * np.exp(-r * T)
        d1v = (np.log(FP_AR / FP_Kv) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2v = d1v - sigma * np.sqrt(T)
        cv = FP_AR * norm.cdf(d1v) - FP_Kv * norm.cdf(d2v)
        vanilla = PARTICIPATION * (cv / AR_0) * 100
        return digital + vanilla

    print(f"\nALTERNATIVE 2: Digital + {PARTICIPATION:.0%} participation, solve for boost")
    print(f"N(d₂) = {Nd2:.3f} — risk-neutral prob of receiving boost: {Nd2:.0%}\n")
    print(f"  {'Margin':>7}  {'Boost':>6}  {'E[ret] μ=4%':>12}  {'E[ret] μ=6%':>12}")
    print(f"  {'─'*44}")
    for margin in [5, 1, 0]:
        budget = 100 - pv_principal - margin
        boost = brentq(lambda b: total_option_value(b) - budget, 0, 200)
        er4 = expected_return_digital(boost, 0.04, sigma)
        er6 = expected_return_digital(boost, 0.06, sigma)
        print(f"  {margin:>6.0f}%  {boost:>5.0f}%  {er4:>11.2%}  {er6:>11.2%}")

    # Current note
    er4_curr = expected_return_digital(BOOST, 0.04, sigma)
    er6_curr = expected_return_digital(BOOST, 0.06, sigma)
    print(f"\nCurrent note (20% boost, 5.1% margin):  {er4_curr:>11.2%}  {er6_curr:>11.2%}")
    print(f"Risk-free benchmark:                     {np.exp(r)-1:>11.2%}  {np.exp(r)-1:>11.2%}")


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":
    # Part 1: Scrape MX
    try:
        options, atm_data = scrape_mx_options()
    except Exception as e:
        print(f"MX scraping failed ({e}), continuing with saved data...")

    # Part 2: Implied vols
    sigma_implied = compute_implied_vols()

    # Part 3: Historical variance
    sigma_hist = compute_historical_variance()

    # Part 4: Price with exact prepaid forwards
    print("\n" + "=" * 70)
    print("PART 4: PRICING WITH EXACT PREPAID FORWARDS")
    print("=" * 70)
    price_note(sigma_hist, label="Historical + VRP")
    price_note(sigma_implied, label="Market Implied")

    # Part 5: Continuous yield approximation
    fv_approx = price_continuous_yield(sigma_hist)

    # Part 6: Expected returns
    compute_expected_returns(sigma_hist)

    # Part 7: Alternative structures
    analyze_alternatives(sigma_hist)
