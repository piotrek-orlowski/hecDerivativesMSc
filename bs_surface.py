"""
3D Black-Scholes call price surface C(S, τ) with a simulated GBM path
projected onto the surface.
"""

from pathlib import Path

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ── Parameters ──────────────────────────────────────────────────────
K = 100
r = 0.05
sigma = 0.20
T = 0.5  # 6 months
S0 = 100
n_days = int(T * 365)
dt = T / n_days

np.random.seed(42)


def bs_call(S, K, tau, r, sigma):
    tau = np.maximum(tau, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


# ── Surface grid ────────────────────────────────────────────────────
S_grid = np.linspace(60, 150, 200)
tau_grid = np.linspace(0.005, T, 200)
S_mesh, tau_mesh = np.meshgrid(S_grid, tau_grid)
C_mesh = bs_call(S_mesh, K, tau_mesh, r, sigma)

# ── Simulate GBM paths ──────────────────────────────────────────────
tau_path = np.linspace(T, 0, n_days + 1)
seeds = [42, 64, 99]
colors = ["black", "#d62728", "#2ca02c"]
labels = ["Path 1", "Path 2", "Path 3"]
paths = []

for seed in seeds:
    rng = np.random.RandomState(seed)
    Z = rng.randn(n_days)
    S_p = np.zeros(n_days + 1)
    S_p[0] = S0
    for i in range(n_days):
        S_p[i + 1] = S_p[i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i])
    C_p = bs_call(S_p, K, np.maximum(tau_path, 1e-10), r, sigma)
    paths.append((S_p, C_p))

# ── Plot ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 8), dpi=150)
ax = fig.add_subplot(111, projection="3d")

# Surface
ax.plot_surface(S_mesh, tau_mesh, C_mesh, cmap="coolwarm", alpha=0.55,
                edgecolor="none", rstride=4, cstride=4)

# Payoff at maturity (τ = 0)
payoff = np.maximum(S_grid - K, 0)
ax.plot(S_grid, np.zeros_like(S_grid), payoff, color="#1f77b4", lw=2.5,
        zorder=12, label="Payoff at maturity")

# Cross-section slices at τ = 0.1, 0.2, ..., 0.5
for tau_slice in np.arange(0.1, T + 0.05, 0.1):
    C_slice = bs_call(S_grid, K, tau_slice, r, sigma)
    ax.plot(S_grid, np.full_like(S_grid, tau_slice), C_slice,
            color="lightblue", lw=1.2, ls="--", alpha=0.85, zorder=5)

# GBM trajectories on the surface
for (S_p, C_p), color, label in zip(paths, colors, labels):
    ax.plot(S_p, tau_path, C_p, color=color, lw=2.2, zorder=10,
            label=f"{label} (S_T={S_p[-1]:.1f})")
    ax.scatter(*[[v] for v in (S_p[0], tau_path[0], C_p[0])],
               color=color, s=50, zorder=11, marker="o")
    ax.scatter(*[[v] for v in (S_p[-1], tau_path[-1], C_p[-1])],
               color=color, s=50, zorder=11, marker="D")
    ax.plot(S_p, tau_path, np.zeros_like(C_p), color=color, lw=0.6,
            ls="--", alpha=0.35)

ax.set_xlabel("Stock price $S$", fontsize=11, labelpad=10)
ax.set_ylabel("Time to maturity $\\tau$", fontsize=11, labelpad=10)
ax.set_zlabel("Call price $C$", fontsize=11, labelpad=8)
ax.set_title("Black-Scholes Call Surface with GBM Sample Path", fontsize=13, pad=15)
ax.legend(loc="upper left", fontsize=9)

ax.view_init(elev=25, azim=-140)
fig.tight_layout()

outfile = Path(__file__).with_name("bs_surface.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved → {outfile}")
