"""
Animation of Black-Scholes call option price as time-to-maturity shrinks
from 6 months to 0, one frame per calendar day, 6 fps.
"""

from pathlib import Path

import imageio_ffmpeg
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# ── Parameters ──────────────────────────────────────────────────────
K = 100        # strike
r = 0.05       # risk-free rate
sigma = 0.20   # volatility
T_start = 0.5  # 6 months

S = np.linspace(60, 150, 500)

days_total = int(T_start * 365)  # ~182 frames
taus = np.array([(days_total - d) / 365 for d in range(days_total + 1)])


def bs_call(S, K, tau, r, sigma):
    """Black-Scholes call price. Returns intrinsic value at tau=0."""
    if tau <= 0:
        return np.maximum(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


# ── Figure setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
# Ensure even pixel dimensions for h264
w, h = fig.get_size_inches() * 150
fig.set_size_inches(w // 2 * 2 / 150, h // 2 * 2 / 150)
fig.set_facecolor("white")

(line_bs,) = ax.plot([], [], lw=2.5, color="#1f77b4", label="BS call price")
(line_iv,) = ax.plot(S, np.maximum(S - K, 0), lw=1.2, ls="--", color="grey",
                     alpha=0.6, label="Intrinsic value")

ax.set_xlim(S[0], S[-1])
ax.set_ylim(-2, 55)
ax.set_xlabel("Stock price $S$", fontsize=12)
ax.set_ylabel("Call price $C(S, \\tau)$", fontsize=12)
ax.axvline(K, color="grey", lw=0.7, ls=":")
ax.legend(loc="upper left", fontsize=11)

title = ax.set_title("", fontsize=13)
time_text = ax.text(0.97, 0.93, "", transform=ax.transAxes,
                    ha="right", va="top", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)


def init():
    line_bs.set_data([], [])
    title.set_text("")
    time_text.set_text("")
    return line_bs, title, time_text


def animate(i):
    tau = taus[i]
    C = bs_call(S, K, tau, r, sigma)
    line_bs.set_data(S, C)
    days_left = days_total - i
    title.set_text(
        f"Black-Scholes Call  |  K={K},  r={r:.0%},  σ={sigma:.0%}"
    )
    time_text.set_text(f"Day {i}  |  {days_left} days to maturity")
    return line_bs, title, time_text


anim = FuncAnimation(fig, animate, init_func=init,
                     frames=len(taus), interval=1000 / 6, blit=True)

outfile = Path(__file__).with_name("bs_call_animation.mov")
print(f"Saving {len(taus)} frames at 6 fps → {outfile} ...")
anim.save(outfile, writer="ffmpeg", fps=6, dpi=150)
print("Done.")
