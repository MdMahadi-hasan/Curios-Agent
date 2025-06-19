import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# ------------------------- 1. Parameters -------------------------------- #
T = 200
N = 10_000
rho = 0.05
alpha = 1.3
bar_delta = 1.2
mu_C, sigma_C = 0.1, 0.5
p_h = 0.20
mu_H, sigma_H = 0.5, 0.7
lam = 0.000005
beta = 2
phi = 0.5

# ------------------- 1A. Expected-value constants (for B) --------------- #
E_delta_raw = alpha / (alpha - 1)                     # E[1+X] for Pareto(shape=1.3)
E_delta      = 0.4 * bar_delta * (E_delta_raw - 1)    # sign-adjusted mean benefit
E_C          = np.exp(mu_C + sigma_C**2 / 2)
E_H          = np.exp(mu_H + sigma_H**2 / 2)
B            = E_delta - E_C - p_h * E_H              # ≈ –0.074 < 0

# --------------------- 2. Memory Allocation ----------------------------- #
P_cur  = np.ones(N)
P_lazy = np.ones(N)
P_opt  = np.ones(N)
A_opt  = np.zeros(N, dtype=int)

crash_counts_cur  = np.zeros(N, dtype=int)
crash_counts_opt  = np.zeros(N, dtype=int)

rng = np.random.default_rng(42)

# --------------------- 3. Simulation Loop ------------------------------- #
for t in range(1, T + 1):

    # --- stochastic primitives shared by every agent this period -------- #
    C_t       = rng.lognormal(mu_C, sigma_C, N)
    Δ_raw     = rng.pareto(alpha, N) + 1.0             # support ≥1
    Δ_t       = bar_delta * (Δ_raw - 1)                # min 0, heavy right tail
    Δ_t      *= rng.choice([-1, 1], N, p=[0.30, 0.70]) # 30 % negative draws
    H_t       = np.zeros(N)
    mask_h    = rng.random(N) < p_h
    H_t[mask_h] = rng.lognormal(mu_H, sigma_H, mask_h.sum())

    # ---------- Curious agent (always adopts) --------------------------- #
    P_cur = (1 - rho) * P_cur + rho * (P_cur + Δ_t - C_t - H_t)
    crash_prob_cur = lam * (t ** beta)                 # A_cur = t
    mask_crash_cur = rng.random(N) < crash_prob_cur
    P_cur[mask_crash_cur] *= phi
    crash_counts_cur += mask_crash_cur.astype(int)

    # ---------- Lazy agent (never adopts) ------------------------------- #
    # No change (P_lazy stays at 1.0)

    # ---------- Optimal agent (threshold rule from Prop. 1) ------------- #
    denom        = lam * (1 - phi) * ((A_opt + 1) ** beta - A_opt ** beta)
    P_threshold  = (rho * B) / denom                   # same shape as A_opt
    adopt_mask   = P_opt <= P_threshold                # True if threshold met

    # performance update conditional on adoption
    increment    = adopt_mask * (Δ_t - C_t - H_t)
    P_opt        = (1 - rho) * P_opt + rho * (P_opt + increment)

    # tally adoptions
    A_opt       += adopt_mask.astype(int)

    # crash risk conditional on current adoption history
    crash_prob_opt = lam * (A_opt ** beta)
    mask_crash_opt = rng.random(N) < crash_prob_opt
    P_opt[mask_crash_opt] *= phi
    crash_counts_opt += mask_crash_opt.astype(int)

# --------------------- 4. Diagnostics ----------------------------------- #
def summary(P, lbl):
    pct = np.percentile(P, [1, 5, 50, 95, 99])
    return pd.Series({
        f'{lbl} Mean':      P.mean(),
        f'{lbl} Std Dev':   P.std(ddof=1),
        f'{lbl} P1':        pct[0],
        f'{lbl} P5':        pct[1],
        f'{lbl} Median':    pct[2],
        f'{lbl} P95':       pct[3],
        f'{lbl} P99':       pct[4],
        f'{lbl} ES 5%':     P[P <= pct[1]].mean(),
        f'{lbl} Sharpe':    P.mean() / P.std(ddof=1) if P.std(ddof=1) else np.nan,
        f'{lbl} Skew':      skew(P),
        f'{lbl} Kurtosis':  kurtosis(P, fisher=False)
    })

out = pd.concat([
    summary(P_cur,  "Curious"),
    summary(P_lazy, "Lazy"),
    summary(P_opt,  "Optimal"),
    pd.Series({
        "Pr(Lazy > Curious)":     np.mean(P_lazy > P_cur),
        "Pr(Optimal > Curious)":  np.mean(P_opt  > P_cur),
        "Pr(Optimal > Lazy)":     np.mean(P_opt  > P_lazy),
        "Crash Freq (Curious)":   np.mean(crash_counts_cur > 0),
        "Crash Freq (Optimal)":   np.mean(crash_counts_opt > 0),
        "Avg Crashes (Curious)":  crash_counts_cur.mean(),
        "Avg Crashes (Optimal)":  crash_counts_opt.mean()
    })
])

print(f"\n=== Monte-Carlo Summary (N = {N:,}, T = {T}) ===")
print(out.to_string(float_format="%.4f"))

# --------------------- 5. CDF Plot -------------------------------------- #
cdf = np.linspace(0, 1, N)
plt.figure(figsize=(6, 4))
plt.plot(np.sort(P_cur),  cdf, label="Curious",  color="#2c7fb8")
plt.plot(np.sort(P_lazy), cdf, label="Lazy",    ls="--", color="#f46d43")
plt.plot(np.sort(P_opt),  cdf, label="Optimal", ls="-.", color="#4daf4a")
plt.xlabel(r"Terminal performance $P_T$")
plt.ylabel("Cumulative probability")
plt.legend()
plt.tight_layout()
plt.savefig("CDF_with_optimal.png", dpi=300)
plt.show()
