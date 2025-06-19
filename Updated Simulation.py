import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ------------------------------------------------------------------
# A.  CORE SIMULATION FUNCTION (now Curious · Lazy · Optimal)
# ------------------------------------------------------------------
def simulate(params, rng_seed=42):
    """
    Monte-Carlo simulation of three agents:
        • Curious  (always adopts)
        • Lazy     (never adopts)
        • Optimal  (Prop-1 threshold)
    Returns a dict of summary statistics.
    """
    # Unpack --------------------------------------------------------
    (T, N, rho, alpha, bar_delta,
     mu_C, sigma_C, p_h, mu_H, sigma_H,
     lam, beta, phi) = (
        params['T'],     params['N'],     params['rho'],
        params['alpha'], params['bar_delta'],
        params['mu_C'],  params['sigma_C'],
        params['p_h'],   params['mu_H'],  params['sigma_H'],
        params['lam'],   params['beta'],  params['phi'])

    # ---- Expected values needed for B ----------------------------
    E_delta_raw = alpha / (alpha - 1)              # E[1 + X],  X~Pareto(α)
    E_delta     = 0.4 * bar_delta * (E_delta_raw - 1)  # 70 % positive, 30 % negative
    E_C         = np.exp(mu_C + sigma_C**2 / 2)
    E_H         = np.exp(mu_H + sigma_H**2 / 2)
    B           = E_delta - E_C - p_h * E_H        # per-period net benefit

    # 1. Memory allocation -----------------------------------------
    P_cur  = np.ones(N)
    P_lazy = np.ones(N)
    P_opt  = np.ones(N)
    A_opt  = np.zeros(N, dtype=int)

    crashes_cur = np.zeros(N, dtype=int)
    crashes_opt = np.zeros(N, dtype=int)

    rng = np.random.default_rng(rng_seed)

    # 2. Simulation loop -------------------------------------------
    for t in range(1, T + 1):

        # --- primitives (shared draws so agents differ only by actions) --
        C_t       = rng.lognormal(mu_C, sigma_C, N)
        Δ_raw     = rng.pareto(alpha, N) + 1.0
        Δ_t       = bar_delta * (Δ_raw - 1)             # min 0
        Δ_t      *= rng.choice([-1, 1], N, p=[0.30, 0.70])

        H_t       = np.zeros(N)
        mask_h    = rng.random(N) < p_h
        H_t[mask_h] = rng.lognormal(mu_H, sigma_H, mask_h.sum())

        # ---- Curious (always adopt) ----------------------------------
        P_cur = (1 - rho) * P_cur + rho * (P_cur + Δ_t - C_t - H_t)
        crash_prob_cur = lam * (t ** beta)              # A_cur = t
        mask_crash_cur = rng.random(N) < crash_prob_cur
        P_cur[mask_crash_cur] *= phi
        crashes_cur += mask_crash_cur.astype(int)

        # ---- Lazy (never adopt) --------------------------------------
        # P_lazy stays constant at 1 (no update needed)

        # ---- Optimal (threshold rule) --------------------------------
        denom        = lam * (1 - phi) * ((A_opt + 1) ** beta - A_opt ** beta)
        P_thresh     = (rho * B) / denom
        adopt_mask   = P_opt <= P_thresh                # True ➜ adopt this period

        inc_opt      = adopt_mask * (Δ_t - C_t - H_t)   # zero if no adoption
        P_opt        = (1 - rho) * P_opt + rho * (P_opt + inc_opt)

        A_opt       += adopt_mask.astype(int)

        crash_prob_opt = lam * (A_opt ** beta)
        mask_crash_opt = rng.random(N) < crash_prob_opt
        P_opt[mask_crash_opt] *= phi
        crashes_opt += mask_crash_opt.astype(int)

    # 3. Diagnostics ------------------------------------------------
    pct_cur = np.percentile(P_cur,  [1, 5, 50, 95, 99])
    pct_opt = np.percentile(P_opt,  [1, 5, 50, 95, 99])

    summary = {
        # --- Curious metrics ----------------------------------------
        'MeanCur'   : P_cur.mean(),
        'MedianCur' : pct_cur[2],
        'ES5Cur'    : P_cur[P_cur <= pct_cur[1]].mean(),
        'SharpeCur' : P_cur.mean() / P_cur.std(ddof=1),
        'PrLazy>Cur': np.mean(P_lazy > P_cur),

        # --- Optimal vs others --------------------------------------
        'MeanOpt'   : P_opt.mean(),
        'MedianOpt' : pct_opt[2],
        'PrOpt>Cur' : np.mean(P_opt  > P_cur),
        'PrOpt>Lazy': np.mean(P_opt  > P_lazy),

        # --- Crash diagnostics --------------------------------------
        'AvgCrashCur': crashes_cur.mean(),
        'AvgCrashOpt': crashes_opt.mean(),
    }
    return summary


# ------------------------------------------------------------------
# B.  BASELINE & SWEEP DEFINITIONS  (unchanged)
# ------------------------------------------------------------------
baseline = dict(
    T=200, N=10_000,
    rho=0.05,
    alpha=1.3,
    bar_delta=1.2,
    mu_C=0.1, sigma_C=0.5,
    p_h=0.20,
    mu_H=0.5, sigma_H=0.7,
    lam=5e-6,
    beta=2,
    phi=0.5,
)

sweep = {
    'alpha' : [1.1, 1.3, 1.5, 1.7],
    'lam'   : [2e-6, 5e-6, 1e-5, 2e-5],
    'rho'   : [0.01, 0.05, 0.10],
    'phi'   : [0, 0.2, 0.5, 0.95],
    'p_h'   : [0, 0.10, 0.20, 0.50],
    'beta'  : [1, 2, 3, 4]
}

# ------------------------------------------------------------------
# C.  OAT DRIVER
# ------------------------------------------------------------------
records = []
for param, values in sweep.items():
    for v in values:
        params = baseline.copy()
        params[param] = v
        rec = simulate(params, rng_seed=42)
        rec['Param'] = param
        rec['Value'] = v
        records.append(rec)

results = pd.DataFrame(records)

# ------------------------------------------------------------------
# D.  SAVE SUMMARY TABLE
# ------------------------------------------------------------------
results.to_csv("OAT_summary_with_optimal.csv", index=False)
print("\n===== OAT SENSITIVITY SUMMARY (Curious · Lazy · Optimal) =====")
print(results.to_string(index=False))




import numpy as np
import matplotlib.pyplot as plt

sorted_cur  = np.sort(P_cur)
sorted_lazy = np.sort(P_lazy)
sorted_opt  = np.sort(P_opt)
cdf = np.linspace(0.0, 1.0, N)

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

ax.plot(sorted_cur,  cdf, label="Curious Agent",  color="#2c7fb8")
ax.plot(sorted_lazy, cdf, label="Lazy Agent",     linestyle='--', color="red")   # now bright red
ax.plot(sorted_opt,  cdf, label="Optimal Agent",  linestyle='-.', color="#4daf4a")

ax.set_xlabel(r"Terminal performance $P_T$")
ax.set_ylabel("Cumulative probability")

ax.legend(
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

fig.savefig("CDFDad.png", dpi=300, bbox_inches='tight')

