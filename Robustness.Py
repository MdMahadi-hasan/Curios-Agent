import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import matplotlib as mpl   # <-- used only in Section E

# ------------------------------------------------------------------
# A.  CORE SIMULATION FUNCTION
# ------------------------------------------------------------------
def simulate(params, rng_seed=42):
    """
    Runs one Monte-Carlo simulation of the Curious vs Lazy agents.
    Returns a dict of summary statistics.
    """
    # Unpack --------------------------------------------------------
    T, N          = params['T'], params['N']
    rho           = params['rho']
    alpha         = params['alpha']
    bar_delta     = params['bar_delta']
    mu_C          = params['mu_C']
    sigma_C       = params['sigma_C']
    p_h           = params['p_h']
    mu_H          = params['mu_H']
    sigma_H       = params['sigma_H']
    lam           = params['lam']
    beta          = params['beta']
    phi           = params['phi']

    # 1. Memory allocation -----------------------------------------
    P_cur   = np.ones(N)
    P_lazy  = np.ones(N)
    crashes = np.zeros(N, dtype=int)
    rng     = np.random.default_rng(rng_seed)

    # 2. Simulation loop -------------------------------------------
    for t in range(1, T + 1):
        C_t         = rng.lognormal(mu_C, sigma_C, N)
        Delta_raw   = rng.pareto(alpha, N) + 1.0
        Delta_t     = bar_delta * Delta_raw - bar_delta
        Delta_t    *= rng.choice([-1, 1], N, p=[0.30, 0.70])

        H_t         = np.zeros(N)
        mask_h      = rng.random(N) < p_h
        H_t[mask_h] = rng.lognormal(mu_H, sigma_H, mask_h.sum())

        # Curious update
        P_cur = (1 - rho) * P_cur + rho * (P_cur + Delta_t - C_t - H_t)

        # Crash
        crash_prob  = lam * t ** beta
        mask_crash  = rng.random(N) < crash_prob
        P_cur[mask_crash] *= phi
        crashes    += mask_crash.astype(int)

        # Lazy update
        P_lazy = (1 - rho) * P_lazy + rho * P_lazy

    # 3. Diagnostics ------------------------------------------------
    pctls_cur = np.percentile(P_cur, [1, 5, 50, 95, 99])
    summary = {
        'MeanCur'   : P_cur.mean(),
        'MedianCur' : pctls_cur[2],
        'ES5Cur'    : P_cur[P_cur <= pctls_cur[1]].mean(),
        'SharpeCur' : P_cur.mean() / P_cur.std(ddof=1),
        'SkewCur'   : skew(P_cur),
        'KurtCur'   : kurtosis(P_cur, fisher=False),
        'PrLazy>Cur': np.mean(P_lazy > P_cur),
        'AvgCrashes': crashes.mean()
    }
    return summary


# ------------------------------------------------------------------
# B.  BASELINE & SWEEP DEFINITIONS
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

# Parameter ranges to sweep ----------------------------------------
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
        summ = simulate(params, rng_seed=42)
        summ['Param'] = param
        summ['Value'] = v
        records.append(summ)

results = pd.DataFrame(records)


# ------------------------------------------------------------------
# D.  SAVE SUMMARY TABLE
# ------------------------------------------------------------------
results.to_csv("OAT_summary.csv", index=False)
print("\n===== OAT SENSITIVITY SUMMARY =====")
print(results.to_string(index=False))


# ------------------------------------------------------------------
# E.  VISUALISATIONS  ── publication-ready 3×2 composite figure
# ------------------------------------------------------------------
# 1) Global style (serif font, thicker lines, boxed axes, no grid)
mpl.rcParams.update({
    "font.family"       : "serif",
    "font.size"         : 12,
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 12,
    "axes.grid"         : False,
    "axes.spines.top"   : True,   # keep all four spines visible
    "axes.spines.right" : True,
    "lines.linewidth"   : 2.0,
})

# 2) Mapping from parameter names to math symbols for labels
param_labels = {
    "alpha": r"$\alpha$",
    "lam"  : r"$\lambda$",
    "rho"  : r"$\rho$",
    "phi"  : r"$\phi$",
    "p_h"  : r"$p_h$",
    "beta" : r"$\beta$",
}

# 3) Subdued colour palette (Tableau 10 works well in print/PDF)
palette = ["#4E79A7", "#F28E2B", "#76B7B2",
           "#E15759", "#59A14F", "#EDC948"]

# 4) Create a single 3 × 2 panel figure
fig, axes = plt.subplots(nrows=3,
                         ncols=2,
                         figsize=(7, 9),
                         sharey=False)

panel_letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for i, param in enumerate(sweep.keys()):
    ax      = axes.flatten()[i]
    subset  = results[results["Param"] == param]

    ax.plot(subset["Value"],
            subset["PrLazy>Cur"],
            marker="o",
            color=palette[i % len(palette)])

    # Single math-symbol x-label (no redundant title)
    ax.set_xlabel(param_labels.get(param, param))
    ax.set_ylabel(r"$\Pr(\mathrm{Lazy} > \mathrm{Curious})$")

    # Panel identifier in upper-left
    ax.text(0.02, 0.96, panel_letters[i],
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="top", ha="left")

    # Make sure all four spines are visible
    for spine in ax.spines.values():
        spine.set_visible(True)

fig.tight_layout()
fig.savefig("Sensitivitylineplot.png", dpi=500)

print("\nComposite figure saved: Sensitivitylineplot.png")


# ------------------------------------------------------------------
# (Optional) CDF overlay for extreme ends of a parameter
# ------------------------------------------------------------------
# Example: lowest vs highest alpha – left here in case you need it,
# but commented out to keep the script fast.  Uncomment if desired.

# alpha_low  = baseline.copy(); alpha_low['alpha']  = sweep['alpha'][0]
# alpha_high = baseline.copy(); alpha_high['alpha'] = sweep['alpha'][-1]
# cur_low  = simulate(alpha_low , rng_seed=7)
# cur_high = simulate(alpha_high, rng_seed=8)
# # Re-plot CDFs using the same style settings if needed.
