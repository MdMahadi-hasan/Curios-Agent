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

# --------------------- 2. Memory Allocation ----------------------------- #
P_cur = np.ones(N)
P_lazy = np.ones(N)
crash_counts = np.zeros(N, dtype=int)
rng = np.random.default_rng(42)

# --------------------- 3. Simulation Loop ------------------------------- #
for t in range(1, T + 1):
    C_t = rng.lognormal(mu_C, sigma_C, N)
    Delta_raw = rng.pareto(alpha, N) + 1.0
    Delta_t = bar_delta * Delta_raw - bar_delta
    Delta_t *= rng.choice([-1, 1], N, p=[0.30, 0.70])

    H_t = np.zeros(N)
    mask_h = rng.random(N) < p_h
    H_t[mask_h] = rng.lognormal(mu_H, sigma_H, mask_h.sum())

    # Curious agent update
    P_cur = (1 - rho) * P_cur + rho * (P_cur + Delta_t - C_t - H_t)

    # Crash event
    crash_prob = lam * t ** beta
    mask_crash = rng.random(N) < crash_prob
    P_cur[mask_crash] *= phi
    crash_counts += mask_crash.astype(int)

    # Lazy agent update
    P_lazy = (1 - rho) * P_lazy + rho * P_lazy  # constant adjustment

# --------------------- 4. Diagnostics ----------------------------------- #
def agent_summary(P, label):
    pctls = np.percentile(P, [1, 5, 50, 95, 99])
    return pd.Series({
        f"{label} Mean": P.mean(),
        f"{label} Std Dev": P.std(ddof=1),
        f"{label} P1": pctls[0],
        f"{label} P5": pctls[1],
        f"{label} Median": pctls[2],
        f"{label} P95": pctls[3],
        f"{label} P99": pctls[4],
        f"{label} ES 5%": P[P <= pctls[1]].mean(),
        f"{label} Sharpe": P.mean() / P.std(ddof=1),
        f"{label} Skewness": skew(P),
        f"{label} Kurtosis": kurtosis(P, fisher=False),
    })

summary_cur = agent_summary(P_cur, "Curious")
summary_lazy = agent_summary(P_lazy, "Lazy")

# Additional comparison metrics
comparison = pd.Series({
    "Pr(Lazy > Curious)": np.mean(P_lazy > P_cur),
    "Crash Freq (Curious)": np.mean(crash_counts > 0),
    "Avg Crashes (Curious)": crash_counts.mean()
})

# Merge all summaries
full_summary = pd.concat([summary_cur, summary_lazy, comparison])

# --------------------- 5. Print Results --------------------------------- #
print("\n=== Monte-Carlo Summary (N = {:,}, T = {}) ===".format(N, T))
print(full_summary.to_string(float_format="{:0.4f}".format))

# --------------------- 6. CDF Plot -------------------------------------- #
sorted_cur = np.sort(P_cur)
sorted_lazy = np.sort(P_lazy)
cdf = np.linspace(0.0, 1.0, N)

plt.figure(figsize=(6, 4))
plt.plot(sorted_cur, cdf, label="Curious Agent", color="#2c7fb8")   # modern teal-blue
plt.plot(sorted_lazy, cdf, label="Lazy Agent", linestyle='--', color="#f46d43")  # modern coral-orange
plt.xlabel(r"Terminal performance $P_T$")
plt.ylabel("Cumulative probability")
plt.legend()
plt.tight_layout()


# Save the figure in high resolution (e.g., 300 DPI)
plt.savefig("CDF.png", dpi=300)

plt.show()
