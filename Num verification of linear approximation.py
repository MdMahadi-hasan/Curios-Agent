import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Calibration ----------------------------------------------------------
rho       = 0.10
alpha     = 1.55
bar_delta = 1.15

p_h       = 0.25
mu_H, sigma_H = 0.6, 0.8

lam   = 5.00e-06
beta  = 2
phi   = 0.92

p_s        = 0.02
mu_S, sigma_S = 1.00, 0.9

mu_C_hw,    sigma_C_hw    = -1.69, 0.50
mu_C_pers,  sigma_C_pers  = -0.83, 0.40
mu_C_data,  sigma_C_data  = -1.69, 0.40
mu_C_maint, sigma_C_maint = -1.10, 0.50

# ---- Helpers --------------------------------------------------------------
def lognorm_mean(mu, sigma):
    return np.exp(mu + 0.5 * sigma**2)

# ---- Expected primitives --------------------------------------------------
# (using the “no-α multiplier” version you supplied)
E_delta = bar_delta / (alpha - 1)

E_C = sum([
    lognorm_mean(mu_C_hw,    sigma_C_hw),
    lognorm_mean(mu_C_pers,  sigma_C_pers),
    lognorm_mean(mu_C_data,  sigma_C_data),
    lognorm_mean(mu_C_maint, sigma_C_maint),
])

E_H = lognorm_mean(mu_H, sigma_H)
E_S = lognorm_mean(mu_S, sigma_S)

# Net expected gain per adoption
B = E_delta - E_C - p_h * E_H - p_s * E_S

# ---- Threshold functions ---------------------------------------------------
def P_closed(A):
    return (rho * B) / (lam * (1 - phi) * ((A + 1)**beta - A**beta))

def P_exact(A):
    factor = 1 - lam * (A + 1)**beta
    denom  = lam * (1 - phi) * ((A + 1)**beta - A**beta)
    return (rho * B * factor) / denom

# ---- Compute thresholds ----------------------------------------------------
A_vals = np.arange(0, 401)
df = pd.DataFrame({
    "A":        A_vals,
    "P_exact":  [P_exact(a)  for a in A_vals],
    "P_closed": [P_closed(a) for a in A_vals],
})
df["Abs_Error"]   = df["P_exact"] - df["P_closed"]
df["Rel_Error_%"] = 100 * df["Abs_Error"] / df["P_exact"]

# ---- Plot ------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(df["A"], df["P_exact"], label="Exact (Bellman)")
plt.plot(df["A"], df["P_closed"], linestyle="--", label="Closed-form approx.")
plt.xlabel("Cumulative adoptions $A_t$")
plt.ylabel("Adoption threshold $P^*(A_t)$")
plt.title("Adoption thresholds (no ace_tools)")
plt.legend()
plt.tight_layout()
plt.show()

# ---- Quick look at the first 25 rows ---------------------------------------
print("\nThreshold comparison (first 25 A):")
print(df.head(25).to_string(index=False))
