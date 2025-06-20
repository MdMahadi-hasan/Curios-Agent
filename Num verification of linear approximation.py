import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Baseline parameters (provided by the user) ----
rho = 0.05
alpha = 1.3
bar_delta = 1.2
mu_C, sigma_C = 0.1, 0.5
p_h = 0.20
mu_H, sigma_H = 0.5, 0.7
lam = 5e-6
beta = 2
phi = 0.5

# ---- Derived primitives ----
# Expected benefit Δ (Pareto with minimum bar_delta and shape alpha)
E_delta = alpha * bar_delta / (alpha - 1)

# Expected cost C (Log‑normal)
E_C = np.exp(mu_C + (sigma_C ** 2) / 2)

# Expected hallucination penalty H (Log‑normal)
E_H = np.exp(mu_H + (sigma_H ** 2) / 2)

# Net expected gain of *one* adoption
B = E_delta - E_C - p_h * E_H

# ---- Closed‑form threshold from Lemma 6 / Proposition 1 ----
def P_closed(A):
    """Analytical approximation P*(A) from Lemma 6"""
    return (rho * B) / (lam * (1 - phi) * ((A + 1) ** beta - A ** beta))

# ---- “Exact” threshold from the full indifference condition (no small‑λ approximation) ----
def P_exact(A):
    factor = (1 - lam * (A + 1) ** beta)
    denom = lam * (1 - phi) * ((A + 1) ** beta - A ** beta)
    return (rho * B * factor) / denom

# ---- Compute over a relevant range of adoption counts ----
A_vals = np.arange(0, 401)  # 0 … 400 keeps crash‑prob < 1 (λA² ≤ 1)
df = pd.DataFrame({
    "A": A_vals,
    "P_exact": [P_exact(a) for a in A_vals],
    "P_closed": [P_closed(a) for a in A_vals],
})
df["Abs_Error"] = df["P_exact"] - df["P_closed"]
df["Rel_Error_%"] = 100 * df["Abs_Error"] / df["P_exact"]

# ---- Plot exact vs approximate thresholds ----
plt.figure(figsize=(6, 4))
plt.plot(df["A"], df["P_exact"], label="Exact (Bellman)")
plt.plot(df["A"], df["P_closed"], linestyle="--", label="Closed‑form approx.")
plt.xlabel("Cumulative adoptions $A_t$")
plt.ylabel("Adoption threshold $P^*(A_t)$")
plt.title("Analytical vs. numerical adoption thresholds")
plt.legend()
plt.tight_layout()
plt.show()

# ---- Show the first few rows of the table ----
import ace_tools as tools; tools.display_dataframe_to_user(name="Threshold comparison (first 25 A)", dataframe=df.head(25))
