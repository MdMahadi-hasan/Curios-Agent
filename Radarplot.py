# ------------------------------------------------------------------
# F. RADAR PLOT — MODERN STYLE: Sensitivity spread of Pr(Lazy > Curious)
# ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# 1. Compute sensitivity spread ΔPr = max Pr – min Pr
sens = (
    results
    .groupby('Param')['PrLazy>Cur']
    .agg(lambda x: x.max() - x.min())
    .reindex(list(sweep.keys()))
)

# 2. Map to LaTeX labels
latex_map = {
    'alpha': r'$\alpha$',
    'lam'  : r'$\lambda$',
    'rho'  : r'$\rho$',
    'phi'  : r'$\phi$',
    'p_h'  : r'$p_h$',
    'beta' : r'$\beta$'
}
labels = [latex_map[p] for p in sens.index]

# 3. Prepare circular data
values = sens.values.tolist()
values += values[:1]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

# 4. Modern color palette
line_color = "#005F73"
fill_color = "#0A9396"

# 5. Start polar plot
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("white")

# Plot line & fill
ax.plot(angles, values,
        color=line_color,
        linewidth=3,
        linestyle="-",
        zorder=3)
ax.fill(angles, values,
        color=fill_color,
        alpha=0.3,
        zorder=2)

# Scatter markers
ax.scatter(angles[:-1], sens.values,
           color=line_color,
           s=70,
           edgecolors='white',
           linewidths=1.5,
           zorder=4)

# 6. Ticks and labels
ax.set_theta_offset(np.pi / 2)   # start at 12 o'clock
ax.set_theta_direction(-1)       # clockwise

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels,
                   fontsize=12,
                   fontweight='bold',
                   color="#001219")

# Radial grid
max_range = round(sens.max() + 0.02, 2)
yticks = np.linspace(0, max_range, 4)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{y:.2f}" for y in yticks],
                   fontsize=10,
                   color="#333")
ax.yaxis.grid(True,
              linestyle="--",
              linewidth=0.5,
              color="#ccc")
ax.xaxis.grid(True,
              linestyle="--",
              linewidth=0.6,
              color="#bbb")

# 7. Title with math
ax.set_title(
    r"Sensitivity Spread of $\Pr(\mathrm{Lazy} > \mathrm{Curious})$" "\nAcross One-at-a-Time Sweeps",
    size=15,
    weight='bold',
    pad=25,
    color="#005F73"
)

plt.tight_layout()
plt.savefig("OAT_sensitivity_radar_modern.png", dpi=300)
print("Plot saved: OAT_sensitivity_radar_modern.png")
