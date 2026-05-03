import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Unfrozen, vlg_linear head, VLG warm-start, no outside penalty
alphas_vlg = [0.0, 0.1, 0.3, 0.6, 0.8, 1.0]
accs_vlg =   [0.7594, 0.7610, 0.7603, 0.7544, 0.7509, 0.7463]

# Unfrozen, spatial_pool head, VLG warm-start, no outside penalty
alphas_sp = [0.0, 0.1, 0.2, 0.4, 0.6, 1.0]
accs_sp =   [0.7604, 0.7620, 0.7610, 0.7570, 0.7528, 0.7499]

fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")

ax.plot(alphas_vlg, accs_vlg, "o-", color="#2196F3", linewidth=2.5, markersize=9,
        label="vlg_linear head", zorder=5)
for a, acc in zip(alphas_vlg, accs_vlg):
    offset = 12 if acc > 0.755 else -15
    ax.annotate(f"{acc:.2%}", (a, acc), textcoords="offset points",
                xytext=(0, offset), ha="center", fontsize=9, fontweight="bold", color="#2196F3")

ax.plot(alphas_sp, accs_sp, "D-", color="#FF9800", linewidth=2, markersize=9,
        label="spatial_pool head", zorder=6)
for a, acc in zip(alphas_sp, accs_sp):
    offset = -15 if a in (0.0, 0.8) else 12
    ha = "left" if a == 0.0 else "center"
    ax.annotate(f"{acc:.2%}", (a, acc), textcoords="offset points",
                xytext=(5 if a == 0.0 else 0, offset), ha=ha, fontsize=9, fontweight="bold", color="#FF9800")

ax.axhline(y=0.7594, color="gray", linestyle=":", linewidth=1, alpha=0.7)
ax.annotate("VLG-CBM baseline", (0.7, 0.7594), textcoords="offset points",
            xytext=(0, 8), ha="center", fontsize=8, color="gray")

ax.set_xlabel("Spatial coupling α", fontsize=12)
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("Effect of α on Classification Accuracy\n"
             "(unfrozen global head, VLG warm-start, soft_align, CUB-200)",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower left", fontsize=9)
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(0.744, 0.764)
ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "results", "alpha_ablation_unfrozen.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
