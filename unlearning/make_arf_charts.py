import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(BASE, "probe_results_new", "reprobe2")
OUT = os.path.join(BASE, "arf_analysis_out")
os.makedirs(OUT, exist_ok=True)

summary = pd.read_csv(os.path.join(IN, "summary.csv"))
curves = pd.read_csv(os.path.join(IN, "fewshot_curves.csv"))

ORDER = ["baseline", "wic_unmasked@800", "random_label@800", "amnesiac@800",
         "delete@800", "fisher@800", "arf@800", "wic_mgda@800", "rl_mgda@800"]
LABEL = {
    "baseline": "Baseline\n(nothing unlearned)",
    "wic_unmasked@800": "WIC",
    "random_label@800": "Random\nLabeling",
    "amnesiac@800": "Amnesiac",
    "delete@800": "DELETE",
    "fisher@800": "Fisher\n(broken)",
    "arf@800": "ARF\n(new)",
    "wic_mgda@800": "WIC\n+MGDA",
    "rl_mgda@800": "Random Label\n+MGDA",
}
COL = {
    "baseline": "#B8BCC8",
    "wic_unmasked@800": "#3B6FA0", "random_label@800": "#3B6FA0",
    "amnesiac@800": "#3B6FA0", "delete@800": "#3B6FA0",
    "fisher@800": "#3A3D46",
    "arf@800": "#2E9E6B",
    "wic_mgda@800": "#8A5EC0", "rl_mgda@800": "#8A5EC0",
}

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#888", "axes.labelcolor": "#222",
    "text.color": "#222", "xtick.color": "#222", "ytick.color": "#222",
})

# ── Fig 1: 1-shot CLS-probe residual gap (forget - test_reference), all 9 arms ──
cls1 = summary[(summary["feat"] == "cls") & (summary["shots_lowshot"] == 1)]
piv = cls1.pivot(index="arm", columns="group", values="acc_lowshot")
piv = piv.reindex(ORDER)
gap = piv["forget"] - piv["test_reference"]

fig, ax = plt.subplots(figsize=(10, 5.5))
colors = [COL[a] for a in ORDER]
bars = ax.bar(range(len(ORDER)), gap.values, color=colors, width=0.62)
ax.axhline(0, color="#888", linewidth=1)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([LABEL[a] for a in ORDER], fontsize=9.5)
ax.set_ylabel("Residual gap: forget-writer probe acc. minus\ngenuinely-unseen-writer probe acc. (1-shot, pts)")
ax.set_title("How much writer identity survives a FRESH post-hoc probe\n(0 = indistinguishable from a writer the model never saw at all)")
for i, (a, v) in enumerate(zip(ORDER, gap.values)):
    ax.text(i, v + (0.6 if v >= 0 else -1.4), f"{v:+.1f}", ha="center", fontsize=9, color="#333")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(-6, 42)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_residual_gap_all_methods.png"), dpi=160, bbox_inches="tight")
plt.close()

# ── Fig 2: forget vs retain_control probe accuracy scatter -- separates
#    genuine targeted forgetting from general representational collapse ──
fig, ax = plt.subplots(figsize=(7.5, 6.5))
for a in ORDER:
    x = piv.loc[a, "retain_control"]
    y = piv.loc[a, "forget"]
    ax.scatter(x, y, s=170, color=COL[a], edgecolor="white", linewidth=1.3, zorder=3)
    dx, dy = 1.6, 1.6
    if a == "wic_mgda@800":
        dy = -4.5
    if a == "baseline":
        dx, dy = -13, -3.2
    ax.annotate(LABEL[a].replace("\n", " "), (x, y), xytext=(x + dx, y + dy),
                fontsize=8.7, color="#333")
ax.plot([0, 100], [0, 100], "--", color="#ccc", linewidth=1, zorder=1)
ax.text(60, 63, "identity line\n(no selective effect)", fontsize=7.6, color="#999", rotation=32)
ax.set_xlabel("Retain-writer probe accuracy (1-shot, %) -- higher = healthier representation")
ax.set_ylabel("Forget-writer probe accuracy (1-shot, %)")
ax.set_title("A low forget-probe score only means real forgetting if\nretain writers stay identifiable -- otherwise it's collapse, not targeting")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_targeted_vs_collapse.png"), dpi=160, bbox_inches="tight")
plt.close()

# ── Fig 3: full multi-shot curve, forget vs test_reference, for the 3 NEW methods ──
new_arms = ["arf@800", "wic_mgda@800", "rl_mgda@800"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
for ax, a in zip(axes, new_arms):
    for grp, ls, col in [("forget", "-", COL[a]), ("test_reference", "--", "#9C9FAB")]:
        d = curves[(curves.arm == a) & (curves.feat == "cls") & (curves.group == grp)].sort_values("shots")
        ax.plot(d["shots"], d["acc"], ls, color=col, linewidth=2.2, marker="o", markersize=3.5,
                label="Forget writers" if grp == "forget" else "Genuinely unseen writers")
    ax.set_title(LABEL[a].replace("\n", " "), fontsize=11)
    ax.set_xlabel("Support examples per writer (shots)")
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eee", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Frozen-encoder probe accuracy (%)")
axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
plt.suptitle("Even at the saturated end, forget writers stay well above the unseen-writer floor", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_probe_curves_new_methods.png"), dpi=160, bbox_inches="tight")
plt.close()

print("wrote 3 figures to", OUT)
print(gap.round(1).to_string())
