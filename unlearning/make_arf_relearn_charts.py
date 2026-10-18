import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "arf_analysis_out")
os.makedirs(OUT, exist_ok=True)

new = pd.read_csv(os.path.join(BASE, "relearn_new_out", "relearn_arf_mgda", "relearn_summary.csv"))
k5 = new[new["k"] == 5].set_index("arm")

# Original 5-method numbers, k=5, from the earlier session's relearn_summary.csv
# (baseline / wic_unmasked@800 / random_label@800 / fisher@800 / amnesiac@800 / delete@800)
orig_k5 = {
    "wic_unmasked@800": {"start": 8.500000089406967, "final": 90.12500047683716, "retain_start": 96.0, "retain_final": 94.0},
    "random_label@800": {"start": 0.24999999441206455, "final": 89.12500143051147, "retain_start": 94.75, "retain_final": 94.75},
    "fisher@800":       {"start": 0.0, "final": 0.0, "retain_start": 0.75, "retain_final": 0.75},
    "amnesiac@800":     {"start": 55.12499809265137, "final": 88.37500214576721, "retain_start": 95.0, "retain_final": 96.25},
    "delete@800":       {"start": 42.625001072883606, "final": 86.87499761581421, "retain_start": 95.75, "retain_final": 94.75},
}

ORDER = ["wic_unmasked@800", "random_label@800", "amnesiac@800", "delete@800", "fisher@800",
         "arf@800", "wic_mgda@800", "rl_mgda@800"]
LABEL = {
    "wic_unmasked@800": "WIC", "random_label@800": "Random\nLabeling",
    "amnesiac@800": "Amnesiac", "delete@800": "DELETE", "fisher@800": "Fisher\n(broken)",
    "arf@800": "ARF\n(new)", "wic_mgda@800": "WIC\n+MGDA", "rl_mgda@800": "Random Label\n+MGDA",
}
COL_OLD, COL_NEW = "#D97B3F", "#2E9E6B"
COL_MGDA = "#8A5EC0"

rows = []
for a in ORDER:
    if a in orig_k5:
        d = orig_k5[a]
        rows.append((a, d["start"], d["final"], d["retain_start"], d["retain_final"]))
    else:
        r = k5.loc[a]
        rows.append((a, r["start_f95"], r["final_f95"], r["retain_start"], r["retain_final"]))

df = pd.DataFrame(rows, columns=["arm", "start", "final", "retain_start", "retain_final"])
df = df.set_index("arm").loc[ORDER]

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#888", "axes.labelcolor": "#222",
    "text.color": "#222", "xtick.color": "#222", "ytick.color": "#222",
})

# ── Fig: recovery under the 5-shot relearning attack, all 8 methods ──
fig, ax = plt.subplots(figsize=(11, 5.5))
x = range(len(ORDER))
w = 0.36
bar_colors = [COL_NEW if a == "arf@800" else COL_MGDA if "mgda" in a else COL_OLD for a in ORDER]
ax.bar([i - w/2 for i in x], df["start"], width=w, color="#B8BCC8", label="Right after unlearning")
ax.bar([i + w/2 for i in x], df["final"], width=w, color=bar_colors, label="After 5-shot relearning attack")
ax.set_xticks(list(x))
ax.set_xticklabels([LABEL[a] for a in ORDER], fontsize=9.5)
ax.set_ylabel("Forget-writer identifiability (writer-head accuracy, %)")
ax.set_title("A 5-shot, 150-step relearning attack recovers ARF and the MGDA\nvariants just as fast as every output-level method")
for i, a in enumerate(ORDER):
    ax.text(i + w/2, df.loc[a, "final"] + 1.5, f"{df.loc[a,'final']:.0f}%", ha="center", fontsize=8.5)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_relearn_recovery_all_methods.png"), dpi=160, bbox_inches="tight")
plt.close()

# ── Fig: retain-set stability during the attack, 3 new methods ──
fig, ax = plt.subplots(figsize=(7, 5))
new_arms = ["arf@800", "wic_mgda@800", "rl_mgda@800"]
x = range(len(new_arms))
w = 0.36
ax.bar([i - w/2 for i in x], [df.loc[a, "retain_start"] for a in new_arms], width=w,
       color="#B8BCC8", label="Before attack")
ax.bar([i + w/2 for i in x], [df.loc[a, "retain_final"] for a in new_arms], width=w,
       color=[COL_NEW, COL_MGDA, COL_MGDA], label="After attack (5-shot, 150 steps)")
ax.set_xticks(list(x))
ax.set_xticklabels([LABEL[a] for a in new_arms])
ax.set_ylabel("Retain-writer accuracy (%)")
ax.set_title("Retain accuracy barely moves during the attack\n(rules out “the attack just retrains the whole model”)")
for i, a in enumerate(new_arms):
    s, f = df.loc[a, "retain_start"], df.loc[a, "retain_final"]
    ax.text(i, max(s, f) + 2, f"{s:.1f}→{f:.1f}", ha="center", fontsize=9)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2)
ax.set_ylim(0, 105)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_retain_stability_new_methods.png"), dpi=160, bbox_inches="tight")
plt.close()

print(df.round(1).to_string())
print("wrote 2 figures to", OUT)
