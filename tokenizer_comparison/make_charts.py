import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import regex

_bn_font_path = r"C:\Windows\Fonts\Nirmala.ttc"
font_manager.fontManager.addfont(_bn_font_path)
BN_FONT = font_manager.FontProperties(fname=_bn_font_path)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "analysis_out")
m = pd.read_csv(os.path.join(OUT, "merged.csv"))

COL = {"grapheme": "#2E9E6B", "bpe": "#D97B3F", "char": "#3B6FA0"}
NAME = {"grapheme": "TrOCR + Grapheme", "bpe": "TrOCR + BPE", "char": "TrOCR + Char"}

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#888",
    "axes.labelcolor": "#222",
    "text.color": "#222",
    "xtick.color": "#222",
    "ytick.color": "#222",
})

# ── Fig 1: accuracy of all three tokenizers by word complexity (grapheme count) ──
bins = [0, 2, 4, 6, 8, 100]
labels = ["1-2", "3-4", "5-6", "7-8", "9+"]
m["gcount_bucket"] = pd.cut(m["grapheme_len"], bins=bins, labels=labels, right=True)

tab = m.groupby("gcount_bucket", observed=True).agg(
    n=("ground_truth", "size"),
    grapheme=("correct_grapheme", "mean"),
    bpe=("correct_bpe", "mean"),
    char=("correct_char", "mean"),
) * [1, 100, 100, 100]  # keep n as-is, scale accuracy cols to %
counts = m.groupby("gcount_bucket", observed=True).size()

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(tab.index))
for key in ("bpe", "char", "grapheme"):
    ax.plot(x, tab[key], marker="o", linewidth=2.4, markersize=7, color=COL[key], label=NAME[key])
ax.set_xticks(list(x))
ax.set_xticklabels([f"{l}\ngraphemes\n(n={n})" for l, n in zip(labels, counts)])
ax.set_ylabel("Word accuracy (%)")
ax.set_ylim(60, 100)
ax.set_title("Accuracy diverges as ground-truth word complexity increases")
ax.legend(frameon=False, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#eee", linewidth=1, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_outcome_by_gcount.png"), dpi=160, bbox_inches="tight")
plt.close()

# ── Fig 2: accuracy by conjunct presence, grouped bars ──────────────────
tab2 = m.groupby("has_conjunct", observed=True).agg(
    n=("ground_truth", "size"),
    grapheme=("correct_grapheme", "mean"),
    bpe=("correct_bpe", "mean"),
    char=("correct_char", "mean"),
)
idx_labels = [f"No conjunct\n(n={tab2.loc[False,'n']})", f"\u22651 conjunct\n(n={tab2.loc[True,'n']})"]

fig, ax = plt.subplots(figsize=(6.5, 5))
xpos = [0, 1]
width = 0.26
for i, key in enumerate(("bpe", "char", "grapheme")):
    vals = tab2[key].values * 100
    ax.bar([p + (i - 1) * width for p in xpos], vals, width=width, color=COL[key], label=NAME[key])
ax.set_xticks(xpos)
ax.set_xticklabels(idx_labels)
ax.set_ylabel("Word accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Accuracy by presence of a consonant conjunct")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#eee", linewidth=1, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_outcome_by_conjunct.png"), dpi=160, bbox_inches="tight")
plt.close()

# ── Fig 3: conjuncts driving the triple-divergence cases ────────────────
HASANTA = "্"


def graphemes(word):
    return regex.findall(r"\X", str(word))


def conjuncts_in(word):
    return [g for g in graphemes(word) if HASANTA in g]


grapheme_wins = m[m["outcome3"] == "G"]           # grapheme right, BOTH baselines wrong
baselines_win = m[m["outcome3"] == "BC"]          # both baselines right, grapheme wrong

cnt_g = Counter(c for w in grapheme_wins["ground_truth"] for c in conjuncts_in(w))
cnt_o = Counter(c for w in baselines_win["ground_truth"] for c in conjuncts_in(w))

top = [c for c, _ in cnt_g.most_common(12)]
rate_g = [cnt_g.get(c, 0) / max(len(grapheme_wins), 1) * 100 for c in top]
rate_o = [cnt_o.get(c, 0) / max(len(baselines_win), 1) * 100 for c in top]

fig, ax = plt.subplots(figsize=(9, 5.5))
y = range(len(top))
ax.barh([i + 0.19 for i in y], rate_g, height=0.38, color=COL["grapheme"],
         label=f"Grapheme beats both baselines (n={len(grapheme_wins)})")
ax.barh([i - 0.19 for i in y], rate_o, height=0.38, color="#9C9FAB",
         label=f"Both baselines beat grapheme (n={len(baselines_win)})")
ax.set_yticks(list(y))
ax.set_yticklabels(top, fontproperties=BN_FONT, fontsize=16)
ax.invert_yaxis()
ax.set_xlabel("% of cases in this group containing the conjunct")
ax.set_title("Conjuncts driving the triple-divergence cases")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_top_conjuncts.png"), dpi=160, bbox_inches="tight")
plt.close()

print("wrote 3 figures to", OUT)
print(f"grapheme_wins (G) n={len(grapheme_wins)}   baselines_win (BC) n={len(baselines_win)}")
