"""
make_new_charts.py
===================
Three new figures for the manuscript rewrite, matching the visual style
of make_arf_charts.py. Every number plotted here is copied directly from
a table already in paper/main.tex -- nothing here is re-derived from raw
logs, so this script is a pure plotting pass over already-reported,
already-cited results.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "..", "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#888", "axes.labelcolor": "#222",
    "text.color": "#222", "xtick.color": "#222", "ytick.color": "#222",
})

BLUE = "#3B6FA0"
GREEN = "#2E9E6B"
PURPLE = "#8A5EC0"
GRAY = "#9C9FAB"
DARK = "#3A3D46"
RED = "#C1544A"

# ══════════════════════════════════════════════════════════════════════
# Fig A: extended tokenizer ablation -- word accuracy, 6 variants
# (Table: tab:tokenizer-ablation)
# ══════════════════════════════════════════════════════════════════════
tok_labels = ["Grapheme", "BPE\nvocab 500", "Character", "Unigram\nvocab 500",
              "Unigram\nvocab 2000", "WordPiece\nvocab 2000", "BPE\nvocab 2000",
              "BPE\nvocab 4000"]
tok_acc = [92.21, 91.02, 88.54, 89.02, 89.34, 88.81, 87.77, 86.15]
tok_cer = [2.52, 3.11, 3.30, 3.76, 3.99, 4.10, 4.25, 5.59]
tok_colors = [GREEN, BLUE, DARK, PURPLE, PURPLE, RED, BLUE, BLUE]

fig, ax1 = plt.subplots(figsize=(9, 5.8))
x = np.arange(len(tok_labels))
bars = ax1.bar(x, tok_acc, color=tok_colors, width=0.6, zorder=3)
ax1.set_ylabel("Word accuracy (%)")
ax1.set_ylim(80, 95)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{lbl}\n(CER {c:.2f}%)" for lbl, c in zip(tok_labels, tok_cer)],
                     fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", color="#eee", linewidth=1, zorder=0)
ax1.set_axisbelow(True)
for xi, v in zip(x, tok_acc):
    ax1.text(xi, v + 0.25, f"{v:.2f}%", ha="center", fontsize=9, color="#333")
ax1.set_title("Word accuracy across all eight tokenizer configurations tested\n"
               "(bars colored by vocabulary-construction family: green=grapheme, "
               "dark=character, blue=BPE, purple=Unigram, red=WordPiece)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_tokenizer_sweep.png"), dpi=160, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════
# Fig B: ConvNeXt vs ViT writer-ID accuracy, 3 datasets, log scale vs chance
# (Table: tab:convnext, plus BanglaWriting/English prose numbers)
# ══════════════════════════════════════════════════════════════════════
datasets = ["BnHTRD\n(Bangla, 74/94w)", "BanglaWriting\n(Bangla, 260w)", "IBM campaign\n(English, 300w)"]
vit_acc = [100.00, 68.30, 88.61]
convnext_acc = [5.00, 0.75, 2.91]
chance = [1.35, 0.38, 0.33]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
xg = np.arange(len(datasets))
w = 0.26
ax.bar(xg - w, vit_acc, width=w, color=GREEN, label="ViT (TrOCR encoder)", zorder=3)
ax.bar(xg, convnext_acc, width=w, color=RED, label="ConvNeXt-Base", zorder=3)
ax.bar(xg + w, chance, width=w, color=GRAY, label="Chance level", zorder=3)
ax.set_yscale("log")
ax.set_ylim(0.2, 200)
ax.set_ylabel("Writer-identification accuracy (%, log scale)")
ax.set_xticks(xg)
ax.set_xticklabels(datasets, fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", which="both", color="#eee", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for xi, v in zip(xg - w, vit_acc):
    ax.text(xi, v * 1.15, f"{v:.1f}", ha="center", fontsize=8.5, color="#333")
for xi, v in zip(xg, convnext_acc):
    ax.text(xi, v * 1.25, f"{v:.2f}", ha="center", fontsize=8.5, color="#333")
for xi, v in zip(xg + w, chance):
    ax.text(xi, v * 1.25, f"{v:.2f}", ha="center", fontsize=8, color="#666")
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title("ConvNeXt never approaches ViT-level writer discrimination in any dataset,\n"
              "but sits closest to chance on the two Bangla datasets specifically")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_convnext_three_datasets.png"), dpi=160, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════
# Fig C: ARF capacity ablation -- residual gap vs retain-writer probe acc
# (Table: tab:verdict)
# ══════════════════════════════════════════════════════════════════════
arf_arms = {
    "ARF (original)": (16.2, None),
    "+ deep probe": (8.5, 91.3),
    "+ 3-ensemble": (11.0, 91.6),
    "+ reset25": (12.4, 91.4),
    "+ no reset": (15.6, 89.9),
    "+ 3× weight": (22.4, 76.6),
    "WIC+MGDA": (-3.9, 68.2),
}
arm_colors = {
    "ARF (original)": GRAY, "+ deep probe": GREEN, "+ 3-ensemble": GREEN,
    "+ reset25": BLUE, "+ no reset": BLUE, "+ 3× weight": RED, "WIC+MGDA": PURPLE,
}

fig, ax = plt.subplots(figsize=(7.8, 6.5))
for name, (gap, retain) in arf_arms.items():
    if retain is None:
        continue
    ax.scatter(retain, gap, s=190, color=arm_colors[name], edgecolor="white",
               linewidth=1.3, zorder=3)
    dx, dy = 1.4, 0.6
    if name == "+ deep probe":
        dx, dy = -13, 0.9
    if name == "WIC+MGDA":
        dx, dy = 1.4, -1.6
    ax.annotate(name, (retain, gap), xytext=(retain + dx, gap + dy), fontsize=8.8, color="#333")
ax.axhline(11.2, color="#888", linestyle="--", linewidth=1, zorder=1)
ax.text(69.5, 11.6, "best original output-level method (WIC, +11.2 pt)",
        fontsize=7.6, color="#777")
ax.axvspan(90, 100, color=GREEN, alpha=0.06, zorder=0)
ax.text(95, -5.5, "healthy\nretain range", fontsize=7.6, color=GREEN, ha="center")
ax.set_xlabel("Retain-writer 1-shot probe accuracy (%) -- higher = healthier representation")
ax.set_ylabel("Residual gap: forget-writer probe acc. minus\ngenuinely-unseen-writer probe acc. (1-shot, pts)")
ax.set_xlim(60, 100)
ax.set_ylim(-8, 26)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("A deeper or ensembled adversary lowers the residual gap without\n"
              "the retain-writer collapse that confounds naive loss-weight scaling")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_arf_capacity_ablation.png"), dpi=160, bbox_inches="tight")
plt.close()

print("wrote 3 figures to", OUT)
