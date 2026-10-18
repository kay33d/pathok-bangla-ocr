"""
Per-sample divergence analysis: TrOCR+Grapheme vs TrOCR+BPE vs TrOCR+Char
on the identical BnHTRD test split (6,614 samples, same image_path order in all three CSVs).

Produces:
  - merged.csv               : per-sample outcomes for all three tokenizers + complexity features
  - stats_summary.txt        : McNemar's tests, contingency tables, bucketed win-rate stats
"""
import io
import os
from math import comb

import numpy as np
import pandas as pd
import regex  # extended grapheme clusters, \X -- identical to pathok's GraphemeTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "analysis_out")
os.makedirs(OUT, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────
bpe = pd.read_csv(os.path.join(BASE, "bpe_out", "test_results.csv"))
gra = pd.read_csv(os.path.join(BASE, "grapheme_out", "test_results_trocr_grapheme.csv"))
chr_ = pd.read_csv(os.path.join(BASE, "char_out", "test_results_trocr_char.csv"))

for df in (bpe, gra, chr_):
    df["image_path"] = df["image_path"].astype(str)
    df["ground_truth"] = df["ground_truth"].astype(str)
    df["prediction"] = df["prediction"].astype(str)
    df["correct"] = df["correct"].astype(bool)

same_order_bg = (bpe["image_path"].values == gra["image_path"].values).all()
same_order_cg = (chr_["image_path"].values == gra["image_path"].values).all()
same_set = (set(bpe["image_path"]) == set(gra["image_path"]) == set(chr_["image_path"]))

m = gra.merge(bpe, on="image_path", suffixes=("_grapheme", "_bpe"))
m = m.merge(chr_.add_suffix("_char"), left_on="image_path", right_on="image_path_char")
gt_mismatch = (
    (m["ground_truth_grapheme"] != m["ground_truth_bpe"])
    | (m["ground_truth_grapheme"] != m["ground_truth_char"])
).sum()
m["ground_truth"] = m["ground_truth_grapheme"]
m = m.drop(columns=[c for c in m.columns if c.startswith("ground_truth_")])

# ── complexity features on the ground-truth word ────────────────────────
HASANTA = "্"
NUKTA = "়"
VOWEL_SIGNS = set("ািীুূৃৄেৈোৌৗ")


def graphemes(word: str):
    return regex.findall(r"\X", word)


def n_conjuncts(word: str) -> int:
    return sum(1 for g in graphemes(word) if HASANTA in g)


def n_vowel_signs(word: str) -> int:
    return sum(1 for g in graphemes(word) if any(v in g for v in VOWEL_SIGNS))


m["char_len"] = m["ground_truth"].str.len()
m["grapheme_len"] = m["ground_truth"].apply(lambda w: len(graphemes(w)))
m["n_conjuncts"] = m["ground_truth"].apply(n_conjuncts)
m["has_conjunct"] = m["n_conjuncts"] > 0
m["n_vowel_signs"] = m["ground_truth"].apply(n_vowel_signs)
m["has_nukta"] = m["ground_truth"].str.contains(NUKTA)
m["char_grapheme_ratio"] = m["char_len"] / m["grapheme_len"].replace(0, np.nan)

# ── 3-way outcome label ──────────────────────────────────────────────────
def label3(row):
    g, b, c = row["correct_grapheme"], row["correct_bpe"], row["correct_char"]
    tag = "".join(x for x, v in zip("GBC", (g, b, c)) if v)
    return tag if tag else "none"


m["outcome3"] = m.apply(label3, axis=1)
m["n_models_correct"] = (
    m["correct_grapheme"].astype(int) + m["correct_bpe"].astype(int) + m["correct_char"].astype(int)
)
m["grapheme_beats_both"] = m["correct_grapheme"] & ~m["correct_bpe"] & ~m["correct_char"]

# 2-way outcome vs bpe (kept for continuity with the earlier pass)
def outcome_vs(g, other):
    if g and other:
        return "both_correct"
    if g and not other:
        return "grapheme_only"
    if other and not g:
        return "other_only"
    return "both_wrong"


m["outcome_vs_bpe"] = [outcome_vs(g, o) for g, o in zip(m["correct_grapheme"], m["correct_bpe"])]
m["outcome_vs_char"] = [outcome_vs(g, o) for g, o in zip(m["correct_grapheme"], m["correct_char"])]


def mcnemar(a_only, b_only):
    n = a_only + b_only
    if n == 0:
        return float("nan")
    k = min(a_only, b_only)
    p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)
    return min(p, 1.0)


n_g_vs_bpe = int((m["outcome_vs_bpe"] == "grapheme_only").sum())
n_bpe_vs_g = int((m["outcome_vs_bpe"] == "other_only").sum())
p_g_bpe = mcnemar(n_g_vs_bpe, n_bpe_vs_g)

n_g_vs_char = int((m["outcome_vs_char"] == "grapheme_only").sum())
n_char_vs_g = int((m["outcome_vs_char"] == "other_only").sum())
p_g_char = mcnemar(n_g_vs_char, n_char_vs_g)

m.to_csv(os.path.join(OUT, "merged.csv"), index=False)

# ── stats summary ────────────────────────────────────────────────────────
buf = io.StringIO()
buf.write(f"same order (bpe vs grapheme)  : {same_order_bg}\n")
buf.write(f"same order (char vs grapheme) : {same_order_cg}\n")
buf.write(f"same test-set membership      : {same_set}\n")
buf.write(f"ground-truth mismatches       : {gt_mismatch}\n")
buf.write(f"n merged samples              : {len(m)}\n\n")

buf.write("=== aggregate word accuracy (sanity check vs thesis Table 4.2) ===\n")
for name in ("grapheme", "bpe", "char"):
    buf.write(f"{name:10s} word-acc: {m[f'correct_{name}'].mean()*100:.2f}%   "
              f"mean CER: {m[f'cer_{name}'].mean()*100:.2f}%\n")
buf.write("\n")

buf.write("=== full 3-way outcome distribution (G=grapheme, B=bpe, C=char correct) ===\n")
buf.write(m["outcome3"].value_counts().to_string() + "\n\n")

buf.write("=== triple-divergence: grapheme right, BOTH baselines wrong ===\n")
buf.write(f"count = {int(m['grapheme_beats_both'].sum())}  "
          f"({m['grapheme_beats_both'].mean()*100:.2f}% of test set)\n\n")

buf.write("=== McNemar's exact test: Grapheme vs BPE ===\n")
buf.write(f"grapheme_only = {n_g_vs_bpe}, bpe_only = {n_bpe_vs_g}, p = {p_g_bpe:.3e}\n\n")

buf.write("=== McNemar's exact test: Grapheme vs Char ===\n")
buf.write(f"grapheme_only = {n_g_vs_char}, char_only = {n_char_vs_g}, p = {p_g_char:.3e}\n\n")

buf.write("=== win-rate by grapheme-length bucket (3-way) ===\n")
bins = [0, 2, 4, 6, 8, 100]
labels = ["1-2", "3-4", "5-6", "7-8", "9+"]
m["gcount_bucket"] = pd.cut(m["grapheme_len"], bins=bins, labels=labels, right=True)
tab = m.groupby("gcount_bucket", observed=True).apply(
    lambda d: pd.Series({
        "n": len(d),
        "grapheme_acc_%": d["correct_grapheme"].mean() * 100,
        "bpe_acc_%": d["correct_bpe"].mean() * 100,
        "char_acc_%": d["correct_char"].mean() * 100,
        "grapheme_beats_both_%": d["grapheme_beats_both"].mean() * 100,
    })
)
buf.write(tab.to_string() + "\n\n")

buf.write("=== win-rate by conjunct presence (3-way) ===\n")
tab2 = m.groupby("has_conjunct", observed=True).apply(
    lambda d: pd.Series({
        "n": len(d),
        "grapheme_acc_%": d["correct_grapheme"].mean() * 100,
        "bpe_acc_%": d["correct_bpe"].mean() * 100,
        "char_acc_%": d["correct_char"].mean() * 100,
        "grapheme_beats_both_%": d["grapheme_beats_both"].mean() * 100,
    })
)
buf.write(tab2.to_string() + "\n\n")

with open(os.path.join(OUT, "stats_summary.txt"), "w", encoding="utf-8") as f:
    f.write(buf.getvalue())

print(buf.getvalue())
print(f"\nWrote merged.csv + stats_summary.txt to {OUT}")
