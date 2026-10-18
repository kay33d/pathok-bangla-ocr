import io, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest
from math import comb

BASE = r"C:\Users\LENOVO\Desktop\pathok_tokenizer_analysis"
OUT_TXT = BASE + r"\seqlen_ci_report.txt"
OUT_CSV_CI = BASE + r"\bucket_accuracy_with_ci.csv"
OUT_CSV_MCNEMAR = BASE + r"\mcnemar_multipletest_corrected.csv"

df = pd.read_csv(BASE + r"\merged.csv")
bins = [0, 2, 4, 6, 8, 100]
labels = ["1-2", "3-4", "5-6", "7-8", "9+"]
df["gcount_bucket"] = pd.cut(df["grapheme_len"], bins=bins, labels=labels, right=True)
n = len(df)
rng = np.random.default_rng(42)
N_BOOT = 5000

report = []
def p(s=""):
    report.append(str(s))

p("="*78)
p("SEQUENCE-LENGTH REGRESSION + BOOTSTRAP CIs + MULTIPLE-TESTING CORRECTION")
p(f"n = {n} samples (BnHTRD test split, 3-way merged: grapheme/BPE/char)")
p("="*78)

# ---------------------------------------------------------------------
# 1. Bootstrap 95% CIs on the bucket-level accuracy table (gcount_bucket)
# ---------------------------------------------------------------------
p("\n--- 1. Bootstrap 95% CIs, accuracy by grapheme-length bucket ---\n")

def bucket_acc(sub):
    return {
        "grapheme_acc": sub["correct_grapheme"].mean() * 100,
        "bpe_acc": sub["correct_bpe"].mean() * 100,
        "char_acc": sub["correct_char"].mean() * 100,
    }

def bootstrap_ci(sub, key_fn, n_boot=N_BOOT):
    idx = sub.index.to_numpy()
    vals = {k: [] for k in ["grapheme_acc", "bpe_acc", "char_acc"]}
    m = len(idx)
    for _ in range(n_boot):
        samp = rng.choice(idx, size=m, replace=True)
        r = key_fn(sub.loc[samp])
        for k in vals:
            vals[k].append(r[k])
    out = {}
    for k, v in vals.items():
        v = np.array(v)
        out[k + "_lo"] = np.percentile(v, 2.5)
        out[k + "_hi"] = np.percentile(v, 97.5)
    return out

rows = []
for bucket, sub in df.groupby("gcount_bucket"):
    point = bucket_acc(sub)
    ci = bootstrap_ci(sub, bucket_acc)
    row = {"gcount_bucket": bucket, "n": len(sub)}
    row.update(point)
    row.update(ci)
    rows.append(row)
    p(f"  bucket {bucket:>4}  n={len(sub):5d}  "
      f"grapheme={point['grapheme_acc']:5.1f}% [{ci['grapheme_acc_lo']:.1f},{ci['grapheme_acc_hi']:.1f}]  "
      f"bpe={point['bpe_acc']:5.1f}% [{ci['bpe_acc_lo']:.1f},{ci['bpe_acc_hi']:.1f}]  "
      f"char={point['char_acc']:5.1f}% [{ci['char_acc_lo']:.1f},{ci['char_acc_hi']:.1f}]")

bucket_ci_df = pd.DataFrame(rows)
bucket_ci_df.to_csv(OUT_CSV_CI, index=False)

p("\n--- 1b. Bootstrap 95% CIs, accuracy by conjunct presence ---\n")
rows2 = []
for grp, sub in df.groupby("has_conjunct"):
    point = bucket_acc(sub)
    ci = bootstrap_ci(sub, bucket_acc)
    row = {"has_conjunct": grp, "n": len(sub)}
    row.update(point)
    row.update(ci)
    rows2.append(row)
    p(f"  has_conjunct={str(grp):5}  n={len(sub):5d}  "
      f"grapheme={point['grapheme_acc']:5.1f}% [{ci['grapheme_acc_lo']:.1f},{ci['grapheme_acc_hi']:.1f}]  "
      f"bpe={point['bpe_acc']:5.1f}% [{ci['bpe_acc_lo']:.1f},{ci['bpe_acc_hi']:.1f}]  "
      f"char={point['char_acc']:5.1f}% [{ci['char_acc_lo']:.1f},{ci['char_acc_hi']:.1f}]")

conjunct_ci_df = pd.DataFrame(rows2)

# ---------------------------------------------------------------------
# 2. McNemar tests: overall + per-subgroup, with Holm-Bonferroni correction
# ---------------------------------------------------------------------
p("\n--- 2. McNemar exact tests: overall + per-subgroup, Holm-Bonferroni corrected ---\n")

def mcnemar_exact(a_correct, b_correct):
    only_a = int(((a_correct == 1) & (b_correct == 0)).sum())
    only_b = int(((a_correct == 0) & (b_correct == 1)).sum())
    m = only_a + only_b
    if m == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    pval = sum(comb(m, i) for i in range(0, k + 1)) * 2 / (2 ** m)
    pval = min(pval, 1.0)
    return only_a, only_b, pval

tests = []
pairs = [("grapheme", "bpe", "correct_grapheme", "correct_bpe"),
         ("grapheme", "char", "correct_grapheme", "correct_char"),
         ("bpe", "char", "correct_bpe", "correct_char")]

# overall
for name_a, name_b, col_a, col_b in pairs:
    oa, ob, pv = mcnemar_exact(df[col_a], df[col_b])
    tests.append({"scope": "overall", "subgroup": "all", "comparison": f"{name_a}_vs_{name_b}",
                  f"{name_a}_only": oa, f"{name_b}_only": ob, "p_raw": pv})

# per length-bucket subgroup
for bucket, sub in df.groupby("gcount_bucket"):
    for name_a, name_b, col_a, col_b in pairs:
        oa, ob, pv = mcnemar_exact(sub[col_a], sub[col_b])
        tests.append({"scope": "gcount_bucket", "subgroup": bucket, "comparison": f"{name_a}_vs_{name_b}",
                      f"{name_a}_only": oa, f"{name_b}_only": ob, "p_raw": pv})

# per conjunct subgroup
for grp, sub in df.groupby("has_conjunct"):
    for name_a, name_b, col_a, col_b in pairs:
        oa, ob, pv = mcnemar_exact(sub[col_a], sub[col_b])
        tests.append({"scope": "has_conjunct", "subgroup": grp, "comparison": f"{name_a}_vs_{name_b}",
                      f"{name_a}_only": oa, f"{name_b}_only": ob, "p_raw": pv})

tdf = pd.DataFrame(tests)
# Holm-Bonferroni across the whole family
order = tdf["p_raw"].sort_values().index
m = len(tdf)
holm = pd.Series(index=tdf.index, dtype=float)
running_max = 0.0
for rank, idx in enumerate(order, start=1):
    adj = (m - rank + 1) * tdf.loc[idx, "p_raw"]
    running_max = max(running_max, adj)
    holm[idx] = min(running_max, 1.0)
tdf["p_holm"] = holm
tdf["significant_at_0.05_holm"] = tdf["p_holm"] < 0.05
tdf.to_csv(OUT_CSV_MCNEMAR, index=False)

p(f"  family size (all tests, all scopes): {m}")
p(f"  significant after Holm-Bonferroni (alpha=0.05): {int(tdf['significant_at_0.05_holm'].sum())} / {m}")
p("\n  overall (3 tests):")
for _, r in tdf[tdf.scope == "overall"].iterrows():
    p(f"    {r['comparison']:18s}  p_raw={r['p_raw']:.3e}  p_holm={r['p_holm']:.3e}  sig={r['significant_at_0.05_holm']}")
p("\n  by length bucket (grapheme_vs_bpe, grapheme_vs_char only):")
for _, r in tdf[(tdf.scope == "gcount_bucket") & (tdf.comparison != "bpe_vs_char")].iterrows():
    p(f"    bucket={str(r['subgroup']):5s} {r['comparison']:18s}  p_raw={r['p_raw']:.3e}  p_holm={r['p_holm']:.3e}  sig={r['significant_at_0.05_holm']}")
p("\n  by conjunct presence:")
for _, r in tdf[tdf.scope == "has_conjunct"].iterrows():
    p(f"    conjunct={str(r['subgroup']):5s} {r['comparison']:18s}  p_raw={r['p_raw']:.3e}  p_holm={r['p_holm']:.3e}  sig={r['significant_at_0.05_holm']}")

# ---------------------------------------------------------------------
# 3. Formal regression of correctness on length + tokenizer (+ interaction)
#    -- targets the reviewer's specific ask: formalize the char/BPE crossover
# ---------------------------------------------------------------------
p("\n--- 3. Logistic regression: P(correct) ~ length * tokenizer (char vs bpe) ---\n")

long_rows = []
for _, r in df.iterrows():
    long_rows.append({"length": r["char_len"], "tokenizer": "char", "correct": r["correct_char"]})
    long_rows.append({"length": r["char_len"], "tokenizer": "bpe", "correct": r["correct_bpe"]})
long_df = pd.DataFrame(long_rows)
long_df["correct"] = long_df["correct"].astype(int)
long_df["is_char"] = (long_df["tokenizer"] == "char").astype(int)
long_df["length_c"] = long_df["length"] - long_df["length"].mean()

model = smf.logit("correct ~ length_c * is_char", data=long_df).fit(disp=0)
p(model.summary().as_text())

b0, b_len, b_char, b_int = model.params["Intercept"], model.params["length_c"], model.params["is_char"], model.params["length_c:is_char"]
mean_len = long_df["length"].mean()
# crossover: where char logit == bpe logit  =>  b_char + b_int*length_c = 0
if abs(b_int) > 1e-9:
    crossover_c = -b_char / b_int
    crossover_len = crossover_c + mean_len
    p(f"\n  Model-implied crossover length (char overtakes bpe): "
      f"char_len ~= {crossover_len:.1f}")
else:
    p("\n  No crossover (interaction term ~0): one tokenizer dominates at all lengths.")

p(f"\n  Interaction term (length_c:is_char) p-value = {model.pvalues['length_c:is_char']:.4g}")
p("  Interpretation: a significant negative interaction means char's relative\n"
  "  accuracy advantage over bpe erodes (or reverses) as word length grows --\n"
  "  the formal version of the 'crossover on long words' observation.")

# Same regression for grapheme vs bpe (does grapheme's advantage hold at all lengths?)
long_rows2 = []
for _, r in df.iterrows():
    long_rows2.append({"length": r["grapheme_len"], "tokenizer": "grapheme", "correct": r["correct_grapheme"]})
    long_rows2.append({"length": r["grapheme_len"], "tokenizer": "bpe", "correct": r["correct_bpe"]})
long_df2 = pd.DataFrame(long_rows2)
long_df2["correct"] = long_df2["correct"].astype(int)
long_df2["is_grapheme"] = (long_df2["tokenizer"] == "grapheme").astype(int)
long_df2["length_c"] = long_df2["length"] - long_df2["length"].mean()
model2 = smf.logit("correct ~ length_c * is_grapheme", data=long_df2).fit(disp=0)
p("\n--- 3b. Logistic regression: P(correct) ~ length * tokenizer (grapheme vs bpe) ---\n")
p(model2.summary().as_text())
p(f"\n  Interaction term (length_c:is_grapheme) p-value = {model2.pvalues['length_c:is_grapheme']:.4g}")
b_int2 = model2.params["length_c:is_grapheme"]
if b_int2 > 0:
    p("  Positive interaction: grapheme's advantage over bpe WIDENS with word length\n"
      "  (consistent with longer words containing more conjuncts).")
else:
    p("  Negative interaction: grapheme's advantage over bpe NARROWS with word length.")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\n".join(report))
print(f"\nwrote: {OUT_TXT}\nwrote: {OUT_CSV_CI}\nwrote: {OUT_CSV_MCNEMAR}")
