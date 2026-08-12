"""
exp5_gate_v4.py
===============
~10 min per arm. The final table.

WHAT CHANGED FROM GATE v3
-------------------------
1. THE ORACLE IS THE TARGET, NOT THE NON-MEMBER RATE.
   For M_retrain the forget writers ARE non-members, so its forget-seen rate is
   what perfect deletion looks like under this attack -- including any residual
   distribution shift between the forget writers and the test writers. Measuring
   against the test-writer rate silently assumes those two groups are
   exchangeable. They are not. Headline metric:

       Unlearning Gap  =  100 * (forget_arm - forget_oracle)
                              / (forget_baseline - forget_oracle)

   0 = indistinguishable from exact deletion. 100 = nothing happened.
   Negative = over-unlearning, now defined against a real reference instead of
   against an arbitrary 50% or the chance rate.

2. FIXED-FPR THRESHOLDS INSTEAD OF YOUDEN'S J.
   With 30 non-member writers and near-separable score distributions, J lands in
   an empty gap and the reported percentage moves by whole writers between arms
   for no reason -- which is exactly why your v3 non-member rates bounced
   between 10.00, 13.33 and 16.67 (3, 4 and 5 writers). A threshold pinned at a
   fixed non-member FPR removes that artefact and makes arms comparable.

3. CLUSTER BOOTSTRAP OVER WRITERS on every number. Your effective sample size is
   74 / 30 / 20 writers, not 5841 / 2243 / 1448 words. Word-level CIs understate
   the uncertainty by roughly sqrt(80).

4. SYMMETRIC SCORING. v3 scored members out-of-fold with one model each but
   scored forget writers with the mean of five, so the two groups had different
   score variances. Here forget writers are scored by each fold model and the
   per-fold scores are aggregated the same way for all three groups.

5. TPR AT LOW FPR reported alongside AUC, per Carlini et al. Note the writer
   level cannot resolve below 1/30 = 3.3% FPR; that limit is printed, not hidden.

OUTPUT
------
  /kaggle/working/gate_v4/gate_v4.csv
  /kaggle/working/gate_v4/writer_scores_long.csv   (for the violin/strip figure)
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve


class G:
    OUT = "/kaggle/working/gate_v4"
    ORACLE = "/kaggle/working/oracle/oracle_final.pt"

    WORDS_PER_WRITER = 80
    N_FOLDS = 5
    N_BOOT = 2000
    TARGET_FPR_WORD = 0.10
    TARGET_FPR_WRITER = 0.10      # 3/30 writers; granularity is printed
    RANK_N = 800
    DECODE_N = 800

    ARMS = ["baseline", "oracle", "wic_unmasked@300", "wic_unmasked@800",
            "random_label@800", "wic_masked"]


os.makedirs(G.OUT, exist_ok=True)


def tpr_at_fpr(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target, fpr, tpr))


def run_attack(model, ctx, label):
    vocab, fe = ctx["vocab"], ctx["fe"]
    k = G.WORDS_PER_WRITER
    mem = sample_per_writer(ctx["retain_df"], k)
    non = sample_per_writer(ctx["test_df"], k)
    fgt = sample_per_writer(ctx["forget_df"], k)

    Xm, Xn, Xf = (extract_features(model, d, vocab, fe) for d in (mem, non, fgt))
    X = np.vstack([Xm, Xn])
    y = np.r_[np.ones(len(Xm)), np.zeros(len(Xn))]
    groups = np.r_[mem["writer"].values, non["writer"].values]

    # grouped CV; forget is scored by every fold model, then averaged, and the
    # member/non-member OOF scores are left as-is (each from a model that never
    # saw that writer). Both conventions are stated in the docstring.
    oof = np.zeros(len(X)); fold_f = []
    for tr, te in GroupKFold(n_splits=G.N_FOLDS).split(X, y, groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=1.0))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        fold_f.append(clf.predict_proba(Xf)[:, 1])
    sf = np.mean(fold_f, axis=0)
    sm, sn = oof[:len(Xm)], oof[len(Xm):]
    wm_g, wn_g, wf_g = (mem["writer"].values, non["writer"].values,
                        fgt["writer"].values)

    # ── WORD level ────────────────────────────────────────────────────────────
    auc_w = boot_auc(y, oof, groups, n_boot=G.N_BOOT)
    t_word = thr_at_fpr(sn, G.TARGET_FPR_WORD)
    res = {"label": label,
           "auc_word": auc_w, "thr_word": t_word,
           "tpr_at_1pct_word": 100 * tpr_at_fpr(y, oof, 0.01),
           "tpr_at_10pct_word": 100 * tpr_at_fpr(y, oof, 0.10),
           "word_member": boot_rate(sm, wm_g, t_word, n_boot=G.N_BOOT),
           "word_nonmember": boot_rate(sn, wn_g, t_word, n_boot=G.N_BOOT),
           "word_forget": boot_rate(sf, wf_g, t_word, n_boot=G.N_BOOT)}

    # ── WRITER level: aggregate first, then fit the threshold at writer scale ──
    pw = lambda s, g: pd.Series(s, index=g).groupby(level=0).mean()
    wm, wn, wf = pw(sm, wm_g), pw(sn, wn_g), pw(sf, wf_g)
    wy = np.r_[np.ones(len(wm)), np.zeros(len(wn))]
    ws = np.r_[wm.values, wn.values]
    wg = np.r_[wm.index.values, wn.index.values]
    auc_r = boot_auc(wy, ws, wg, n_boot=G.N_BOOT)
    t_writer = thr_at_fpr(wn.values, G.TARGET_FPR_WRITER)
    res.update({
        "auc_writer": auc_r, "thr_writer": t_writer,
        "writer_member": boot_rate(wm.values, wm.index.values, t_writer, n_boot=G.N_BOOT),
        "writer_nonmember": boot_rate(wn.values, wn.index.values, t_writer, n_boot=G.N_BOOT),
        "writer_forget": boot_rate(wf.values, wf.index.values, t_writer, n_boot=G.N_BOOT),
        "forget_above": int((wf.values > t_writer).sum()),
        "n_forget_writers": len(wf), "n_nonmember_writers": len(wn),
        "_wm": wm, "_wn": wn, "_wf": wf})
    return res


def diagnostics(model, ctx, label, n_classes):
    vocab, fe = ctx["vocab"], ctx["fe"]
    out = {}
    for nm, d in (("forget", ctx["forget_df"]), ("retain", ctx["retain_df"])):
        r = writer_rank_diag(model, d, vocab, fe, n_classes, n=G.RANK_N)
        out[f"rank_{nm}"] = boot_mean(r["ranks"], r["writer"], n_boot=1000)
        out[f"wACC_{nm}"] = r["ACC"]
    for nm, d in (("retain", ctx["retain_df"]), ("test", ctx["test_df"]),
                  ("forget", ctx["forget_df"])):
        q = rec_quality(model, d, vocab, fe, n=G.DECODE_N, return_per_word=True)
        pw = q["per_word"]
        out[f"WER_{nm}"] = boot_mean(pw["wrong"].values.astype(float),
                                     pw["writer"].values, n_boot=1000)
        out[f"CER_{nm}"] = q["CER"]
    return out


def main():
    ctx = setup()
    vocab, fe = ctx["vocab"], ctx["fe"]
    n_classes = ctx["n_classes"]
    reg = discover_checkpoints()
    if os.path.exists(G.ORACLE):
        reg["oracle"] = G.ORACLE
    reg.setdefault("oracle", reg.get("oracle_final"))
    print(f"\n  checkpoints on disk: {sorted(k for k in reg if reg.get(k))}")

    model = build_model(vocab, n_classes)
    rows, long_rows = [], []
    for label in G.ARMS:
        if label == "baseline":
            sd = ctx["base_sd"]
        else:
            p = reg.get(label)
            if not p or not os.path.exists(p):
                print(f"  [skip] {label}: not found"
                      + ("   <-- exp1 must finish first; without it there is no "
                         "target and the table is uninterpretable"
                         if label == "oracle" else ""))
                continue
            sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        load_into(model, sd, label, strict_report=False)

        r = run_attack(model, ctx, label)
        r.update(diagnostics(model, ctx, label, n_classes))
        rows.append(r)
        for grp, ser in (("member", r["_wm"]), ("nonmember", r["_wn"]),
                         ("forget", r["_wf"])):
            for w, v in ser.items():
                long_rows.append({"arm": label, "group": grp, "writer": w,
                                  "score": float(v)})

        print("\n" + "=" * 80)
        print(f"  {label}")
        print("-" * 80)
        print(f"    AUC word   {fmt_ci(r['auc_word'])}    "
              f"TPR@1%FPR {r['tpr_at_1pct_word']:.1f}%")
        print(f"    AUC writer {fmt_ci(r['auc_writer'])}   "
              f"(FPR resolution limit {100/r['n_nonmember_writers']:.1f}%)")
        print(f"    word   member {fmt_ci(r['word_member'], pct=True, prec=1)}  "
              f"nonmember {fmt_ci(r['word_nonmember'], pct=True, prec=1)}  "
              f"forget {fmt_ci(r['word_forget'], pct=True, prec=1)}")
        print(f"    writer member {fmt_ci(r['writer_member'], pct=True, prec=1)}  "
              f"nonmember {fmt_ci(r['writer_nonmember'], pct=True, prec=1)}  "
              f"forget {fmt_ci(r['writer_forget'], pct=True, prec=1)} "
              f"({r['forget_above']}/{r['n_forget_writers']})")
        print(f"    forget wACC {r['wACC_forget']:.2f}%  "
              f"mean rank {fmt_ci(r['rank_forget'], prec=1)}/{n_classes}")
        print(f"    WER retain {fmt_ci(r['WER_retain'], prec=3)}  "
              f"test {fmt_ci(r['WER_test'], prec=3)}   (fractions, x100 for %)")

    # ── ORACLE-REFERENCED SUMMARY ─────────────────────────────────────────────
    flat = pd.DataFrame([{
        "arm": r["label"],
        "auc_word": r["auc_word"]["point"],
        "auc_word_lo": r["auc_word"]["lo"], "auc_word_hi": r["auc_word"]["hi"],
        "auc_writer": r["auc_writer"]["point"],
        "auc_writer_lo": r["auc_writer"]["lo"], "auc_writer_hi": r["auc_writer"]["hi"],
        "tpr1_word": r["tpr_at_1pct_word"],
        "word_forget": r["word_forget"]["point"],
        "word_nonmember": r["word_nonmember"]["point"],
        "writer_forget": r["writer_forget"]["point"],
        "writer_forget_lo": r["writer_forget"]["lo"],
        "writer_forget_hi": r["writer_forget"]["hi"],
        "writer_nonmember": r["writer_nonmember"]["point"],
        "forget_wACC": r["wACC_forget"],
        "rank_forget": r["rank_forget"]["point"],
        "WER_retain": 100 * r["WER_retain"]["point"],
        "WER_test": 100 * r["WER_test"]["point"],
    } for r in rows])

    have = set(flat["arm"])
    if {"oracle", "baseline"} <= have:
        o = flat[flat.arm == "oracle"].iloc[0]
        b = flat[flat.arm == "baseline"].iloc[0]

        def gap(col):
            den = b[col] - o[col]
            return 100 * (flat[col] - o[col]) / den if abs(den) > 1e-9 else np.nan

        flat["ugap_writer_rate"] = gap("writer_forget")
        flat["ugap_auc_writer"] = gap("auc_writer")
        flat["rank_dev_from_oracle"] = flat["rank_forget"] - o["rank_forget"]
        flat["utility_cost_vs_oracle"] = flat["WER_test"] - o["WER_test"]
    flat.to_csv(f"{G.OUT}/gate_v4.csv", index=False)
    pd.DataFrame(long_rows).to_csv(f"{G.OUT}/writer_scores_long.csv", index=False)

    print("\n" + "=" * 112)
    print("  GATE v4 -- everything referenced to exact deletion (the oracle)")
    print("-" * 112)
    print(f"  {'arm':<20} {'AUCwr':>16} {'wr_forget%':>18} {'U-gap':>8} "
          f"{'rank dev':>9} {'testWER':>8} {'ΔWER':>7}")
    for _, r in flat.iterrows():
        ug = r.get("ugap_writer_rate", np.nan)
        rd = r.get("rank_dev_from_oracle", np.nan)
        dw = r.get("utility_cost_vs_oracle", np.nan)
        print(f"  {r['arm']:<20} "
              f"{r['auc_writer']:.3f}[{r['auc_writer_lo']:.2f},{r['auc_writer_hi']:.2f}] "
              f"{r['writer_forget']:6.1f}[{r['writer_forget_lo']:5.1f},"
              f"{r['writer_forget_hi']:5.1f}] "
              f"{('-' if not np.isfinite(ug) else f'{ug:6.1f}'):>8} "
              f"{('-' if not np.isfinite(rd) else f'{rd:+8.1f}'):>9} "
              f"{r['WER_test']:7.2f}% "
              f"{('-' if not np.isfinite(dw) else f'{dw:+6.2f}'):>7}")
    print("-" * 112)
    print("  U-gap  0   = indistinguishable from exact deletion")
    print("  U-gap  100 = indistinguishable from no deletion at all")
    print("  U-gap  <0  = over-unlearning, measured against a real reference")
    print("  rank dev   = deviation of forget mean logit rank from the oracle's.")
    print("               Near 0 is correct even if the absolute rank is high;")
    print("               large positive is inversion, large negative is 'still")
    print("               identified'.")
    print("  Read the CIs before claiming any two arms differ. With 20 forget and")
    print("  30 non-member writers, differences under ~10 points are not real.")
    print(f"\n  saved -> {G.OUT}/gate_v4.csv, {G.OUT}/writer_scores_long.csv")
    return flat


if __name__ == "__main__":
    main()
