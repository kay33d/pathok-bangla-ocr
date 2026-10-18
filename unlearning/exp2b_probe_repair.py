"""
exp2b_probe_repair.py
=====================
~30-50 min. Replaces exp2_reprobe.py.

WHY THE FIRST VERSION HAD NO RESOLUTION
---------------------------------------
Oracle floor 82.89%, baseline ceiling 90.40%, CIs 10-20 points wide. A model
that never saw those 20 writers separates them almost as well as the model that
trained on them. The '% of range' metric was uninterpretable, and so is any
claim built on asymptotic closed-set accuracy.

Three causes, three fixes:

 1. THE TASK WAS TOO EASY. 250 words per writer, 20 classes, 768-d features.
    Fix: probe the SAMPLE-EFFICIENCY curve. Memorisation, if present, shows as
    the model needing fewer examples to place an unseen page of a known writer,
    not as a higher asymptote. Differences live at 1-5 shots, which the original
    design skipped entirely.

 2. CLOSED-SET ACCURACY SATURATES. At 90% there is nowhere to go.
    Fix: VERIFICATION. Cosine similarity over same-writer vs different-writer
    pairs from held-out pages, scored by EER and AUC. This is the standard
    biometric metric and it keeps resolving well past the point where
    identification accuracy flattens.

 3. IT PROBED THE WRONG MODULE. Your own MIA gets per-writer AUC 0.956 from
    DECODER confidence. So memorisation exists and is detectable -- just not in
    the encoder's writer-discriminative geometry, which is dominated by generic
    Bangla style extraction. Fix: probe decoder states too, so the negative
    encoder result is a contrast rather than a dead end.

 4. PER-ARM MAX OVER 20 CONFIGS is a selection artefact. Fix: one fixed config,
    declared up front. Everything else goes in the appendix.

A FOURTH REFERENCE POINT
------------------------
TEST writers under the BASELINE. These 30 were seen by nobody, so their
separability is pure generic style capacity. If forget-under-oracle equals
test-under-baseline, you have measured the encoder's ability to separate any 20
Bangla writers and confirmed that the probe is blind to membership. That is the
publishable claim, and it needs this arm to be made cleanly.

OUTPUT
------
  /kaggle/working/reprobe2/fewshot_curves.csv
  /kaggle/working/reprobe2/verification.csv
  /kaggle/working/reprobe2/summary.csv
"""

import os
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

class P:
    OUT = "/kaggle/working/reprobe2"
    ORACLE = "/kaggle/working/oracle/oracle_final.pt"

    WORDS_PER_WRITER = 300
    SHOTS = [1, 2, 5, 10, 25, 50, 150]     # probe-training words per writer
    N_REPEATS = 5                          # resamples of the k-shot support set
    LAYER = 12                             # fixed. declared before looking.
    POOL = "cls"                           # fixed. 'meanpatch' reported alongside.
    ALSO_POOL = "meanpatch"
    PROBE_DECODER = True                   # the contrast that matters

    N_PAIRS = 40000                        # verification pairs per arm/group
    N_RETAIN_CONTROL = 20

    ARMS = ["baseline", "oracle", "wic_unmasked@300", "wic_unmasked@800",
            "random_label@800", "fisher@800", "amnesiac@800", "delete@800"]
    GROUPS = ["forget", "retain_control", "test_reference"]


os.makedirs(P.OUT, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed(model, df, vocab, fe, cfg=S):
    """Encoder CLS + mean-patch at a FIXED layer, plus a decoder-state pooling.
    The decoder vector is the mean hidden state over the teacher-forced label
    positions -- the same computation the 12-dim MIA features summarise, so an
    encoder/decoder contrast is apples to apples."""
    loader = DataLoader(WDataset(df, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval()
    out = {"cls": [], "meanpatch": [], "decoder": []}
    for b in tqdm(loader, desc="embed", leave=False, dynamic_ncols=True):
        pv = b["pixel_values"].to(DEVICE); lab = b["labels"].to(DEVICE)
        enc = model.ved.encoder(pixel_values=pv, output_hidden_states=True,
                                return_dict=True)
        h = enc.hidden_states[P.LAYER].float()
        out["cls"].append(h[:, 0, :].cpu().numpy())
        out["meanpatch"].append(h[:, 1:, :].mean(1).cpu().numpy())
        if P.PROBE_DECODER:
            d = model.ved(encoder_outputs=enc, labels=lab,
                          output_hidden_states=True, return_dict=True)
            dh = d.decoder_hidden_states[-1].float()
            m = (lab != -100).float().unsqueeze(-1)
            out["decoder"].append(((dh * m).sum(1) /
                                   m.sum(1).clamp(min=1)).cpu().numpy())
    return ({k: np.concatenate(v) for k, v in out.items() if v},
            df["writer"].values, df["page"].values)


def page_partition(pages, writers, seed=S.SEED):
    """Held-out pages per writer. Support sets are always drawn from the train
    pages, evaluation always from the held-out pages."""
    rng = np.random.RandomState(seed)
    te = np.zeros(len(pages), dtype=bool)
    for w in np.unique(writers):
        m = writers == w
        wp = np.unique(pages[m]); rng.shuffle(wp)
        k = max(1, min(len(wp) - 1, int(round(0.3 * len(wp))))) if len(wp) > 1 else 0
        if k:
            te |= np.isin(pages, wp[:k]) & m
        else:
            idx = np.flatnonzero(m); te[idx[: max(1, len(idx) // 3)]] = True
    return ~te, te


def fewshot_curve(X, writers, tr, te, arm, group, feat):
    """Accuracy vs number of support words per writer, averaged over resamples.
    Sample efficiency, not asymptote, is where memorisation would show.

    FIX (2026-08-13): the original version aborted the ENTIRE curve the moment
    ANY one class had an empty train-page pool (page_partition() puts a writer
    with too few total samples entirely into the test split). One under-sampled
    writer was silently zeroing out every arm/group/feat result. Now that writer
    is dropped from the support set only -- it still sits in the held-out eval
    set and is simply never predicted correctly, which is the honest outcome for
    an attacker with zero training shots on it."""
    y, classes = pd.factorize(writers)
    all_classes = np.unique(y)
    train_classes = [c for c in all_classes if (tr & (y == c)).any()]
    no_train = [classes[c] for c in all_classes if c not in train_classes]
    if no_train:
        print(f"    [{arm}|{group}|{feat}] writer(s) with zero train-page rows "
              f"(too few samples to split) -- excluded from the support set, "
              f"still scored as misses in eval: {no_train}")
    rows = []
    for shots in P.SHOTS:
        accs, hits_all, w_all = [], [], []
        for rep in range(P.N_REPEATS):
            rng = np.random.RandomState(S.SEED + rep)
            sel = []
            for c in train_classes:
                pool = np.flatnonzero(tr & (y == c))
                sel.append(rng.choice(pool, min(shots, len(pool)), replace=False))
            if not sel:
                continue
            sel = np.concatenate(sel)
            if len(np.unique(y[sel])) < 2:
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, C=1.0))
            clf.fit(X[sel], y[sel])
            pred = clf.predict(X[te])
            hit = (pred == y[te]).astype(float)
            accs.append(100 * hit.mean())
            hits_all.append(hit); w_all.append(writers[te])
        if not accs:
            continue
        ci = boot_mean(np.concatenate(hits_all), np.concatenate(w_all), n_boot=800)
        rows.append({"arm": arm, "group": group, "feat": feat, "shots": shots,
                     "acc": 100 * ci["point"], "lo": 100 * ci["lo"],
                     "hi": 100 * ci["hi"], "acc_sd_over_reps": float(np.std(accs)),
                     "chance": 100 / len(np.unique(y)), "n_eval": int(te.sum())})
    return rows


def verification(X, writers, pages, te, arm, group, feat, n_pairs=P.N_PAIRS):
    """Same-writer vs different-writer cosine similarity over held-out pages.
    Pairs are always ACROSS pages, so page texture cannot carry the decision."""
    idx = np.flatnonzero(te)
    if len(idx) < 50:
        return None
    Z = X[idx]
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    w = writers[idx]; pg = pages[idx]
    rng = np.random.RandomState(S.SEED)
    a = rng.randint(0, len(idx), n_pairs)
    b = rng.randint(0, len(idx), n_pairs)
    keep = (a != b) & (pg[a] != pg[b])          # never within-page
    a, b = a[keep], b[keep]
    if len(a) < 200:
        return None
    sim = (Z[a] * Z[b]).sum(1)
    lab = (w[a] == w[b]).astype(int)
    if lab.sum() < 20 or (1 - lab).sum() < 20:
        return None
    auc = roc_auc_score(lab, sim)
    fpr, tpr, _ = roc_curve(lab, sim)
    eer = float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])
    ci = boot_auc(lab, sim, w[a], n_boot=600)
    return {"arm": arm, "group": group, "feat": feat, "ver_auc": auc,
            "ver_auc_lo": ci["lo"], "ver_auc_hi": ci["hi"], "eer": 100 * eer,
            "n_pos": int(lab.sum()), "n_neg": int((1 - lab).sum())}


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ctx = setup()
    vocab, fe = ctx["vocab"], ctx["fe"]
    reg = discover_checkpoints()
    if os.path.exists(P.ORACLE):
        reg["oracle"] = P.ORACLE
    reg.setdefault("oracle", reg.get("oracle_final"))
    print(f"\n  checkpoints on disk: {sorted(k for k in reg if reg.get(k))}")
    missing = [a for a in P.ARMS if a not in ("baseline",) and not reg.get(a)]
    if missing:
        print(f"  [!] not found, will be skipped: {missing}")
        print("      Attach the Stage II notebook output before quoting anything.")

    rng = np.random.RandomState(S.SEED)
    rw = rng.choice(sorted(ctx["retain_df"]["writer"].unique()),
                    P.N_RETAIN_CONTROL, replace=False)
    tw = rng.choice(sorted(ctx["test_df"]["writer"].unique()),
                    P.N_RETAIN_CONTROL, replace=False)
    sets = {
        "forget": sample_per_writer(ctx["forget_df"], P.WORDS_PER_WRITER),
        "retain_control": sample_per_writer(
            ctx["retain_df"][ctx["retain_df"]["writer"].isin(rw)]
            .reset_index(drop=True), P.WORDS_PER_WRITER),
        "test_reference": sample_per_writer(
            ctx["test_df"][ctx["test_df"]["writer"].isin(tw)]
            .reset_index(drop=True), P.WORDS_PER_WRITER),
    }
    for k, v in sets.items():
        print(f"  {k:<15} {len(v):>6,} words / {v['writer'].nunique()} writers "
              f"(chance {100/v['writer'].nunique():.2f}%)")

    model = build_model(vocab, ctx["n_classes"])
    curve_rows, ver_rows = [], []
    for arm in P.ARMS:
        if arm == "baseline":
            sd = ctx["base_sd"]
        else:
            p = reg.get(arm)
            if not p or not os.path.exists(p):
                continue
            sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        load_into(model, sd, arm, strict_report=False)
        head = writer_acc(model, ctx["forget_df"], vocab, fe, n=800)
        print(f"\n  --- {arm} ---  head forget wACC {head:.2f} %")

        for group, d in sets.items():
            E, w, pg = embed(model, d, vocab, fe)
            tr, te = page_partition(pg, w)
            for feat, X in E.items():
                if feat == P.ALSO_POOL and group != "forget":
                    continue                    # keep the grid small
                curve_rows += [dict(r, head_forget_wACC=head)
                               for r in fewshot_curve(X, w, tr, te, arm, group, feat)]
                v = verification(X, w, pg, te, arm, group, feat)
                if v:
                    ver_rows.append(dict(v, head_forget_wACC=head))

    C = pd.DataFrame(curve_rows); V = pd.DataFrame(ver_rows)
    C.to_csv(f"{P.OUT}/fewshot_curves.csv", index=False)
    V.to_csv(f"{P.OUT}/verification.csv", index=False)
    if C.empty:
        print("\n  [!!] fewshot_curves is empty -- no arm/group/feat combo "
              "produced a usable curve. Identification tables below will be "
              "blank; verification.csv may still have data.")
        C = pd.DataFrame(columns=["arm", "group", "feat", "shots", "acc", "lo",
                                   "hi", "acc_sd_over_reps", "chance", "n_eval",
                                   "head_forget_wACC"])

    # ── SAMPLE EFFICIENCY TABLE (the point of the rewrite) ────────────────────
    print("\n" + "=" * 92)
    print(f"  FEW-SHOT IDENTIFICATION, forget writers, {P.POOL} L{P.LAYER}")
    print("  Differences, if any, live at 1-5 shots. The asymptote is generic")
    print("  Bangla style capacity and tells you nothing about membership.")
    print("-" * 92)
    q = C[(C.group == "forget") & (C.feat == P.POOL)]
    if len(q):
        piv = q.pivot_table(index="arm", columns="shots", values="acc")
        print(piv.round(1).to_string())
    print("\n  Same table, DECODER states (where your MIA finds signal):")
    qd = C[(C.group == "forget") & (C.feat == "decoder")]
    if len(qd):
        print(qd.pivot_table(index="arm", columns="shots", values="acc")
                .round(1).to_string())

    print("\n" + "=" * 92)
    print("  GENERIC-CAPACITY CONTROL")
    print("-" * 92)
    ref = C[(C.feat == P.POOL) & (C.shots == max(P.SHOTS))]
    for _, r in ref.iterrows():
        print(f"  {r['arm']:<20} {r['group']:<16} {r['acc']:6.2f}% "
              f"[{r['lo']:5.1f}, {r['hi']:5.1f}]  (chance {r['chance']:.1f}%)")
    print("\n  If forget-under-oracle ~= test_reference-under-baseline, the probe")
    print("  is measuring how well a Bangla-trained ViT separates ANY 20 writers,")
    print("  not whether these 20 were deleted. Say that explicitly -- it is the")
    print("  result, and it invalidates every published re-probe audit that was")
    print("  reported without an oracle floor.")

    print("\n" + "=" * 92)
    print("  VERIFICATION (same-writer vs different-writer, across pages)")
    print("-" * 92)
    if len(V):
        print(f"  {'arm':<20} {'group':<16} {'feat':<11} {'AUC':>18} {'EER':>7}")
        for _, r in V.sort_values(["feat", "group", "arm"]).iterrows():
            print(f"  {r['arm']:<20} {r['group']:<16} {r['feat']:<11} "
                  f"{r['ver_auc']:.3f}[{r['ver_auc_lo']:.2f},{r['ver_auc_hi']:.2f}] "
                  f"{r['eer']:6.2f}%")
    print("\n  EER keeps resolving where closed-set accuracy has saturated. If the")
    print("  arms separate here but not in the identification table, report EER as")
    print("  the primary probe metric and say why.")

    summ = (C[C.shots == min(P.SHOTS)]
            .merge(C[C.shots == max(P.SHOTS)], on=["arm", "group", "feat"],
                   suffixes=("_lowshot", "_highshot")))
    summ.to_csv(f"{P.OUT}/summary.csv", index=False)
    print(f"\n  saved -> {P.OUT}/fewshot_curves.csv, verification.csv, summary.csv")
    return C, V


if __name__ == "__main__":
    main()