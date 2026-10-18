"""
exp3_relearn.py
===============
~15 min per (arm, k) pair. With 4 arms x 3 budgets, ~2-3 h.

THE QUESTION
------------
An adversary obtains a handful of words from the deleted writers and fine-tunes
the released model. How fast does writer identity come back?

If knowledge was REMOVED, the unlearned model must relearn these writers from
scratch, at the same rate as a model that never saw them. If knowledge was
merely SUPPRESSED, the unlearned model snaps back far faster, because the
features are already there and only a few output directions need fixing.

THE CONTROL IS THE WHOLE EXPERIMENT
-----------------------------------
The oracle is the from-scratch rate. It never saw the forget writers, but its
head has all 95 units (see exp1), so it is directly fine-tunable on the same
k samples with the same optimiser. Comparing an unlearned arm's recovery curve
against the oracle's is the only way to distinguish "identity was deleted" from
"identity was hidden". A recovery curve reported without this control says
nothing: any model can learn 20 writers from 500 samples given enough steps.

Headline metric: STEPS-TO-HALF -- optimiser steps to reach 50% of the original
baseline's forget wACC. Report the ratio  steps(arm) / steps(oracle).
Ratio ~1 means genuine deletion. Ratio << 1 means suppression.

CONTROLS
--------
 * Relearn and evaluation sets are PAGE-DISJOINT within each writer, so recovery
   is generalisation to unseen pages, not recall of the k images themselves.
 * Retain wACC is tracked throughout: an attack that recovers forget identity by
   wrecking the rest of the model is not a useful attack, and you should be able
   to say so with numbers.
 * The recognition loss is kept in the objective at low weight. Without it the
   encoder drifts, retain wACC collapses, and the curve is uninterpretable.
 * FULL fine-tuning by default. HEAD_ONLY=True gives the weaker, purely linear
   version -- run both if you want to separate "head was broken" from
   "encoder was damaged".

OUTPUT
------
  /kaggle/working/relearn/relearn_curves.csv
  /kaggle/working/relearn/relearn_summary.csv
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader



class R:
    OUT = "/kaggle/working/relearn"
    ORACLE = "/kaggle/working/oracle/oracle_final.pt"

    K_PER_WRITER = [5, 10, 25]     # 100 / 200 / 500 total samples over 20 writers
    STEPS = 200
    EVAL_EVERY = 20
    BATCH = 12
    LR = 1e-5
    WD = 0.0
    CLIP = 1.0
    LAMBDA_REC = 0.5               # keeps the encoder from drifting
    HEAD_ONLY = False
    AMP = True

    EVAL_N = 800                   # held-out forget words
    RETAIN_EVAL_N = 400

    ARMS = ["baseline", "oracle", "wic_unmasked@300", "wic_unmasked@800",
            "random_label@800"]


os.makedirs(R.OUT, exist_ok=True)


def page_disjoint_kshot(forget_df, k, seed=S.SEED):
    """k words per writer for the attacker, drawn from a SUBSET of that writer's
    pages; evaluation uses only the writer's other pages."""
    rng = np.random.RandomState(seed)
    tr_idx, te_idx = [], []
    for w, g in forget_df.groupby("writer"):
        pages = g["page"].unique().copy()
        rng.shuffle(pages)
        if len(pages) < 2:                       # degenerate writer: word-level split
            gi = g.sample(frac=1.0, random_state=seed).index.to_numpy()
            tr_idx.append(gi[:k]); te_idx.append(gi[k:])
            continue
        n_tr_pages = max(1, len(pages) // 3)
        tr_pages = set(pages[:n_tr_pages])
        gtr = g[g["page"].isin(tr_pages)]
        gte = g[~g["page"].isin(tr_pages)]
        take = gtr.sample(min(k, len(gtr)), random_state=seed)
        tr_idx.append(take.index.to_numpy())
        te_idx.append(gte.index.to_numpy())
    tr = forget_df.loc[np.concatenate(tr_idx)].reset_index(drop=True)
    te = forget_df.loc[np.concatenate(te_idx)].reset_index(drop=True)
    return tr, te


def infinite(loader):
    while True:
        for b in loader:
            yield b


def relearn(model, state, ctx, kshot_df, eval_df, tag, cfg=S):
    seed_all(S.SEED)
    model.load_state_dict(state, strict=True)
    vocab, fe = ctx["vocab"], ctx["fe"]

    if R.HEAD_ONLY:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.writer_head.parameters():
            p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=R.LR, weight_decay=R.WD)
    scaler = torch.amp.GradScaler("cuda", enabled=R.AMP and DEVICE == "cuda")

    bs = min(R.BATCH, len(kshot_df))
    it = infinite(DataLoader(WDataset(kshot_df, vocab, fe, cfg), batch_size=bs,
                             shuffle=True, drop_last=len(kshot_df) > bs,
                             num_workers=cfg.NUM_WORKERS))

    rows = [{"step": 0,
             "forget_wACC": writer_acc(model, eval_df, vocab, fe, cfg, n=R.EVAL_N),
             "retain_wACC": writer_acc(model, ctx["retain_df"], vocab, fe, cfg,
                                       n=R.RETAIN_EVAL_N)}]
    print(f"    step   0  forget {rows[0]['forget_wACC']:6.2f} %  "
          f"retain {rows[0]['retain_wACC']:6.2f} %")

    model.train()
    for step in range(1, R.STEPS + 1):
        b = next(it)
        pv = b["pixel_values"].to(DEVICE)
        lab = b["labels"].to(DEVICE)
        wid = b["wid"].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=R.AMP and DEVICE == "cuda"):
            enc = model.ved.encoder(pixel_values=pv, return_dict=True)
            wlog = model.writer_head(enc.last_hidden_state[:, 0, :])
            rec = model.ved(encoder_outputs=enc, labels=lab, return_dict=True)
        loss = F.cross_entropy(wlog.float(), wid) + R.LAMBDA_REC * rec.loss.float()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, R.CLIP)
        scaler.step(opt); scaler.update()

        if step % R.EVAL_EVERY == 0:
            fa = writer_acc(model, eval_df, vocab, fe, cfg, n=R.EVAL_N)
            ra = writer_acc(model, ctx["retain_df"], vocab, fe, cfg,
                            n=R.RETAIN_EVAL_N)
            rows.append({"step": step, "forget_wACC": fa, "retain_wACC": ra})
            print(f"    step {step:3d}  forget {fa:6.2f} %  retain {ra:6.2f} %")
            model.train()

    model.eval()
    q = rec_quality(model, ctx["test_df"], vocab, fe, cfg, n=300)
    for r in rows:
        r["tag"] = tag
    rows[-1]["test_WER_after"] = q["WER"]
    return pd.DataFrame(rows)


def steps_to_half(curve, target):
    """First step at which forget wACC reaches half the original baseline value.
    np.inf if it never does within the budget."""
    hit = curve[curve["forget_wACC"] >= 0.5 * target]
    return int(hit["step"].iloc[0]) if len(hit) else np.inf


def main():
    ctx = setup()
    vocab, fe = ctx["vocab"], ctx["fe"]
    reg = discover_checkpoints()
    if os.path.exists(R.ORACLE):
        reg.setdefault("oracle", R.ORACLE)
    reg.setdefault("oracle", reg.get("oracle_final"))
    print(f"\n  checkpoints on disk: {sorted(k for k in reg if reg[k])}")

    model = build_model(vocab, ctx["n_classes"])

    # the reference the attack is trying to restore
    load_into(model, ctx["base_sd"], "baseline", strict_report=False)
    baseline_forget_acc = writer_acc(model, ctx["forget_df"], vocab, fe, n=800)
    print(f"\n  original baseline forget wACC = {baseline_forget_acc:.2f} % "
          f"(the recovery target)")

    curves, summary = [], []
    for arm in R.ARMS:
        if arm == "baseline":
            sd = ctx["base_sd"]
        else:
            p = reg.get(arm)
            if not p or not os.path.exists(p):
                print(f"\n  [skip] {arm}: not found"
                      + ("  <-- run exp1; without the oracle the recovery curves "
                         "have no control and cannot be interpreted"
                         if arm == "oracle" else ""))
                continue
            sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        load_into(model, sd, arm, strict_report=False)
        state = copy.deepcopy(model.state_dict())

        for k in R.K_PER_WRITER:
            tr, te = page_disjoint_kshot(ctx["forget_df"], k)
            print(f"\n  --- {arm}  |  k={k}/writer  ({len(tr)} attack words, "
                  f"{len(te):,} held-out eval words, page-disjoint) ---")
            c = relearn(model, state, ctx, tr, te, f"{arm}|k{k}")
            c["arm"] = arm; c["k"] = k
            curves.append(c)
            s2h = steps_to_half(c, baseline_forget_acc)
            summary.append({
                "arm": arm, "k": k,
                "start_wACC": c["forget_wACC"].iloc[0],
                "final_wACC": c["forget_wACC"].iloc[-1],
                "recovered_frac": c["forget_wACC"].iloc[-1] / max(baseline_forget_acc, 1e-9),
                "steps_to_half": s2h,
                "retain_wACC_final": c["retain_wACC"].iloc[-1],
                "retain_drop": c["retain_wACC"].iloc[0] - c["retain_wACC"].iloc[-1]})

    C = pd.concat(curves, ignore_index=True)
    Sm = pd.DataFrame(summary)
    C.to_csv(f"{R.OUT}/relearn_curves.csv", index=False)

    # normalise against the oracle: the from-scratch rate
    if "oracle" in set(Sm["arm"]):
        ref = Sm[Sm.arm == "oracle"].set_index("k")
        Sm["steps_ratio_vs_oracle"] = Sm.apply(
            lambda r: r["steps_to_half"] / ref.loc[r["k"], "steps_to_half"]
            if np.isfinite(r["steps_to_half"]) and
               np.isfinite(ref.loc[r["k"], "steps_to_half"]) else np.nan, axis=1)
        Sm["recovery_excess_vs_oracle"] = Sm.apply(
            lambda r: r["final_wACC"] - ref.loc[r["k"], "final_wACC"], axis=1)
    Sm.to_csv(f"{R.OUT}/relearn_summary.csv", index=False)

    print("\n" + "=" * 92)
    print("  FEW-SHOT RELEARNING ATTACK")
    print("-" * 92)
    print(f"  {'arm':<20} {'k':>3} {'start':>7} {'final':>7} {'%recov':>7} "
          f"{'steps@half':>11} {'vs oracle':>10} {'retainDrop':>11}")
    for _, r in Sm.iterrows():
        sr = r.get("steps_ratio_vs_oracle", np.nan)
        s2h = "inf" if not np.isfinite(r["steps_to_half"]) else f"{int(r['steps_to_half'])}"
        print(f"  {r['arm']:<20} {int(r['k']):>3} {r['start_wACC']:6.2f}% "
              f"{r['final_wACC']:6.2f}% {100*r['recovered_frac']:6.1f}% "
              f"{s2h:>11} {('-' if not np.isfinite(sr) else f'{sr:.2f}x'):>10} "
              f"{r['retain_drop']:10.2f}")
    print("-" * 92)
    print("  READ IT LIKE THIS:")
    print("    ratio ~ 1.0  -> the arm relearns no faster than a model that never")
    print("                    saw these writers. Consistent with real deletion.")
    print("    ratio << 1.0 -> identity was suppressed, not removed. The features")
    print("                    survived and only the readout had to be repaired.")
    print("    Also check retainDrop: an attack that recovers forget identity by")
    print("    damaging the retain writers is not a clean recovery.")
    print(f"\n  saved -> {R.OUT}/relearn_curves.csv, {R.OUT}/relearn_summary.csv")
    return C, Sm


if __name__ == "__main__":
    main()
