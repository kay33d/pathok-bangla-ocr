"""
exp1_retrain_oracle.py
======================
THE REFERENCE MODEL. Everything else in the paper is measured as a distance
from this. Run it first; ~4.5-5 h on a T4, i.e. one Kaggle session.

WHAT IT TRAINS
--------------
M_retrain: identical recipe to the clean baseline, but the 20 forget writers
are absent from the training set entirely. Same architecture, same vocabulary,
same preprocessing, same optimiser, same number of EPOCHS.

TWO DESIGN DECISIONS, BOTH DELIBERATE
-------------------------------------
1. THE WRITER HEAD KEEPS ALL 95 OUTPUT UNITS.
   The forget units are never positively supervised, but they still receive
   negative gradient through the softmax denominator -- exactly what happens in
   a model retrained without those writers on a fixed head architecture. The
   payoff is that the oracle is drop-in compatible with every downstream script:
   same w2i, same rank diagnostic out of 95, same relearn setup, same MIA.

   CONSEQUENCE YOU MUST REPORT: the oracle's forget wACC will also be ~0%, and
   its mean true-writer rank will also be high. That means "forget wACC 0.00%,
   below the 1.33% chance rate" is NOT by itself evidence of over-unlearning --
   exact deletion does the same thing. Over-unlearning is a deviation from the
   ORACLE's rank distribution, not from n_classes/2. This single fact
   recontextualises your F4 finding, and it is why this experiment runs first.

2. THE VOCABULARY IS REUSED (948 graphemes, built from retain+forget).
   A strictly exact retrain would rebuild the vocabulary from retain writers
   only. Reusing it isolates the effect of training data on WEIGHTS and keeps
   MIA features numerically comparable across models (the 12 features include
   rank and entropy terms that shift with vocabulary size). Set
   O.REBUILD_VOCAB = True for the strict version, and report both if a reviewer
   pushes. The retain-only vocabulary differs from the full one by only the
   handful of graphemes unique to forget writers, so the effect is small -- but
   say so explicitly rather than leaving it implicit.

OUTPUT
------
  /kaggle/working/oracle/
      oracle_ep01.pt ... oracle_ep08.pt
      oracle_final.pt          <- feed this to exp2, exp3, exp5
      oracle_history.csv
      vocab_oracle.json        (only if REBUILD_VOCAB)
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm



class O:
    OUT = "/kaggle/working/oracle"

    EPOCHS = 5              # match the baseline recipe
    MATCH_STEPS = False      # True -> match total optimiser STEPS instead of epochs
    BATCH = 16
    LR = 5e-5
    WEIGHT_DECAY = 0.01
    WRITER_W = 2.0
    FP16 = True
    GRAD_CKPT = False

    REBUILD_VOCAB = False    # see docstring
    SEEDS = [42]             # add e.g. [42, 43] to bound seed noise (2x cost)

    EVAL_N = 600             # per-epoch monitoring only
    BASELINE_TRAIN_STEPS = 2875 * 8   # for MATCH_STEPS


os.makedirs(O.OUT, exist_ok=True)


def train_oracle(ctx, seed=42, cfg=S):
    seed_all(seed)
    vocab, fe = ctx["vocab"], ctx["fe"]
    retain_df, val_df = ctx["retain_df"], ctx["val_df"]
    n_classes = ctx["n_classes"]

    if O.REBUILD_VOCAB:
        vocab = GraphemeVocab().build(pd.concat([retain_df, val_df])["label"])
        vocab.save(f"{O.OUT}/vocab_oracle.json")
        print(f"  [vocab] rebuilt from retain writers only: {vocab.vocab_size} "
              f"(full baseline vocab was {ctx['vocab'].vocab_size})")

    # sanity: the forget writers must be absent from every row we train on
    forget_writers = set(ctx["forget_df"]["writer"])
    assert not (set(retain_df["writer"]) & forget_writers), "FORGET LEAK into oracle"
    assert not (set(val_df["writer"]) & forget_writers), "FORGET LEAK into oracle val"

    model = build_model(vocab, n_classes, cfg).to(DEVICE)
    if O.GRAD_CKPT:
        model.ved.encoder.gradient_checkpointing_enable()
        model.ved.decoder.gradient_checkpointing_enable()

    loader = DataLoader(WDataset(retain_df, vocab, fe, cfg), batch_size=O.BATCH,
                        shuffle=True, num_workers=cfg.NUM_WORKERS,
                        pin_memory=True, drop_last=True)
    steps_per_epoch = len(loader)
    epochs = O.EPOCHS
    if O.MATCH_STEPS:
        epochs = max(1, round(O.BASELINE_TRAIN_STEPS / steps_per_epoch))
        print(f"  [match-steps] {epochs} epochs x {steps_per_epoch} = "
              f"{epochs*steps_per_epoch} steps (baseline: {O.BASELINE_TRAIN_STEPS})")
    total = steps_per_epoch * epochs

    opt = torch.optim.AdamW(model.parameters(), lr=O.LR, weight_decay=O.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=O.LR, total_steps=total, pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda", enabled=O.FP16 and DEVICE == "cuda")

    print(f"\n[oracle seed={seed}] {len(retain_df):,} samples "
          f"({retain_df['writer'].nunique()} writers) | {steps_per_epoch} steps/epoch "
          f"| {epochs} epochs | head has {n_classes} units, "
          f"{len(forget_writers)} never supervised\n")

    hist, t0 = [], time.time()
    for ep in range(epochs):
        model.train(); agg = defaultdict(float); n = 0
        pbar = tqdm(loader, desc=f"epoch {ep+1}/{epochs}", dynamic_ncols=True)
        for b in pbar:
            pv = b["pixel_values"].to(DEVICE, non_blocking=True)
            lb = b["labels"].to(DEVICE, non_blocking=True)
            wid = b["wid"].to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=O.FP16 and DEVICE == "cuda"):
                rec, wlog = model(pv, lb)
                l_rec = rec.loss
                l_wid = F.cross_entropy(wlog.float(), wid)
                loss = l_rec + O.WRITER_W * l_wid

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()

            agg["rec"] += l_rec.item(); agg["wid"] += l_wid.item()
            agg["wacc"] += (wlog.argmax(-1) == wid).float().mean().item(); n += 1
            pbar.set_postfix(rec=f"{agg['rec']/n:.3f}", wid=f"{agg['wid']/n:.3f}",
                             wacc=f"{agg['wacc']/n:.3f}")

        ck = f"{O.OUT}/oracle_s{seed}_ep{ep+1:02d}.pt"
        torch.save({"model": model.state_dict(), "w2i": ctx["w2i"], "epoch": ep + 1,
                    "seed": seed, "rebuilt_vocab": O.REBUILD_VOCAB}, ck)
        va = writer_acc(model, val_df, vocab, fe, cfg, n=O.EVAL_N)
        vq = rec_quality(model, val_df, vocab, fe, cfg, n=300)
        row = {"epoch": ep + 1, "val_ACC": va, "val_WER": vq["WER"],
               "val_CER": vq["CER"], "train_rec": agg["rec"] / n,
               "train_wacc": agg["wacc"] / n, "mins": (time.time() - t0) / 60}
        hist.append(row)
        print(f"  epoch {ep+1}: val ACC {va:.2f}%  CER {vq['CER']:.2f}%  "
              f"WER {vq['WER']:.2f}%  ({row['mins']:.0f} min)")

    return model, vocab, pd.DataFrame(hist)


def oracle_report(model, ctx, vocab, cfg=S):
    """The numbers that make the oracle useful downstream. Read the forget row
    carefully: this is what EXACT DELETION looks like, and it is the target."""
    fe = ctx["fe"]; n_classes = ctx["n_classes"]
    print("\n" + "=" * 74)
    print("  ORACLE M_retrain -- what exact deletion actually produces")
    print("-" * 74)
    out = {}
    for name, d in (("forget", ctx["forget_df"]), ("retain", ctx["retain_df"]),
                    ("val", ctx["val_df"]), ("test", ctx["test_df"])):
        line = f"  [{name:<6}] "
        if name != "test":
            a = writer_acc(model, d, vocab, fe, cfg, n=1500)
            out[f"ACC_{name}"] = a
            line += f"wACC {a:6.2f} %   "
        q = rec_quality(model, d, vocab, fe, cfg, n=1000)
        out[f"WER_{name}"] = q["WER"]; out[f"CER_{name}"] = q["CER"]
        line += f"WER {q['WER']:6.2f} %  CER {q['CER']:5.2f} %"
        print(line)

    print("\n  TRUE-WRITER LOGIT RANK under exact deletion")
    for name, d in (("forget", ctx["forget_df"]), ("retain", ctx["retain_df"])):
        r = writer_rank_diag(model, d, vocab, fe, n_classes, cfg, n=800)
        out[f"rank_{name}"] = r["mean_rank"]
        out[f"rankmed_{name}"] = r["median_rank"]
        print(f"    [{name:<6}] wACC {r['ACC']:6.2f} %  mean rank "
              f"{r['mean_rank']:5.1f}/{n_classes}  median {r['median_rank']:5.1f}")
    print(f"\n  ==> Report unlearned arms as a DEVIATION from forget mean rank "
          f"{out['rank_forget']:.1f}, not from {n_classes/2:.0f}.")
    print(f"  ==> Oracle forget wACC is {out['ACC_forget']:.2f} %. Any unlearned arm "
          f"near this value is CORRECT, not over-unlearned.")
    print(f"  ==> Oracle test WER {out['WER_test']:.2f} % is the utility ceiling "
          f"unlearning can hope to match.")
    return out


def main():
    ctx = setup()
    all_rows = []
    for seed in O.SEEDS:
        model, vocab, hist = train_oracle(ctx, seed)
        rep = oracle_report(model, ctx, vocab)
        torch.save({"model": model.state_dict(), "w2i": ctx["w2i"],
                    "seed": seed, "report": rep,
                    "rebuilt_vocab": O.REBUILD_VOCAB,
                    "epochs": O.EPOCHS},
                   f"{O.OUT}/oracle_final_s{seed}.pt")
        if seed == O.SEEDS[0]:
            torch.save({"model": model.state_dict(), "w2i": ctx["w2i"],
                        "seed": seed, "report": rep},
                       f"{O.OUT}/oracle_final.pt")   # canonical name for exp2/3/5
        hist.to_csv(f"{O.OUT}/oracle_history_s{seed}.csv", index=False)
        all_rows.append(dict(seed=seed, **rep))
        del model
        torch.cuda.empty_cache()

    t = pd.DataFrame(all_rows)
    t.to_csv(f"{O.OUT}/oracle_report.csv", index=False)
    if len(t) > 1:
        print("\n  SEED SPREAD (this is the noise floor for every claim you make):")
        print(t.describe().loc[["mean", "std", "min", "max"]].T)
    print(f"\nsaved -> {O.OUT}/oracle_final.pt")
    return t


if __name__ == "__main__":
    main()
