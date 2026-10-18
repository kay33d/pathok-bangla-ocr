# ═══════════════════════════════════════════════════════════════════════════
#  stage1_null_control
#  Run AFTER stage1_pruning.py, in the same session (it reuses `ctx`, S,
#  collect_stats, neuron_scores, rank_scores, apply_pruning, restore, WDataset).
#
#  Three things this adds to the Stage I sweep:
#    1. Effect sizes (Cohen's d) -- what frac|t|>2 should have been read as.
#    2. A sensitive metric (logit margin and NLL), not just argmax accuracy.
#    3. THE NULL CONTROL: rank neurons on retain-vs-retain, with nobody
#       deleted, and sweep identically. If the real gap sits inside the null
#       band, the ranking carries no deletion-specific signal.
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np, torch

# ── 1. effect sizes: what frac|t|>2 should have been ────────────────────────
st = torch.load(f"{S.OUT}/act_stats.pt", weights_only=False)

def cohens_d(sf, sr):
    out = {}
    for n in sf:
        mf, vf, nf = sf[n]; mr, vr, nr = sr[n]
        sp = torch.sqrt(((nf-1)*vf + (nr-1)*vr) / max(nf+nr-2, 1) + 1e-12)
        out[n] = (mf - mr) / sp
    return out

dd = cohens_d(st["forget"], st["retain"])
allv = torch.cat([v.abs() for v in dd.values()]).numpy()
n_per = st["forget"]["L5.ffn"][2]
print(f"  n per group = {n_per}  ->  |t|>2 means |d|>{2*np.sqrt(2/n_per):.3f}")
print(f"  |d|  p50 {np.percentile(allv,50):.3f}  p95 {np.percentile(allv,95):.3f}  "
      f"max {allv.max():.3f}")
for thr, lab in [(0.2,"small"), (0.5,"medium"), (0.8,"large")]:
    print(f"    frac |d| > {thr} ({lab:6}): {(allv>thr).mean():.4f}")

# ── 2. sensitive metric: margin and NLL, not just argmax ────────────────────
@torch.no_grad()
def writer_diag(model, df, vocab, fe, cfg=S, n=None):
    n = n or cfg.EVAL_N
    d = df.sample(min(n, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval(); A, M, N = [], [], []
    for b in loader:
        pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
        enc = model.ved.encoder(pixel_values=pv, return_dict=True)
        wl = model.writer_head(enc.last_hidden_state[:, 0, :]).float()
        lp = torch.log_softmax(wl, -1)
        true = lp.gather(1, wid[:, None]).squeeze(1)
        tmp = lp.clone(); tmp.scatter_(1, wid[:, None], -1e9)
        A.append((wl.argmax(-1) == wid).float().cpu())
        M.append((true - tmp.max(1).values).cpu())
        N.append((-true).cpu())
    return {"ACC": 100*torch.cat(A).mean().item(),
            "margin": torch.cat(M).mean().item(),
            "NLL": torch.cat(N).mean().item()}

# ── 3. NULL CONTROL: rank on retain-vs-retain, sweep identically ───────────
def null_control(ctx, cfg=S, group="ffn", rates=(0.05, 0.15, 0.30, 0.40),
                 n_pseudo=20):
    model, vocab, fe = ctx["model"], ctx["vocab"], ctx["fe"]
    rng = np.random.RandomState(cfg.SEED)
    rw = sorted(ctx["retain_df"]["writer"].unique())
    pf = set(rng.choice(rw, n_pseudo, replace=False))
    A = ctx["retain_df"][ctx["retain_df"]["writer"].isin(pf)].reset_index(drop=True)
    B = ctx["retain_df"][~ctx["retain_df"]["writer"].isin(pf)].reset_index(drop=True)
    print(f"  pseudo-forget {len(A):,}/{n_pseudo}w  vs  "
          f"pseudo-retain {len(B):,}/{len(rw)-n_pseudo}w  (neither deleted)")

    restore(model, ctx["state"])
    sa = collect_stats(model, A, vocab, fe, cfg, "pseudo-F")
    sb = collect_stats(model, B, vocab, fe, cfg, "pseudo-R")
    ranked = rank_scores(neuron_scores(sa, sb, "welch"), cfg)

    print(f"\n  NULL CONTROL -- {group}")
    print(f"    {'rate':>6} {'ACC_A':>8} {'ACC_B':>8} {'gap':>7} "
          f"{'marg_A':>8} {'marg_B':>8}")
    rows = []
    for r in rates:
        restore(model, ctx["state"])
        apply_pruning(model, ranked, {group: r}, verbose=False)
        da = writer_diag(model, A, vocab, fe, cfg)
        db = writer_diag(model, B, vocab, fe, cfg)
        print(f"    {r:6.2f} {da['ACC']:7.2f}% {db['ACC']:7.2f}% "
              f"{db['ACC']-da['ACC']:6.2f} {da['margin']:8.2f} {db['margin']:8.2f}")
        rows.append({"rate": r, "ACC_A": da["ACC"], "ACC_B": db["ACC"],
                     "gap": db["ACC"]-da["ACC"], "margin_A": da["margin"],
                     "margin_B": db["margin"]})
    restore(model, ctx["state"])
    return pd.DataFrame(rows)

# ── 4. real arm, same rates, same sensitive metrics ────────────────────────
def real_arm(ctx, cfg=S, group="ffn", rates=(0.05, 0.15, 0.30, 0.40)):
    model, vocab, fe = ctx["model"], ctx["vocab"], ctx["fe"]
    print(f"\n  REAL -- {group}")
    print(f"    {'rate':>6} {'ACC_f':>8} {'ACC_r':>8} {'gap':>7} "
          f"{'marg_f':>8} {'marg_r':>8}")
    rows = []
    for r in rates:
        restore(model, ctx["state"])
        apply_pruning(model, ctx["ranked"], {group: r}, verbose=False)
        df_ = writer_diag(model, ctx["forget_df"], vocab, fe, cfg)
        dr_ = writer_diag(model, ctx["retain_df"], vocab, fe, cfg)
        print(f"    {r:6.2f} {df_['ACC']:7.2f}% {dr_['ACC']:7.2f}% "
              f"{dr_['ACC']-df_['ACC']:6.2f} {df_['margin']:8.2f} {dr_['margin']:8.2f}")
        rows.append({"rate": r, "ACC_f": df_["ACC"], "ACC_r": dr_["ACC"],
                     "gap": dr_["ACC"]-df_["ACC"], "margin_f": df_["margin"],
                     "margin_r": dr_["margin"]})
    restore(model, ctx["state"])
    return pd.DataFrame(rows)

real = real_arm(ctx)
null = null_control(ctx)
real.to_csv(f"{S.OUT}/real_arm.csv", index=False)
null.to_csv(f"{S.OUT}/null_control.csv", index=False)
print("\n  gap comparison (real should exceed null if Stage I works):")
print(pd.DataFrame({"rate": real["rate"], "real_gap": real["gap"].values,
                    "null_gap": null["gap"].values,
                    "excess": real["gap"].values - null["gap"].values}))
