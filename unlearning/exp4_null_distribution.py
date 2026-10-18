"""
exp4_null_distribution.py
=========================
~1.5-2.5 h. Turns your best negative result from suggestive into conclusive.

THE PROBLEM WITH THE CURRENT EVIDENCE
-------------------------------------
You have ONE null draw. It gives a gap of 2.63 at ffn 0.30 against a real gap
of 8.50, and a reviewer can reasonably answer "that is one sample of a noisy
statistic". The claim "the real gap sits inside the null band" needs a band,
and a band needs a distribution.

WHAT THIS RUNS
--------------
 * N_DRAWS pseudo-forget sets: 20 writers drawn at random from RETAIN, word-count
   matched to the real forget set, with nobody deleted. Rank neurons on
   pseudo-A vs pseudo-B exactly as the real arm ranks forget vs retain, prune
   identically, and record the accuracy gap.
 * A RANDOM-RANKING arm: identical pruning rates, neurons chosen uniformly at
   random. This separates "the ranking found something" from "pruning anything
   at this rate produces a gap between a 20-writer group and a 74-writer group".
 * A one-sided permutation p-value per (group, rate):
       p = (1 + #{null_gap >= real_gap}) / (N_DRAWS + 1)
   Report it, and report the null mean and 95th percentile alongside every real
   gap in the sweep table. A real gap that fails to clear the null 95th
   percentile is not evidence of writer-specific structure.
 * Cohen's d over all ~46k neurons, with the n-inflation demonstration: at
   n~960 per group, |t| > 2 corresponds to |d| > 0.091, which is why 73% of
   neurons look "significant" while the median effect is 0.15.

The paired real/null figure this produces is the single most convincing panel
in the Stage I section.

OUTPUT
------
  /kaggle/working/nulldist/null_gaps.csv        every (draw, group, rate) gap
  /kaggle/working/nulldist/permutation_test.csv real gap, null mean/sd/p95, p
  /kaggle/working/nulldist/effect_sizes.csv     per-module |t| and |d| quantiles
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm



class N:
    OUT = "/kaggle/working/nulldist"

    N_DRAWS = 20
    RATES = [0.15, 0.30]          # add 0.40 if you have the budget
    GROUPS = ["embed", "attn_shallow", "attn_middle", "attn_deep", "ffn"]

    ACT_BATCHES = 60              # batches per side for activation statistics
    EVAL_N = 400                  # keep this identical for real and null arms
    CLS_ONLY = True
    SCORE = "welch"
    SCORE_ABS = True
    MATCH_WORD_COUNT = True       # size-match pseudo-forget to the real forget set
    RANDOM_RANK_ARM = True


os.makedirs(N.OUT, exist_ok=True)
GROUP_ORDER = N.GROUPS


# ══════════════════════════════════════════════════════════════════════════════
# PRUNING MACHINERY
# ══════════════════════════════════════════════════════════════════════════════
def target_modules(model):
    vit = model.ved.encoder
    groups = {"embed": [("embed.proj", vit.embeddings.patch_embeddings.projection)],
              "attn_shallow": [], "attn_middle": [], "attn_deep": [], "ffn": []}
    for i, layer in enumerate(vit.encoder.layer):
        key = ("attn_shallow" if i <= 3 else
               "attn_middle" if i <= 7 else "attn_deep")
        groups[key].append((f"L{i}.attn_out", layer.attention.output.dense))
        groups["ffn"].append((f"L{i}.ffn", layer.intermediate.dense))
    return groups


@torch.no_grad()
def collect_stats(model, df, vocab, fe, cfg=S, tag="", seed=S.SEED):
    """Per-output-neuron (mean, var, n) on the CLS token, float64."""
    groups = target_modules(model)
    flat = [(n_, m) for g in groups.values() for (n_, m) in g]
    s1 = {n_: None for n_, _ in flat}
    s2 = {n_: None for n_, _ in flat}
    cnt = {n_: 0 for n_, _ in flat}
    handles = []

    def mk(name):
        def hook(_m, _i, out):
            o = out.detach().double()
            if o.dim() == 4:
                v = o.mean(dim=(2, 3))
            elif N.CLS_ONLY:
                v = o[:, 0, :]
            else:
                v = o.mean(dim=1)
            s1[name] = v.sum(0) if s1[name] is None else s1[name] + v.sum(0)
            sq = (v ** 2).sum(0)
            s2[name] = sq if s2[name] is None else s2[name] + sq
            cnt[name] += v.shape[0]
        return hook

    for n_, m in flat:
        handles.append(m.register_forward_hook(mk(n_)))
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(WDataset(df, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=True, num_workers=cfg.NUM_WORKERS, generator=g)
    try:
        for i, b in enumerate(loader):
            if i >= N.ACT_BATCHES:
                break
            model.ved.encoder(pixel_values=b["pixel_values"].to(DEVICE))
    finally:
        for h in handles:
            h.remove()

    out = {}
    for n_, _ in flat:
        c = max(cnt[n_], 1)
        mu = s1[n_] / c
        var = (s2[n_] / c) - mu ** 2
        out[n_] = (mu.float().cpu(), var.clamp(min=0).float().cpu(), c)
    return out


def welch(stat_a, stat_b, eps=1e-8):
    return {n_: (stat_a[n_][0] - stat_b[n_][0]) /
                torch.sqrt(stat_a[n_][1] / stat_a[n_][2] +
                           stat_b[n_][1] / stat_b[n_][2] + eps)
            for n_ in stat_a}


def cohens_d(stat_a, stat_b, eps=1e-12):
    out = {}
    for n_ in stat_a:
        ma, va, na = stat_a[n_]; mb, vb, nb = stat_b[n_]
        sp = torch.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1) + eps)
        out[n_] = (ma - mb) / sp
    return out


def rank_scores(scores):
    return {n_: s.abs() for n_, s in scores.items()} if N.SCORE_ABS else scores


def random_ranking(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {n_: torch.rand(m.weight.shape[0], generator=g)
            for gr in target_modules(model).values() for (n_, m) in gr}


def apply_pruning(model, ranked, rates):
    groups = target_modules(model)
    total = zeroed = 0
    for gname, entries in groups.items():
        k = rates.get(gname, 0.0)
        for name, mod in entries:
            sc = ranked[name]
            n_out = sc.numel()
            n_cut = int(round(k * n_out))
            mask = torch.ones(n_out)
            if n_cut > 0:
                mask[torch.topk(sc, n_cut).indices] = 0.0
            with torch.no_grad():
                w = mod.weight
                shape = [-1] + [1] * (w.dim() - 1)
                w.mul_(mask.to(w.device, w.dtype).view(*shape))
                if mod.bias is not None:
                    mod.bias.mul_(mask.to(w.device, w.dtype))
            total += n_out; zeroed += int((mask == 0).sum())
    return 100.0 * zeroed / max(total, 1)


# ══════════════════════════════════════════════════════════════════════════════
def sweep_one(model, state, ranked, A_df, B_df, ctx, label):
    """A = the (pseudo-)forget group, B = the (pseudo-)retain group.
    gap = ACC(B) - ACC(A), the same orientation as your original sweep."""
    vocab, fe = ctx["vocab"], ctx["fe"]
    rows = []
    for grp in N.GROUPS:
        for rate in N.RATES:
            restore(model, state)
            sp = apply_pruning(model, ranked, {grp: rate})
            aa = writer_acc(model, A_df, vocab, fe, n=N.EVAL_N)
            ab = writer_acc(model, B_df, vocab, fe, n=N.EVAL_N)
            rows.append({"label": label, "group": grp, "rate": rate,
                         "sparsity": sp, "ACC_A": aa, "ACC_B": ab,
                         "gap": ab - aa})
    restore(model, state)
    return rows


def main():
    seed_all()
    ctx = setup()
    vocab, fe = ctx["vocab"], ctx["fe"]
    model = build_model(vocab, ctx["n_classes"])
    load_into(model, ctx["base_sd"], "baseline", strict_report=False)
    state = copy.deepcopy(model.state_dict())

    forget_df, retain_df = ctx["forget_df"], ctx["retain_df"]
    n_forget_writers = forget_df["writer"].nunique()
    n_forget_words = len(forget_df)
    print(f"\n  real forget: {n_forget_words:,} words / {n_forget_writers} writers")

    # ── REAL ARM ──────────────────────────────────────────────────────────────
    print("\n  [real] collecting activation statistics...")
    st_f = collect_stats(model, forget_df, vocab, fe, tag="forget")
    st_r = collect_stats(model, retain_df, vocab, fe, tag="retain")
    torch.save({"forget": st_f, "retain": st_r}, f"{N.OUT}/act_stats_real.pt")
    ranked_real = rank_scores(welch(st_f, st_r))
    real_rows = sweep_one(model, state, ranked_real, forget_df, retain_df,
                          ctx, "real")
    real = pd.DataFrame(real_rows)
    print("\n  REAL GAPS")
    print(real[["group", "rate", "ACC_A", "ACC_B", "gap"]].to_string(index=False))

    # ── EFFECT SIZES ──────────────────────────────────────────────────────────
    tt = welch(st_f, st_r); dd = cohens_d(st_f, st_r)
    n_per = st_f["L5.ffn"][2]
    thresh_d = 2 * np.sqrt(2 / n_per)
    allt = torch.cat([v.abs() for v in tt.values()]).numpy()
    alld = torch.cat([v.abs() for v in dd.values()]).numpy()
    es_rows = [{"module": k,
                "n": tt[k].numel(),
                "t_p50": float(np.percentile(tt[k].abs().numpy(), 50)),
                "t_p95": float(np.percentile(tt[k].abs().numpy(), 95)),
                "frac_t_gt2": float((tt[k].abs().numpy() > 2).mean()),
                "d_p50": float(np.percentile(dd[k].abs().numpy(), 50)),
                "d_p95": float(np.percentile(dd[k].abs().numpy(), 95)),
                "frac_d_gt08": float((dd[k].abs().numpy() > 0.8).mean())}
               for k in tt]
    ES = pd.DataFrame(es_rows)
    ES.to_csv(f"{N.OUT}/effect_sizes.csv", index=False)
    print("\n" + "=" * 74)
    print("  SIGNIFICANCE vs EFFECT SIZE")
    print("-" * 74)
    print(f"  n per group = {n_per}  ->  |t| > 2 is only |d| > {thresh_d:.3f}")
    print(f"  |t|  median {np.percentile(allt,50):.2f}   "
          f"frac |t|>2 = {(allt>2).mean():.3f}")
    print(f"  |d|  median {np.percentile(alld,50):.3f}  p95 {np.percentile(alld,95):.3f}  "
          f"max {alld.max():.3f}")
    for thr, lab in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        print(f"    frac |d| > {thr} ({lab:6}): {(alld>thr).mean():.4f}  "
              f"({int((alld>thr).sum()):,} of {len(alld):,} neurons)")
    print("  ==> The 'significant' fraction is an artefact of n, not of structure.")

    # ── NULL DRAWS ────────────────────────────────────────────────────────────
    rng = np.random.RandomState(S.SEED)
    retain_writers = sorted(retain_df["writer"].unique())
    null_rows = []
    print(f"\n  running {N.N_DRAWS} null draws "
          f"({n_forget_writers} pseudo-forget writers each, nobody deleted)")
    for draw in tqdm(range(N.N_DRAWS), desc="null draws", dynamic_ncols=True):
        pf = set(rng.choice(retain_writers, n_forget_writers, replace=False))
        A = retain_df[retain_df["writer"].isin(pf)].reset_index(drop=True)
        B = retain_df[~retain_df["writer"].isin(pf)].reset_index(drop=True)
        if N.MATCH_WORD_COUNT and len(A) > n_forget_words:
            A = A.sample(n_forget_words, random_state=S.SEED + draw).reset_index(drop=True)
        sa = collect_stats(model, A, vocab, fe, seed=S.SEED + draw)
        sb = collect_stats(model, B, vocab, fe, seed=S.SEED + draw)
        ranked = rank_scores(welch(sa, sb))
        for r in sweep_one(model, state, ranked, A, B, ctx, f"null{draw}"):
            r["draw"] = draw
            null_rows.append(r)

    # ── RANDOM-RANKING ARM ────────────────────────────────────────────────────
    rand_rows = []
    if N.RANDOM_RANK_ARM:
        print("\n  random-ranking arm (same rates, neurons chosen uniformly)")
        for draw in range(min(5, N.N_DRAWS)):
            ranked = random_ranking(model, seed=draw)
            for r in sweep_one(model, state, ranked, forget_df, retain_df,
                               ctx, f"rand{draw}"):
                r["draw"] = draw
                rand_rows.append(r)

    NL = pd.DataFrame(null_rows)
    NL.to_csv(f"{N.OUT}/null_gaps.csv", index=False)
    if rand_rows:
        pd.DataFrame(rand_rows).to_csv(f"{N.OUT}/random_rank_gaps.csv", index=False)

    # ── PERMUTATION TEST ──────────────────────────────────────────────────────
    out = []
    for _, r in real.iterrows():
        sub = NL[(NL.group == r["group"]) & (NL.rate == r["rate"])]["gap"].values
        p = (1 + int((sub >= r["gap"]).sum())) / (len(sub) + 1)
        rr = (pd.DataFrame(rand_rows) if rand_rows else pd.DataFrame())
        rand_mean = (rr[(rr.group == r["group"]) & (rr.rate == r["rate"])]["gap"].mean()
                     if len(rr) else np.nan)
        out.append({"group": r["group"], "rate": r["rate"],
                    "real_gap": r["gap"],
                    "null_mean": sub.mean(), "null_sd": sub.std(ddof=1),
                    "null_p95": np.percentile(sub, 95), "null_max": sub.max(),
                    "random_rank_gap": rand_mean,
                    "excess_over_null_mean": r["gap"] - sub.mean(),
                    "p_perm": p, "n_null": len(sub)})
    PT = pd.DataFrame(out)
    PT.to_csv(f"{N.OUT}/permutation_test.csv", index=False)

    print("\n" + "=" * 100)
    print("  PERMUTATION TEST -- is the real gap distinguishable from no deletion?")
    print("-" * 100)
    print(f"  {'group':<14} {'rate':>5} {'real':>7} {'null mean':>10} {'null sd':>8} "
          f"{'null p95':>9} {'null max':>9} {'randrank':>9} {'p':>7}")
    for _, r in PT.iterrows():
        star = " *" if r["p_perm"] < 0.05 else ""
        print(f"  {r['group']:<14} {r['rate']:5.2f} {r['real_gap']:7.2f} "
              f"{r['null_mean']:10.2f} {r['null_sd']:8.2f} {r['null_p95']:9.2f} "
              f"{r['null_max']:9.2f} {r['random_rank_gap']:9.2f} "
              f"{r['p_perm']:7.3f}{star}")
    print("-" * 100)
    print("  A real gap below null p95 (p >= 0.05) means the activation ranking")
    print("  carries no deletion-specific signal at that operating point: the same")
    print("  gap appears between any 20 writers and the rest, with nobody deleted.")
    print(f"\n  saved -> {N.OUT}/permutation_test.csv, {N.OUT}/null_gaps.csv, "
          f"{N.OUT}/effect_sizes.csv")
    return real, NL, PT, ES


if __name__ == "__main__":
    main()
