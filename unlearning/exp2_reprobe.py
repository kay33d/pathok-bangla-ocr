"""
exp2_reprobe.py
===============
THE CHEAP ONE. ~20-40 min for all arms. Run it the same night as exp1 starts.

THE QUESTION
------------
Stage II drives forget-set writer accuracy to 0%. Did it remove writer identity
from the ENCODER, or did it only lobotomise the 95-way linear head that reads
the CLS token?

THE TEST
--------
Freeze the post-unlearning encoder. Throw the writer head away. Train a FRESH
20-way classifier on forget-writer embeddings. If it recovers high accuracy,
the biometric information never left the representation and the reported
"0.00% forget accuracy" is a property of one linear layer, not of the model.

WHAT MAKES THIS INTERPRETABLE
-----------------------------
Three reference points, all measured with the identical probe:

  ORACLE  -- a model that never saw these writers. Its probe accuracy is NOT
             chance: a ViT trained on other handwriting still produces
             generically style-discriminative features. This is the FLOOR.
             Anything above it is residual memorisation of the forget writers.
  BASELINE-- the un-unlearned model. This is the CEILING.
  RETAIN  -- the same probe over 20 randomly chosen retain writers, who were
             never deleted. Confirms the probe itself works on this arm and is
             not being defeated by generic damage to the encoder.

Report every arm on the axis   floor(oracle) -> ceiling(baseline).
"Probe accuracy 91% vs oracle floor 34%" is a claim. "Probe accuracy 91%" alone
is not, because nobody knows what a clean model would have scored.

FOUR CONTROLS THAT MATTER
-------------------------
 1. SPLIT BY PAGE, not by word. Two crops from the same line share ink weight,
    lighting, pen pressure and paper texture. A within-page split measures page
    recall and inflates every number.
 2. PROBE BOTH POOLINGS. Stage II optimises the CLS -> head path specifically.
    Mean-pooled PATCH tokens are what the decoder cross-attends to and receive
    no direct pressure, so they may retain more. If mean-pool >> CLS after
    unlearning, that gap IS the evidence of targeted suppression.
 3. PROBE EVERY DEPTH. A depth profile shows where identity survives and
    substitutes for the neuron-level localisation that failed your null control.
 4. NCC alongside logistic regression. Nearest-class-centroid trains nothing,
    so it cannot be accused of re-fitting the information back in.

OUTPUT
------
  /kaggle/working/reprobe/probe_results.csv     arm x pool x layer x probe
  /kaggle/working/reprobe/probe_summary.csv     headline table
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm




class P:
    OUT = "/kaggle/working/reprobe"
    ORACLE = "/kaggle/working/oracle/oracle_final.pt"   # or an attached input path

    WORDS_PER_WRITER = 250
    LAYERS = [0, 3, 6, 9, 12]      # 0 = patch embeddings, 12 = final block
    POOLS = ["cls", "meanpatch"]
    PROBES = ["logreg", "ncc"]     # add "mlp" if you want a nonlinear probe too
    TEST_FRAC_PAGES = 0.3
    N_RETAIN_CONTROL = 20          # random retain writers, matched to |forget|

    ARMS = [                        # (label, path or None for baseline)
        ("baseline", None),
        ("oracle", ORACLE),
        ("wic_unmasked@300", None), # filled from discover_checkpoints()
        ("wic_unmasked@800", None),
        ("random_label@800", None),
    ]


os.makedirs(P.OUT, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed(model, df, vocab, fe, layers=P.LAYERS, cfg=S):
    """Return {(pool, layer): (N, 768)} plus aligned writer/page arrays."""
    loader = DataLoader(WDataset(df, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval()
    buf = {(p, l): [] for p in P.POOLS for l in layers}
    for b in tqdm(loader, desc="embed", leave=False, dynamic_ncols=True):
        pv = b["pixel_values"].to(DEVICE)
        hs = model.ved.encoder(pixel_values=pv, output_hidden_states=True,
                               return_dict=True).hidden_states
        for l in layers:
            h = hs[l].float()
            if "cls" in P.POOLS:
                buf[("cls", l)].append(h[:, 0, :].cpu().numpy())
            if "meanpatch" in P.POOLS:
                buf[("meanpatch", l)].append(h[:, 1:, :].mean(1).cpu().numpy())
    return ({k: np.concatenate(v) for k, v in buf.items()},
            df["writer"].values, df["page"].values)


def page_split(pages, writers, test_frac=P.TEST_FRAC_PAGES, seed=S.SEED):
    """Hold out whole PAGES inside each writer. Guarantees at least one page on
    each side per writer, otherwise a writer vanishes from train or test and the
    20-way problem silently becomes a 19-way one."""
    rng = np.random.RandomState(seed)
    te = np.zeros(len(pages), dtype=bool)
    for w in np.unique(writers):
        m = writers == w
        wp = np.unique(pages[m])
        rng.shuffle(wp)
        k = max(1, int(round(test_frac * len(wp))))
        k = min(k, len(wp) - 1) if len(wp) > 1 else 1   # keep >=1 page for train
        te |= np.isin(pages, wp[:k]) & m
    return ~te, te


def ncc_predict(Xtr, ytr, Xte):
    """Nearest class centroid on L2-normalised features. Trains nothing."""
    n = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Xtr, Xte = n(Xtr), n(Xte)
    classes = np.unique(ytr)
    C = np.stack([Xtr[ytr == c].mean(0) for c in classes])
    C = n(C)
    return classes[np.argmax(Xte @ C.T, axis=1)]


def run_probes(E, writers, pages, tag, chance):
    tr, te = page_split(pages, writers)
    y = pd.factorize(writers)[0]
    rows = []
    for (pool, layer), X in E.items():
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        for probe in P.PROBES:
            if probe == "logreg":
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=3000, C=1.0,
                                                       n_jobs=-1))
                clf.fit(Xtr, ytr); pred = clf.predict(Xte)
            elif probe == "mlp":
                clf = make_pipeline(StandardScaler(),
                                    MLPClassifier(hidden_layer_sizes=(256,),
                                                  max_iter=400, random_state=S.SEED))
                clf.fit(Xtr, ytr); pred = clf.predict(Xte)
            elif probe == "ncc":
                pred = ncc_predict(Xtr, ytr, Xte)
            else:
                raise ValueError(probe)
            hit = (pred == yte).astype(float)
            ci = boot_mean(hit, writers[te], n_boot=1000)
            rows.append({"arm": tag, "pool": pool, "layer": layer, "probe": probe,
                         "acc": 100 * ci["point"], "lo": 100 * ci["lo"],
                         "hi": 100 * ci["hi"], "chance": chance,
                         "n_train": int(tr.sum()), "n_test": int(te.sum())})
    return rows


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ctx = setup()
    vocab, fe = ctx["vocab"], ctx["fe"]
    reg = discover_checkpoints()
    print(f"\n  checkpoints on disk: {sorted(reg)}")

    arms = []
    for label, path in P.ARMS:
        if label == "baseline":
            arms.append((label, None)); continue
        if label == "oracle":
            p = path if os.path.exists(path) else reg.get("oracle_final")
            if p is None:
                print(f"  [skip] oracle not found -- run exp1 first. Without it "
                      f"you have a ceiling but NO FLOOR and the numbers below "
                      f"cannot be interpreted.")
                continue
            arms.append((label, p)); continue
        p = path or reg.get(label)
        if p is None:
            print(f"  [skip] {label}: not found")
            continue
        arms.append((label, p))

    # forget writers (the deletion request) and a matched retain control group
    fgt = sample_per_writer(ctx["forget_df"], P.WORDS_PER_WRITER)
    rng = np.random.RandomState(S.SEED)
    rw = rng.choice(sorted(ctx["retain_df"]["writer"].unique()),
                    P.N_RETAIN_CONTROL, replace=False)
    ret = sample_per_writer(
        ctx["retain_df"][ctx["retain_df"]["writer"].isin(rw)].reset_index(drop=True),
        P.WORDS_PER_WRITER)
    n_f = fgt["writer"].nunique(); n_r = ret["writer"].nunique()
    print(f"\n  forget probe set: {len(fgt):,} words / {n_f} writers "
          f"(chance {100/n_f:.2f}%)")
    print(f"  retain control  : {len(ret):,} words / {n_r} writers "
          f"(chance {100/n_r:.2f}%)")

    model = build_model(vocab, ctx["n_classes"])
    rows = []
    for label, path in arms:
        sd = (ctx["base_sd"] if path is None else
              torch.load(path, map_location="cpu", weights_only=False)["model"])
        load_into(model, sd, label, strict_report=False)

        head_acc = writer_acc(model, ctx["forget_df"], vocab, fe, n=800)
        print(f"\n  --- {label} ---   head-reported forget wACC: {head_acc:.2f} %")

        Ef, wf, pf = embed(model, fgt, vocab, fe)
        rows += [dict(r, group="forget", head_acc=head_acc)
                 for r in run_probes(Ef, wf, pf, label, 100 / n_f)]

        Er, wr, pr = embed(model, ret, vocab, fe)
        rows += [dict(r, group="retain_control", head_acc=float("nan"))
                 for r in run_probes(Er, wr, pr, label, 100 / n_r)]

        best = max([r for r in rows if r["arm"] == label and r["group"] == "forget"],
                   key=lambda r: r["acc"])
        print(f"      best forget probe: {best['acc']:.2f}% "
              f"[{best['lo']:.1f}, {best['hi']:.1f}]  "
              f"({best['pool']} L{best['layer']}, {best['probe']})")

    t = pd.DataFrame(rows)
    t.to_csv(f"{P.OUT}/probe_results.csv", index=False)

    # ── headline table ────────────────────────────────────────────────────────
    f = t[t.group == "forget"]
    summ = (f.loc[f.groupby("arm")["acc"].idxmax()]
              [["arm", "head_acc", "pool", "layer", "probe", "acc", "lo", "hi",
                "chance"]]
              .reset_index(drop=True))
    order = [a for a, _ in P.ARMS if a in set(summ["arm"])]
    summ["__o"] = summ["arm"].map({a: i for i, a in enumerate(order)})
    summ = summ.sort_values("__o").drop(columns="__o")

    floor = summ.loc[summ.arm == "oracle", "acc"]
    floor = float(floor.iloc[0]) if len(floor) else float("nan")
    ceil = summ.loc[summ.arm == "baseline", "acc"]
    ceil = float(ceil.iloc[0]) if len(ceil) else float("nan")
    summ["residual_pct_of_range"] = 100 * (summ["acc"] - floor) / max(ceil - floor, 1e-9)
    summ.to_csv(f"{P.OUT}/probe_summary.csv", index=False)

    print("\n" + "=" * 82)
    print("  FROZEN-ENCODER RE-PROBE  (20-way over forget writers)")
    print("-" * 82)
    print(f"  {'arm':<20} {'head wACC':>10} {'probe acc':>18} {'best config':<22} "
          f"{'% of range':>10}")
    for _, r in summ.iterrows():
        cfgs = f"{r['pool']} L{int(r['layer'])} {r['probe']}"
        print(f"  {r['arm']:<20} {r['head_acc']:9.2f}% "
              f"{r['acc']:7.2f}% [{r['lo']:5.1f},{r['hi']:5.1f}] {cfgs:<22} "
              f"{r['residual_pct_of_range']:9.1f}%")
    print("-" * 82)
    print(f"  chance {summ['chance'].iloc[0]:.2f}%   "
          f"oracle floor {floor:.2f}%   baseline ceiling {ceil:.2f}%")
    print("  '% of range' = where the arm sits between exact deletion (0) and no")
    print("  deletion at all (100). This is the number to put in the abstract.")

    # CLS vs mean-patch: the signature of head-targeted suppression
    print("\n  CLS vs MEAN-PATCH at the final layer (logreg), forget group:")
    q = f[(f.layer == max(P.LAYERS)) & (f.probe == "logreg")]
    piv = q.pivot_table(index="arm", columns="pool", values="acc")
    if {"cls", "meanpatch"} <= set(piv.columns):
        piv["meanpatch - cls"] = piv["meanpatch"] - piv["cls"]
    print(piv.round(2).to_string())
    print("\n  A LARGE positive gap only in the unlearned arms means Stage II")
    print("  suppressed the CLS->head path specifically while leaving the patch")
    print("  representation the decoder reads untouched. That is targeted")
    print("  suppression, not deletion.")

    print(f"\n  saved -> {P.OUT}/probe_results.csv, {P.OUT}/probe_summary.csv")
    return t, summ


if __name__ == "__main__":
    main()
