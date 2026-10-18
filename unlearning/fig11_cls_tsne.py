r"""
fig11_cls_tsne.py
=================
ONE file: extracts the CLS embeddings, writes the data table, draws the figure.
Supersedes the exp11_cls_tsne_table.py / make_fig11_tsne.py split.

Six panels: Baseline, Stage I pruning only, WIC, Random Labeling, Amnesiac
(the negative-gradient checkpoint -- see NAMING below), DELETE.

THREE MODES, PICKED AUTOMATICALLY
---------------------------------
  1. cls_tsne_table.csv exists      -> plot it. No torch, no GPU, no dataset.
  2. cls_embeddings.npz exists      -> re-fit t-SNE on CPU, write the CSV, plot.
     (--force-tsne to take this path even when the CSV is already there)
  3. neither exists                 -> full extraction from the checkpoints,
     write the NPZ and the CSV, then plot. Needs a GPU, the BnHTRD dataset and
     the 6 checkpoints.

So the first run on Kaggle does everything; every later run, anywhere, is
seconds off the table. The table stays the artifact -- restyle the figure
without ever re-reading 11 GB of checkpoints.

    # Kaggle, first run: extract -> table -> figure
    python fig11_cls_tsne.py --out figs/fig_cls_tsne_arms

    # anywhere after that, from the table
    python fig11_cls_tsne.py --table cls_tsne/cls_tsne_table.csv --show

    # re-fit t-SNE with a different perplexity, no GPU
    python fig11_cls_tsne.py --npz cls_tsne/cls_embeddings.npz \
                             --force-tsne --perplexity 50

RUNNING MODE 3
--------------
Paste this BELOW the pathok_common cell and the prune_unlearn_5methods cell,
exactly like every other exp*.py here. It consumes S, DEVICE, setup(),
build_model(), load_into(), WDataset, sample_per_writer(), prune_by_ratio() and
seed_all() from those cells rather than redefining them, so the split, the
preprocessing and the writer->index map are the ones the rest of the audit uses.
Modes 1 and 2 need none of that -- only numpy/pandas/matplotlib (+ sklearn for
mode 2).

NAMING, STATED ONCE SO THE FIGURE CANNOT MISLEAD
------------------------------------------------
unlearned_amnesiac_step800.pt is NOT Graves et al.'s record-and-subtract
Amnesiac Unlearning. prune_unlearn_5methods.amnesiac_shock() applies reversed
gradient steps on the forget set -- gradient ascent, i.e. the Negative Gradient
baseline, which an earlier version of this project filed under the wrong name.
The panel label says so. To title it plainly "Amnesiac", edit that one string in
ARMS; the checkpoint it reads is the same file either way.

pruned_only.pt was written by prune_unlearn_5methods.py to
/kaggle/working/prune_unlearn/ but was never downloaded. If the glob misses it,
Stage I is reconstructed from the baseline via prune_by_ratio() under
seed_all() -- the same procedure that produced it. Loaded vs. reconstructed is
recorded per row in the `provenance` column, so the two never blur together.

--out takes NO extension; one file per --format. The PDF keeps the coloured
forget points vector and rasterises the grey retain cloud, so it stays a few
hundred KB while the part carrying the evidence stays sharp. Fonts embed as
TrueType, not Type 3 (Elsevier preflight rejects Type 3).
"""

import argparse
import glob
import json
import os
import sys

import matplotlib

# Backend must be chosen before pyplot is imported; only force the headless one
# when no window is wanted, or --show silently draws nothing.
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# ==============================================================================
# WHAT GETS PLOTTED
# ==============================================================================
ARMS = [
    dict(key="baseline",     file="baseline_clean_final.pt",
         label="Baseline (nothing unlearned)"),
    dict(key="pruned_only",  file="pruned_only.pt",
         label="Stage I: pruning only"),
    dict(key="wic",          file="unlearned_wic_unmasked_step800.pt",
         label="WIC"),
    dict(key="random_label", file="unlearned_random_label_step800.pt",
         label="Random Labeling"),
    dict(key="amnesiac",     file="unlearned_amnesiac_step800.pt",
         label="Negative Gradient\n(previously labelled Amnesiac)"),
    dict(key="delete",       file="unlearned_delete_step800.pt",
         label="DELETE"),
]
PANEL_ORDER = [a["key"] for a in ARMS]
LABELS = {a["key"]: a["label"] for a in ARMS}

# Sampling. The 20 forget writers are coloured individually; retain writers sit
# behind them in grey purely as context, so they get fewer words each.
WORDS_PER_FORGET_WRITER = 40
WORDS_PER_RETAIN_WRITER = 8
MAX_WORDS_PER_PAGE = 6      # stops one page's ink weight / lighting from forming
                            # a cluster that reads as writer identity
EXTRACT_BATCH = 32

RETAIN_KW = dict(s=6, c="#c8c8c8", marker="^", linewidths=0, alpha=0.55,
                 zorder=1, rasterized=True)
FORGET_KW = dict(s=14, marker="o", linewidths=0.3, edgecolors="white",
                 alpha=0.95, zorder=2)


# ==============================================================================
# STAGE A -- EXTRACTION (mode 3; needs GPU + dataset + checkpoints)
# ==============================================================================
def _require(name):
    """Pull a symbol out of the notebook's globals, with an error that says what
    to do instead of an opaque NameError halfway through a GPU run."""
    if name not in globals():
        raise NameError(
            "`{}` is not defined. Extraction (mode 3) expects this file to be "
            "run BELOW the pathok_common and prune_unlearn_5methods cells, the "
            "same way every other exp*.py here is. If you only want to draw the "
            "figure, pass --table <cls_tsne_table.csv> or --npz "
            "<cls_embeddings.npz> and no GPU is needed.".format(name))
    return globals()[name]


def find_ckpt(filename):
    """Locate a checkpoint anywhere under /kaggle/input, /kaggle/working or the
    cwd. Mirrors discover_checkpoints()'s glob, so attached-dataset layout
    changes need no edits here."""
    pats = ["/kaggle/input/**/" + filename,
            "/kaggle/working/**/" + filename,
            "./**/" + filename]
    hits = sorted({h for p in pats for h in glob.glob(p, recursive=True)})
    return hits[0] if hits else None


def load_arm(arm, ctx):
    """(model, provenance). Reconstructs Stage I if pruned_only.pt is missing;
    raises for any other missing checkpoint -- a silently substituted arm is
    worse than a missing panel."""
    import torch
    build_model, load_into = _require("build_model"), _require("load_into")
    vocab, fe, n_classes = ctx["vocab"], ctx["fe"], ctx["n_classes"]
    path = find_ckpt(arm["file"])

    if path is not None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(vocab, n_classes)
        model = load_into(model, ck["model"], arm["key"])
        return model, "loaded:" + os.path.basename(path)

    if arm["key"] != "pruned_only":
        raise FileNotFoundError(
            arm["file"] + " not found under /kaggle/input, /kaggle/working or "
            "the cwd. Attach the checkpoint dataset -- do not drop the panel "
            "silently.")

    print("  [pruned_only] checkpoint not found -- re-running Stage I pruning "
          "from the baseline (prune_by_ratio, seeded)")
    _require("seed_all")()
    model = build_model(vocab, n_classes)
    model = load_into(model, ctx["base_sd"], "baseline-for-pruning")
    _require("prune_by_ratio")(model, ctx["forget_df"], ctx["retain_df"],
                               vocab, fe)
    return model.eval(), "reconstructed:prune_by_ratio"


def cap_per_page(df, k, seed):
    """At most k crops per page. Shuffle-then-head rather than groupby.apply:
    groupby.sample(n=k) raises on pages holding fewer than k words, and
    groupby.apply(...sample...) trips pandas 2.2's grouping-column deprecation."""
    if k is None:
        return df
    return (df.sample(frac=1.0, random_state=seed)
              .groupby("page", group_keys=False)
              .head(k)
              .reset_index(drop=True))


def build_panel_frame(ctx):
    """The fixed crop set every arm is scored on -- same rows, same order, every
    panel, so differences between panels are the encoder and not the draw."""
    S = _require("S")
    sample_per_writer = _require("sample_per_writer")

    forget = cap_per_page(ctx["forget_df"], MAX_WORDS_PER_PAGE, S.SEED)
    forget = sample_per_writer(forget, WORDS_PER_FORGET_WRITER)
    forget["group"] = "forget"

    retain = cap_per_page(ctx["retain_df"], MAX_WORDS_PER_PAGE, S.SEED)
    retain = sample_per_writer(retain, WORDS_PER_RETAIN_WRITER)
    retain["group"] = "retain"

    panel = (pd.concat([forget, retain], ignore_index=True)
               .sort_values(["group", "writer", "image_path"])
               .reset_index(drop=True))
    f, r = panel[panel["group"] == "forget"], panel[panel["group"] == "retain"]
    print("  panel: {:,} crops | forget {:,} ({}w) | retain {:,} ({}w)".format(
        len(panel), len(f), f["writer"].nunique(),
        len(r), r["writer"].nunique()))
    return panel


def extract_cls(model, df, ctx):
    """768-d CLS per crop -- the exact tensor writer_acc() feeds to
    model.writer_head, so the figure and the Table 11 numbers read the same
    representation."""
    import torch
    from torch.utils.data import DataLoader
    S, WDataset, DEVICE = _require("S"), _require("WDataset"), _require("DEVICE")

    loader = DataLoader(WDataset(df, ctx["vocab"], ctx["fe"]),
                        batch_size=EXTRACT_BATCH, shuffle=False,
                        num_workers=S.NUM_WORKERS)
    model.eval()
    out = []
    with torch.no_grad():
        for b in loader:
            enc = model.ved.encoder(pixel_values=b["pixel_values"].to(DEVICE),
                                    return_dict=True)
            out.append(enc.last_hidden_state[:, 0, :].float().cpu().numpy())
    return np.concatenate(out, axis=0)


def run_extraction(npz_path):
    """Mode 3. Writes cls_embeddings.npz and returns (panel_df, embeddings,
    provenance). t-SNE is NOT fitted here -- that is stage B, so perplexity can
    be retuned later without a GPU."""
    import torch
    ctx = _require("setup")()
    panel = build_panel_frame(ctx)

    embeddings, provenance = {}, {}
    for i, arm in enumerate(ARMS):
        print("\n[{}/{}] {}".format(i + 1, len(ARMS), arm["key"]))
        model, prov = load_arm(arm, ctx)
        provenance[arm["key"]] = prov
        print("  provenance: " + prov)

        Z = extract_cls(model, panel, ctx)
        embeddings[arm["key"]] = Z.astype(np.float32)
        print("  CLS: {}".format(Z.shape))

        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(npz_path)) or ".", exist_ok=True)
    np.savez_compressed(
        npz_path,
        writer=panel["writer"].values.astype(str),
        wid=panel["wid"].values.astype(np.int64),
        group=panel["group"].values.astype(str),
        page=panel["page"].values.astype(str),
        image_path=panel["image_path"].values.astype(str),
        provenance=np.array(json.dumps(provenance)),
        **embeddings)
    print("\nwrote " + npz_path)
    return panel, embeddings, provenance


# ==============================================================================
# STAGE B -- t-SNE -> TABLE (CPU only)
# ==============================================================================
def fit_tsne(Z, perplexity, n_iter, seed):
    """PCA-50 then t-SNE-2, fitted per arm and never jointly."""
    import inspect
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n = Z.shape[0]
    # Stay in float32. Upcasting the full (n, 768) block to float64 doubles peak
    # memory for no accuracy that survives t-SNE anyway, and it is what actually
    # blew up on a 6 GB box. One copy, then subtract in place.
    Z = np.array(Z, dtype=np.float32, copy=True)
    Z -= Z.mean(0, dtype=np.float32, keepdims=True)
    Z = PCA(n_components=min(50, Z.shape[1], n), random_state=seed).fit_transform(Z)
    perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    # sklearn renamed n_iter -> max_iter in 1.5 and dropped n_iter in 1.7;
    # Kaggle images straddle that boundary, so use whichever this build takes.
    kw = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    ts = TSNE(n_components=2, perplexity=perp, init="pca",
              learning_rate="auto", random_state=seed, **{kw: n_iter})
    return ts.fit_transform(Z)


def load_npz(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    meta = {k: z[k] for k in ("writer", "wid", "group", "page", "image_path")
            if k in z.files}
    panel = pd.DataFrame(meta)
    # An npz written before `wid` was stored still plots fine -- the figure
    # colours by `writer`, not by class index.
    for col in ("wid", "page", "image_path"):
        if col not in panel.columns:
            panel[col] = -1 if col == "wid" else ""
    prov = json.loads(str(z["provenance"])) if "provenance" in z.files else {}
    emb = {a["key"]: z[a["key"]] for a in ARMS if a["key"] in z.files}
    return panel, emb, prov


def build_table(panel, embeddings, provenance, perplexity, n_iter, seed):
    rows = []
    for i, arm in enumerate(ARMS):
        key = arm["key"]
        if key not in embeddings:
            print("WARNING: no embeddings for " + key + " -- panel omitted.")
            continue
        print("  t-SNE: {} ({} points)".format(key, embeddings[key].shape[0]))
        xy = fit_tsne(embeddings[key], perplexity, n_iter, seed)
        rows.append(pd.DataFrame({
            "arm": key,
            "arm_label": arm["label"],
            "panel": i,
            "provenance": provenance.get(key, "unknown"),
            "writer": panel["writer"].values,
            "wid": panel["wid"].values,
            "group": panel["group"].values,
            "page": panel["page"].values,
            "image_path": panel["image_path"].values,
            "tsne_x": xy[:, 0],
            "tsne_y": xy[:, 1],
        }))
    if not rows:
        raise RuntimeError("no arms produced embeddings -- nothing to plot.")
    return pd.concat(rows, ignore_index=True)


def get_table(args):
    """Mode selection. Returns the table, extracting only as far back as needed."""
    if os.path.exists(args.table) and not args.force_tsne and not args.force_extract:
        print("reading table: " + args.table)
        return pd.read_csv(args.table, dtype={"writer": str, "page": str})

    if os.path.exists(args.npz) and not args.force_extract:
        print("reading embeddings: " + args.npz + "  (re-fitting t-SNE)")
        panel, emb, prov = load_npz(args.npz)
    else:
        print("no table and no embeddings -- running full extraction.")
        print("  (needs a GPU, the BnHTRD dataset and the 6 checkpoints; see "
              "the module docstring if this is the wrong machine for that)")
        panel, emb, prov = run_extraction(args.npz)

    table = build_table(panel, emb, prov, args.perplexity, args.n_iter, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.table)) or ".", exist_ok=True)
    table.to_csv(args.table, index=False)
    print("wrote {}  ({:,} rows)".format(args.table, len(table)))
    return table


# ==============================================================================
# STAGE C -- FIGURE
# ==============================================================================
def save_figure(fig, out, formats, dpi):
    """One render, one file per format. `out`'s own extension is dropped -- the
    formats list is the authority, so --out fig.png --format pdf cannot produce
    a PNG named .pdf."""
    stem = os.path.splitext(os.path.abspath(out))[0]
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in formats:
        path = stem + "." + ext
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print("wrote " + path)


def writer_palette(writers):
    """One stable colour per forget writer. tab20 gives 20 distinguishable hues,
    exactly the forget-set size; beyond that it wraps through tab20b so the
    figure degrades rather than erroring."""
    base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    return {w: base[i % len(base)] for i, w in enumerate(sorted(writers, key=str))}


def draw_panel(ax, sub, palette, title):
    retain = sub[sub["group"] == "retain"]
    forget = sub[sub["group"] == "forget"]

    if len(retain):
        ax.scatter(retain["tsne_x"], retain["tsne_y"], **RETAIN_KW)
    for w, g in forget.groupby("writer", sort=True):
        ax.scatter(g["tsne_x"], g["tsne_y"], color=palette[w], **FORGET_KW)

    ax.set_title(title, fontsize=8.5, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#999999")
    ax.margins(0.06)


def draw_figure(df, ncols):
    arms = [a for a in PANEL_ORDER if a in set(df["arm"])]
    missing = [a for a in PANEL_ORDER if a not in set(df["arm"])]
    if missing:
        print("WARNING: no rows for " + ", ".join(missing) +
              " -- those panels are omitted, not blank-filled.")

    palette = writer_palette(df[df["group"] == "forget"]["writer"].unique())
    nrows = int(np.ceil(len(arms) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, arm in zip(axes, arms):
        sub = df[df["arm"] == arm]
        draw_panel(ax, sub, palette, sub["arm_label"].iloc[0])
    for ax in axes[len(arms):]:
        ax.set_visible(False)

    n_fw = df[df["group"] == "forget"]["writer"].nunique()
    fig.suptitle("CLS-embedding t-SNE per unlearning arm "
                 "— forget writers coloured individually, "
                 "retain writers in grey", fontsize=10, y=0.985)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=5,
               markerfacecolor="#4878cf", markeredgecolor="white",
               label="forget writers ({}, one colour each)".format(n_fw)),
        Line2D([], [], marker="^", linestyle="", markersize=5,
               markerfacecolor="#c8c8c8", markeredgecolor="none",
               label="retain writers (context)")],
        loc="lower center", ncol=2, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.005,
             "Tight, separately-coloured clumps mean the encoder still "
             "separates the writers the method claims to have forgotten.  "
             "Each panel is fitted independently; positions are comparable "
             "within a panel, not across panels.",
             ha="center", fontsize=6.5, color="#444444")
    fig.tight_layout(rect=[0, 0.075, 1, 0.96])
    return fig, palette


# ==============================================================================
# MAIN
# ==============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Extract CLS embeddings per unlearning arm and plot them.")
    ap.add_argument("--table", default="cls_tsne/cls_tsne_table.csv",
                    help="data table; built if absent")
    ap.add_argument("--npz", default="cls_tsne/cls_embeddings.npz",
                    help="raw CLS embeddings; built if absent")
    ap.add_argument("--out", default="figs/fig_cls_tsne_arms",
                    help="output path WITHOUT extension; one file per --format")
    ap.add_argument("--format", "--formats", dest="formats", default="png,pdf",
                    help="comma-separated: png,pdf,svg,eps (default png,pdf)")
    ap.add_argument("--show", action="store_true",
                    help="open the figure in a window after writing it")
    ap.add_argument("--force-tsne", action="store_true",
                    help="re-fit t-SNE from the npz even if the table exists")
    ap.add_argument("--force-extract", action="store_true",
                    help="re-run extraction from the checkpoints (needs a GPU)")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--legend", action="store_true",
                    help="also write a per-writer colour key (off by default: "
                         "the clumping is the evidence, not which writer is which)")
    args = ap.parse_args(argv)
    formats = [f.strip().lstrip(".").lower()
               for f in args.formats.split(",") if f.strip()]

    df = get_table(args)
    fig, palette = draw_figure(df, args.ncols)
    save_figure(fig, args.out, formats, args.dpi)

    leg_fig = None
    if args.legend:
        leg_fig, la = plt.subplots(figsize=(6, 2.2))
        la.axis("off")
        la.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=6,
                                  markerfacecolor=c, markeredgecolor="white",
                                  label="writer " + str(w))
                           for w, c in palette.items()],
                  loc="center", ncol=5, frameon=False, fontsize=7)
        save_figure(leg_fig, os.path.splitext(args.out)[0] + "_writer_legend",
                    formats, args.dpi)

    keys = ["panel", "arm", "arm_label", "provenance"]
    summary = df.groupby(keys, as_index=False).agg(n_points=("tsne_x", "size"))
    fw = (df[df["group"] == "forget"].groupby(keys, as_index=False)
          .agg(n_forget_writers=("writer", "nunique"),
               n_forget_points=("tsne_x", "size")))
    print()
    print(summary.merge(fw, on=keys, how="left")
                 .sort_values("panel").to_string(index=False))

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        if leg_fig is not None:
            plt.close(leg_fig)
    return df


if __name__ == "__main__":
    main()
