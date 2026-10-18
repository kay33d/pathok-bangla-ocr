r"""
make_relearn_unseen_chart.py
============================
"Forget and Unseen-Base Accuracy Across Relearning Steps" -- rebuilt.

The original PDF (thesis files/Tableau visuals/forget_unseen_relearning_accuracy.pdf,
matplotlib 3.8.2) was produced by a notebook that is not on this machine, so this
is a fresh script over the same source data:

    exp3c_batch1_real/exp2_unseen_control/relearn_curves_with_unseen.csv
        wic_unmasked@800, random_label@800, amnesiac@800  (+ baseline, oracle)
    exp3c_batch2_real/exp2_unseen_control/relearn_curves_with_unseen.csv
        delete@800, arf@800, fisher@800, wic_mgda@800, rl_mgda@800

Both files are needed for the four-arm figure -- DELETE lives in batch 2, the
other three in batch 1. Solid lines are `forget_full95` on group="forget";
dashed are `unseen_acc` on group="unseen_control", the matched genuinely-unseen
writers under the identical attack recipe. At k=5 step 0 this reproduces
8.50 / 0.25 / 55.12 / 42.63, matching the original figure.

AESTHETIC (following the Tableau relearning_steps chart)
--------------------------------------------------------
  * legends outside the plot box, right-hand side, stacked
  * Method legend as square colour swatches; plain lines, no markers
  * uniform x ticks (0, 25, ... 150) on a linear axis, instead of one crowded
    label per sampled step -- the samples are at 0,2,5,10,15,20,30,45,60,90,
    120,150, which bunch six labels into the first eighth of the axis
  * horizontal gridlines only, top/right spines dropped

    python make_relearn_unseen_chart.py --out figs/fig_relearn_unseen
    python make_relearn_unseen_chart.py --k 1 --xtick-step 10 --show

--out takes NO extension; one file per --format. Fonts embed as TrueType, not
Type 3 (Elsevier preflight rejects Type 3).

NOTE ON "AMNESIAC": amnesiac@800 is the reversed-gradient arm, which the
unlearning literature calls Negative Gradient; the paper's Table 11 renames it
accordingly. ARMS keeps the original figure's wording -- change the label there
if this figure should match the table instead.
"""

import argparse
import os
import sys

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"   # Times-compatible maths glyphs


# Order fixes both the colour assignment and the legend order.
ARMS = [
    ("wic_unmasked@800",  "WIC",             "#1f77b4"),
    ("random_label@800",  "Random Labeling", "#ff7f0e"),
    ("amnesiac@800",      "Amnesiac",        "#2ca02c"),
    ("delete@800",        "DELETE",          "#d62728"),
]

DEFAULT_CURVES = [
    r"D:\claude files\kaggle_downloads\exp3c_batch1_real\exp2_unseen_control\relearn_curves_with_unseen.csv",
    r"D:\claude files\kaggle_downloads\exp3c_batch2_real\exp2_unseen_control\relearn_curves_with_unseen.csv",
]

FORGET_KW = dict(linewidth=2.0, linestyle="-", zorder=3)
UNSEEN_KW = dict(linewidth=1.8, linestyle="--", zorder=2, alpha=0.95)


def load_curves(paths, k):
    frames = []
    for p in paths:
        if not os.path.exists(p):
            print("WARNING: missing " + p + " -- arms it holds will be absent.")
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(
            "none of the curve CSVs exist. Pass --curves with the paths to "
            "relearn_curves_with_unseen.csv (batch 1 AND batch 2 -- DELETE is "
            "only in batch 2).")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["k"].astype(int) == int(k)]

    if "valid" in df.columns:
        bad = df[df["valid"].astype(str).str.lower() != "true"]
        if len(bad):
            print("dropping {} rows with valid != True (retain-replay gate "
                  "failed)".format(len(bad)))
            df = df.drop(bad.index)

    df["step"] = df["step"].astype(int)
    return df


def series(df, arm, group, value_col):
    sub = df[(df["arm"] == arm) & (df["group"] == group)]
    sub = sub.dropna(subset=[value_col]).sort_values("step")
    return sub["step"].to_numpy(), sub[value_col].to_numpy()


def draw(df, k, xtick_step, figsize):
    fig, ax = plt.subplots(figsize=figsize)

    plotted = []
    for arm, label, colour in ARMS:
        xf, yf = series(df, arm, "forget", "forget_full95")
        xu, yu = series(df, arm, "unseen_control", "unseen_acc")
        if not len(xf) and not len(xu):
            print("WARNING: no rows for " + arm + " at k=" + str(k))
            continue
        if len(xf):
            ax.plot(xf, yf, color=colour, **FORGET_KW)
        if len(xu):
            ax.plot(xu, yu, color=colour, **UNSEEN_KW)
        plotted.append((label, colour))

    if not plotted:
        raise RuntimeError("nothing to plot at k=" + str(k))

    ax.set_xlabel("Relearning Steps", fontsize=11, fontweight="bold",
                  labelpad=8)
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Forget and Unseen-Base Accuracy Across Relearning Steps",
                 fontsize=12.5, fontweight="bold", pad=14)

    # Uniform ticks on a linear axis. The measured steps stay where they are --
    # the markers show them -- but the labels no longer crowd the left edge.
    xmax = int(df["step"].max())
    ticks = list(range(0, xmax + 1, xtick_step))
    if ticks[-1] != xmax:
        ticks.append(xmax)
    ax.set_xticks(ticks)
    ax.set_xlim(-xmax * 0.02, xmax * 1.02)

    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))

    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b0b0b0")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#333333", labelsize=9.5, length=3)

    # Both legends outside the axes, stacked at the right.
    # Method: square colour swatches, Tableau-style. Evaluation Set stays as line
    # samples -- solid vs dashed is the whole distinction, a box cannot show it.
    method_handles = [Patch(facecolor=c, edgecolor="none", label=lab)
                      for lab, c in plotted]
    set_handles = [
        Line2D([], [], color="#444444", linewidth=2.0, linestyle="-",
               label="Forget"),
        Line2D([], [], color="#444444", linewidth=1.8, linestyle="--",
               label="Unseen Base"),
    ]

    # alignment="left" puts each title flush with its entries, as Tableau does.
    leg1 = ax.legend(handles=method_handles, title="Method",
                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     frameon=False, fontsize=9.5, borderaxespad=0,
                     alignment="left", handlelength=0.9, handleheight=0.9)
    leg1.get_title().set_fontsize(10)
    leg1.get_title().set_fontweight("bold")
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=set_handles, title="Evaluation Set",
                     loc="upper left", bbox_to_anchor=(1.02, 0.58),
                     frameon=False, fontsize=9.5, borderaxespad=0,
                     alignment="left")
    leg2.get_title().set_fontsize(10)
    leg2.get_title().set_fontweight("bold")

    fig.tight_layout()
    # Returned so savefig can count them: a legend attached via ax.add_artist is
    # invisible to bbox_inches="tight", which is what clipped "Random Labeling".
    return fig, [leg1, leg2]


def save_figure(fig, out, formats, dpi, extra_artists=()):
    stem = os.path.splitext(os.path.abspath(out))[0]
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in formats:
        path = stem + "." + ext
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    bbox_extra_artists=list(extra_artists))
        print("wrote " + path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Forget vs unseen-base accuracy across relearning steps.")
    ap.add_argument("--curves", nargs="+", default=DEFAULT_CURVES,
                    help="relearn_curves_with_unseen.csv paths (batch1 batch2)")
    ap.add_argument("--k", type=int, default=5, help="shots per writer")
    ap.add_argument("--out", default="figs/fig_relearn_unseen",
                    help="output path WITHOUT extension; one file per --format")
    ap.add_argument("--format", "--formats", dest="formats", default="pdf,png",
                    help="comma-separated: pdf,png,svg,eps")
    ap.add_argument("--xtick-step", type=int, default=25,
                    help="spacing of the uniform x ticks (default 25)")
    ap.add_argument("--figsize", default="9,5")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--dump-table", default=None,
                    help="also write the exact plotted rows to this CSV")
    args = ap.parse_args(argv)

    formats = [f.strip().lstrip(".").lower()
               for f in args.formats.split(",") if f.strip()]
    figsize = tuple(float(v) for v in args.figsize.split(","))

    df = load_curves(args.curves, args.k)
    fig, legends = draw(df, args.k, args.xtick_step, figsize)
    save_figure(fig, args.out, formats, args.dpi, legends)

    keep = [a for a, _, _ in ARMS]
    plotted = df[df["arm"].isin(keep)].copy()
    if args.dump_table:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_table)) or ".",
                    exist_ok=True)
        plotted.to_csv(args.dump_table, index=False)
        print("wrote " + args.dump_table)

    label = {a: lab for a, lab, _ in ARMS}
    summary = (plotted[plotted["step"].isin([0, plotted["step"].max()])]
               .assign(method=lambda d: d["arm"].map(label),
                       value=lambda d: d["forget_full95"].fillna(d["unseen_acc"]))
               .pivot_table(index=["method", "group"], columns="step",
                            values="value")
               .round(2))
    print()
    print(summary.to_string())

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return df


if __name__ == "__main__":
    main()
