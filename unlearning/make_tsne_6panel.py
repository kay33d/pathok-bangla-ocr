r"""
make_tsne_6panel.py
===================
Six-panel CLS-embedding t-SNE figure, composed from the eight-panel PNG that
main.tex already ships (figs/fig_tsne_panel_all_methods.png).

WHY THIS EXISTS RATHER THAN fig11_cls_tsne.py
---------------------------------------------
fig11_cls_tsne.py draws the figure from embeddings. Those need a GPU and the
BnHTRD images, i.e. Kaggle. This script needs neither: it cuts the six wanted
panels straight out of the already-rendered PNG and re-lays them out with new
typeset titles. The dots are the original pixels, untouched -- no resampling
that invents detail, no AI upscaling (on a scatter plot that can move or
invent points, which would falsify a results figure).

WHAT IT CHANGES vs. THE SOURCE PNG
-----------------------------------
  * six panels instead of eight, in the order set in PANELS below
  * titles retypeset in Times New Roman (vector, sharp at any zoom)
  * the source PNG's baked-in headline and footnote are dropped -- the crop
    boxes start at the panel frames, so that text is never included
  * each panel enlarged UP x by exact pixel replication, so PDF viewers that
    smooth images do not blur the dots. This adds no detail; the source is
    650x540 per panel and that is the real resolution ceiling.

    python make_tsne_6panel.py
    python make_tsne_6panel.py --src ... --out ... --png

If numpy dies with "OpenBLAS ... memory allocation", this box is low on RAM:
prefix the command with  OPENBLAS_NUM_THREADS=1  (set OPENBLAS_NUM_THREADS=1
on cmd). The PNG is rendered from the finished PDF via PyMuPDF rather than by
matplotlib, because matplotlib's raster path resamples in float64 and is what
runs out of memory first.
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # TrueType, never Type 3
})

DEFAULT_SRC = r"C:\Users\LENOVO\Desktop\pathok_paper\figs\fig_tsne_panel_all_methods.png"
DEFAULT_OUT = r"C:\Users\LENOVO\Desktop\pathok_paper\figs\fig_tsne_panel_6methods_hq"

# Panel frame boxes measured from the source PNG's own grey borders (2 px lines).
COL_X = [18, 695, 1372, 2049]
ROW_Y = [171, 801]
PW, PH = 650, 540

# (source row, source col) in the 8-panel grid -> title in the new figure.
# Row/col refer to the SOURCE layout:
#   row 0: Baseline, Stage I, WIC, Random Labeling
#   row 1: Fisher, Negative Gradient, Amnesiac (Graves), DELETE
PANELS = [
    ((0, 0), "Baseline (nothing unlearned)"),
    ((0, 1), "Stage I: pruning only"),
    ((0, 2), "WIC"),
    # The negative-gradient run. Titled "Amnesiac Unlearning" at the author's
    # request; Table 11 lists Negative Gradient and Amnesiac Unlearning
    # (Graves et al.) as separate methods, so the caption must say which this is.
    ((1, 1), "Amnesiac Unlearning"),
    ((0, 3), "Random Labeling"),
    ((1, 3), "DELETE"),
]

UP = 4                       # integer nearest-neighbour upscale
PANEL_W_IN = 3.45            # printed width of one panel
MX, GAP, TITLE, ROWGAP, BOT = 0.05, 0.20, 0.36, 0.14, 0.05   # layout bands, inches


def load_panels(src):
    im = Image.open(src).convert("RGBA")
    flat = Image.new("RGBA", im.size, (255, 255, 255, 255))
    flat.alpha_composite(im)                      # flatten transparency onto white
    flat = flat.convert("RGB")

    def crop(r, c):
        p = flat.crop((COL_X[c], ROW_Y[r], COL_X[c] + PW, ROW_Y[r] + PH))
        return np.asarray(p.resize((PW * UP, PH * UP), Image.NEAREST))

    return [(crop(r, c), title) for (r, c), title in PANELS]


def draw(panels, ncols=3):
    pw = PANEL_W_IN
    ph = pw * PH / PW
    nrows = int(np.ceil(len(panels) / ncols))
    W = 2 * MX + ncols * pw + (ncols - 1) * GAP
    H = BOT + nrows * (ph + TITLE) + (nrows - 1) * ROWGAP

    fig = plt.figure(figsize=(W, H))
    # Bottom edge of each row, top row first.
    row_y = [BOT + (nrows - 1 - r) * (ph + TITLE + ROWGAP) for r in range(nrows)]
    for i, (img, title) in enumerate(panels):
        row, col = divmod(i, ncols)
        ax = fig.add_axes([(MX + col * (pw + GAP)) / W, row_y[row] / H,
                           pw / W, ph / H])
        ax.imshow(img, interpolation="none", aspect="auto")
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=6)
    return fig


def verify(pdf_path):
    raw = open(pdf_path, "rb").read()
    fonts = sorted(set(m.decode() for m in
                       re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-]+)", raw)))
    print("  fonts:", fonts)
    print("  images:", len(re.findall(rb"/Subtype\s*/Image", raw)),
          "| JPEG:", raw.count(b"/DCTDecode"),
          "| Type3:", raw.count(b"/CharProcs"))
    bad = [f for f in fonts if "TimesNewRoman" not in f]
    if bad:
        print("  WARNING: non-Times font embedded:", bad)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output path WITHOUT extension")
    ap.add_argument("--png", action="store_true",
                    help="also render a PNG from the finished PDF (PyMuPDF)")
    ap.add_argument("--png-dpi", type=int, default=600)
    ap.add_argument("--ncols", type=int, default=3)
    args = ap.parse_args(argv)

    fig = draw(load_panels(args.src), args.ncols)
    stem = os.path.splitext(os.path.abspath(args.out))[0]
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    pdf = stem + ".pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", pdf, os.path.getsize(pdf), "bytes")
    verify(pdf)

    if args.png:
        import pymupdf
        doc = pymupdf.open(pdf)
        pix = doc[0].get_pixmap(dpi=args.png_dpi, alpha=False)
        pix.save(stem + ".png")
        print("wrote", stem + ".png", pix.width, "x", pix.height)


if __name__ == "__main__":
    main()
