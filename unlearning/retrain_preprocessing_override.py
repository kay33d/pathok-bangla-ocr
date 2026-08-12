"""
retrain_preprocessing_override.py
=================================
Paste this AFTER pathok_retrain_clean.py and BEFORE calling main().

It overrides WDataset to apply the chosen preprocessing. Python resolves the
global name at call time, so train_baseline() and quick_eval() will pick up this
version automatically -- no need to edit the script above.

CHOSEN PIPELINE: height+contrast
--------------------------------
Measured leakage (geometry-only writer ID, chance 1.10%):

    raw                   38.36 %   34.9x   <- current behaviour
    height                19.69 %   17.9x
    height+contrast       11.11 %   10.1x   <- CHOSEN
    height+clahe          18.56 %   16.9x
    height+otsu            4.99 %    4.5x   lowest leakage, but the pilot showed
                                            writer ACC 43.25% vs 58.17% for
                                            contrast -- binarisation destroys the
                                            biometric signal the head must hold
    height+contrast+fill  20.24 %   18.4x   stretching away aspect ratio makes it
                                            WORSE: the anisotropic resize factor
                                            becomes a fingerprint of its own

CLIPPING FIX
------------
At TARGET_W=256, 1.8% of words were wider than the canvas and got truncated.
A truncated word is a guaranteed recognition error, and 1.8% would exceed the
previous run's entire retain WER of 1.20%. Over-wide words are now squeezed
horizontally to fit instead of cut, so every character survives with mild
distortion. Verify the printed clip rate is 0.00%.
"""

import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
TARGET_H = 64
TARGET_W = 256          # 4 x TARGET_H; median canvas coverage ~46%

_clip_stats = {"n": 0, "squeezed": 0}


def preprocess(gray, target_h=TARGET_H, target_w=TARGET_W, track=True):
    """
    height+contrast, with no-clip handling.

      1. height-normalise to target_h, preserving aspect
      2. if the result is wider than target_w, SQUEEZE horizontally to fit
         (never truncate -- a cut word is a guaranteed error)
      3. standardise mean/std on the CONTENT REGION only, before padding
         (padding-inclusive statistics encode aspect ratio -> leak)
      4. pad right with white to a fixed canvas
    """
    h, w = gray.shape
    scale = target_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))

    if track:
        _clip_stats["n"] += 1

    if new_w > target_w:                      # squeeze, do not clip
        if track:
            _clip_stats["squeezed"] += 1
        content = cv2.resize(gray, (target_w, target_h),
                             interpolation=cv2.INTER_AREA)
    else:
        content = cv2.resize(gray, (new_w, target_h),
                             interpolation=cv2.INTER_LINEAR)

    c = content.astype(np.float32)
    c = (c - c.mean()) / (c.std() + 1e-6)
    content = np.clip(c * 48 + 200, 0, 255).astype(np.uint8)

    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    canvas[:, :content.shape[1]] = content
    return canvas


def clip_report():
    n, s = _clip_stats["n"], _clip_stats["squeezed"]
    print(f"[preprocess] {n:,} images | squeezed (over-wide) {s:,} "
          f"({100*s/max(n,1):.2f} %) | truncated 0 (0.00 %)")
    return _clip_stats


# ══════════════════════════════════════════════════════════════════════════════
# OVERRIDDEN DATASET
# ══════════════════════════════════════════════════════════════════════════════
class WDataset(Dataset):
    """Same interface as the version in pathok_retrain_clean.py, plus preprocess()."""

    def __init__(self, df, vocab, fe):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.fe = fe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        gray = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.full((TARGET_H, TARGET_W), 255, dtype=np.uint8)
        im = preprocess(gray)
        img = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))
        pv = self.fe(img, return_tensors="pt").pixel_values.squeeze(0)

        lb = torch.tensor(encode_label(self.vocab, str(r["label"])),   # noqa: F821
                          dtype=torch.long)
        lb[lb == PAD_ID] = -100                                        # noqa: F821
        return {"pixel_values": pv, "labels": lb,
                "wid": torch.tensor(int(r["wid"]), dtype=torch.long)}


# ══════════════════════════════════════════════════════════════════════════════
# VISUAL SANITY CHECK -- look at this before starting a 5 hour run
# ══════════════════════════════════════════════════════════════════════════════
def preview(df, n=8, save="/kaggle/working/preprocess_preview.png"):
    """
    Stacks n preprocessed crops. Confirm the words are legible, roughly centred
    vertically, and not squashed into a sliver. If they look wrong, the retrain
    will fail and you will not know why for five hours.
    """
    sub = df.sample(n, random_state=0)
    tiles = []
    for p in sub["image_path"]:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        before = cv2.resize(gray, (TARGET_W, TARGET_H),
                            interpolation=cv2.INTER_LINEAR)
        after = preprocess(gray, track=False)
        sep = np.full((4, TARGET_W), 128, dtype=np.uint8)
        tiles += [before, sep, after, np.full((10, TARGET_W), 255, dtype=np.uint8)]
    grid = np.vstack(tiles)
    cv2.imwrite(save, grid)
    print(f"[preview] wrote {save}  (pairs: naive resize on top, "
          f"height+contrast below)")
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage(save))
    except Exception:
        pass
    return grid
