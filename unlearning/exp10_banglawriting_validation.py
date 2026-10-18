"""
pathok-exp10-banglawriting.py
==============================
Second cross-dataset validation of the CNN-vs-ViT writer-ID dissociation
found on BnHTRD: ConvNeXt-Base matches the TrOCR ViT encoder on
recognition but writer identification never rises above chance, and a
targeted tuning attempt (20x LR, 5x WRITER_W) produced zero measurable
change there. Experiment 9 tested this on English (Felix92/ibm-
handwriting-campaign-word, 302 writers, Latin script). This experiment
tests it on a SECOND, INDEPENDENT Bangla dataset -- BanglaWriting
(Mridha et al. 2021, reasat/banglawriting on Kaggle): 260 writers,
21,234 words, single-page handwriting with LabelMe-style word bounding
boxes, collected under entirely different conditions than BnHTRD (no
shared writers, no shared acquisition pipeline). Same script this time,
different dataset -- isolates dataset/writer-population effects from
script effects, complementing Experiment 9's isolation of script effects.

Unlike Experiment 9, this dataset has no pretrained-vocabulary shortcut
(it's Bangla, not English) and no pre-cropped word images -- both are
built here: a character-level vocabulary (not the full grapheme
segmentation used in the main study; recognition is an auxiliary
objective here, not what's being compared) and on-the-fly word crops
from the LabelMe bounding boxes.

Both arms share the same freshly-initialized decoder vocabulary and the
same joint recognition + writer-classification recipe; only the encoder
differs, and the ConvNeXt arm again uses the pre-proven-tuned recipe
(separate high-LR param group, WRITER_W=10) from the start.

OUTPUT
------
Per-epoch printed table for both arms and a FINAL RESULT banner directly
comparing them, plus a running three-way comparison against BnHTRD and
the English validation.
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, AutoImageProcessor
from transformers.modeling_outputs import BaseModelOutput
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

BW_SMOKE_TEST = False  # flip to False for the real run once this passes clean
CLS_LR_MULT = 20      # ConvNeXt fresh-params LR multiplier (proven-tuned recipe)


class C:
    ROOT = "/kaggle/input/banglawriting/converted/converted"
    TROCR_MODEL = "microsoft/trocr-base-handwritten"
    OUT = "/kaggle/working/banglawriting"

    SEED = 42
    VAL_FRAC = 0.10
    BATCH = 16
    EPOCHS = 2 if BW_SMOKE_TEST else 6
    LR = 5e-5
    WEIGHT_DECAY = 0.01
    WRITER_W = 10.0
    FP16 = True
    NUM_WORKERS = 2
    MAX_LEN = 24
    TARGET_H, TARGET_W = 64, 256
    TRAIN_SUBSAMPLE = 300 if BW_SMOKE_TEST else None

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(C.OUT, exist_ok=True)
np.random.seed(C.SEED); torch.manual_seed(C.SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. INDEX THE PAGES: parse every LabelMe JSON into one row per word crop
# ══════════════════════════════════════════════════════════════════════════════
import re as _re
_PAGE_JSON_RE = _re.compile(r"^\d+_\d+_\d+\.json$")


def resolve_root(cfg=C):
    """Kaggle's dataset_sources mount path doesn't always match the slug-based
    guess (seen before with the pathok-frozen-split dataset). Try the expected
    path first, then fall back to a recursive search under /kaggle/input for
    the <writer>_<age>_<gender>.json naming pattern."""
    candidates = [
        cfg.ROOT,
        "/kaggle/input/datasets/reasat/banglawriting/converted/converted",
        "/kaggle/input/banglawriting",
        "/kaggle/input/datasets/reasat/banglawriting",
    ]
    for c in candidates:
        if glob.glob(os.path.join(c, "*.json")):
            print(f"[data] root resolved (direct): {c}")
            return c
    print("[data] direct candidates empty, falling back to recursive search under /kaggle/input")
    for jp in glob.iglob("/kaggle/input/**/*.json", recursive=True):
        if _PAGE_JSON_RE.match(os.path.basename(jp)):
            root = os.path.dirname(jp)
            print(f"[data] root resolved (recursive): {root}")
            return root
    raise FileNotFoundError("Could not locate BanglaWriting annotation JSONs under /kaggle/input")


def build_index(cfg=C):
    root = resolve_root(cfg)
    jsons = sorted(glob.glob(os.path.join(root, "*.json")))
    print(f"[data] {len(jsons)} page annotation files found")
    rows = []
    for jp in jsons:
        stem = os.path.splitext(os.path.basename(jp))[0]
        writer_id = stem.split("_")[0]
        img_path = jp[:-5] + ".jpg"
        if not os.path.exists(img_path):
            continue
        try:
            with open(jp, encoding="utf-8") as f:
                ann = json.load(f)
        except Exception:
            continue
        for shp in ann.get("shapes", []):
            label = str(shp.get("label", "")).strip()
            pts = shp.get("points", [])
            if not label or len(pts) < 2:
                continue
            (x1, y1), (x2, y2) = pts[0], pts[1]
            x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            rows.append({"image_path": img_path, "writer_id": writer_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label})
    df = pd.DataFrame(rows)
    print(f"[data] {len(df):,} word crops from {df['writer_id'].nunique()} writers")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHARACTER VOCAB -- recognition is an auxiliary objective here, not the
#    thing being compared, so a plain Unicode-codepoint vocab is enough
# ══════════════════════════════════════════════════════════════════════════════
class CharVocab:
    def __init__(self):
        self.c2i = {}; self.i2c = {PAD_ID: "<pad>", UNK_ID: "<unk>", BOS_ID: "<s>", EOS_ID: "</s>"}
        self.vocab_size = SPECIAL_TOKENS

    def build(self, labels):
        chars = sorted({ch for w in labels for ch in str(w)})
        for i, ch in enumerate(chars):
            t = i + SPECIAL_TOKENS
            self.c2i[ch] = t; self.i2c[t] = ch
        self.vocab_size = SPECIAL_TOKENS + len(chars)
        print(f"[vocab] {len(chars)} unique characters (vocab_size={self.vocab_size})")
        return self

    def encode(self, w, max_len=C.MAX_LEN):
        ids = [BOS_ID] + [self.c2i.get(ch, UNK_ID) for ch in str(w)] + [EOS_ID]
        ids = ids[:max_len]
        ids += [PAD_ID] * (max_len - len(ids))
        return ids


def preprocess(gray, target_h=C.TARGET_H, target_w=C.TARGET_W):
    h, w = gray.shape
    scale = target_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    if new_w > target_w:
        content = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        content = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    canvas[:, :content.shape[1]] = content
    return canvas


class WDataset(Dataset):
    def __init__(self, df, vocab, fe, w2i, cfg=C):
        self.df = df.reset_index(drop=True)
        self.vocab, self.fe, self.w2i, self.cfg = vocab, fe, w2i, cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        crop = img[max(0, y1):y2, max(0, x1):x2] if img is not None else None
        if crop is None or crop.size == 0:
            crop = np.full((self.cfg.TARGET_H, self.cfg.TARGET_W), 255, np.uint8)
        else:
            crop = preprocess(crop, self.cfg.TARGET_H, self.cfg.TARGET_W)
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB))
        pv = self.fe(pil, return_tensors="pt").pixel_values.squeeze(0)
        lb = torch.tensor(self.vocab.encode(r["label"]), dtype=torch.long)
        lb[lb == PAD_ID] = -100
        wid = self.w2i[r["writer_id"]]
        return {"pixel_values": pv, "labels": lb, "wid": torch.tensor(wid, dtype=torch.long)}


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODELS -- ViT arm and ConvNeXt arm, identical adapter to Experiments 5/9
# ══════════════════════════════════════════════════════════════════════════════
class MultiHead(nn.Module):
    def __init__(self, ved, n_writers):
        super().__init__()
        self.ved = ved
        self.writer_head = nn.Linear(ved.config.encoder.hidden_size, n_writers)

    def forward(self, pixel_values, labels=None):
        enc = self.ved.encoder(pixel_values=pixel_values, return_dict=True)
        wlog = self.writer_head(enc.last_hidden_state[:, 0, :])
        rec = (self.ved(encoder_outputs=enc, labels=labels, return_dict=True)
               if labels is not None else None)
        return rec, wlog


def sinusoidal_2d_pos_embed(h, w, dim, device):
    assert dim % 4 == 0
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                          indexing="ij")
    y, x = y.flatten().float(), x.flatten().float()
    quarter = dim // 4
    div = torch.exp(torch.arange(quarter, device=device).float() * (-np.log(10000.0) / quarter))
    pe = torch.zeros(h * w, dim, device=device)
    pe[:, 0 * quarter:1 * quarter] = torch.sin(x[:, None] * div)
    pe[:, 1 * quarter:2 * quarter] = torch.cos(x[:, None] * div)
    pe[:, 2 * quarter:3 * quarter] = torch.sin(y[:, None] * div)
    pe[:, 3 * quarter:4 * quarter] = torch.cos(y[:, None] * div)
    return pe


class ConvNeXtAsViT(nn.Module):
    def __init__(self, hidden_size=768, pretrained=True):
        super().__init__()
        from torchvision.models import convnext_base, ConvNeXt_Base_Weights
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_base(weights=weights)
        self.stem = backbone.features
        self.proj = nn.Linear(1024, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.hidden_size = hidden_size

    def forward(self, pixel_values, return_dict=True):
        feat = self.stem(pixel_values)
        B, _, h, w = feat.shape
        feat = feat.flatten(2).transpose(1, 2)
        feat = self.proj(feat)
        pos = sinusoidal_2d_pos_embed(h, w, self.hidden_size, feat.device)
        feat = feat + pos.unsqueeze(0)
        cls = self.cls_token.expand(B, -1, -1)
        feat = torch.cat([cls, feat], dim=1)
        return BaseModelOutput(last_hidden_state=feat)


class ConvNeXtMultiHead(nn.Module):
    def __init__(self, ved, n_writers, hidden_size):
        super().__init__()
        self.ved = ved
        self.cnn_encoder = ConvNeXtAsViT(hidden_size=hidden_size)
        self.writer_head = nn.Linear(hidden_size, n_writers)

    def forward(self, pixel_values, labels=None):
        enc = self.cnn_encoder(pixel_values)
        wlog = self.writer_head(enc.last_hidden_state[:, 0, :])
        rec = (self.ved(encoder_outputs=enc, labels=labels, return_dict=True)
               if labels is not None else None)
        return rec, wlog


def _build_ved(vocab_size, cfg=C):
    ved = VisionEncoderDecoderModel.from_pretrained(cfg.TROCR_MODEL)
    ved.decoder.resize_token_embeddings(vocab_size)
    ved.config.vocab_size = vocab_size
    ved.config.decoder.vocab_size = vocab_size
    for c_ in (ved.config, ved.generation_config):
        c_.decoder_start_token_id = BOS_ID
        c_.pad_token_id = PAD_ID
        c_.eos_token_id = EOS_ID
        c_.bos_token_id = BOS_ID
    return ved


def build_vit(n_writers, vocab_size, cfg=C):
    ved = _build_ved(vocab_size, cfg)
    m = MultiHead(ved, n_writers).to(DEVICE)
    n_par = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"[model] ViT arm | {n_par:.1f}M params | writers={n_writers}")
    return m


def build_convnext(n_writers, vocab_size, cfg=C):
    ved = _build_ved(vocab_size, cfg)
    hidden_size = ved.config.encoder.hidden_size
    m = ConvNeXtMultiHead(ved, n_writers, hidden_size).to(DEVICE)
    n_par = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"[model] ConvNeXt arm | {n_par:.1f}M params | writers={n_writers}")
    return m


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def writer_acc(model, loader, is_convnext, n_batches=None):
    model.eval()
    hits, seen = 0, 0
    for bi, b in enumerate(loader):
        if n_batches and bi >= n_batches:
            break
        pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
        if is_convnext:
            enc = model.cnn_encoder(pv)
        else:
            enc = model.ved.encoder(pixel_values=pv, return_dict=True)
        wl = model.writer_head(enc.last_hidden_state[:, 0, :])
        hits += (wl.argmax(-1) == wid).sum().item(); seen += wid.numel()
    model.train()
    return 100.0 * hits / max(seen, 1)


def train_arm(model, train_loader, val_loader, is_convnext, tag, cfg=C):
    if is_convnext:
        fresh = ([model.cnn_encoder.cls_token] + list(model.cnn_encoder.proj.parameters())
                 + list(model.writer_head.parameters()))
        fresh_ids = {id(p) for p in fresh}
        rest = [p for p in model.parameters() if id(p) not in fresh_ids]
        opt = torch.optim.AdamW(
            [{"params": fresh, "lr": cfg.LR * CLS_LR_MULT},
             {"params": rest, "lr": cfg.LR}], weight_decay=cfg.WEIGHT_DECAY)
        max_lr = [cfg.LR * CLS_LR_MULT, cfg.LR]
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        max_lr = cfg.LR

    total = len(train_loader) * cfg.EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total,
                                                pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.FP16 and DEVICE == "cuda")

    print(f"\n[train] {tag} | {len(train_loader)} steps/epoch | {cfg.EPOCHS} epochs")
    rows = []
    for ep in range(cfg.EPOCHS):
        model.train()
        agg_rec, agg_wid, agg_wacc, n = 0.0, 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"{tag} epoch {ep+1}/{cfg.EPOCHS}", dynamic_ncols=True)
        for b in pbar:
            pv = b["pixel_values"].to(DEVICE, non_blocking=True)
            lb = b["labels"].to(DEVICE, non_blocking=True)
            wid = b["wid"].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.FP16 and DEVICE == "cuda"):
                rec, wlog = model(pv, lb)
                l_rec = rec.loss
                l_wid = F.cross_entropy(wlog, wid)
                loss = l_rec + cfg.WRITER_W * l_wid
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            agg_rec += l_rec.item(); agg_wid += l_wid.item()
            agg_wacc += (wlog.argmax(-1) == wid).float().mean().item(); n += 1
            pbar.set_postfix(rec=f"{agg_rec/n:.3f}", wid=f"{agg_wid/n:.3f}",
                             wacc=f"{100*agg_wacc/n:.1f}%")
        val_wacc = writer_acc(model, val_loader, is_convnext,
                              n_batches=None if not BW_SMOKE_TEST else 5)
        row = {"tag": tag, "epoch": ep + 1, "train_rec_loss": agg_rec / n,
               "train_wacc": 100 * agg_wacc / n, "val_wacc": val_wacc}
        rows.append(row)
        print(f"  [{tag}] epoch {ep+1}: rec_loss {row['train_rec_loss']:.3f}  "
              f"train_wacc {row['train_wacc']:6.2f}%  val_wacc {val_wacc:6.2f}%")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"device: {DEVICE}\n")
    df = build_index()
    if C.TRAIN_SUBSAMPLE is not None:
        # keep writer diversity in the smoke pass rather than truncating rows
        df = df.groupby("writer_id", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(2, C.TRAIN_SUBSAMPLE // df["writer_id"].nunique())),
                               random_state=C.SEED))

    writers = sorted(df["writer_id"].unique())
    w2i = {w: i for i, w in enumerate(writers)}
    n_writers = len(writers)
    chance = 100.0 / n_writers
    print(f"[data] {len(df):,} word crops | {n_writers} writers | chance = {chance:.2f}%\n")

    vocab = CharVocab().build(df["label"])

    n = len(df)
    perm = np.random.RandomState(C.SEED).permutation(n)
    n_val = max(1, int(n * C.VAL_FRAC))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(f"[split] train={len(train_df):,}  val={len(val_df):,}\n")

    fe = AutoImageProcessor.from_pretrained(C.TROCR_MODEL)
    train_loader = DataLoader(WDataset(train_df, vocab, fe, w2i), batch_size=C.BATCH,
                              shuffle=True, num_workers=C.NUM_WORKERS, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(WDataset(val_df, vocab, fe, w2i), batch_size=C.BATCH,
                            shuffle=False, num_workers=C.NUM_WORKERS)

    all_rows = []

    vit = build_vit(n_writers, vocab.vocab_size)
    all_rows += train_arm(vit, train_loader, val_loader, is_convnext=False, tag="ViT")
    vit_final = all_rows[-1]["val_wacc"]
    del vit; torch.cuda.empty_cache()

    convnext = build_convnext(n_writers, vocab.vocab_size)
    all_rows += train_arm(convnext, train_loader, val_loader, is_convnext=True, tag="ConvNeXt")
    convnext_final = all_rows[-1]["val_wacc"]
    del convnext; torch.cuda.empty_cache()

    pd.DataFrame(all_rows).to_csv(f"{C.OUT}/banglawriting_curves.csv", index=False)

    smoke_tag = "SMOKE TEST" if BW_SMOKE_TEST else "REAL RUN"
    print("\n" + "#" * 90)
    print(f"#  FINAL RESULT -- Cross-dataset validation on BanglaWriting ({smoke_tag})")
    print(f"#  Bangla, {n_writers} writers, chance = {chance:.2f}%")
    print("#" * 90)
    print(f"#  ViT       final val writer-ID acc.: {vit_final:6.2f}%")
    print(f"#  ConvNeXt  final val writer-ID acc.: {convnext_final:6.2f}%")
    print("#" * 90)
    print("#  Three-way read (fill in from Experiments 5 and 9's logs):")
    print("#    BnHTRD    (Bangla, orig.): ViT ~100%  vs  ConvNeXt ~5%")
    print("#    IBM-campaign (English)   : ViT ?      vs  ConvNeXt ?")
    print(f"#    BanglaWriting (Bangla)   : ViT {vit_final:.1f}%  vs  ConvNeXt {convnext_final:.1f}%")
    if vit_final > 5 * chance and convnext_final < 2 * chance:
        print("#  RESULT: dissociation REPLICATES on a second, independent Bangla")
        print("#  dataset with a disjoint writer population and acquisition pipeline.")
    elif vit_final > 5 * chance and convnext_final > 5 * chance:
        print("#  RESULT: dissociation does NOT replicate here -- ConvNeXt learns")
        print("#  writer identity on this dataset. Read alongside Experiment 9 to")
        print("#  see whether this tracks script, dataset, or something else.")
    else:
        print("#  RESULT: inconclusive by this simple threshold -- read the per-epoch")
        print("#  table above/CSV directly.")
    print("#" * 90 + "\n")
    return all_rows


if __name__ == "__main__":
    main()
