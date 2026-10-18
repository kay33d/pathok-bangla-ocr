"""
pathok_retrain_clean.py
=======================
SELF-CONTAINED. Nothing needs to be defined beforehand. Run it as a script,
or paste the whole thing into one Kaggle cell and call main().

WHAT THIS REPLACES
------------------
Your original training script split the data with train_test_split() over ROWS,
so all 150 BN-HTRd writers appeared in train, val AND test. That makes:
  - the 92.21% word accuracy partly writer memorisation, not generalisation
  - the MIA "non-member" class invalid (they were all members)
  - RQ1 unanswerable

This script retrains from raw TrOCR on a WRITER-DISJOINT split. The 30 test
writers are never seen by the encoder, the decoder, or the vocabulary.

ALSO FIXED HERE
---------------
  1. vocab built from TRAINING writers only (was leaking test-writer graphemes)
  2. labels no longer carry a leading BOS -- VisionEncoderDecoderModel
     shift-rights them and prepends decoder_start_token_id itself, so
     [BOS, g..., EOS] became decoder input [BOS, BOS, g...]
  3. writer head trained jointly from step 0 -> ONE run, not two
  4. split frozen to split.json so every later experiment is comparable
  5. explicit check that decoder.output_projection is tied to embed_tokens
     (your old log showed it as a missing key on load)

OUTPUT
------
  /kaggle/working/pathok_clean/
      split.json                 <- frozen writer assignment, reuse forever
      vocab_clean.json
      baseline_ep01.pt ... baseline_ep08.pt
      baseline_clean_final.pt    <- feed this to the unlearning pipeline
"""

import os
import glob
import json
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, AutoImageProcessor
from sklearn.model_selection import train_test_split
import editdistance
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CONVNEXT_SMOKE_TEST = False  # smoke test passed clean: pipeline mechanics confirmed working

# THE ACTUAL BUG, DIAGNOSED (not a tuning problem): a fresh-probe experiment
# (pathok-convnext-writer-probe) fit a brand-new logistic-regression writer
# classifier directly on a jointly-trained checkpoint's frozen embeddings --
# no retraining -- and found writer identity sitting at 77-87% on
# MEAN-POOLED PATCH TOKENS (vs.\ 5.26% chance), while the exact same
# checkpoint's CLS token (the learned nn.Parameter every earlier ConvNeXt
# run pooled through) read at or BELOW chance (3-4%) on every group,
# including retain writers the model was jointly trained on. The
# representation was never missing the signal; the CLS slot just never
# learned to aggregate it.
#
# The fix is structural: delete the learned cls_token parameter entirely
# and set position 0 of last_hidden_state to the mean of the patch tokens
# directly, computed fresh every forward pass. Every existing
# pathok_common.py / unlearning script already reads
# last_hidden_state[:, 0, :] expecting a pooled representation -- this
# makes that read correct, with zero changes needed anywhere else.
#
# SECOND ROUND, hyperparameters reverted: a first run of this exact
# architecture, still carrying exp5b's WRITER_W=10.0 / CLS_LR_MULT=20
# tuning (built to force signal into a writer objective that COULDN'T
# learn), confirmed the fix too well in one respect and broke another:
# writer-ACC hit 99.6%/99.7% (matching ViT), but recognition CER collapsed
# to 29-41% (vs.\ ~7.6% for the original CLS-based checkpoint), still
# visibly degrading at epoch 8. A loss weighted 10x that finally CAN
# decrease dominates a shared decoder's gradient budget; a 20x LR on the
# projection layer both heads read through compounds it. This version
# reverts to the ORIGINAL, untuned recipe (WRITER_W=2.0, single optimizer
# group, plain cfg.LR everywhere) -- the tuning was solving the same
# problem the pooling fix already solves, from a different angle, and now
# actively fights it.


class C:
    ROOT = "/kaggle/input/datasets/kaidul007/bnhtrd/Dataset"
    TROCR_MODEL = "microsoft/trocr-base-handwritten"
    OUT = "/kaggle/working/convnext_meanpool_untuned"

    # Reuse the FROZEN split/vocab from the original baseline run byte-for-byte,
    # same convention as the Experiment 1 oracle -- Experiment 3 must differ
    # from the TrOCR ViT arm in exactly one respect (the encoder), not in
    # split, vocab, or writer-index space too. Published as a small public
    # dataset (not a kernel_source) because this account has no collaborator
    # access to kayed007's private training kernel.
    SRC = "/kaggle/input/pathok-frozen-split"

    SEED = 42
    N_TEST_WRITERS = 30       # NON-members. never seen by anything.
    N_FORGET_WRITERS = 20     # members, to be unlearned in stage II
    VAL_FRAC = 0.05           # sampled within retain writers

    EPOCHS = 1 if CONVNEXT_SMOKE_TEST else 8
    BATCH = 16
    LR = 5e-5
    WEIGHT_DECAY = 0.01
    WRITER_W = 2.0            # back to the ORIGINAL (untuned) value -- see
                              # the banner in train_baseline() for why the
                              # 10x tuning attempt is reverted here
    FP16 = True
    GRAD_CKPT = False         # True if OOM; ~30% slower
    NUM_WORKERS = 2

    EVAL_N = 30 if CONVNEXT_SMOKE_TEST else 1500  # samples per split in the gate check
    TRAIN_SUBSAMPLE = 400 if CONVNEXT_SMOKE_TEST else None


PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = 4
MAX_LEN = 64
PUNCT = {'.', ':', '।', ',', ';', '!', '?', '-', '–', '—'}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(C.OUT, exist_ok=True)
random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BANGLA GRAPHEME SEGMENTER  (unchanged from your script)
# ══════════════════════════════════════════════════════════════════════════════
HASANTA = '\u09CD'
CHANDRABINDU = '\u0981'
ANUSVARA = '\u0982'
VISARGA = '\u0983'
NUKTA = '\u09BC'

VOWEL_SIGNS = set('\u09BE\u09BF\u09C0\u09C1\u09C2\u09C3\u09C4'
                  '\u09C7\u09C8\u09CB\u09CC\u09D7')
INDEPENDENT_VOWELS = set('\u0985\u0986\u0987\u0988\u0989\u098A\u098B'
                         '\u098F\u0990\u0993\u0994')
CONSONANTS = set(chr(c) for c in range(0x0995, 0x09BA)) | {
    '\u09DC', '\u09DD', '\u09DF', '\u09CE', '\u09B0'}


def segment_graphemes(word: str) -> list:
    graphemes = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in INDEPENDENT_VOWELS:
            g = ch; i += 1
            while i < n and word[i] in {CHANDRABINDU, ANUSVARA, VISARGA}:
                g += word[i]; i += 1
            graphemes.append(g)
        elif ch in CONSONANTS:
            g = ch; i += 1
            if i < n and word[i] == NUKTA:
                g += word[i]; i += 1
            while i + 1 < n and word[i] == HASANTA and word[i + 1] in CONSONANTS:
                g += word[i] + word[i + 1]; i += 2
            if i < n and word[i] in VOWEL_SIGNS:
                g += word[i]; i += 1
            graphemes.append(g)
        else:
            graphemes.append(ch); i += 1
    return graphemes


def is_bangla_word(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    return sum(1 for c in chars if '\u0980' <= c <= '\u09FF') / len(chars) > 0.8


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA + FROZEN WRITER-DISJOINT SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def writer_of(path, root):
    """BN-HTRd: <ROOT>/<writer_folder>/Words/<page>/<word>.jpg"""
    return os.path.relpath(path, root).split(os.sep)[0]


def inspect_layout(root=C.ROOT, n=5):
    """Run this FIRST. Expect ~150 top-level folders (one per writer)."""
    imgs = glob.glob(root + "/*/Words/*/*.jpg") + glob.glob(root + "/*/Words/*/*.JPG")
    tops = sorted({writer_of(p, root) for p in imgs})
    print(f"images: {len(imgs):,} | top-level folders: {len(tops)}")
    print("examples:", tops[:n])
    return tops


def load_with_writers(root=C.ROOT):
    label_dfs = []
    for f in glob.glob(root + "/*/*.xlsx"):
        try:
            for d in pd.read_excel(f, sheet_name=None, engine="openpyxl").values():
                if d.shape[1] < 2:
                    continue
                d = d.iloc[:, :2].copy(); d.columns = ["Id", "Word"]
                label_dfs.append(d)
        except Exception:
            continue

    labels_df = pd.concat(label_dfs, ignore_index=True)
    labels_df = labels_df.drop_duplicates(subset="Id", keep="first")
    imgs = glob.glob(root + "/*/Words/*/*.jpg") + glob.glob(root + "/*/Words/*/*.JPG")
    image_df = pd.DataFrame({
        "Id": [os.path.basename(p).split(".")[0] for p in imgs],
        "image_path": imgs,
    })

    df = (image_df.merge(labels_df, on="Id", how="inner")
          .rename(columns={"Word": "label"})[["image_path", "label"]]
          .dropna().reset_index(drop=True))
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"] != ""]
    df = df[~df["label"].isin(PUNCT)]
    df = df[df["label"].apply(is_bangla_word)]
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    df["writer"] = df["image_path"].apply(lambda p: writer_of(p, root))

    print(f"samples: {len(df):,} | writers: {df['writer'].nunique()}")
    return df


def verify_split(retain_df, forget_df, val_df, test_df):
    W = lambda d: set(d["writer"])
    assert not (W(retain_df) & W(forget_df)), "retain/forget writer overlap"
    assert not (W(test_df) & (W(retain_df) | W(forget_df) | W(val_df))), \
        "TEST WRITER LEAK -- this is exactly the bug being fixed"
    P = lambda d: set(d["image_path"])
    assert not (P(retain_df) & P(val_df)), "retain/val image overlap"
    print("[split] integrity OK")
    print(f"  retain {len(retain_df):>6,} / {len(W(retain_df))} writers  (members)")
    print(f"  forget {len(forget_df):>6,} / {len(W(forget_df))} writers  (members, to unlearn)")
    print(f"  val    {len(val_df):>6,}")
    print(f"  test   {len(test_df):>6,} / {len(W(test_df))} writers  (NON-members)")


def build_frozen_split(df, cfg=C):
    """Deterministic and saved. Reload for every later experiment."""
    path = os.path.join(cfg.OUT, "split.json")
    if os.path.exists(path):
        spec = json.load(open(path))
        print(f"[split] reloaded from {path}")
    else:
        writers = sorted(df["writer"].unique())
        random.Random(cfg.SEED).shuffle(writers)
        a, b = cfg.N_TEST_WRITERS, cfg.N_TEST_WRITERS + cfg.N_FORGET_WRITERS
        spec = {"test": writers[:a], "forget": writers[a:b], "retain": writers[b:]}
        json.dump(spec, open(path, "w"), indent=2)
        print(f"[split] created and frozen -> {path}")

    test_w, forget_w, retain_w = map(set, (spec["test"], spec["forget"], spec["retain"]))
    test_df = df[df["writer"].isin(test_w)].reset_index(drop=True)
    forget_df = df[df["writer"].isin(forget_w)].reset_index(drop=True)
    retain_all = df[df["writer"].isin(retain_w)].reset_index(drop=True)
    retain_df, val_df = train_test_split(retain_all, test_size=cfg.VAL_FRAC,
                                         random_state=cfg.SEED)
    retain_df, val_df = retain_df.reset_index(drop=True), val_df.reset_index(drop=True)

    w2i = {w: i for i, w in enumerate(sorted(retain_w | forget_w))}
    for d in (retain_df, forget_df, val_df):
        d["wid"] = d["writer"].map(w2i)
    test_df["wid"] = -1

    verify_split(retain_df, forget_df, val_df, test_df)
    return retain_df, forget_df, val_df, test_df, w2i


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VOCAB -- TRAINING WRITERS ONLY
# ══════════════════════════════════════════════════════════════════════════════
class GraphemeVocab:
    def __init__(self):
        self.g2i = {}
        self.i2g = {PAD_ID: "<pad>", UNK_ID: "<unk>", BOS_ID: "<s>", EOS_ID: "</s>"}
        self.vocab_size = SPECIAL_TOKENS

    def build(self, labels):
        gs = sorted({g for w in labels for g in segment_graphemes(str(w))})
        for i, g in enumerate(gs):
            t = i + SPECIAL_TOKENS
            self.g2i[g] = t; self.i2g[t] = g
        self.vocab_size = SPECIAL_TOKENS + len(gs)
        print(f"[vocab] {len(gs)} graphemes from TRAINING writers only "
              f"(vocab_size={self.vocab_size})")
        return self

    def encode(self, w):
        return [self.g2i.get(g, UNK_ID) for g in segment_graphemes(str(w))]

    def decode(self, ids):
        out = []
        for t in ids:
            t = int(t)
            if t == EOS_ID:
                break
            if t in (PAD_ID, BOS_ID, UNK_ID):
                continue
            out.append(self.i2g.get(t, ""))
        return "".join(out)

    def save(self, p):
        json.dump({"g2i": self.g2i,
                   "i2g": {str(k): v for k, v in self.i2g.items()},
                   "vocab_size": self.vocab_size},
                  open(p, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, p):
        d = json.load(open(p, encoding="utf-8"))
        v = cls(); v.g2i = d["g2i"]
        v.i2g = {int(k): x for k, x in d["i2g"].items()}
        v.vocab_size = d["vocab_size"]
        return v


def report_oov(vocab, test_df):
    """Expected and fine -- but must be reported alongside test CER."""
    tot = oov = words = 0
    for w in test_df["label"].astype(str):
        gs = segment_graphemes(w)
        k = sum(1 for g in gs if g not in vocab.g2i)
        tot += len(gs); oov += k; words += (k > 0)
    print(f"[vocab] test OOV: {100*oov/max(tot,1):.2f}% of graphemes, "
          f"{100*words/max(len(test_df),1):.2f}% of words affected")
    return oov / max(tot, 1)


def encode_label(vocab, text):
    """
    FIX: no leading BOS. The model prepends decoder_start_token_id itself.
    labels = [g..., EOS, PAD...]
    """
    seq = (vocab.encode(text) + [EOS_ID])[:MAX_LEN]
    seq += [PAD_ID] * (MAX_LEN - len(seq))
    return seq


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATASET / MODEL
# ══════════════════════════════════════════════════════════════════════════════
class WDataset(Dataset):
    def __init__(self, df, vocab, fe):
        self.df = df.reset_index(drop=True); self.vocab = vocab; self.fe = fe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r["image_path"])
        img = (Image.new("RGB", (384, 384), 255) if img is None
               else Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        pv = self.fe(img, return_tensors="pt").pixel_values.squeeze(0)
        lb = torch.tensor(encode_label(self.vocab, str(r["label"])), dtype=torch.long)
        lb[lb == PAD_ID] = -100
        return {"pixel_values": pv, "labels": lb,
                "wid": torch.tensor(int(r["wid"]), dtype=torch.long)}


class MultiHead(nn.Module):
    """ViT encoder shared by the TrOCR decoder and a linear writer head on CLS."""

    def __init__(self, ved, n_writers):
        super().__init__()
        self.ved = ved
        self.writer_head = nn.Linear(ved.config.encoder.hidden_size, n_writers)
        self.n_writers = n_writers

    @property
    def vit(self):
        return self.ved.encoder

    def forward(self, pixel_values, labels=None, need_rec=True):
        enc = self.ved.encoder(pixel_values=pixel_values, return_dict=True)
        wlog = self.writer_head(enc.last_hidden_state[:, 0, :])
        rec = (self.ved(encoder_outputs=enc, labels=labels, return_dict=True)
               if (need_rec and labels is not None) else None)
        return rec, wlog


from transformers.modeling_outputs import BaseModelOutput
# The kayed007/pathok-exp3-resnet50-base smoke test crashed with
# "TypeError: 'ResNetEncOutput' object is not subscriptable": HF's
# VisionEncoderDecoderModel internals sometimes index encoder_outputs[0]
# rather than only accessing .last_hidden_state, so a plain attribute-only
# stand-in isn't enough -- use the real ModelOutput subclass, which supports
# both attribute and tuple-style access.


def sinusoidal_2d_pos_embed(h, w, dim, device):
    """Standard 2D sinusoidal positional encoding (half the channels per
    spatial axis). Computed from the actual feature-map size every forward
    pass rather than a fixed learned parameter, so it never assumes a
    specific input resolution -- unlike ViT's patch grid, nothing here breaks
    if TrOCR's feature extractor ever changes its target size."""
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
    return pe  # (h*w, dim)


class ConvNeXtAsViT(nn.Module):
    """Wraps a ConvNeXt-Base backbone to mimic the ViT encoder's call
    interface: forward(pixel_values) -> ModelOutput with .last_hidden_state
    of shape (B, 1+h*w, hidden_size) -- so writer_head and
    ved(encoder_outputs=...) work completely unchanged everywhere else in
    this pipeline (Stage I pruning, all five Stage II unlearning objectives,
    every audit instrument). Position 0 is the MEAN of the patch tokens,
    computed fresh every forward pass -- NOT a learned cls_token parameter.
    exp5/exp5b used a learned cls_token here, which a fresh-probe diagnostic
    (pathok-convnext-writer-probe) showed never learned to aggregate writer
    identity (3-4%, at/below chance) despite the mean-pooled patch tokens
    themselves carrying it strongly (77-87%). Every downstream script reads
    last_hidden_state[:, 0, :] as "the pooled representation"; this class
    makes that read point at the representation that actually has the
    signal, rather than an empty learned slot.
    This is Experiment 5: ConvNeXt is specifically designed to match
    ViT-level performance using only convolutional operations, so it
    isolates whether TrOCR's forgetting resistance is about Transformer
    self-attention specifically or about pretrained representational
    richness in general. Tokenizer, dataset, split, writer head, decoder
    init, and every downstream script are held identical; only this class
    is swapped in for ved.encoder."""

    def __init__(self, hidden_size=768, pretrained=True):
        super().__init__()
        from torchvision.models import convnext_base, ConvNeXt_Base_Weights
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_base(weights=weights)
        self.stem = backbone.features  # -> (B, 1024, H/32, W/32)
        self.proj = nn.Linear(1024, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, pixel_values, return_dict=True):
        feat = self.stem(pixel_values)                     # (B, 1024, h, w)
        B, _, h, w = feat.shape
        feat = feat.flatten(2).transpose(1, 2)             # (B, h*w, 1024)
        feat = self.proj(feat)                             # (B, h*w, hidden)
        pos = sinusoidal_2d_pos_embed(h, w, self.hidden_size, feat.device)
        feat = feat + pos.unsqueeze(0)
        pooled = feat.mean(dim=1, keepdim=True)             # (B, 1, hidden) -- position 0
        feat = torch.cat([pooled, feat], dim=1)             # (B, 1+h*w, hidden)
        return BaseModelOutput(last_hidden_state=feat)


class ConvNeXtMultiHead(nn.Module):
    """Same call interface as MultiHead (forward returns (rec, wlog) from
    pixel_values), but pixel_values are routed through ConvNeXtAsViT instead
    of ved.encoder. ved is kept only for its pretrained decoder and its
    .forward(encoder_outputs=..., labels=...) convenience method -- ved's own
    ViT encoder is deleted immediately after construction and never called."""

    def __init__(self, ved, n_writers, hidden_size):
        super().__init__()
        self.ved = ved
        self.cnn_encoder = ConvNeXtAsViT(hidden_size=hidden_size)
        self.writer_head = nn.Linear(hidden_size, n_writers)
        self.n_writers = n_writers

    def forward(self, pixel_values, labels=None, need_rec=True):
        enc = self.cnn_encoder(pixel_values)
        wlog = self.writer_head(enc.last_hidden_state[:, 0, :])
        rec = (self.ved(encoder_outputs=enc, labels=labels, return_dict=True)
               if (need_rec and labels is not None) else None)
        return rec, wlog


def build_fresh_convnext(vocab_size, n_writers, cfg=C):
    ved = VisionEncoderDecoderModel.from_pretrained(cfg.TROCR_MODEL)
    ved.decoder.resize_token_embeddings(vocab_size)
    ved.config.vocab_size = vocab_size
    ved.config.decoder.vocab_size = vocab_size
    for c_ in (ved.config, ved.generation_config):
        c_.decoder_start_token_id = BOS_ID
        c_.pad_token_id = PAD_ID
        c_.eos_token_id = EOS_ID
        c_.bos_token_id = BOS_ID
    ved.generation_config.max_new_tokens = MAX_LEN

    hidden_size = ved.config.encoder.hidden_size  # 768 -- must match what the
                                                    # pretrained decoder's cross-attention expects
    # ved.encoder is never called for its actual forward pass (ConvNeXt-Base
    # replaces that entirely), but it CANNOT be deleted: HF's
    # VisionEncoderDecoderModel.forward() does a config sanity check
    # (self.encoder.config.hidden_size != self.decoder.config.hidden_size)
    # unconditionally, even when encoder_outputs is already supplied and the
    # encoder's own computation is skipped. Left in place, inert, ~86M params.

    try:
        tied = (ved.decoder.output_projection.weight.data_ptr()
                == ved.decoder.model.decoder.embed_tokens.weight.data_ptr())
        print(f"[tie check] output_projection tied to embed_tokens: {tied}")
    except AttributeError:
        print("[tie check] could not resolve output_projection; verify manually")

    m = ConvNeXtMultiHead(ved, n_writers, hidden_size).to(DEVICE)
    n_par = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"[model] fresh ConvNeXt-Base encoder + {cfg.TROCR_MODEL} decoder | "
          f"{n_par:.1f}M params | vocab={vocab_size} | writers={n_writers}")
    return m


def build_fresh(vocab_size, n_writers, cfg=C):
    ved = VisionEncoderDecoderModel.from_pretrained(cfg.TROCR_MODEL)
    ved.decoder.resize_token_embeddings(vocab_size)
    ved.config.vocab_size = vocab_size
    ved.config.decoder.vocab_size = vocab_size
    for c_ in (ved.config, ved.generation_config):
        c_.decoder_start_token_id = BOS_ID
        c_.pad_token_id = PAD_ID
        c_.eos_token_id = EOS_ID
        c_.bos_token_id = BOS_ID
    ved.generation_config.max_new_tokens = MAX_LEN

    if cfg.GRAD_CKPT:
        ved.encoder.gradient_checkpointing_enable()
        ved.decoder.gradient_checkpointing_enable()

    try:
        tied = (ved.decoder.output_projection.weight.data_ptr()
                == ved.decoder.model.decoder.embed_tokens.weight.data_ptr())
        print(f"[tie check] output_projection tied to embed_tokens: {tied}")
    except AttributeError:
        print("[tie check] could not resolve output_projection; verify manually")

    m = MultiHead(ved, n_writers).to(DEVICE)
    n_par = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"[model] fresh from {cfg.TROCR_MODEL} | {n_par:.1f}M params | "
          f"vocab={vocab_size} | writers={n_writers}")
    return m


# ══════════════════════════════════════════════════════════════════════════════
# 5.  METRICS / EVAL
# ══════════════════════════════════════════════════════════════════════════════
def corpus_cer_wer(preds, gts):
    d = c = we = 0
    for p, g in zip(preds, gts):
        d += editdistance.eval(list(p), list(g))
        c += max(len(g), 1)
        we += (p != g)
    return d / max(c, 1), we / max(len(gts), 1)


@torch.no_grad()
def quick_eval(model, df, vocab, fe, n=None, cfg=C, tag="", num_beams=1):
    """Encoder-agnostic: MultiHead has a working model.ved.encoder, so
    generate() can take raw pixel_values and run it internally as usual.
    ResNetMultiHead deleted that encoder, so both the writer-head forward
    and generate() must be given pre-computed encoder_outputs from
    model.cnn_encoder instead -- HF's generate() skips its own internal
    encoder call whenever encoder_outputs is already present in kwargs."""
    n = n or cfg.EVAL_N
    d = df.sample(min(n, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval()
    is_cnn = hasattr(model, "cnn_encoder")
    preds, wc, wt = [], 0, 0
    for b in tqdm(loader, desc=f"eval {tag}", leave=False, dynamic_ncols=True):
        pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
        if is_cnn:
            enc = model.cnn_encoder(pv)
            wl = model.writer_head(enc.last_hidden_state[:, 0, :])
            gen = model.ved.generate(encoder_outputs=enc, max_new_tokens=MAX_LEN,
                                     num_beams=num_beams)
        else:
            enc = model.ved.encoder(pixel_values=pv, return_dict=True)
            wl = model.writer_head(enc.last_hidden_state[:, 0, :])
            gen = model.ved.generate(pv, max_new_tokens=MAX_LEN, num_beams=num_beams)
        m = wid >= 0
        if m.any():
            wc += (wl[m].argmax(-1) == wid[m]).sum().item(); wt += int(m.sum())
        preds += [vocab.decode(r) for r in gen.cpu().tolist()]
    cer, wer = corpus_cer_wer(preds, d["label"].astype(str).tolist())
    return {"ACC": 100 * wc / wt if wt else float("nan"),
            "CER": 100 * cer, "WER": 100 * wer, "n": len(d)}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def train_baseline(model, retain_df, forget_df, val_df, vocab, fe, cfg=C):
    train_df = pd.concat([retain_df, forget_df], ignore_index=True)
    if cfg.TRAIN_SUBSAMPLE is not None:
        train_df = train_df.sample(min(cfg.TRAIN_SUBSAMPLE, len(train_df)),
                                   random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(train_df, vocab, fe), batch_size=cfg.BATCH,
                        shuffle=True, num_workers=cfg.NUM_WORKERS,
                        pin_memory=True, drop_last=True)

    # SINGLE param group, plain cfg.LR everywhere, WRITER_W back to the
    # original 2.0 -- matching exp5_convnext_base.py's UNTUNED recipe
    # exactly, not exp5b's two-group/10x-loss-weight tuning attempt.
    #
    # Why revert instead of keeping the "stronger" tuning: that tuning was
    # built to compensate for a writer objective that COULDN'T learn
    # anything (broken CLS readout, loss stuck near chance forever). With
    # the mean-pool fix, the writer objective is fully learnable again --
    # the first meanpool run (WRITER_W=10, CLS_LR_MULT=20) confirmed this
    # too well: writer-ACC hit 99.6%/99.7% (matching ViT), but recognition
    # CER collapsed to 29-41% (vs.\ ~7.6% for the original CLS-based
    # checkpoint), still visibly degrading at epoch 8. A loss weighted 10x
    # that finally CAN decrease will dominate a shared decoder's gradient
    # budget; a 20x LR on the shared projection layer that both heads read
    # through compounds it. The fix (correct pooling) and the compensating
    # tuning (aggressive weighting) were solving the same problem from two
    # different angles -- now that the fix works, the tuning is not just
    # unnecessary, it actively hurts. This run isolates that: same
    # architecture as the meanpool run above, original untuned recipe.
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                            weight_decay=cfg.WEIGHT_DECAY)
    total = len(loader) * cfg.EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.LR, total_steps=total, pct_start=0.1,
        anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.FP16 and DEVICE == "cuda")

    print(f"\n[train] {len(train_df):,} samples | {len(loader)} steps/epoch "
          f"| {cfg.EPOCHS} epochs\n")

    for ep in range(cfg.EPOCHS):
        model.train(); agg = defaultdict(float); n = 0
        pbar = tqdm(loader, desc=f"epoch {ep+1}/{cfg.EPOCHS}", dynamic_ncols=True)
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

            agg["rec"] += l_rec.item(); agg["wid"] += l_wid.item()
            agg["wacc"] += (wlog.argmax(-1) == wid).float().mean().item(); n += 1
            pbar.set_postfix(rec=f"{agg['rec']/n:.3f}", wid=f"{agg['wid']/n:.3f}",
                             wacc=f"{agg['wacc']/n:.3f}")

        ck = os.path.join(cfg.OUT, f"convnext_meanpool_untuned_ep{ep+1:02d}.pt")
        torch.save({"model": model.state_dict(), "epoch": ep + 1}, ck)
        v = quick_eval(model, val_df, vocab, fe, n=600, tag="val")
        print(f"  epoch {ep+1}: val ACC {v['ACC']:.2f}%  CER {v['CER']:.2f}%  "
              f"WER {v['WER']:.2f}%  -> {ck}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 7.  GATE -- do not proceed to unlearning unless this passes
# ══════════════════════════════════════════════════════════════════════════════
def gate_check(model, retain_df, forget_df, test_df, vocab, fe, cfg=C):
    r = {t: quick_eval(model, d, vocab, fe, tag=t) for t, d in
         [("forget", forget_df), ("retain", retain_df), ("test", test_df)]}

    print("\n" + "=" * 68)
    print("  CLEAN BASELINE  (model M)")
    print("-" * 68)
    for t in ("forget", "retain", "test"):
        print(f"  {t:<7} ACC {r[t]['ACC']:6.2f} %   CER {r[t]['CER']:6.2f} %"
              f"   WER {r[t]['WER']:6.2f} %   (n={r[t]['n']})")
    print("=" * 68)

    gap = r["test"]["CER"] - r["retain"]["CER"]
    checks = [
        ("writer ACC on forget >= 95%", r["forget"]["ACC"] >= 95,
         "RQ1 needs the model to actually identify its training writers; "
         "raise WRITER_W or train longer"),
        ("writer ACC on retain >= 95%", r["retain"]["ACC"] >= 95,
         "same fix"),
        ("member/non-member CER gap >= 5 pts", gap >= 5.0,
         f"gap is {gap:.2f}; too small means no memorisation signal for MIA"),
    ]
    print("\n  GATE:")
    ok = True
    for name, passed, note in checks:
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}"
              + (f"\n           -> {note}" if note and not passed else ""))
        ok &= passed
    print(f"\n  member/non-member CER gap: {gap:.2f} points")
    print("  ->", "PROCEED to MIA + Stage I" if ok else
          "DO NOT proceed; fix the failures above first")
    return r, ok


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(cfg=C):
    print(f"device: {DEVICE}\n")
    inspect_layout(cfg.ROOT)

    os.makedirs(cfg.OUT, exist_ok=True)
    import shutil, glob as _glob
    print(f"[diag] /kaggle/input contents: {os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'MISSING'}")
    print(f"[diag] expected SRC={cfg.SRC} exists={os.path.isdir(cfg.SRC)}")
    hits = _glob.glob("/kaggle/input/**/split.json", recursive=True)
    print(f"[diag] split.json found anywhere under /kaggle/input: {hits}")
    real_src = os.path.dirname(hits[0]) if hits else cfg.SRC
    for fname in ("split.json", "vocab_clean.json"):
        src, dst = os.path.join(real_src, fname), os.path.join(cfg.OUT, fname)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"[reuse] copied frozen {fname} from {src} -> {dst}")

    df = load_with_writers(cfg.ROOT)
    # build_frozen_split() reloads split.json from cfg.OUT if present (just
    # copied above), reproducing the identical 30/20/74 writer partition
    # every other checkpoint in this project uses.
    retain_df, forget_df, val_df, test_df, w2i = build_frozen_split(df, cfg)

    vocab = GraphemeVocab.load(os.path.join(cfg.OUT, "vocab_clean.json"))
    print(f"[vocab] reloaded frozen vocab, vocab_size={vocab.vocab_size}")
    report_oov(vocab, test_df)

    fe = AutoImageProcessor.from_pretrained(cfg.TROCR_MODEL)
    model = build_fresh_convnext(vocab.vocab_size, len(w2i), cfg)
    model = train_baseline(model, retain_df, forget_df, val_df, vocab, fe, cfg)

    res, ok = gate_check(model, retain_df, forget_df, test_df, vocab, fe, cfg)
    torch.save({"model": model.state_dict(), "w2i": w2i, "gate": res, "ok": ok},
               os.path.join(cfg.OUT, "convnext_meanpool_untuned_final.pt"))

    smoke_tag = "SMOKE TEST" if CONVNEXT_SMOKE_TEST else "REAL RUN"
    print("\n" + "#" * 78)
    print(f"#  FINAL RESULT -- ConvNeXt-Base mean-pool baseline ({smoke_tag})")
    print("#" * 78)
    for t in ("forget", "retain", "test"):
        print(f"#  {t:<7} writer-ACC {res[t]['ACC']:6.2f}%   CER {res[t]['CER']:6.2f}%   "
              f"WER {res[t]['WER']:6.2f}%   (n={res[t]['n']})")
    print(f"#  gate passed: {ok}")
    print(f"#  checkpoint saved -> {cfg.OUT}/convnext_meanpool_untuned_final.pt")
    print("#" * 78 + "\n")

    return model, (retain_df, forget_df, val_df, test_df, w2i), vocab, fe, res, ok




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

if __name__ == "__main__":
    main()