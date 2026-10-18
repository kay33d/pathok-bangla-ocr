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
class C:
    ROOT = "/kaggle/input/datasets/kaidul007/bnhtrd/Dataset"
    TROCR_MODEL = "microsoft/trocr-base-handwritten"
    OUT = "/kaggle/working/pathok_clean"

    SEED = 42
    N_TEST_WRITERS = 30       # NON-members. never seen by anything.
    N_FORGET_WRITERS = 20     # members, to be unlearned in stage II
    VAL_FRAC = 0.05           # sampled within retain writers

    EPOCHS = 8
    BATCH = 16
    LR = 5e-5
    WEIGHT_DECAY = 0.01
    WRITER_W = 2.0            # weight on writer CE. raise if writer ACC < 95%
    FP16 = True
    GRAD_CKPT = False         # True if OOM; ~30% slower
    NUM_WORKERS = 2

    EVAL_N = 1500             # samples per split in the gate check


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
    n = n or cfg.EVAL_N
    d = df.sample(min(n, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval()
    preds, wc, wt = [], 0, 0
    for b in tqdm(loader, desc=f"eval {tag}", leave=False, dynamic_ncols=True):
        pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
        enc = model.ved.encoder(pixel_values=pv, return_dict=True)
        wl = model.writer_head(enc.last_hidden_state[:, 0, :])
        m = wid >= 0
        if m.any():
            wc += (wl[m].argmax(-1) == wid[m]).sum().item(); wt += int(m.sum())
        gen = model.ved.generate(pv, max_new_tokens=MAX_LEN, num_beams=num_beams)
        preds += [vocab.decode(r) for r in gen.cpu().tolist()]
    cer, wer = corpus_cer_wer(preds, d["label"].astype(str).tolist())
    return {"ACC": 100 * wc / wt if wt else float("nan"),
            "CER": 100 * cer, "WER": 100 * wer, "n": len(d)}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def train_baseline(model, retain_df, forget_df, val_df, vocab, fe, cfg=C):
    train_df = pd.concat([retain_df, forget_df], ignore_index=True)
    loader = DataLoader(WDataset(train_df, vocab, fe), batch_size=cfg.BATCH,
                        shuffle=True, num_workers=cfg.NUM_WORKERS,
                        pin_memory=True, drop_last=True)

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

        ck = os.path.join(cfg.OUT, f"baseline_ep{ep+1:02d}.pt")
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

    df = load_with_writers(cfg.ROOT)
    retain_df, forget_df, val_df, test_df, w2i = build_frozen_split(df, cfg)

    vocab = GraphemeVocab().build(pd.concat([retain_df, forget_df])["label"])
    vocab.save(os.path.join(cfg.OUT, "vocab_clean.json"))
    report_oov(vocab, test_df)

    fe = AutoImageProcessor.from_pretrained(cfg.TROCR_MODEL)
    model = build_fresh(vocab.vocab_size, len(w2i), cfg)
    model = train_baseline(model, retain_df, forget_df, val_df, vocab, fe, cfg)

    res, ok = gate_check(model, retain_df, forget_df, test_df, vocab, fe, cfg)
    torch.save({"model": model.state_dict(), "w2i": w2i, "gate": res, "ok": ok},
               os.path.join(cfg.OUT, "baseline_clean_final.pt"))
    print(f"\nsaved -> {cfg.OUT}/baseline_clean_final.pt")

    return model, (retain_df, forget_df, val_df, test_df, w2i), vocab, fe, res, ok


if __name__ == "__main__":
    main()
