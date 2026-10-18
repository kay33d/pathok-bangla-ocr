"""
mia_gate_v2.py
==============
SELF-CONTAINED. GPU. ~15 min.
Run on the NEW baseline (height+contrast preprocessing).

THREE FIXES OVER v1
-------------------
1. MATCHING PREPROCESSING  [critical]
   v1 fed RAW crops. The new model has only ever seen height-normalised,
   contrast-standardised 64x256 canvases. Feeding it raw images is an
   out-of-distribution shift and every number would be meaningless.

2. WRITER-LEVEL CROSS-VALIDATION  [methodological]
   v1 split WORDS randomly into attacker-train / attacker-test, so the same
   writer's words appeared on both sides. The attacker could memorise
   writer-specific quirks rather than membership, inflating AUC. Folds are now
   split by WRITER, so the attacker is always evaluated on writers it has never
   scored.

3. PER-WRITER AGGREGATION  [threat model]
   Deletion requests are per-user. A realistic adversary asks "was writer 53 in
   the training set?" using all of that writer's words at once, not one word in
   isolation. Per-word AUC is reported for comparability with Kang et al.;
   per-writer AUC is the honest evaluation and should be substantially higher.

CALIBRATION
-----------
v1 read the argmax, and non-members came out at 63% "seen" instead of ~50%.
The decision threshold is now placed at Youden's J on out-of-fold scores, so
"seen"/"unseen" percentages are interpretable.

READING THE OUTPUT
------------------
The Stage II target is the NON-MEMBER rate, not 50%. After unlearning,
forget_seen should land near nonmember_seen. Driving it below would be
over-unlearning -- making deleted writers look less member-like than writers
the model never saw, which is its own detectable signature.

SET
---
    M.CKPT_ROOT -> folder with baseline_clean_final.pt, split.json,
                   vocab_clean.json
"""

import os
import glob
import json
import random
import warnings

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import editdistance
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, AutoImageProcessor
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

warnings.filterwarnings("ignore")


class M:
    # ---- EDIT THIS ----------------------------------------------------------
    CKPT_ROOT = "/kaggle/input/notebooks/kayed007/fork-of-retrain-pathok/pathok_clean"
    # -------------------------------------------------------------------------
    ROOT = "/kaggle/input/datasets/kaidul007/bnhtrd/Dataset"
    TROCR_MODEL = "microsoft/trocr-base-handwritten"
    CKPT_FILE = "baseline_clean_final.pt"
    SPLIT_FILE = "split.json"
    VOCAB_FILE = "vocab_clean.json"

    SEED = 42
    VAL_FRAC = 0.05           # must match the training run
    BATCH = 16
    NUM_WORKERS = 2

    WORDS_PER_WRITER = 80     # balanced sampling -> stable per-writer means
    DECODE_N = 1500
    N_FOLDS = 5

    TARGET_H = 64
    TARGET_W = 256


PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = 4
MAX_LEN = 64
PUNCT = {'.', ':', '।', ',', ';', '!', '?', '-', '–', '—'}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(M.SEED); np.random.seed(M.SEED); torch.manual_seed(M.SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  GRAPHEMES / VOCAB
# ══════════════════════════════════════════════════════════════════════════════
HASANTA, CHANDRABINDU, ANUSVARA = '\u09CD', '\u0981', '\u0982'
VISARGA, NUKTA = '\u0983', '\u09BC'
VOWEL_SIGNS = set('\u09BE\u09BF\u09C0\u09C1\u09C2\u09C3\u09C4'
                  '\u09C7\u09C8\u09CB\u09CC\u09D7')
INDEPENDENT_VOWELS = set('\u0985\u0986\u0987\u0988\u0989\u098A\u098B'
                         '\u098F\u0990\u0993\u0994')
CONSONANTS = set(chr(c) for c in range(0x0995, 0x09BA)) | {
    '\u09DC', '\u09DD', '\u09DF', '\u09CE', '\u09B0'}


def segment_graphemes(word):
    out, i, n = [], 0, len(word)
    while i < n:
        ch = word[i]
        if ch in INDEPENDENT_VOWELS:
            g = ch; i += 1
            while i < n and word[i] in {CHANDRABINDU, ANUSVARA, VISARGA}:
                g += word[i]; i += 1
            out.append(g)
        elif ch in CONSONANTS:
            g = ch; i += 1
            if i < n and word[i] == NUKTA:
                g += word[i]; i += 1
            while i + 1 < n and word[i] == HASANTA and word[i + 1] in CONSONANTS:
                g += word[i] + word[i + 1]; i += 2
            if i < n and word[i] in VOWEL_SIGNS:
                g += word[i]; i += 1
            out.append(g)
        else:
            out.append(ch); i += 1
    return out


def is_bangla_word(t):
    c = [x for x in t if not x.isspace()]
    if not c:
        return False
    return sum(1 for x in c if '\u0980' <= x <= '\u09FF') / len(c) > 0.8


class GraphemeVocab:
    def __init__(self):
        self.g2i = {}
        self.i2g = {PAD_ID: "<pad>", UNK_ID: "<unk>", BOS_ID: "<s>", EOS_ID: "</s>"}
        self.vocab_size = SPECIAL_TOKENS

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

    @classmethod
    def load(cls, p):
        d = json.load(open(p, encoding="utf-8"))
        v = cls(); v.g2i = d["g2i"]
        v.i2g = {int(k): x for k, x in d["i2g"].items()}
        v.vocab_size = d["vocab_size"]
        print(f"[vocab] {p} | vocab_size={v.vocab_size}")
        return v


def encode_label(vocab, text):
    seq = (vocab.encode(text) + [EOS_ID])[:MAX_LEN]
    return seq + [PAD_ID] * (MAX_LEN - len(seq))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PREPROCESSING  -- MUST match the retrain exactly
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(gray, cfg=M):
    """height+contrast, squeeze over-wide words rather than truncating."""
    h, w = gray.shape
    scale = cfg.TARGET_H / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    if new_w > cfg.TARGET_W:
        content = cv2.resize(gray, (cfg.TARGET_W, cfg.TARGET_H),
                             interpolation=cv2.INTER_AREA)
    else:
        content = cv2.resize(gray, (new_w, cfg.TARGET_H),
                             interpolation=cv2.INTER_LINEAR)
    c = content.astype(np.float32)
    c = (c - c.mean()) / (c.std() + 1e-6)
    content = np.clip(c * 48 + 200, 0, 255).astype(np.uint8)
    canvas = np.full((cfg.TARGET_H, cfg.TARGET_W), 255, dtype=np.uint8)
    canvas[:, :content.shape[1]] = content
    return canvas


class WDataset(Dataset):
    def __init__(self, df, vocab, fe, cfg=M):
        self.df = df.reset_index(drop=True)
        self.vocab, self.fe, self.cfg = vocab, fe, cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        gray = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.full((self.cfg.TARGET_H, self.cfg.TARGET_W), 255, np.uint8)
        img = Image.fromarray(cv2.cvtColor(preprocess(gray, self.cfg),
                                           cv2.COLOR_GRAY2RGB))
        pv = self.fe(img, return_tensors="pt").pixel_values.squeeze(0)
        lb = torch.tensor(encode_label(self.vocab, str(r["label"])), dtype=torch.long)
        lb[lb == PAD_ID] = -100
        return {"pixel_values": pv, "labels": lb,
                "wid": torch.tensor(int(r["wid"]), dtype=torch.long)}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA / SPLIT RELOAD
# ══════════════════════════════════════════════════════════════════════════════
def writer_of(p, root):
    return os.path.relpath(p, root).split(os.sep)[0]


def load_with_writers(cfg=M):
    label_dfs = []
    for f in glob.glob(cfg.ROOT + "/*/*.xlsx"):
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

    imgs = (glob.glob(cfg.ROOT + "/*/Words/*/*.jpg")
            + glob.glob(cfg.ROOT + "/*/Words/*/*.JPG"))
    image_df = pd.DataFrame({"Id": [os.path.basename(p).split(".")[0] for p in imgs],
                             "image_path": imgs})
    df = (image_df.merge(labels_df, on="Id", how="inner")
          .rename(columns={"Word": "label"})[["image_path", "label"]]
          .dropna().reset_index(drop=True))
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"] != ""]
    df = df[~df["label"].isin(PUNCT)]
    df = df[df["label"].apply(is_bangla_word)]
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    df["writer"] = df["image_path"].apply(lambda p: writer_of(p, cfg.ROOT))
    print(f"samples: {len(df):,} | writers: {df['writer'].nunique()}")
    return df


def reload_split(df, cfg=M):
    spec = json.load(open(os.path.join(cfg.CKPT_ROOT, cfg.SPLIT_FILE)))
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

    W = lambda d: set(d["writer"])
    assert not (W(test_df) & (W(retain_df) | W(forget_df))), "TEST WRITER LEAK"

    empty = sorted(set(w2i) - (W(retain_df) | W(forget_df)))
    if empty:
        print(f"  [note] {len(empty)} writer class(es) with no surviving images "
              f"after the label dedupe: {empty} -- dead output units in the "
              f"writer head, harmless but worth stating in the writeup")
    print(f"  retain {len(retain_df):>6,} / {len(W(retain_df))} writers")
    print(f"  forget {len(forget_df):>6,} / {len(W(forget_df))} writers")
    print(f"  test   {len(test_df):>6,} / {len(W(test_df))} writers  (NON-members)")
    return retain_df, forget_df, val_df, test_df, w2i


def sample_per_writer(df, n, cfg=M):
    """Balanced sampling so per-writer means are comparable across writers."""
    return (df.groupby("writer", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), n), random_state=cfg.SEED))
              .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL
# ══════════════════════════════════════════════════════════════════════════════
class MultiHead(nn.Module):
    def __init__(self, ved, n_writers):
        super().__init__()
        self.ved = ved
        self.writer_head = nn.Linear(ved.config.encoder.hidden_size, n_writers)
        self.n_writers = n_writers

    def forward(self, pv, labels=None):
        enc = self.ved.encoder(pixel_values=pv, return_dict=True)
        wlog = self.writer_head(enc.last_hidden_state[:, 0, :])
        rec = (self.ved(encoder_outputs=enc, labels=labels, return_dict=True)
               if labels is not None else None)
        return rec, wlog


def load_model(vocab, cfg=M):
    ck = torch.load(os.path.join(cfg.CKPT_ROOT, cfg.CKPT_FILE),
                    map_location="cpu", weights_only=False)
    n_writers = (len(ck["w2i"]) if "w2i" in ck
                 else ck["model"]["writer_head.weight"].shape[0])
    ved = VisionEncoderDecoderModel.from_pretrained(cfg.TROCR_MODEL)
    ved.decoder.resize_token_embeddings(vocab.vocab_size)
    ved.config.vocab_size = vocab.vocab_size
    ved.config.decoder.vocab_size = vocab.vocab_size
    for c in (ved.config, ved.generation_config):
        c.decoder_start_token_id = BOS_ID
        c.pad_token_id = PAD_ID
        c.eos_token_id = EOS_ID
        c.bos_token_id = BOS_ID
    ved.generation_config.max_new_tokens = MAX_LEN

    model = MultiHead(ved, n_writers)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing:
        print(f"  [warn] missing: {list(missing)[:5]}")
    model = model.to(DEVICE).eval()
    print(f"[model] loaded | writers={n_writers}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DECODE DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
def corpus_cer_wer(preds, gts):
    d = c = we = 0
    for p, g in zip(preds, gts):
        d += editdistance.eval(list(p), list(g))
        c += max(len(g), 1); we += (p != g)
    return d / max(c, 1), we / max(len(gts), 1)


@torch.no_grad()
def decode_diagnostic(model, df, vocab, fe, cfg=M, tag=""):
    d = df.sample(min(cfg.DECODE_N, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    preds, wc, wt = [], 0, 0
    for b in tqdm(loader, desc=f"decode {tag}", leave=False, dynamic_ncols=True):
        pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
        enc = model.ved.encoder(pixel_values=pv, return_dict=True)
        wl = model.writer_head(enc.last_hidden_state[:, 0, :])
        m = wid >= 0
        if m.any():
            wc += (wl[m].argmax(-1) == wid[m]).sum().item(); wt += int(m.sum())
        gen = model.ved.generate(pv, max_new_tokens=MAX_LEN, num_beams=1)
        preds += [vocab.decode(r) for r in gen.cpu().tolist()]

    gts = d["label"].astype(str).tolist()
    dists = np.array([editdistance.eval(list(p), list(g)) for p, g in zip(preds, gts)])
    gtlen = np.array([max(len(g), 1) for g in gts])
    plen = np.array([len(p) for p in preds])
    wrong = np.array([p != g for p, g in zip(preds, gts)])
    runaway = plen > (3 * gtlen + 5)
    cer_raw = 100 * dists.sum() / gtlen.sum()
    keep = ~runaway
    cer_t = 100 * dists[keep].sum() / gtlen[keep].sum() if keep.any() else float("nan")

    print(f"\n  [{tag}] n={len(d)}  writer ACC "
          f"{100*wc/max(wt,1):.2f} %" if wt else f"\n  [{tag}] n={len(d)}")
    print(f"    WER {100*wrong.mean():6.2f} %   CER raw {cer_raw:6.2f} %   "
          f"CER trimmed {cer_t:6.2f} %   runaway {runaway.sum()}")
    return {"wer": 100 * wrong.mean(), "cer_raw": cer_raw, "cer_trimmed": cer_t,
            "runaway": int(runaway.sum()),
            "writer_acc": 100 * wc / wt if wt else float("nan")}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MIA FEATURES
# ══════════════════════════════════════════════════════════════════════════════
MIA_FEAT_DIM = 12


@torch.no_grad()
def mia_features(model, df, vocab, fe, cfg=M, tag=""):
    """Returns (features, writer_array). Teacher-forced confidence signature."""
    loader = DataLoader(WDataset(df, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    feats = []
    for b in tqdm(loader, desc=f"MIA {tag}", leave=False, dynamic_ncols=True):
        pv = b["pixel_values"].to(DEVICE); lb = b["labels"].to(DEVICE)
        rec, _ = model(pv, lb)
        logp = F.log_softmax(rec.logits.float(), dim=-1)
        probs = logp.exp()
        mask = (lb != -100)
        tok_lp = logp.gather(-1, lb.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        ent = -(probs * logp).sum(-1)
        top2 = probs.topk(2, dim=-1).values
        m = mask.float(); cnt = m.sum(1).clamp(min=1)
        mmean = lambda x: (x * m).sum(1) / cnt
        mmax = lambda x: x.masked_fill(~mask, -1e9).max(1).values
        mmin = lambda x: x.masked_fill(~mask, 1e9).min(1).values
        mean_lp = mmean(tok_lp)
        feats.append(torch.stack([
            mean_lp, mmin(tok_lp), mmax(tok_lp),
            torch.sqrt(mmean((tok_lp - mean_lp.unsqueeze(1)) ** 2) + 1e-8),
            mmean(ent), mmax(ent),
            mmean(top2[..., 0]), mmean(top2[..., 0] - top2[..., 1]),
            mmean((tok_lp.exp() > 0.9).float()),
            mmean((tok_lp.exp() < 0.1).float()),
            cnt / MAX_LEN, mean_lp * cnt / MAX_LEN,
        ], dim=1).cpu())
    return torch.cat(feats, 0).numpy(), df["writer"].values


# ══════════════════════════════════════════════════════════════════════════════
# 7.  WRITER-LEVEL CROSS-VALIDATED MIA
# ══════════════════════════════════════════════════════════════════════════════
def writer_cv_mia(Xr, Wr, Xt, Wt, Xf, Wf, cfg=M):
    """
    Folds are split by WRITER, so the attacker never scores a writer it was
    trained on. v1 split words randomly, letting the attacker memorise
    writer-specific quirks instead of membership.
    """
    X = np.vstack([Xr, Xt])
    y = np.r_[np.ones(len(Xr)), np.zeros(len(Xt))]
    W = np.r_[Wr, Wt]

    writers = np.unique(W)
    wlab = np.array([y[W == w][0] for w in writers])

    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    for tr_i, te_i in skf.split(writers, wlab):
        tr_m = np.isin(W, writers[tr_i])
        te_m = np.isin(W, writers[te_i])
        sc = StandardScaler().fit(X[tr_m])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(X[tr_m]), y[tr_m])
        oof[te_m] = clf.predict_proba(sc.transform(X[te_m]))[:, 1]

    word_auc = roc_auc_score(y, oof)

    # per-writer: average the out-of-fold member probability over the writer
    wscore = np.array([oof[W == w].mean() for w in writers])
    writer_auc = roc_auc_score(wlab, wscore)

    # calibrated threshold: Youden's J on out-of-fold scores
    fpr, tpr, thr = roc_curve(y, oof)
    thr_star = float(thr[np.argmax(tpr - fpr)])

    # final attacker on everything, to score the forget set
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(X), y)
    pf = clf.predict_proba(sc.transform(Xf))[:, 1]
    fwriters = np.unique(Wf)
    fscore = np.array([pf[Wf == w].mean() for w in fwriters])

    seen = lambda p: 100 * (p >= thr_star).mean()
    return {
        "word_auc": word_auc,
        "writer_auc": writer_auc,
        "threshold": thr_star,
        "member_seen": seen(oof[y == 1]),
        "nonmember_seen": seen(oof[y == 0]),
        "forget_seen": seen(pf),
        "member_writer_seen": 100 * (wscore[wlab == 1] >= thr_star).mean(),
        "nonmember_writer_seen": 100 * (wscore[wlab == 0] >= thr_star).mean(),
        "forget_writer_seen": 100 * (fscore >= thr_star).mean(),
        "n_writers_member": int((wlab == 1).sum()),
        "n_writers_nonmember": int((wlab == 0).sum()),
        "n_writers_forget": len(fwriters),
        "forget_writer_scores": {str(w): float(s)
                                 for w, s in zip(fwriters, fscore)},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(cfg=M):
    print(f"device: {DEVICE}")
    assert os.path.isdir(cfg.CKPT_ROOT), f"CKPT_ROOT not found: {cfg.CKPT_ROOT}"

    vocab = GraphemeVocab.load(os.path.join(cfg.CKPT_ROOT, cfg.VOCAB_FILE))
    df = load_with_writers(cfg)
    retain_df, forget_df, val_df, test_df, w2i = reload_split(df, cfg)
    model = load_model(vocab, cfg)
    fe = AutoImageProcessor.from_pretrained(cfg.TROCR_MODEL)

    print("\n" + "=" * 70)
    print("  DECODE DIAGNOSTIC  (height+contrast preprocessing)")
    print("-" * 70)
    dg = {t: decode_diagnostic(model, d, vocab, fe, cfg, t)
          for t, d in [("forget", forget_df), ("retain", retain_df),
                       ("test", test_df)]}

    print("\n" + "=" * 70)
    print("  MEMBERSHIP INFERENCE  (writer-level CV, calibrated)")
    print("-" * 70)
    Sr = sample_per_writer(retain_df, cfg.WORDS_PER_WRITER, cfg)
    St = sample_per_writer(test_df, cfg.WORDS_PER_WRITER, cfg)
    Sf = sample_per_writer(forget_df, cfg.WORDS_PER_WRITER, cfg)
    print(f"  sampled {len(Sr):,} retain / {len(St):,} test / {len(Sf):,} forget "
          f"words ({cfg.WORDS_PER_WRITER} per writer)")

    Xr, Wr = mia_features(model, Sr, vocab, fe, cfg, "retain")
    Xt, Wt = mia_features(model, St, vocab, fe, cfg, "test")
    Xf, Wf = mia_features(model, Sf, vocab, fe, cfg, "forget")

    r = writer_cv_mia(Xr, Wr, Xt, Wt, Xf, Wf, cfg)

    print(f"\n  PER-WORD   AUC : {r['word_auc']:.3f}   "
          f"(comparable to Kang et al.)")
    print(f"  PER-WRITER AUC : {r['writer_auc']:.3f}   "
          f"(the realistic threat model; n={r['n_writers_member']} member / "
          f"{r['n_writers_nonmember']} non-member writers)")
    print(f"  decision threshold (Youden's J): {r['threshold']:.3f}")
    print("\n  word level:")
    print(f"    members    -> seen : {r['member_seen']:6.2f} %")
    print(f"    NON-MEMBERS-> seen : {r['nonmember_seen']:6.2f} %   "
          f"<-- Stage II TARGET for forget")
    print(f"    forget     -> seen : {r['forget_seen']:6.2f} %")
    print("\n  writer level:")
    print(f"    members    -> seen : {r['member_writer_seen']:6.2f} %")
    print(f"    NON-MEMBERS-> seen : {r['nonmember_writer_seen']:6.2f} %")
    print(f"    forget     -> seen : {r['forget_writer_seen']:6.2f} %  "
          f"({r['n_writers_forget']} writers)")

    wer_gap = dg["test"]["wer"] - dg["retain"]["wer"]
    checks = [
        ("per-word AUC >= 0.60", r["word_auc"] >= 0.60, ""),
        ("per-writer AUC >= 0.70", r["writer_auc"] >= 0.70,
         "weak per-writer attacker makes the unlearning claim easy to satisfy"),
        ("forget seen >= member/non-member midpoint",
         r["forget_seen"] >= (r["member_seen"] + r["nonmember_seen"]) / 2,
         "forget must start clearly member-like"),
        ("member/non-member WER gap >= 5 pts", wer_gap >= 5.0, f"gap {wer_gap:.2f}"),
    ]
    print("\n  GATE:")
    ok = True
    for name, passed, note in checks:
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}"
              + (f"\n           -> {note}" if note and not passed else ""))
        ok &= passed
    print(f"\n  member/non-member WER gap: {wer_gap:.2f} points")
    print(f"\n  STAGE II TARGETS (do not aim for 50%):")
    print(f"    word   level: drive forget_seen {r['forget_seen']:.1f} % "
          f"-> ~{r['nonmember_seen']:.1f} %")
    print(f"    writer level: drive forget_seen {r['forget_writer_seen']:.1f} % "
          f"-> ~{r['nonmember_writer_seen']:.1f} %")
    print("  ->", "PROCEED to Stage I pruning" if ok else "investigate first")

    out = {"decode": dg, "mia": r, "wer_gap": wer_gap, "ok": bool(ok)}
    json.dump(out, open("/kaggle/working/mia_gate_v2.json", "w"),
              indent=2, default=float)
    print("\nsaved -> /kaggle/working/mia_gate_v2.json")
    return model, (retain_df, forget_df, val_df, test_df, w2i), vocab, fe, out


if __name__ == "__main__":
    main()
