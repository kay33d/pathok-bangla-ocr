"""
pathok_common.py
================
Shared machinery for the Tier-1 experiments. Import this first, then run any
of exp1..exp5. On Kaggle: paste this whole file into one cell and run it, then
paste the experiment file into the next cell.

    %%writefile pathok_common.py
    ...            # or just paste and exec

Everything here is lifted from your existing scripts with three changes:

 1. reload_split() no longer requires a checkpoint. w2i can be supplied, taken
    from a checkpoint, or rebuilt with sorted(). All experiments must use the
    SAME w2i, so pass it explicitly once you have it.

 2. page_of() is added. BN-HTRd paths are <ROOT>/<writer>/Words/<page>/<word>.jpg,
    so page id is recoverable. Probe train/test splits are made BY PAGE inside
    each writer, otherwise two crops from the same line share ink thickness,
    lighting and pen pressure and the probe reports memorisation of the page,
    not of the writer.

 3. Cluster bootstrap over WRITERS. Every headline number in the paper is an
    average over 20-74 writers, not over 1500 independent words, so word-level
    CIs understate the uncertainty by roughly sqrt(words_per_writer).

NOTHING here trains anything. It is safe to import repeatedly.
"""

import os
import glob
import json
import copy
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class S:
    # ---- EDIT THESE ---------------------------------------------------------
    CKPT_ROOT = "/kaggle/input/notebooks/kayed007/fork-of-retrain-pathok/pathok_clean"
    ROOT      = "/kaggle/input/datasets/kaidul007/bnhtrd/Dataset"
    # -------------------------------------------------------------------------
    TROCR_MODEL = "microsoft/trocr-base-handwritten"
    CKPT_FILE   = "baseline_clean_final.pt"
    SPLIT_FILE  = "split.json"
    VOCAB_FILE  = "vocab_clean.json"

    SEED = 42
    VAL_FRAC = 0.05          # must match the original retrain
    BATCH = 16
    NUM_WORKERS = 2
    TARGET_H = 64
    TARGET_W = 256

    CACHE = "/kaggle/working/df_cache.parquet"


PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = 4
MAX_LEN = 64
PUNCT = {'.', ':', '।', ',', ';', '!', '?', '-', '–', '—'}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed=S.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_all()


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHEMES / VOCAB
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
    return bool(c) and sum(1 for x in c if '\u0980' <= x <= '\u09FF') / len(c) > 0.8


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


def encode_label(vocab, text):
    seq = (vocab.encode(text) + [EOS_ID])[:MAX_LEN]
    return seq + [PAD_ID] * (MAX_LEN - len(seq))


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (height + contrast, squeeze not truncate)
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(gray, cfg=S):
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
    def __init__(self, df, vocab, fe, cfg=S):
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
# DATA / SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def writer_of(p, root):
    return os.path.relpath(p, root).split(os.sep)[0]


def page_of(p, root):
    """<ROOT>/<writer>/Words/<page>/<word>.jpg -> '<writer>/<page>'.
    Probe splits go by page: two crops from the same line share ink weight,
    lighting and pen pressure, so a within-page split measures page recall."""
    parts = os.path.relpath(p, root).split(os.sep)
    return f"{parts[0]}/{parts[2]}" if len(parts) >= 4 else parts[0]


def load_with_writers(cfg=S, use_cache=True):
    if use_cache and cfg.CACHE and os.path.exists(cfg.CACHE):
        df = pd.read_parquet(cfg.CACHE)
        print(f"  [cache] {len(df):,} samples | writers {df['writer'].nunique()}")
        return df

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
    if not label_dfs:
        raise FileNotFoundError(f"No .xlsx under {cfg.ROOT}/*/*.xlsx -- check S.ROOT")
    labels_df = (pd.concat(label_dfs, ignore_index=True)
                 .drop_duplicates(subset="Id", keep="first"))
    imgs = (glob.glob(cfg.ROOT + "/*/Words/*/*.jpg")
            + glob.glob(cfg.ROOT + "/*/Words/*/*.JPG"))
    image_df = pd.DataFrame({"Id": [os.path.basename(p).split(".")[0] for p in imgs],
                             "image_path": imgs})
    df = (image_df.merge(labels_df, on="Id", how="inner")
          .rename(columns={"Word": "label"})[["image_path", "label"]].dropna()
          .reset_index(drop=True))
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["label"] != "") & (~df["label"].isin(PUNCT))]
    df = df[df["label"].apply(is_bangla_word)]
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    df["writer"] = df["image_path"].apply(lambda p: writer_of(p, cfg.ROOT))
    df["page"] = df["image_path"].apply(lambda p: page_of(p, cfg.ROOT))
    print(f"  samples: {len(df):,} | writers: {df['writer'].nunique()} "
          f"| pages: {df['page'].nunique()}")
    if cfg.CACHE:
        try:
            df.to_parquet(cfg.CACHE)
        except Exception as e:
            print(f"  [cache] skipped: {e}")
    return df


def reload_split(df, cfg=S, w2i=None, ck=None, verbose=True):
    """Rebuild the frozen split. Pass w2i explicitly (preferred) so every
    experiment agrees on writer->index. Falls back to the checkpoint, then to
    sorted() -- note string sort puts '10' before '2'."""
    spec = json.load(open(os.path.join(cfg.CKPT_ROOT, cfg.SPLIT_FILE)))
    test_w, forget_w, retain_w = (set(map(str, spec[k]))
                                  for k in ("test", "forget", "retain"))
    if w2i is None:
        if ck is not None and "w2i" in ck:
            w2i = {str(k): int(v) for k, v in ck["w2i"].items()}
        else:
            w2i = {w: i for i, w in enumerate(sorted(retain_w | forget_w))}

    test_df = df[df["writer"].isin(test_w)].reset_index(drop=True)
    forget_df = df[df["writer"].isin(forget_w)].reset_index(drop=True)
    retain_all = df[df["writer"].isin(retain_w)].reset_index(drop=True)
    retain_df, val_df = train_test_split(retain_all, test_size=cfg.VAL_FRAC,
                                         random_state=cfg.SEED)
    retain_df = retain_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    for d in (retain_df, forget_df, val_df):
        d["wid"] = d["writer"].map(w2i).fillna(-1).astype(int)
    test_df["wid"] = -1

    assert not (set(test_df["writer"]) & (set(retain_df["writer"])
                                          | set(forget_df["writer"]))), "TEST LEAK"
    if verbose:
        print(f"  retain {len(retain_df):,}/{retain_df['writer'].nunique()}w (MEMBERS) | "
              f"forget {len(forget_df):,}/{forget_df['writer'].nunique()}w | "
              f"val {len(val_df):,} | "
              f"test {len(test_df):,}/{test_df['writer'].nunique()}w (NON-MEMBERS)")
    return retain_df, forget_df, val_df, test_df, w2i


def sample_per_writer(df, k, seed=S.SEED):
    return (df.groupby("writer", group_keys=False)
              .apply(lambda g: g.sample(min(k, len(g)), random_state=seed))
              .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
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


def build_model(vocab, n_writers, cfg=S):
    ved = VisionEncoderDecoderModel.from_pretrained(cfg.TROCR_MODEL)
    ved.decoder.resize_token_embeddings(vocab.vocab_size)
    ved.config.vocab_size = vocab.vocab_size
    ved.config.decoder.vocab_size = vocab.vocab_size
    for c in (ved.config, ved.generation_config):
        c.decoder_start_token_id = BOS_ID; c.pad_token_id = PAD_ID
        c.eos_token_id = EOS_ID; c.bos_token_id = BOS_ID
    ved.generation_config.max_new_tokens = MAX_LEN
    return MultiHead(ved, n_writers)


def load_into(model, sd, label="", strict_report=True):
    res = model.load_state_dict(sd, strict=False)
    crit = [k for k in res.missing_keys if not k.startswith("ved.encoder.pooler")]
    if crit:
        raise RuntimeError(f"[{label}] {len(crit)} critical keys missing, "
                           f"e.g. {crit[:5]} -- every number would be fiction")
    if strict_report and res.unexpected_keys:
        print(f"  [{label}] unexpected keys: {list(res.unexpected_keys)[:5]}")
    return model.to(DEVICE).eval()


def restore(model, state):
    model.load_state_dict(state, strict=True)
    return model


def get_feature_extractor(cfg=S):
    return AutoImageProcessor.from_pretrained(cfg.TROCR_MODEL)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def writer_acc(model, df, vocab, fe, cfg=S, n=800, return_per_writer=False):
    d = (df if n is None else
         df.sample(min(n, len(df)), random_state=cfg.SEED)).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    was = model.training; model.eval()
    hits = []
    for b in loader:
        pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
        enc = model.ved.encoder(pixel_values=pv, return_dict=True)
        wl = model.writer_head(enc.last_hidden_state[:, 0, :])
        hits.append((wl.argmax(-1) == wid).float().cpu())
    if was:
        model.train()
    hits = torch.cat(hits).numpy() if hits else np.array([])
    acc = 100.0 * hits.mean() if len(hits) else float("nan")
    if return_per_writer:
        return acc, pd.DataFrame({"writer": d["writer"].values, "hit": hits})
    return acc


@torch.no_grad()
def rec_quality(model, df, vocab, fe, cfg=S, n=300, return_per_word=False):
    d = (df if n is None else
         df.sample(min(n, len(df)), random_state=cfg.SEED)).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    was = model.training; model.eval(); preds = []
    for b in loader:
        gen = model.ved.generate(b["pixel_values"].to(DEVICE),
                                 max_new_tokens=MAX_LEN, num_beams=1)
        preds += [vocab.decode(r) for r in gen.cpu().tolist()]
    if was:
        model.train()
    gts = d["label"].astype(str).tolist()
    dist = np.array([editdistance.eval(list(p), list(g)) for p, g in zip(preds, gts)])
    gl = np.array([max(len(g), 1) for g in gts])
    keep = np.array([len(p) for p in preds]) <= 3 * gl + 5
    out = {"WER": 100 * np.mean([p != g for p, g in zip(preds, gts)]),
           "CER": 100 * dist[keep].sum() / gl[keep].sum() if keep.any() else float("nan"),
           "runaway": int((~keep).sum())}
    if return_per_word:
        out["per_word"] = pd.DataFrame({"writer": d["writer"].values,
                                        "dist": dist, "gtlen": gl,
                                        "wrong": [p != g for p, g in zip(preds, gts)]})
    return out


@torch.no_grad()
def writer_rank_diag(model, df, vocab, fe, n_classes, cfg=S, n=400):
    """Rank of the TRUE writer's logit among all classes, 1-indexed.
    Interpret against the ORACLE's rank, never against n_classes/2 -- a model
    retrained without these writers also ranks them low, so 'inversion' is only
    meaningful as a deviation from what exact deletion produces."""
    d = df.sample(min(n, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval(); ranks, acc, writers = [], [], []
    for b in loader:
        pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
        enc = model.ved.encoder(pixel_values=pv, return_dict=True)
        wl = model.writer_head(enc.last_hidden_state[:, 0, :]).float()
        true = wl.gather(1, wid[:, None])
        ranks.append(((wl > true).sum(1) + 1).float().cpu())
        acc.append((wl.argmax(-1) == wid).float().cpu())
    r = torch.cat(ranks).numpy()
    return {"mean_rank": float(r.mean()), "median_rank": float(np.median(r)),
            "frac_last": float((r == n_classes).mean()),
            "ACC": 100 * float(torch.cat(acc).mean()),
            "ranks": r, "writer": d["writer"].values}


# ══════════════════════════════════════════════════════════════════════════════
# MIA FEATURES  (12-dim teacher-forced decoder confidence, no writer head)
# ══════════════════════════════════════════════════════════════════════════════
FEATNAMES = ["mean_lp", "min_lp", "max_lp", "std_lp", "sum_lp", "mean_ent",
             "mean_margin", "mean_top1p", "frac_correct", "mean_logrank",
             "n_tokens", "last_lp"]


@torch.no_grad()
def extract_features(model, df, vocab, fe, cfg=S):
    loader = DataLoader(WDataset(df, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval(); feats = []
    for b in loader:
        pv = b["pixel_values"].to(DEVICE); lab = b["labels"].to(DEVICE)
        logits = model.ved(pixel_values=pv, labels=lab, return_dict=True).logits.float()
        lp = torch.log_softmax(logits, -1)
        mask = (lab != -100); gt = lab.clamp(min=0)
        tok = lp.gather(2, gt.unsqueeze(-1)).squeeze(-1)
        top2 = lp.topk(2, dim=-1).values
        ent = -(lp.exp() * lp).sum(-1)
        rank = (lp > tok.unsqueeze(-1)).sum(-1).float()
        m = mask.float(); n = m.sum(1).clamp(min=1)
        mean_lp = (tok * m).sum(1) / n
        f = torch.stack([
            mean_lp,
            tok.masked_fill(~mask, 1e9).min(1).values,
            tok.masked_fill(~mask, -1e9).max(1).values,
            torch.sqrt((((tok - mean_lp[:, None]) ** 2) * m).sum(1) / n),
            (tok * m).sum(1),
            (ent * m).sum(1) / n,
            ((top2[..., 0] - top2[..., 1]) * m).sum(1) / n,
            (tok.exp() * m).sum(1) / n,
            ((rank == 0).float() * m).sum(1) / n,
            (torch.log1p(rank) * m).sum(1) / n,
            n,
            tok.gather(1, (mask.sum(1) - 1).clamp(min=0)[:, None]).squeeze(1),
        ], dim=1)
        feats.append(f.cpu().numpy())
    return np.nan_to_num(np.concatenate(feats), nan=0.0, posinf=0.0, neginf=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER BOOTSTRAP OVER WRITERS
# ══════════════════════════════════════════════════════════════════════════════
def _resample_writers(groups, rng):
    """Resample writer ids with replacement, return row indices of the
    resampled panel. Duplicated writers contribute their rows twice."""
    uniq = np.unique(groups)
    picked = rng.choice(uniq, size=len(uniq), replace=True)
    idx = []
    lut = {w: np.flatnonzero(groups == w) for w in uniq}
    for w in picked:
        idx.append(lut[w])
    return np.concatenate(idx)


def boot_auc(y, scores, groups, n_boot=2000, seed=S.SEED, alpha=0.05):
    """AUC with a writer-level (cluster) bootstrap CI. Members and non-members
    are resampled independently so class balance is preserved."""
    y = np.asarray(y); scores = np.asarray(scores); groups = np.asarray(groups)
    rng = np.random.RandomState(seed)
    pos, neg = y == 1, y == 0
    vals = []
    for _ in range(n_boot):
        ip = _resample_writers(groups[pos], rng)
        ineg = _resample_writers(groups[neg], rng)
        yy = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
        ss = np.r_[scores[pos][ip], scores[neg][ineg]]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, ss))
    vals = np.array(vals)
    return {"point": roc_auc_score(y, scores),
            "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "boot": vals}


def boot_rate(scores, groups, thr, n_boot=2000, seed=S.SEED, alpha=0.05):
    """Percentage of items above thr, with a writer-level bootstrap CI."""
    scores = np.asarray(scores); groups = np.asarray(groups)
    rng = np.random.RandomState(seed)
    vals = [100.0 * (scores[_resample_writers(groups, rng)] > thr).mean()
            for _ in range(n_boot)]
    vals = np.array(vals)
    return {"point": 100.0 * float((scores > thr).mean()),
            "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2)))}


def boot_mean(values, groups, n_boot=2000, seed=S.SEED, alpha=0.05):
    values = np.asarray(values, dtype=float); groups = np.asarray(groups)
    rng = np.random.RandomState(seed)
    vals = [values[_resample_writers(groups, rng)].mean() for _ in range(n_boot)]
    vals = np.array(vals)
    return {"point": float(values.mean()),
            "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2)))}


def thr_at_fpr(nonmember_scores, target_fpr=0.10):
    """Fixed-FPR operating point. Prefer this to Youden's J: with 30 non-member
    writers and near-separable score distributions, J lands in an empty gap and
    the reported percentage swings by whole writers between arms for no reason."""
    return float(np.quantile(np.asarray(nonmember_scores), 1.0 - target_fpr))


def fmt_ci(d, pct=False, prec=3):
    u = "%" if pct else ""
    return f"{d['point']:.{prec}f}{u} [{d['lo']:.{prec}f}, {d['hi']:.{prec}f}]"


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
def discover_checkpoints(patterns=("unlearned_*.pt", "oracle*.pt")):
    reg = {}
    for pat in patterns:
        hits = sorted(set(glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
                          + glob.glob(f"/kaggle/working/**/{pat}", recursive=True)))
        for p in hits:
            stem = os.path.basename(p)[:-3]
            stem = stem[len("unlearned_"):] if stem.startswith("unlearned_") else stem
            if "_step" in stem:
                tag, step = stem.rsplit("_step", 1)
                reg[f"{tag}@{step}"] = p
            else:
                reg[stem] = p
    return reg


def setup(cfg=S, verbose=True):
    """Load vocab, data, split, feature extractor and the baseline state dict.
    Returns a ctx dict every experiment consumes."""
    if verbose:
        print("=" * 74); print(f"  SETUP  device={DEVICE}"); print("-" * 74)
    vocab = GraphemeVocab.load(os.path.join(cfg.CKPT_ROOT, cfg.VOCAB_FILE))
    base_ck = torch.load(os.path.join(cfg.CKPT_ROOT, cfg.CKPT_FILE),
                         map_location="cpu", weights_only=False)
    w2i = {str(k): int(v) for k, v in base_ck["w2i"].items()}
    df = load_with_writers(cfg)
    retain_df, forget_df, val_df, test_df, w2i = reload_split(df, cfg, w2i=w2i,
                                                              verbose=verbose)
    fe = get_feature_extractor(cfg)
    if verbose:
        print(f"  vocab_size={vocab.vocab_size} | writer classes={len(w2i)}")
    return {"vocab": vocab, "fe": fe, "df": df, "w2i": w2i,
            "retain_df": retain_df, "forget_df": forget_df,
            "val_df": val_df, "test_df": test_df,
            "base_sd": base_ck["model"], "n_classes": len(w2i), "cfg": cfg}


print("pathok_common loaded | device:", DEVICE)

"""
exp3b_relearn_repair.py
=======================
~2 h. Replaces exp3_relearn.py.

WHAT WAS BROKEN
---------------
The relearn objective had a recognition term but NO retain writer term, so
fine-tuning on 20 classes demolished the 95-way head. Oracle retain wACC went
88.75% -> 5.00% in 200 steps. The reported "64.62% forget accuracy" was mostly
20-way accuracy on a wrecked model, not recovery of anything, and the summary
metrics degenerated with it (recovered_frac 277%, steps_to_half 0).

FOUR REPAIRS
------------
 1. RETAIN REPLAY. Every step interleaves a retain batch with writer CE, so the
    95-way head has to stay intact while the attacker pushes on 20 classes.
    Recovery now has to happen WITHOUT collateral, which is both the realistic
    threat model and the only way the number means anything. Retain wACC is a
    hard gate: any step where it drops below RETAIN_FLOOR invalidates the row,
    and the script says so instead of quietly reporting the curve.

 2. TWO EVALUATION MODES, both reported.
      full95     : argmax over all 95 logits -- comparable to every other
                   number in the paper.
      restrict20 : argmax over the 20 forget logits only -- isolates "can the
                   model tell these writers apart" from "is the head healthy".
    Divergence between the two is itself diagnostic of head damage.

 3. SMALLER BUDGETS, DENSER EARLY EVALS. A model that never saw the forget
    writers reached 65% from 5 words each in 200 steps, so the discriminative
    region is k in {1,2,3} and steps < 50. The original grid measured only the
    saturated tail.

 4. HONEST METRICS.
      - The recovery target is the ORIGINAL BASELINE forget wACC, and the
        baseline arm is excluded from ratios (it starts at the target, so every
        ratio involving it is 0.00x or 1.00x by construction).
      - Primary statistic is EXCESS OVER ORACLE AT MATCHED STEPS: the area
        between an arm's curve and the oracle's, integrated over the budget.
        This does not degenerate when either curve saturates.
      - steps_to_target is reported only where it is finite and the arm did not
        start above target.

READ THE RESULT THIS WAY
------------------------
The oracle curve is the from-scratch rate. If it is already steep -- and it is --
then few-shot relearning has limited discriminative power here regardless of the
repairs, for the same reason the encoder probe does: generic style features make
20 Bangla writers easy to separate from almost nothing. Report that as the
finding, with the oracle curve as the evidence, rather than reporting an
unlearned arm's curve on its own.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


EXP2_SMOKE_TEST = False  # flip to False for the real run once this passes clean

class R:
    OUT = "/kaggle/working/exp2_unseen_control"
    ORACLE = "/kaggle/working/oracle/oracle_final.pt"

    K_PER_WRITER = [1, 5] if EXP2_SMOKE_TEST else [1, 2, 3, 5]
    STEPS = 6 if EXP2_SMOKE_TEST else 150
    EVAL_STEPS = [0, 2, 6] if EXP2_SMOKE_TEST else [0, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150]
    MICRO_F = 8               # forget/unseen images per step
    MICRO_R = 8               # retain replay images per step
    LR = 1e-5
    WD = 0.0
    CLIP = 1.0
    LAMBDA_F = 1.0            # forget/unseen writer CE (the attack)
    LAMBDA_R = 1.0            # retain writer CE (the replay -- this was missing)
    LAMBDA_REC = 0.5
    AMP = True

    EVAL_N = 30 if EXP2_SMOKE_TEST else 800
    RETAIN_EVAL_N = 30 if EXP2_SMOKE_TEST else 400
    RETAIN_FLOOR = 0.90       # fraction of the arm's own starting retain wACC

    # Experiment 2 -- Controlled Relearning: every arm that has a saved
    # checkpoint, so the forget-vs-unseen speed gap can be reported for the
    # full method roster, not just the 3 newest ones.
    ARMS = (["baseline", "arf@800"] if EXP2_SMOKE_TEST else
            ["oracle", "baseline", "wic_unmasked@800", "random_label@800",
             "amnesiac@800", "delete@800", "fisher@800",
             "arf@800", "wic_mgda@800", "rl_mgda@800"])

    N_UNSEEN_WRITERS = 20     # matched to len(forget writers)


os.makedirs(R.OUT, exist_ok=True)


def page_disjoint_kshot(forget_df, k, seed=S.SEED):
    rng = np.random.RandomState(seed)
    tr_idx, te_idx = [], []
    for w, g in forget_df.groupby("writer"):
        pages = g["page"].unique().copy(); rng.shuffle(pages)
        if len(pages) < 2:
            gi = g.sample(frac=1.0, random_state=seed).index.to_numpy()
            tr_idx.append(gi[:k]); te_idx.append(gi[k:]); continue
        n_tr = max(1, len(pages) // 3)
        tr_pages = set(pages[:n_tr])
        gtr = g[g["page"].isin(tr_pages)]
        gte = g[~g["page"].isin(tr_pages)]
        tr_idx.append(gtr.sample(min(k, len(gtr)), random_state=seed).index.to_numpy())
        te_idx.append(gte.index.to_numpy())
    return (forget_df.loc[np.concatenate(tr_idx)].reset_index(drop=True),
            forget_df.loc[np.concatenate(te_idx)].reset_index(drop=True))


def infinite(loader):
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def forget_acc_two_ways(model, df, vocab, fe, forget_idx, cfg=S, n=R.EVAL_N):
    """full95 = argmax over all classes. restrict20 = argmax over forget logits
    only. Divergence between them means the head is damaged, not that identity
    returned."""
    d = df.sample(min(n, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    was = model.training; model.eval()
    f95, f20 = [], []
    fi = forget_idx.to(DEVICE)
    for b in loader:
        pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
        enc = model.ved.encoder(pixel_values=pv, return_dict=True)
        wl = model.writer_head(enc.last_hidden_state[:, 0, :]).float()
        f95.append((wl.argmax(-1) == wid).float().cpu())
        sub = wl[:, fi]
        pred = fi[sub.argmax(-1)]
        f20.append((pred == wid).float().cpu())
    if was:
        model.train()
    return (100 * float(torch.cat(f95).mean()), 100 * float(torch.cat(f20).mean()))


def relearn(model, state, ctx, kshot_df, eval_df, forget_idx, tag, cfg=S):
    seed_all(S.SEED)
    model.load_state_dict(state, strict=True)
    vocab, fe = ctx["vocab"], ctx["fe"]

    opt = torch.optim.AdamW(model.parameters(), lr=R.LR, weight_decay=R.WD)
    scaler = torch.amp.GradScaler("cuda", enabled=R.AMP and DEVICE == "cuda")

    bf = min(R.MICRO_F, len(kshot_df))
    f_it = infinite(DataLoader(WDataset(kshot_df, vocab, fe, cfg), batch_size=bf,
                               shuffle=True, drop_last=len(kshot_df) > bf,
                               num_workers=cfg.NUM_WORKERS))
    r_it = infinite(DataLoader(WDataset(ctx["retain_df"], vocab, fe, cfg),
                               batch_size=R.MICRO_R, shuffle=True, drop_last=True,
                               num_workers=cfg.NUM_WORKERS))

    def snap(step):
        a95, a20 = forget_acc_two_ways(model, eval_df, vocab, fe, forget_idx, cfg)
        ra = writer_acc(model, ctx["retain_df"], vocab, fe, cfg, n=R.RETAIN_EVAL_N)
        return {"step": step, "forget_full95": a95, "forget_restrict20": a20,
                "retain_wACC": ra, "tag": tag}

    rows = [snap(0)]
    r0 = rows[0]["retain_wACC"]
    print(f"    step   0  f95 {rows[0]['forget_full95']:6.2f}  "
          f"f20 {rows[0]['forget_restrict20']:6.2f}  retain {r0:6.2f}")

    model.train()
    for step in range(1, R.STEPS + 1):
        b_f, b_r = next(f_it), next(r_it)
        nf = b_f["pixel_values"].shape[0]
        pv = torch.cat([b_f["pixel_values"], b_r["pixel_values"]]).to(DEVICE)
        lab = torch.cat([b_f["labels"], b_r["labels"]]).to(DEVICE)
        wid = torch.cat([b_f["wid"], b_r["wid"]]).to(DEVICE)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=R.AMP and DEVICE == "cuda"):
            enc = model.ved.encoder(pixel_values=pv, return_dict=True)
            wlog = model.writer_head(enc.last_hidden_state[:, 0, :])
            rec = model.ved(encoder_outputs=enc, labels=lab, return_dict=True)
        wl = wlog.float()
        l_f = F.cross_entropy(wl[:nf], wid[:nf])
        l_r = F.cross_entropy(wl[nf:], wid[nf:])      # THE REPAIR
        loss = R.LAMBDA_F * l_f + R.LAMBDA_R * l_r + R.LAMBDA_REC * rec.loss.float()

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), R.CLIP)
        scaler.step(opt); scaler.update()

        if step in R.EVAL_STEPS:
            s = snap(step); rows.append(s)
            warn = "  [!] retain collapsed" if s["retain_wACC"] < R.RETAIN_FLOOR * r0 else ""
            print(f"    step {step:3d}  f95 {s['forget_full95']:6.2f}  "
                  f"f20 {s['forget_restrict20']:6.2f}  "
                  f"retain {s['retain_wACC']:6.2f}{warn}")
            model.train()

    model.eval()
    c = pd.DataFrame(rows)
    c["retain_start"] = r0
    c["valid"] = c["retain_wACC"] >= R.RETAIN_FLOOR * r0
    return c


class UnseenHead(nn.Module):
    """A from-scratch linear head over 20 genuinely never-trained-on test-split
    writers. Fresh init per relearn run, mirroring the fact that the base
    model's architecture never had classes for these individuals -- this is
    the correct analog to 'learning a brand new writer's identity from k
    shots', as opposed to forget_df's writer_head which DID once have these
    95 classes and may retain redundant information about them."""
    def __init__(self, hidden, n_unseen):
        super().__init__()
        self.head = nn.Linear(hidden, n_unseen)

    def forward(self, cls):
        return self.head(cls)


def relearn_unseen(model, state, ctx, kshot_df, eval_df, n_unseen, tag, cfg=S):
    """Identical attack recipe to relearn() -- same optimizer, same joint
    forward pass through the shared encoder, same retain-replay CE, same
    reconstruction loss, same step/eval budget -- run against a group of
    writers the model has NEVER been trained on (test_df), through a fresh
    linear head instead of the pretrained 95-way writer_head. Comparing this
    curve to relearn()'s forget_restrict20 curve at matched (arm, k) isolates
    whether forget writers relearn faster than genuinely novel writers, i.e.
    whether the earlier relearning-attack numbers reflect residual
    writer-specific information or just 'any k-shot fine-tune on this
    architecture recovers fast'."""
    seed_all(S.SEED)
    model.load_state_dict(state, strict=True)
    vocab, fe = ctx["vocab"], ctx["fe"]
    hidden = model.writer_head.in_features
    uhead = UnseenHead(hidden, n_unseen).to(DEVICE)

    opt = torch.optim.AdamW(list(model.parameters()) + list(uhead.parameters()),
                            lr=R.LR, weight_decay=R.WD)
    scaler = torch.amp.GradScaler("cuda", enabled=R.AMP and DEVICE == "cuda")

    bf = min(R.MICRO_F, len(kshot_df))
    f_it = infinite(DataLoader(WDataset(kshot_df, vocab, fe, cfg), batch_size=bf,
                               shuffle=True, drop_last=len(kshot_df) > bf,
                               num_workers=cfg.NUM_WORKERS))
    r_it = infinite(DataLoader(WDataset(ctx["retain_df"], vocab, fe, cfg),
                               batch_size=R.MICRO_R, shuffle=True, drop_last=True,
                               num_workers=cfg.NUM_WORKERS))

    @torch.no_grad()
    def uacc(n=R.EVAL_N):
        d = eval_df.sample(min(n, len(eval_df)), random_state=cfg.SEED).reset_index(drop=True)
        loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                            shuffle=False, num_workers=cfg.NUM_WORKERS)
        was = model.training; model.eval(); uhead.eval()
        hits = []
        for b in loader:
            pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
            enc = model.ved.encoder(pixel_values=pv, return_dict=True)
            ul = uhead(enc.last_hidden_state[:, 0, :])
            hits.append((ul.argmax(-1) == wid).float().cpu())
        if was:
            model.train(); uhead.train()
        hits = torch.cat(hits).numpy() if hits else np.array([])
        return 100.0 * hits.mean() if len(hits) else float("nan")

    def snap(step):
        ua = uacc()
        ra = writer_acc(model, ctx["retain_df"], vocab, fe, cfg, n=R.RETAIN_EVAL_N)
        return {"step": step, "unseen_acc": ua, "retain_wACC": ra, "tag": tag}

    rows = [snap(0)]
    r0 = rows[0]["retain_wACC"]
    print(f"    step   0  unseen_acc {rows[0]['unseen_acc']:6.2f}  retain {r0:6.2f}")

    model.train(); uhead.train()
    for step in range(1, R.STEPS + 1):
        b_f, b_r = next(f_it), next(r_it)
        nf = b_f["pixel_values"].shape[0]
        pv = torch.cat([b_f["pixel_values"], b_r["pixel_values"]]).to(DEVICE)
        lab = torch.cat([b_f["labels"], b_r["labels"]]).to(DEVICE)
        wid_f = b_f["wid"].to(DEVICE)
        wid_r = b_r["wid"].to(DEVICE)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=R.AMP and DEVICE == "cuda"):
            enc = model.ved.encoder(pixel_values=pv, return_dict=True)
            cls = enc.last_hidden_state[:, 0, :]
            ul = uhead(cls[:nf]).float()
            wl = model.writer_head(cls[nf:]).float()
            rec = model.ved(encoder_outputs=enc, labels=lab, return_dict=True)
        l_u = F.cross_entropy(ul, wid_f)
        l_r = F.cross_entropy(wl, wid_r)
        loss = R.LAMBDA_F * l_u + R.LAMBDA_R * l_r + R.LAMBDA_REC * rec.loss.float()

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(uhead.parameters()), R.CLIP)
        scaler.step(opt); scaler.update()

        if step in R.EVAL_STEPS:
            s = snap(step); rows.append(s)
            warn = "  [!] retain collapsed" if s["retain_wACC"] < R.RETAIN_FLOOR * r0 else ""
            print(f"    step {step:3d}  unseen_acc {s['unseen_acc']:6.2f}  "
                  f"retain {s['retain_wACC']:6.2f}{warn}")
            model.train(); uhead.train()

    model.eval()
    c = pd.DataFrame(rows)
    c["retain_start"] = r0
    c["valid"] = c["retain_wACC"] >= R.RETAIN_FLOOR * r0
    return c


def auc_curve(c, col="forget_full95"):
    """Normalised area under the recovery curve over the step budget. Robust
    where steps_to_target degenerates."""
    x = c["step"].values.astype(float); y = c[col].values.astype(float)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(_trapz(y, x) / max(x.max() - x.min(), 1e-9))


def main():
    ctx = setup()
    vocab, fe = ctx["vocab"], ctx["fe"]
    reg = discover_checkpoints()
    if os.path.exists(R.ORACLE):
        reg["oracle"] = R.ORACLE
    reg.setdefault("oracle", reg.get("oracle_final"))
    print(f"\n  checkpoints on disk: {sorted(k for k in reg if reg.get(k))}")

    forget_idx = torch.tensor(sorted(ctx["forget_df"]["wid"].unique()),
                              dtype=torch.long)
    print(f"  {len(forget_idx)} forget classes; restrict20 chance = "
          f"{100/len(forget_idx):.2f} %")

    rng_u = np.random.RandomState(S.SEED)
    all_test_writers = sorted(ctx["test_df"]["writer"].unique())
    n_unseen = min(R.N_UNSEEN_WRITERS, len(all_test_writers))
    unseen_writers = sorted(rng_u.choice(all_test_writers, size=n_unseen, replace=False))
    unseen_df = ctx["test_df"][ctx["test_df"]["writer"].isin(unseen_writers)].copy()
    local_map = {w: i for i, w in enumerate(unseen_writers)}
    unseen_df["wid"] = unseen_df["writer"].map(local_map).astype(int)
    print(f"  unseen-writer control: {n_unseen} writers never seen at any "
          f"training stage, {len(unseen_df):,} words, fresh chance = "
          f"{100/n_unseen:.2f} %")

    model = build_model(vocab, ctx["n_classes"])
    load_into(model, ctx["base_sd"], "baseline", strict_report=False)
    target = writer_acc(model, ctx["forget_df"], vocab, fe, n=800)
    print(f"  recovery target (original baseline forget wACC): {target:.2f} %")
    if target < 80:
        print("  [!!] The baseline is not at ~100% on its own members. Run "
              "exp0_diagnose_baseline.py first -- this target is not credible "
              "and every ratio below inherits the problem.")

    def save_progress(curves, summ):
        """Overwrite the three output CSVs with everything accumulated so far.
        Cheap (this data is tiny) and cheap insurance: a Kaggle session that
        gets cut off mid-run (time limit, preemption) still leaves every
        arm that finished its k-loop fully saved, instead of losing all
        compute because to_csv only ran once at the very end."""
        C = pd.concat(curves, ignore_index=True, sort=False)
        Sm = pd.DataFrame(summ)
        C.to_csv(f"{R.OUT}/relearn_curves_with_unseen.csv", index=False)

        if "oracle" in set(Sm["arm"]):
            ref = Sm[(Sm.arm == "oracle") & (Sm.group == "forget")].set_index("k")
            Sm["excess_auc_vs_oracle"] = Sm.apply(
                lambda r: (r["auc"] - ref.loc[r["k"], "auc"])
                if (r["group"] == "forget" and r["k"] in ref.index) else np.nan,
                axis=1)

        piv = Sm.pivot_table(index=["arm", "k"], columns="group",
                              values=["final_acc", "auc"])
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        if "final_acc_forget" in piv.columns and "final_acc_unseen_control" in piv.columns:
            piv["final_acc_gap_forget_minus_unseen"] = (
                piv["final_acc_forget"] - piv["final_acc_unseen_control"])
        if "auc_forget" in piv.columns and "auc_unseen_control" in piv.columns:
            piv["auc_gap_forget_minus_unseen"] = piv["auc_forget"] - piv["auc_unseen_control"]
        piv = piv.reset_index()
        piv.to_csv(f"{R.OUT}/exp2_speed_gap_summary.csv", index=False)
        Sm.to_csv(f"{R.OUT}/relearn_summary_with_unseen.csv", index=False)
        return C, Sm, piv

    curves, summ = [], []
    for arm in R.ARMS:
        if arm == "baseline":
            sd = ctx["base_sd"]
        else:
            p = reg.get(arm)
            if not p or not os.path.exists(p):
                print(f"\n  [skip] {arm}: not found")
                continue
            sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        load_into(model, sd, arm, strict_report=False)
        state = copy.deepcopy(model.state_dict())

        for k in R.K_PER_WRITER:
            tr, te = page_disjoint_kshot(ctx["forget_df"], k)
            print(f"\n  --- {arm} | k={k}/writer | FORGET ({len(tr)} attack words, "
                  f"{len(te):,} held-out, page-disjoint) ---")
            c = relearn(model, state, ctx, tr, te, forget_idx, f"{arm}|k{k}|forget")
            c["arm"] = arm; c["k"] = k; c["group"] = "forget"
            curves.append(c)
            summ.append({
                "arm": arm, "k": k, "group": "forget",
                "start_acc": c["forget_full95"].iloc[0],
                "final_acc": c["forget_full95"].iloc[-1],
                "final_restrict": c["forget_restrict20"].iloc[-1],
                "auc": auc_curve(c, "forget_full95"),
                "retain_start": c["retain_start"].iloc[0],
                "retain_final": c["retain_wACC"].iloc[-1],
                "all_steps_valid": bool(c["valid"].all()),
                "recovered_frac": c["forget_full95"].iloc[-1] / max(target, 1e-9)})

            tr_u, te_u = page_disjoint_kshot(unseen_df, k)
            print(f"  --- {arm} | k={k}/writer | UNSEEN CONTROL ({len(tr_u)} attack "
                  f"words, {len(te_u):,} held-out, page-disjoint) ---")
            cu = relearn_unseen(model, state, ctx, tr_u, te_u, n_unseen, f"{arm}|k{k}|unseen")
            cu["arm"] = arm; cu["k"] = k; cu["group"] = "unseen_control"
            # unseen_acc IS the restricted (n_unseen-way) accuracy -- there is no
            # separate full/restrict distinction because the fresh head only ever
            # has n_unseen output classes to begin with.
            curves.append(cu)
            summ.append({
                "arm": arm, "k": k, "group": "unseen_control",
                "start_acc": cu["unseen_acc"].iloc[0],
                "final_acc": cu["unseen_acc"].iloc[-1],
                "final_restrict": cu["unseen_acc"].iloc[-1],
                "auc": auc_curve(cu, "unseen_acc"),
                "retain_start": cu["retain_start"].iloc[0],
                "retain_final": cu["retain_wACC"].iloc[-1],
                "all_steps_valid": bool(cu["valid"].all()),
                "recovered_frac": np.nan})

        C, Sm, piv = save_progress(curves, summ)
        print(f"  [checkpoint] progress saved through arm={arm} "
              f"({len(curves)} curve blocks so far)")

    # THE key statistic for Experiment 2: forget-group speed minus
    # unseen-control speed, at matched (arm, k). ~0 means forget writers
    # relearn no faster than writers the model never encountered at any
    # training stage -- undercutting the residual-information reading of
    # every earlier relearning-attack number. Large and positive means the
    # opposite: direct, attack-recipe-level evidence (not just probe-level)
    # that forget-writer information is specifically retained.
    C, Sm, piv = save_progress(curves, summ)

    print("\n" + "=" * 104)
    print("  EXPERIMENT 2 -- forget-writer vs.\\ genuinely-unseen-writer relearning speed")
    print("-" * 104)
    for _, r in piv.sort_values(["arm", "k"]).iterrows():
        print(f"  {r['arm']:<20} k={int(r['k'])}  "
              f"final: forget={r['final_acc_forget']:6.2f}%  unseen={r['final_acc_unseen_control']:6.2f}%  "
              f"gap={r['final_acc_gap_forget_minus_unseen']:+6.2f}pt   "
              f"AUC gap={r['auc_gap_forget_minus_unseen']:+6.2f}")
    print("-" * 104)
    print("  gap ~ 0 at a given (arm, k) means that arm's forget writers relearn")
    print("  no faster than writers the model never encountered at any training")
    print("  stage -- undercutting the residual-information reading of the")
    print("  original relearning-attack table. Large positive gap is direct,")
    print("  attack-level evidence (not just probe-level) that forget-writer")
    print("  information specifically survives that arm's unlearning procedure.")
    print(f"\n  saved -> {R.OUT}/exp2_speed_gap_summary.csv "
          f"(the headline table), {R.OUT}/relearn_summary_with_unseen.csv, "
          f"{R.OUT}/relearn_curves_with_unseen.csv")
    return C, Sm, piv


if __name__ == "__main__":
    main()