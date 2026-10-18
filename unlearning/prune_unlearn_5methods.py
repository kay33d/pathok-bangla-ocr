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
prune_unlearn.py  --  Stage I (neural pruning) + Stage II (WIC, Random Labeling)
for the PATHOK multi-head TrOCR-grapheme model, reproducing Kang et al.
"Preserving Privacy Without Compromising Accuracy" (Pattern Recognition 2025/26)
on the BnHTRD Bangla handwriting dataset.

Builds on pathok_common (pasted above this cell verbatim, already proven working
in mohammedkai/pathok-common-revised): S config, MultiHead model, build_model,
load_into, setup(), writer_acc, rec_quality, extract_features (12-dim,
RECOGNITION-HEAD based -- this is what the paper's MIA uses), boot_auc,
discover_checkpoints.

Canonical split: the frozen split.json used by setup()/reload_split() -- 20
forget writers, same split the few-shot-identification and relearning-attack
audits (exp2b/exp3b in pathok-common-revised) already depend on. Do NOT
introduce a second ad-hoc split; that was the bug in the old unlearning-privacy
kernel that made its results incomparable with the rest of this project's
tooling.

STATUS: Stage I (pruning) plus all five Stage II methods from the paper --
WIC, Random Labeling, Fisher Forgetting, Amnesiac Unlearning, DELETE.
WIC/Random Labeling were verified end-to-end (smoke test, then a full run)
before Fisher/Amnesiac/DELETE were added, per the agreed sequencing.

Checkpoint naming matches discover_checkpoints()'s expected convention so the
existing audit kernels pick these up with zero changes:
    unlearned_<tag>_step<N>.pt   ->   reg["<tag>@<N>"]
"""

import copy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set to True for a ~few-minute end-to-end sanity pass (tiny batches/iters/
# eval-n) before committing a full GPU run. Flip to False for the real run.
SMOKE_TEST = False

# ══════════════════════════════════════════════════════════════════════════════
# STAGE I  --  neural pruning by forget/retain activation ratio
# ══════════════════════════════════════════════════════════════════════════════
class PConfig:
    OUT = "/kaggle/working/prune_unlearn"

    # Half the paper's rates: theirs is a from-scratch T5 encoder, this is a
    # pretrained ViT-base and far less redundant (same reasoning as the earlier
    # unlearning-privacy draft, which got this part right).
    PRUNE_RATES = {"shallow": 0.10, "middle": 0.20, "deep": 0.05}
    SHALLOW_LAYERS = (0, 1, 2, 3)
    MIDDLE_LAYERS = (4, 5, 6, 7)
    DEEP_LAYERS = (8, 9, 10, 11)
    PRUNE_BATCHES = 40
    PRUNE_EPS = 1e-8
    FREEZE_PRUNED = True

    # ── Stage II ────────────────────────────────────────────────────────────
    UNLEARN_BATCH_SIZE = 8            # THE BUG: this was never defined before
    UNLEARN_LR = 1e-5
    LAMBDA_FORGET = 1.0
    LAMBDA_RETAIN = 1.0
    LAMBDA_REC = 0.5
    AMP = True
    EVAL_EVERY = 100
    EVAL_N = 500
    CER_EVAL_N = 300


EVAL_LARGE_N = 30 if SMOKE_TEST else 1500
EVAL_MED_N = 20 if SMOKE_TEST else 800

if SMOKE_TEST:
    PConfig.PRUNE_BATCHES = 2
    PConfig.EVAL_EVERY = 5
    PConfig.EVAL_N = 20
    PConfig.CER_EVAL_N = 20

os.makedirs(PConfig.OUT, exist_ok=True)


def _target_linears(model):
    """All nn.Linear inside the ViT encoder blocks, grouped by depth.
    Returns {name: (module, group)}."""
    out = {}
    for name, mod in model.ved.encoder.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1]); break
        if layer_idx is None:
            continue                      # skip pooler / patch-embed projection
        if layer_idx in PConfig.SHALLOW_LAYERS: g = "shallow"
        elif layer_idx in PConfig.MIDDLE_LAYERS: g = "middle"
        else: g = "deep"
        out[name] = (mod, g)
    return out


@torch.no_grad()
def collect_activations(model, loader, targets, n_batches):
    """Mean |activation| per output unit, averaged over batch and tokens."""
    acc = {n: torch.zeros(m.out_features, device=DEVICE) for n, (m, _) in targets.items()}
    seen = 0
    handles = []

    def mk(name):
        def hook(mod, inp, out):
            acc[name] += out.detach().abs().float().mean(dim=tuple(range(out.dim() - 1)))
        return hook

    for name, (mod, _) in targets.items():
        handles.append(mod.register_forward_hook(mk(name)))

    model.eval()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        model.ved.encoder(pixel_values=batch["pixel_values"].to(DEVICE))
        seen += 1

    for h in handles:
        h.remove()
    return {n: (v / max(seen, 1)) for n, v in acc.items()}


def prune_by_ratio(model, forget_df, retain_df, vocab, fe,
                    rates=None, n_batches=PConfig.PRUNE_BATCHES, verbose=True):
    """importance(u) = (act_forget(u) + eps) / (act_retain(u) + eps).
    Top-K% by importance within each linear module (grouped by depth) are
    zeroed. Returns a mask dict {name: BoolTensor[out_features]}."""
    rates = rates or PConfig.PRUNE_RATES
    targets = _target_linears(model)

    f_loader = DataLoader(WDataset(forget_df, vocab, fe), batch_size=S.BATCH,
                          shuffle=True, num_workers=S.NUM_WORKERS)
    r_loader = DataLoader(WDataset(retain_df, vocab, fe), batch_size=S.BATCH,
                          shuffle=True, num_workers=S.NUM_WORKERS)

    print("[stage I] estimating forget-set activations")
    a_f = collect_activations(model, f_loader, targets, n_batches)
    print("[stage I] estimating retain-set activations")
    a_r = collect_activations(model, r_loader, targets, n_batches)

    masks, tot, pruned_tot = {}, 0, 0
    for name, (mod, group) in targets.items():
        rate = rates.get(group, 0.0)
        imp = (a_f[name] + PConfig.PRUNE_EPS) / (a_r[name] + PConfig.PRUNE_EPS)
        k = int(round(rate * mod.out_features))
        mask = torch.zeros(mod.out_features, dtype=torch.bool, device=imp.device)
        if k > 0:
            mask[torch.topk(imp, k).indices] = True
            with torch.no_grad():
                mod.weight[mask] = 0.0
                if mod.bias is not None:
                    mod.bias[mask] = 0.0
        masks[name] = mask
        tot += mod.out_features; pruned_tot += int(mask.sum())

    if verbose:
        print(f"[stage I] pruned {pruned_tot:,}/{tot:,} encoder-linear output "
              f"units ({100*pruned_tot/tot:.2f}%)  rates={rates}")
    return masks


def apply_masks(model, masks):
    """Re-zero pruned units. Call after every optimizer.step() when
    FREEZE_PRUNED, or the forget-writer info Stage I removed can drift back."""
    targets = _target_linears(model)
    with torch.no_grad():
        for name, mask in masks.items():
            if not mask.any():
                continue
            mod = targets[name][0]
            mod.weight[mask] = 0.0
            if mod.bias is not None:
                mod.bias[mask] = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# STAGE II  --  WIC and Random Labeling
# ══════════════════════════════════════════════════════════════════════════════
def _infinite(loader):
    while True:
        for b in loader:
            yield b


def unlearn(model, ctx, masks, method, iters, checkpoint_steps, tag,
            cfg=PConfig):
    """method in {"wic", "random_label"}.
    checkpoint_steps: iterable of iteration numbers at which to save
    /kaggle/working/prune_unlearn/unlearned_<tag>_step<N>.pt (matches
    discover_checkpoints()'s "<tag>@<N>" convention).
    """
    assert method in ("wic", "random_label")
    vocab, fe = ctx["vocab"], ctx["fe"]
    forget_df, retain_df = ctx["forget_df"], ctx["retain_df"]
    n_classes = ctx["n_classes"]

    # paper eq. 2: perm maps W_forget -> W \ W_forget, i.e. random labeling
    # must land on a RETAIN writer, never back on another forget writer.
    forget_idx_set = set(int(i) for i in forget_df["wid"].unique())
    retain_idx_pool = np.array([i for i in range(n_classes) if i not in forget_idx_set])

    f_iter = _infinite(DataLoader(WDataset(forget_df, vocab, fe), batch_size=cfg.UNLEARN_BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=S.NUM_WORKERS))
    r_iter = _infinite(DataLoader(WDataset(retain_df, vocab, fe), batch_size=cfg.UNLEARN_BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=S.NUM_WORKERS))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.UNLEARN_LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP and DEVICE == "cuda")
    checkpoint_steps = set(checkpoint_steps)
    saved = []

    model.train()
    for it in range(1, iters + 1):
        bf, br = next(f_iter), next(r_iter)
        opt.zero_grad(set_to_none=True)

        # ── forget batch ────────────────────────────────────────────────────
        pv_f = bf["pixel_values"].to(DEVICE); lb_f = bf["labels"].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=cfg.AMP and DEVICE == "cuda"):
            rec_f, wlog_f = model(pv_f, lb_f)
            if method == "wic":
                logp_f = F.log_softmax(wlog_f.float(), dim=-1)
                uniform = torch.full_like(logp_f, 1.0 / n_classes)
                l_forget = F.kl_div(logp_f, uniform, reduction="batchmean")
            else:  # random_label
                rnd_np = np.random.choice(retain_idx_pool, size=wlog_f.size(0))
                rnd = torch.tensor(rnd_np, dtype=torch.long, device=DEVICE)
                l_forget = F.cross_entropy(wlog_f.float(), rnd)
            loss1 = cfg.LAMBDA_FORGET * l_forget + cfg.LAMBDA_REC * rec_f.loss.float()
        scaler.scale(loss1).backward()
        del pv_f, lb_f, rec_f, wlog_f

        # ── retain batch ─────────────────────────────────────────────────────
        pv_r = br["pixel_values"].to(DEVICE); lb_r = br["labels"].to(DEVICE)
        wr_r = br["wid"].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=cfg.AMP and DEVICE == "cuda"):
            rec_r, wlog_r = model(pv_r, lb_r)
            l_retain = F.cross_entropy(wlog_r.float(), wr_r)
            loss2 = cfg.LAMBDA_RETAIN * l_retain + cfg.LAMBDA_REC * rec_r.loss.float()
        scaler.scale(loss2).backward()
        del pv_r, lb_r, rec_r

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        if masks is not None and PConfig.FREEZE_PRUNED:
            apply_masks(model, masks)

        if it % cfg.EVAL_EVERY == 0 or it == iters or it in checkpoint_steps:
            wacc_f = writer_acc(model, forget_df, vocab, fe, n=cfg.EVAL_N)
            wacc_r = writer_acc(model, retain_df, vocab, fe, n=cfg.EVAL_N)
            rq_f = rec_quality(model, forget_df, vocab, fe, n=cfg.CER_EVAL_N)
            print(f"  [{tag}] it {it:>4}  wacc_forget {wacc_f:6.2f}%  "
                  f"wacc_retain {wacc_r:6.2f}%  CER_forget {rq_f['CER']:6.2f}%")
            model.train()

        if it in checkpoint_steps:
            path = f"{PConfig.OUT}/unlearned_{tag}_step{it}.pt"
            torch.save({"model": model.state_dict(), "iter": it, "method": method,
                       "tag": tag}, path)
            print(f"    saved -> {path}")
            saved.append(path)

    return model, saved


# ══════════════════════════════════════════════════════════════════════════════
# STAGE II (cont'd)  --  Fisher Forgetting, Amnesiac Unlearning, DELETE
#
# Unlike WIC/Random Labeling (paper eq. 2, 9: forget loss + retain CE + beta*rec,
# all jointly optimised every step), these three are SHOCK-then-REPAIR methods
# in the paper: a short forget-only phase (no recognition term, no retain term)
# that visibly wrecks CER/WER (Tables 12-14), followed by a separate retain-only
# fine-tune phase that recovers it. Replicating that two-phase structure, not
# smoothing it into one joint objective, because the paper's own numbers only
# make sense as two distinct phases.
# ══════════════════════════════════════════════════════════════════════════════
def _writer_ce_on_batch(model, batch):
    pv = batch["pixel_values"].to(DEVICE)
    wid = batch["wid"].to(DEVICE)
    _, wlog = model(pv, labels=None)
    return F.cross_entropy(wlog.float(), wid), wlog.size(0)


def fisher_forget_shock(model, ctx, shock_iters, alpha=0.01, mu=1e-8, cfg=PConfig):
    """Paper eq. 4 (their simplified, sequence-task version of Fisher Forgetting):
    theta_scrub = theta - alpha * g_f / (F + mu), where F is the DIAGONAL Fisher
    Information Matrix and g_f the gradient, both estimated on the forget set via
    the writer classification loss. A single step, using shock_iters forget
    batches to estimate g_f/F (paper: forget-set-size/batch ~= 32 iterations)."""
    vocab, fe = ctx["vocab"], ctx["fe"]
    loader = DataLoader(WDataset(ctx["forget_df"], vocab, fe), batch_size=cfg.UNLEARN_BATCH_SIZE,
                        shuffle=True, drop_last=True, num_workers=S.NUM_WORKERS)
    params = [p for p in model.parameters() if p.requires_grad]
    g_sum = [torch.zeros_like(p) for p in params]
    g2_sum = [torch.zeros_like(p) for p in params]

    model.train()
    it = 0
    for batch in _infinite(loader):
        if it >= shock_iters:
            break
        model.zero_grad(set_to_none=True)
        loss, _ = _writer_ce_on_batch(model, batch)
        loss.backward()
        with torch.no_grad():
            for gs, g2s, p in zip(g_sum, g2_sum, params):
                if p.grad is not None:
                    gs += p.grad
                    g2s += p.grad * p.grad
        it += 1
    print(f"  [fisher] estimated g_f/F over {it} forget batches")

    with torch.no_grad():
        for gs, g2s, p in zip(g_sum, g2_sum, params):
            g_f = gs / max(it, 1)
            F_diag = g2s / max(it, 1)
            p -= alpha * g_f / (F_diag + mu)
    model.zero_grad(set_to_none=True)
    return model


def amnesiac_shock(model, ctx, shock_iters, alpha=1e-6, masks=None, cfg=PConfig):
    """Paper eq. 5: theta(t+1) = theta(t) + alpha * grad(L(forget batch)), i.e.
    gradient ASCENT on the writer-CE loss for the forget set -- reversing the
    training dynamics rather than descending them. alpha is tiny (paper: ~1e-6)
    because this is applied every step, unlike Fisher's single big step."""
    vocab, fe = ctx["vocab"], ctx["fe"]
    loader = DataLoader(WDataset(ctx["forget_df"], vocab, fe), batch_size=cfg.UNLEARN_BATCH_SIZE,
                        shuffle=True, drop_last=True, num_workers=S.NUM_WORKERS)
    params = [p for p in model.parameters() if p.requires_grad]

    model.train()
    it = 0
    for batch in _infinite(loader):
        if it >= shock_iters:
            break
        model.zero_grad(set_to_none=True)
        loss, _ = _writer_ce_on_batch(model, batch)
        loss.backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p += alpha * p.grad          # '+=' is the reversal, not '-='
        if masks is not None and PConfig.FREEZE_PRUNED:
            apply_masks(model, masks)
        it += 1
    model.zero_grad(set_to_none=True)
    print(f"  [amnesiac] applied {it} reversed-gradient steps on the forget set")
    return model


def delete_phase(model, teacher_state, ctx, masks, delete_iters, forget_idx,
                 lambda_kd=1.0, cfg=PConfig):
    """Paper eq. 6: frozen teacher + trainable student. Forget-class logits are
    masked to -inf in BOTH teacher and student before softmax, so the
    distributions being distilled live entirely on the retain sub-space -- no
    gradient ever flows through the forgotten identities, and any image's
    forget-class probability is forced to ~0 by construction, not by pattern-
    matching the forget writers' handwriting."""
    vocab, fe = ctx["vocab"], ctx["fe"]
    n_classes = ctx["n_classes"]

    teacher = build_model(vocab, n_classes).to(DEVICE)
    teacher.load_state_dict(teacher_state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    mask_vec = torch.zeros(n_classes, device=DEVICE)
    mask_vec[list(forget_idx)] = float("-inf")

    f_iter = _infinite(DataLoader(WDataset(ctx["forget_df"], vocab, fe), batch_size=cfg.UNLEARN_BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=S.NUM_WORKERS))
    r_iter = _infinite(DataLoader(WDataset(ctx["retain_df"], vocab, fe), batch_size=cfg.UNLEARN_BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=S.NUM_WORKERS))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.UNLEARN_LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP and DEVICE == "cuda")

    model.train()
    for it in range(1, delete_iters + 1):
        batch = next(f_iter) if it % 2 == 0 else next(r_iter)
        pv = batch["pixel_values"].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=cfg.AMP and DEVICE == "cuda"):
            with torch.no_grad():
                _, wlog_t = teacher(pv, labels=None)
                p_t_r = F.softmax(wlog_t.float() + mask_vec, dim=-1)
            _, wlog_s = model(pv, labels=None)
            logp_s_r = F.log_softmax(wlog_s.float() + mask_vec, dim=-1)
            loss = lambda_kd * F.kl_div(logp_s_r, p_t_r, reduction="batchmean")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        if masks is not None and PConfig.FREEZE_PRUNED:
            apply_masks(model, masks)

        if it % cfg.EVAL_EVERY == 0 or it == delete_iters:
            wacc_f = writer_acc(model, ctx["forget_df"], vocab, fe, n=cfg.EVAL_N)
            print(f"  [delete] it {it:>4}  loss {loss.item():.4f}  "
                  f"wacc_forget {wacc_f:6.2f}%")
            model.train()

    del teacher
    torch.cuda.empty_cache()
    return model


def fine_tune_retain(model, ctx, masks, iters, tag, cfg=PConfig):
    """Paper section 3.6.5, as actually needed for CER/WER to recover in
    Tables 10/12/13/14 (the minimal L_FT = CE(retain) alone cannot explain the
    reported recovery, so recognition loss must also be present -- same
    beta=1 coupling used everywhere else in this paper): retain-only writer CE
    + recognition loss, no forget batches at all."""
    vocab, fe = ctx["vocab"], ctx["fe"]
    r_iter = _infinite(DataLoader(WDataset(ctx["retain_df"], vocab, fe), batch_size=cfg.UNLEARN_BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=S.NUM_WORKERS))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.UNLEARN_LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP and DEVICE == "cuda")

    model.train()
    for it in range(1, iters + 1):
        br = next(r_iter)
        pv_r = br["pixel_values"].to(DEVICE); lb_r = br["labels"].to(DEVICE)
        wr_r = br["wid"].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=cfg.AMP and DEVICE == "cuda"):
            rec_r, wlog_r = model(pv_r, lb_r)
            loss = F.cross_entropy(wlog_r.float(), wr_r) + cfg.LAMBDA_REC * rec_r.loss.float()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        if masks is not None and PConfig.FREEZE_PRUNED:
            apply_masks(model, masks)

        if it % cfg.EVAL_EVERY == 0 or it == iters:
            rq = rec_quality(model, ctx["retain_df"], ctx["vocab"], fe, n=cfg.CER_EVAL_N)
            print(f"  [{tag}+FT] it {it:>4}  loss {loss.item():.4f}  "
                  f"CER_retain {rq['CER']:6.2f}%")
            model.train()
    return model


def unlearn_shock_then_ft(model, ctx, masks, method, total_iters, shock_iters,
                          tag, cfg=PConfig):
    """Dispatcher for fisher / amnesiac: shock phase (forget-only, no
    recognition term) then fine_tune_retain for the rest of the budget.
    Saves one checkpoint at the end, matching how the paper reports these
    (Baseline -> +method -> +FT, no intermediate points)."""
    assert method in ("fisher", "amnesiac")
    if method == "fisher":
        model = fisher_forget_shock(model, ctx, shock_iters)
    else:
        model = amnesiac_shock(model, ctx, shock_iters, masks=masks)

    ft_iters = max(0, total_iters - shock_iters)
    if ft_iters > 0:
        model = fine_tune_retain(model, ctx, masks, ft_iters, tag)

    path = f"{PConfig.OUT}/unlearned_{tag}_step{total_iters}.pt"
    torch.save({"model": model.state_dict(), "iter": total_iters, "method": method,
               "tag": tag}, path)
    print(f"    saved -> {path}")
    return model, [path]


def unlearn_delete(model, ctx, masks, total_iters, delete_iters, tag, cfg=PConfig):
    """Dispatcher for DELETE: KD phase against a frozen teacher (a snapshot of
    the model BEFORE this phase, matching paper eq. 6) then fine_tune_retain."""
    teacher_state = copy.deepcopy(model.state_dict())
    forget_idx = sorted(int(i) for i in ctx["forget_df"]["wid"].unique())

    model = delete_phase(model, teacher_state, ctx, masks, delete_iters, forget_idx)

    ft_iters = max(0, total_iters - delete_iters)
    if ft_iters > 0:
        model = fine_tune_retain(model, ctx, masks, ft_iters, tag)

    path = f"{PConfig.OUT}/unlearned_{tag}_step{total_iters}.pt"
    torch.save({"model": model.state_dict(), "iter": total_iters, "method": "delete",
               "tag": tag}, path)
    print(f"    saved -> {path}")
    return model, [path]


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION  (writer acc, CER/WER, MIA -- matches the paper's Tables 10/11/16)
# ══════════════════════════════════════════════════════════════════════════════
def run_mia(model, ctx, n=None):
    n = n or EVAL_MED_N
    """Recognition-head MIA, exactly as the paper defines it (Section 3.7):
    train a 3-linear-layer-equivalent classifier (here: logistic regression on
    the 12-dim teacher-forced confidence signature, which subsumes it) on
    retain=member / test=non-member, 80/20 split, then classify the FULL
    forget set. Reports % forget classified as 'seen' (member) -- ideally near
    50% (chance) after unlearning, near 100% at baseline."""
    vocab, fe = ctx["vocab"], ctx["fe"]
    retain_n = ctx["retain_df"].sample(min(n, len(ctx["retain_df"])), random_state=S.SEED)
    test_n = ctx["test_df"].sample(min(n, len(ctx["test_df"])), random_state=S.SEED)

    Xr = extract_features(model, retain_n, vocab, fe)
    Xt = extract_features(model, test_n, vocab, fe)
    X = np.concatenate([Xr, Xt]); y = np.concatenate([np.ones(len(Xr)), np.zeros(len(Xt))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=S.SEED,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)

    Xf = extract_features(model, ctx["forget_df"], vocab, fe)
    p_forget = clf.predict_proba(sc.transform(Xf))[:, 1]
    p_test_eval = clf.predict_proba(sc.transform(Xte))[:, 1]
    auc_holdout = roc_auc_score(yte, p_test_eval)
    return {
        "mia_forget_pct_seen": float((p_forget > 0.5).mean() * 100),
        "mia_holdout_auc": float(auc_holdout),
    }


def full_report(model, ctx, tag=""):
    vocab, fe = ctx["vocab"], ctx["fe"]
    r = {"tag": tag}
    r["wacc_forget"] = writer_acc(model, ctx["forget_df"], vocab, fe, n=EVAL_LARGE_N)
    r["wacc_retain"] = writer_acc(model, ctx["retain_df"], vocab, fe, n=EVAL_LARGE_N)
    for key in ("forget", "retain", "test"):
        rq = rec_quality(model, ctx[key + "_df"], vocab, fe, n=EVAL_MED_N)
        r[f"cer_{key}"] = rq["CER"]; r[f"wer_{key}"] = rq["WER"]
    mia = run_mia(model, ctx)
    r.update(mia)

    print(f"\n{'='*78}\n  {tag}\n{'-'*78}")
    print(f"  writer acc  forget/retain : {r['wacc_forget']:6.2f}% / {r['wacc_retain']:6.2f}%")
    print(f"  CER  forget/retain/test   : {r['cer_forget']:6.2f}% / {r['cer_retain']:6.2f}% / {r['cer_test']:6.2f}%")
    print(f"  WER  forget/retain/test   : {r['wer_forget']:6.2f}% / {r['wer_retain']:6.2f}% / {r['wer_test']:6.2f}%")
    print(f"  MIA  forget seen-as-member: {r['mia_forget_pct_seen']:6.2f}%   (chance=50%)")
    print(f"  MIA  holdout AUC          : {r['mia_holdout_auc']:.3f}")
    print(f"{'='*78}\n")
    return r


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════
def _fresh_pruned_model(vocab, n_classes, pruned_state):
    m = build_model(vocab, n_classes).to(DEVICE)
    m.load_state_dict(pruned_state, strict=True)
    return m


def main():
    ctx = setup()
    vocab, fe, n_classes = ctx["vocab"], ctx["fe"], ctx["n_classes"]
    results = []
    all_checkpoints = []

    # ── baseline ─────────────────────────────────────────────────────────────
    model = build_model(vocab, n_classes)
    load_into(model, ctx["base_sd"], "baseline", strict_report=True)
    results.append(full_report(model, ctx, "BASELINE (multi-head, nothing unlearned)"))

    # ── Stage I: prune ───────────────────────────────────────────────────────
    masks = prune_by_ratio(model, ctx["forget_df"], ctx["retain_df"], vocab, fe)
    results.append(full_report(model, ctx, "AFTER STAGE I (pruning only)"))
    torch.save({"model": model.state_dict(), "masks": {k: v.cpu() for k, v in masks.items()}},
               f"{PConfig.OUT}/pruned_only.pt")
    pruned_state = copy.deepcopy(model.state_dict())
    del model; torch.cuda.empty_cache()

    if SMOKE_TEST:
        wic_iters, wic_ckpts = 10, (5, 10)
        rl_iters, rl_ckpts = 10, (10,)
        fisher_total, fisher_shock = 10, 4
        amnesiac_total, amnesiac_shock = 10, 4
        delete_total, delete_iters = 10, 5
    else:
        wic_iters, wic_ckpts = 800, (300, 800)
        rl_iters, rl_ckpts = 800, (800,)
        fisher_total, fisher_shock = 800, 40
        amnesiac_total, amnesiac_shock = 800, 40
        delete_total, delete_iters = 800, 400

    # ── Stage II: WIC ────────────────────────────────────────────────────────
    m = _fresh_pruned_model(vocab, n_classes, pruned_state)
    m, saved = unlearn(m, ctx, masks, method="wic", iters=wic_iters,
                       checkpoint_steps=wic_ckpts, tag="wic_unmasked")
    results.append(full_report(m, ctx, f"Prune + WIC ({wic_iters} it)"))
    all_checkpoints += saved
    del m; torch.cuda.empty_cache()

    # ── Stage II: Random Labeling (ablation) ────────────────────────────────
    m = _fresh_pruned_model(vocab, n_classes, pruned_state)
    m, saved = unlearn(m, ctx, masks, method="random_label", iters=rl_iters,
                       checkpoint_steps=rl_ckpts, tag="random_label")
    results.append(full_report(m, ctx, f"Prune + Random Labeling ({rl_iters} it)"))
    all_checkpoints += saved
    del m; torch.cuda.empty_cache()

    # ── Stage II: Fisher Forgetting ─────────────────────────────────────────
    m = _fresh_pruned_model(vocab, n_classes, pruned_state)
    m, saved = unlearn_shock_then_ft(m, ctx, masks, method="fisher",
                                     total_iters=fisher_total, shock_iters=fisher_shock,
                                     tag="fisher")
    results.append(full_report(m, ctx, f"Prune + Fisher Forgetting ({fisher_total} it)"))
    all_checkpoints += saved
    del m; torch.cuda.empty_cache()

    # ── Stage II: Amnesiac Unlearning ───────────────────────────────────────
    m = _fresh_pruned_model(vocab, n_classes, pruned_state)
    m, saved = unlearn_shock_then_ft(m, ctx, masks, method="amnesiac",
                                     total_iters=amnesiac_total, shock_iters=amnesiac_shock,
                                     tag="amnesiac")
    results.append(full_report(m, ctx, f"Prune + Amnesiac Unlearning ({amnesiac_total} it)"))
    all_checkpoints += saved
    del m; torch.cuda.empty_cache()

    # ── Stage II: DELETE ─────────────────────────────────────────────────────
    m = _fresh_pruned_model(vocab, n_classes, pruned_state)
    m, saved = unlearn_delete(m, ctx, masks, total_iters=delete_total,
                              delete_iters=delete_iters, tag="delete")
    results.append(full_report(m, ctx, f"Prune + DELETE ({delete_total} it)"))
    all_checkpoints += saved
    del m; torch.cuda.empty_cache()

    out = pd.DataFrame(results)
    out.to_csv(f"{PConfig.OUT}/comparison_table.csv", index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nCheckpoints saved: {all_checkpoints}")
    print(f"Comparison table -> {PConfig.OUT}/comparison_table.csv")
    return results


if __name__ == "__main__":
    main()
