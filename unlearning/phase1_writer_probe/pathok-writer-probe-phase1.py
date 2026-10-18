"""
pathok-writer-probe-phase1.py
==============================
Phase 1 of the writer-identity / Bangla-feature analysis:

    Bangla image -> TrOCR encoder -> CLS/pooled representation -> writer classifier

...run across three model checkpoints:
    - baseline  (original, jointly trained on all 94 retain+forget writers)
    - oracle    (retrained with the 20 forget writers absent entirely --
                 the exact-deletion reference point)
    - arf       (this work's headline unlearning method, applied to baseline;
                 forget-writer head accuracy collapses 100% -> 14.4% while
                 CER stays within 1pt of baseline -- see tab:unlearn-all)

For each model, a FRESH writer classifier (logistic regression on frozen,
page-disjoint-split CLS embeddings) is fit per group, rather than reusing
each checkpoint's own jointly-trained writer_head -- this measures what is
linearly decodable from the encoder's representation geometry itself,
independent of whatever a specific head happened to learn (the same
methodology exp2b_probe_repair.py already validated for a wider arm set).

Three writer groups (matching exp2b's design, for the same reason: a
forget-only number is uninterpretable without a floor to compare it to):
    forget          -- the 20 deleted writers (the case that matters)
    retain_control  -- 20 sampled writers still in the training population
    test_reference  -- 30 writers NO model has ever seen (generic capacity)

If forget-under-oracle ~= test_reference-under-baseline, the probe is
separating "any 20 Bangla writers" on generic style, not detecting
membership -- exactly the caveat exp2b's own writeup states.

Both CLS-token and mean-patch pooling are extracted and probed (the CLS
head is the literal ask; mean-patch is a robustness check on the same
representation). Embeddings + writer/page/label metadata are saved to
/kaggle/working/phase1_out/ so Phase 2 (same-word, different-writer
controlled comparison) can reuse them without re-running the encoder.
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
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, AutoImageProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

PHASE1_SMOKE_TEST = False  # smoke test passed clean: w2i verified IDENTICAL, all 3 arms x 3 groups produced results

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PUNCT = {'.', ':', '।', ',', ';', '!', '?', '-', '–', '—'}


class S:
    ROOT = "/kaggle/input/datasets/kaidul007/bnhtrd/Dataset"
    CKPT_DIR = "/kaggle/input/datasets/kaidul007/pathok-writer-probe-ckpts"
    TROCR_MODEL = "microsoft/trocr-base-handwritten"

    SEED = 42
    BATCH = 24
    NUM_WORKERS = 2
    TARGET_H = 64
    TARGET_W = 256

    WORDS_PER_WRITER = 300
    N_RETAIN_CONTROL = 20
    PAGE_TEST_FRAC = 0.30

    OUT = "/kaggle/working/phase1_out"
    CACHE = "/kaggle/working/df_cache.parquet"


os.makedirs(S.OUT, exist_ok=True)
np.random.seed(S.SEED)
torch.manual_seed(S.SEED)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
def is_bangla_word(t):
    c = [x for x in t if not x.isspace()]
    return bool(c) and sum(1 for x in c if 'ঀ' <= x <= '৿') / len(c) > 0.8


def writer_of(p, root):
    return os.path.relpath(p, root).split(os.sep)[0]


def page_of(p, root):
    """<ROOT>/<writer>/Words/<page>/<word>.jpg -> '<writer>/<page>'."""
    parts = os.path.relpath(p, root).split(os.sep)
    return f"{parts[0]}/{parts[2]}" if len(parts) >= 4 else parts[0]


def load_with_writers(cfg=S):
    if cfg.CACHE and os.path.exists(cfg.CACHE):
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


def sample_per_writer(df, k, seed=S.SEED):
    return (df.groupby("writer", group_keys=False)
              .apply(lambda g: g.sample(min(k, len(g)), random_state=seed))
              .reset_index(drop=True))


def preprocess(gray, cfg=S):
    h, w = gray.shape
    scale = cfg.TARGET_H / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    if new_w > cfg.TARGET_W:
        content = cv2.resize(gray, (cfg.TARGET_W, cfg.TARGET_H), interpolation=cv2.INTER_AREA)
    else:
        content = cv2.resize(gray, (new_w, cfg.TARGET_H), interpolation=cv2.INTER_LINEAR)
    c = content.astype(np.float32)
    c = (c - c.mean()) / (c.std() + 1e-6)
    content = np.clip(c * 48 + 200, 0, 255).astype(np.uint8)
    canvas = np.full((cfg.TARGET_H, cfg.TARGET_W), 255, dtype=np.uint8)
    canvas[:, :content.shape[1]] = content
    return canvas


class EmbedDataset(Dataset):
    def __init__(self, df, fe, cfg=S):
        self.df = df.reset_index(drop=True)
        self.fe, self.cfg = fe, cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        gray = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.full((self.cfg.TARGET_H, self.cfg.TARGET_W), 255, np.uint8)
        img = Image.fromarray(cv2.cvtColor(preprocess(gray, self.cfg), cv2.COLOR_GRAY2RGB))
        pv = self.fe(img, return_tensors="pt").pixel_values.squeeze(0)
        return {"pixel_values": pv}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL -- MultiHead wrapper matches the checkpoint's own state-dict layout
# exactly (ved.* + writer_head.*), even though only ved.encoder is used here.
# ══════════════════════════════════════════════════════════════════════════════
class MultiHead(nn.Module):
    def __init__(self, ved, n_writers):
        super().__init__()
        self.ved = ved
        self.writer_head = nn.Linear(ved.config.encoder.hidden_size, n_writers)

    def forward(self, pv):
        return self.ved.encoder(pixel_values=pv, return_dict=True)


def build_model(vocab_size, n_writers, cfg=S):
    ved = VisionEncoderDecoderModel.from_pretrained(cfg.TROCR_MODEL)
    ved.decoder.resize_token_embeddings(vocab_size)
    ved.config.vocab_size = vocab_size
    ved.config.decoder.vocab_size = vocab_size
    return MultiHead(ved, n_writers)


def load_into(model, sd, label=""):
    res = model.load_state_dict(sd, strict=False)
    crit = [k for k in res.missing_keys if not k.startswith("ved.encoder.pooler")]
    if crit:
        raise RuntimeError(f"[{label}] {len(crit)} critical keys missing, "
                           f"e.g. {crit[:5]}")
    return model.to(DEVICE).eval()


@torch.no_grad()
def embed(model, df, fe, cfg=S):
    loader = DataLoader(EmbedDataset(df, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval()
    cls_out, mp_out = [], []
    for b in tqdm(loader, desc="embed", leave=False, dynamic_ncols=True):
        pv = b["pixel_values"].to(DEVICE)
        enc = model(pv)
        h = enc.last_hidden_state.float()
        cls_out.append(h[:, 0, :].cpu().numpy())
        mp_out.append(h[:, 1:, :].mean(1).cpu().numpy())
    return np.concatenate(cls_out), np.concatenate(mp_out)


def page_partition(pages, writers, seed=S.SEED, frac=S.PAGE_TEST_FRAC):
    """Held-out pages per writer -- classifier never trains and tests on the
    same page (which would leak ink weight/lighting/pen pressure, not writer
    identity)."""
    rng = np.random.RandomState(seed)
    te = np.zeros(len(pages), dtype=bool)
    for w in np.unique(writers):
        m = writers == w
        wp = np.unique(pages[m]); rng.shuffle(wp)
        k = max(1, min(len(wp) - 1, int(round(frac * len(wp))))) if len(wp) > 1 else 0
        if k:
            te |= np.isin(pages, wp[:k]) & m
        else:
            idx = np.flatnonzero(m); te[idx[: max(1, len(idx) // 3)]] = True
    return ~te, te


def writer_classifier_acc(X, writers, tr, te):
    y, classes = pd.factorize(writers)
    train_classes = set(y[tr])
    keep_te = te & np.isin(y, list(train_classes))
    if not tr.any() or not keep_te.any() or len(train_classes) < 2:
        return float("nan"), 100.0 / max(len(np.unique(y)), 1), int(keep_te.sum())
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[keep_te])
    acc = 100.0 * (pred == y[keep_te]).mean()
    chance = 100.0 / len(train_classes)
    return acc, chance, int(keep_te.sum())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 78)
    print(f"  PHASE 1: writer-identity probe across baseline / oracle / arf")
    print(f"  device={DEVICE}  SMOKE_TEST={PHASE1_SMOKE_TEST}")
    print("=" * 78)

    fe = AutoImageProcessor.from_pretrained(S.TROCR_MODEL)
    vocab = json.load(open(f"{S.CKPT_DIR}/baseline/vocab_clean.json"))
    vocab_size = vocab["vocab_size"]
    split = json.load(open(f"{S.CKPT_DIR}/baseline/split.json"))
    test_w = set(map(str, split["test"]))
    forget_w = set(map(str, split["forget"]))
    retain_w = set(map(str, split["retain"]))

    base_ck = torch.load(f"{S.CKPT_DIR}/baseline/baseline_clean_final.pt",
                         map_location="cpu", weights_only=False)
    w2i_ck = {str(k): int(v) for k, v in base_ck.get("w2i", {}).items()}
    w2i_recon = {w: i for i, w in enumerate(sorted(retain_w | forget_w))}
    print(f"  w2i from checkpoint: {len(w2i_ck)} writers | "
          f"reconstructed from split.json: {len(w2i_recon)} writers | "
          f"IDENTICAL={w2i_ck == w2i_recon}")
    if w2i_ck and w2i_ck != w2i_recon:
        raise RuntimeError("w2i mismatch between checkpoint and split.json reconstruction "
                           "-- do not trust any downstream number until this is resolved.")
    n_writers = len(w2i_ck) if w2i_ck else len(w2i_recon)

    df = load_with_writers(S)

    rng = np.random.RandomState(S.SEED)
    rw = rng.choice(sorted(df[df["writer"].isin(retain_w)]["writer"].unique()),
                    S.N_RETAIN_CONTROL, replace=False)
    tw = rng.choice(sorted(df[df["writer"].isin(test_w)]["writer"].unique()),
                    S.N_RETAIN_CONTROL, replace=False)

    wpp = 20 if PHASE1_SMOKE_TEST else S.WORDS_PER_WRITER
    sets = {
        "forget": sample_per_writer(df[df["writer"].isin(forget_w)].reset_index(drop=True), wpp),
        "retain_control": sample_per_writer(
            df[df["writer"].isin(rw)].reset_index(drop=True), wpp),
        "test_reference": sample_per_writer(
            df[df["writer"].isin(tw)].reset_index(drop=True), wpp),
    }
    for k, v in sets.items():
        print(f"  {k:<15} {len(v):>6,} words / {v['writer'].nunique()} writers "
              f"(chance {100/v['writer'].nunique():.2f}%)")

    arms = {
        "baseline": f"{S.CKPT_DIR}/baseline/baseline_clean_final.pt",
        "oracle":   f"{S.CKPT_DIR}/oracle/oracle_final.pt",
        "arf":      f"{S.CKPT_DIR}/arf/unlearned_arf_step800.pt",
    }

    model = build_model(vocab_size, n_writers)
    rows = []
    for arm, path in arms.items():
        sd = (base_ck["model"] if arm == "baseline" else
             torch.load(path, map_location="cpu", weights_only=False)["model"])
        load_into(model, sd, arm)
        print(f"\n  --- {arm} ---")

        for group, d in sets.items():
            cls_X, mp_X = embed(model, d, fe)
            writers = d["writer"].values
            pages = d["page"].values
            labels = d["label"].values
            tr, te = page_partition(pages, writers)

            for feat, X in (("cls", cls_X), ("meanpatch", mp_X)):
                acc, chance, n_eval = writer_classifier_acc(X, writers, tr, te)
                rows.append({"arm": arm, "group": group, "feat": feat,
                            "acc": acc, "chance": chance, "n_eval": n_eval,
                            "n_train": int(tr.sum()), "n_writers": d["writer"].nunique()})
                print(f"    [{group:<15}|{feat:<9}] acc={acc:6.2f}%  "
                      f"chance={chance:5.2f}%  n_eval={n_eval}")

            np.savez(f"{S.OUT}/embed_{arm}_{group}.npz",
                    cls=cls_X, meanpatch=mp_X, writer=writers, page=pages,
                    label=labels, train_mask=tr, test_mask=te)

    R = pd.DataFrame(rows)
    R.to_csv(f"{S.OUT}/phase1_writer_classifier_results.csv", index=False)

    print("\n" + "=" * 78)
    print("  PHASE 1 SUMMARY -- writer-classifier accuracy, CLS pooling")
    print("-" * 78)
    piv = R[R.feat == "cls"].pivot_table(index="arm", columns="group", values="acc")
    print(piv.round(2).to_string())
    print("\n  If forget-under-oracle ~= test_reference-under-baseline, the probe is")
    print("  separating any handful of Bangla writers on generic style, not")
    print("  detecting who was actually deleted -- report that explicitly.")
    print("=" * 78)
    print(f"\n  saved -> {S.OUT}/phase1_writer_classifier_results.csv")
    print(f"  saved -> {S.OUT}/embed_<arm>_<group>.npz (for Phase 2 reuse)")


if __name__ == "__main__":
    main()
