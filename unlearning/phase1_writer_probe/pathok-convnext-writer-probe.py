"""
pathok-convnext-writer-probe.py
================================
Does ConvNeXt's encoder actually fail to encode writer identity, or did its
jointly-trained writer_head just fail to read out identity that IS there?

The existing ConvNeXt result (exp5b_convnext_tuning_attempt.py, checkpoint
convnext_tuned_final.pt) reports 5.00% retain-writer accuracy and 0.00%
forget-writer accuracy from its OWN jointly-trained linear writer_head --
barely above the ~1.35% chance floor for 74-75 writers, versus TrOCR ViT's
near-100% in the same setup. That number alone conflates two different
claims: (a) the representation doesn't encode writer style at all, or
(b) it does, but this specific head -- trained under a shared/conflicting
gradient with the recognition objective -- never learned to extract it.

This script disambiguates by running the EXACT same fresh-probe methodology
already validated on the ViT models in Phase 1
(pathok-writer-probe-phase1.py): freeze the encoder, extract CLS-pooled
embeddings, fit a BRAND NEW logistic-regression classifier on a
page-disjoint split, independent of the jointly-trained head entirely.

Three groups, same reasoning as Phase 1 (a forget-only number is
uninterpretable without a floor):
    forget          -- the 20 deleted writers
    retain_control  -- 20 sampled writers still in the training population
    test_reference  -- 30 writers no model has ever seen (generic capacity)

If the fresh probe ALSO comes back near chance on all three groups, that is
real evidence for genuine architectural dissociation (the paper's existing
claim). If it comes back well above chance -- especially on retain_control,
where the jointly-trained head could still only reach 5% despite these
writers being in the training set -- that means ConvNeXt's representation
DOES encode writer style, and the dissociation claim needs to be revised to
"this specific joint-training recipe fails to extract it," which is a very
different (and considerably more interesting) finding.

No training happens here. This is a pure inference + sklearn-probe script,
cheap compared to the multi-hour unlearning-arm runs this question is a
prerequisite for.
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
from transformers.modeling_outputs import BaseModelOutput
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PUNCT = {'.', ':', '।', ',', ';', '!', '?', '-', '–', '—'}


class S:
    ROOT = "/kaggle/input/datasets/kaidul007/bnhtrd/Dataset"
    CKPT_DIR = "/kaggle/input/datasets/kaidul007/pathok-convnext-probe-ckpt"
    TROCR_MODEL = "microsoft/trocr-base-handwritten"

    SEED = 42
    BATCH = 24
    NUM_WORKERS = 2

    WORDS_PER_WRITER = 300
    N_RETAIN_CONTROL = 20
    PAGE_TEST_FRAC = 0.30

    OUT = "/kaggle/working/convnext_probe_out"
    CACHE = "/kaggle/working/df_cache.parquet"


os.makedirs(S.OUT, exist_ok=True)
np.random.seed(S.SEED)
torch.manual_seed(S.SEED)


# ══════════════════════════════════════════════════════════════════════════════
# DATA (identical convention to every other script in this project)
# ══════════════════════════════════════════════════════════════════════════════
def is_bangla_word(t):
    c = [x for x in t if not x.isspace()]
    return bool(c) and sum(1 for x in c if 'ঀ' <= x <= '৿') / len(c) > 0.8


def writer_of(p, root):
    return os.path.relpath(p, root).split(os.sep)[0]


def page_of(p, root):
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


class EmbedDataset(Dataset):
    def __init__(self, df, fe, cfg=S):
        self.df = df.reset_index(drop=True)
        self.fe, self.cfg = fe, cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img_bgr = cv2.imread(r["image_path"])
        if img_bgr is None:
            img_rgb = Image.new("RGB", (384, 384), color=255)
        else:
            img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pv = self.fe(img_rgb, return_tensors="pt").pixel_values.squeeze(0)
        return {"pixel_values": pv}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL -- exact reproduction of exp5b_convnext_tuning_attempt.py's
# ConvNeXtAsViT / ConvNeXtMultiHead, so convnext_tuned_final.pt's state dict
# (cnn_encoder.stem.* / cnn_encoder.proj.* / cnn_encoder.cls_token /
# writer_head.* / ved.*) loads with the SAME attribute names it was saved
# under -- no key remapping, no risk of silently loading the wrong (inert
# stock ViT) weights into the wrong place.
# ══════════════════════════════════════════════════════════════════════════════
def sinusoidal_2d_pos_embed(h, w, dim, device):
    quarter = dim // 4
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                          indexing="ij")
    x = x.flatten().float(); y = y.flatten().float()
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


def build_fresh_convnext(vocab_size, n_writers, cfg=S):
    ved = VisionEncoderDecoderModel.from_pretrained(cfg.TROCR_MODEL)
    ved.decoder.resize_token_embeddings(vocab_size)
    ved.config.vocab_size = vocab_size
    ved.config.decoder.vocab_size = vocab_size
    for c_ in (ved.config, ved.generation_config):
        c_.decoder_start_token_id = BOS_ID
        c_.pad_token_id = PAD_ID
        c_.eos_token_id = EOS_ID
        c_.bos_token_id = BOS_ID
    ved.generation_config.max_new_tokens = 64
    hidden_size = ved.config.encoder.hidden_size
    return ConvNeXtMultiHead(ved, n_writers, hidden_size)


def load_into(model, sd, label=""):
    res = model.load_state_dict(sd, strict=False)
    crit = [k for k in res.missing_keys if not k.startswith("ved.encoder.pooler")]
    if crit:
        raise RuntimeError(f"[{label}] {len(crit)} critical keys missing, e.g. {crit[:8]}")
    unexpected = [k for k in res.unexpected_keys]
    print(f"  [{label}] loaded OK. unexpected_keys (should be none/harmless): {unexpected[:8]}")
    return model.to(DEVICE).eval()


@torch.no_grad()
def embed(model, df, fe, cfg=S):
    loader = DataLoader(EmbedDataset(df, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval()
    cls_out, mp_out = [], []
    for b in tqdm(loader, desc="embed", leave=False, dynamic_ncols=True):
        pv = b["pixel_values"].to(DEVICE)
        enc = model.cnn_encoder(pv)
        h = enc.last_hidden_state.float()
        cls_out.append(h[:, 0, :].cpu().numpy())
        mp_out.append(h[:, 1:, :].mean(1).cpu().numpy())
    return np.concatenate(cls_out), np.concatenate(mp_out)


def page_partition(pages, writers, seed=S.SEED, frac=S.PAGE_TEST_FRAC):
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
    print(f"  Does ConvNeXt's ENCODER encode writer identity, even though its")
    print(f"  own jointly-trained head only reaches 5.00%/0.00% retain/forget?")
    print(f"  device={DEVICE}")
    print("=" * 78)

    split = json.load(open(f"{S.CKPT_DIR}/split.json"))
    vocab = json.load(open(f"{S.CKPT_DIR}/vocab_clean.json"))
    vocab_size = vocab["vocab_size"]
    test_w = set(map(str, split["test"]))
    forget_w = set(map(str, split["forget"]))
    retain_w = set(map(str, split["retain"]))
    w2i = {w: i for i, w in enumerate(sorted(retain_w | forget_w))}
    n_writers = len(w2i)
    print(f"  frozen split: {len(retain_w)} retain / {len(forget_w)} forget / "
          f"{len(test_w)} test writers -> n_writers={n_writers}")

    fe = AutoImageProcessor.from_pretrained(S.TROCR_MODEL)
    ckpt = torch.load(f"{S.CKPT_DIR}/convnext_tuned_final.pt", map_location="cpu",
                      weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    if "w2i" in ckpt:
        ck_w2i = {str(k): int(v) for k, v in ckpt["w2i"].items()}
        print(f"  checkpoint's own w2i: {len(ck_w2i)} writers | "
              f"reconstructed: {n_writers} | IDENTICAL={ck_w2i == w2i}")
        if ck_w2i != w2i:
            raise RuntimeError("w2i mismatch -- do not trust any downstream number "
                               "until this is resolved")
        n_writers = len(ck_w2i)

    model = build_fresh_convnext(vocab_size, n_writers)
    model = load_into(model, sd, "convnext_tuned")

    print(f"\n  Jointly-trained head's OWN reported result (from training log):")
    print(f"    forget writer-ACC 0.00%   retain writer-ACC 5.00%   "
          f"(chance ~= {100/n_writers:.2f}%)")

    df = load_with_writers(S)
    rng = np.random.RandomState(S.SEED)
    rw = rng.choice(sorted(df[df["writer"].isin(retain_w)]["writer"].unique()),
                    S.N_RETAIN_CONTROL, replace=False)
    tw = rng.choice(sorted(df[df["writer"].isin(test_w)]["writer"].unique()),
                    S.N_RETAIN_CONTROL, replace=False)

    sets = {
        "forget": sample_per_writer(df[df["writer"].isin(forget_w)].reset_index(drop=True),
                                    S.WORDS_PER_WRITER),
        "retain_control": sample_per_writer(
            df[df["writer"].isin(rw)].reset_index(drop=True), S.WORDS_PER_WRITER),
        "test_reference": sample_per_writer(
            df[df["writer"].isin(tw)].reset_index(drop=True), S.WORDS_PER_WRITER),
    }
    for k, v in sets.items():
        print(f"  {k:<15} {len(v):>6,} words / {v['writer'].nunique()} writers "
              f"(chance {100/v['writer'].nunique():.2f}%)")

    rows = []
    for group, d in sets.items():
        cls_X, mp_X = embed(model, d, fe)
        writers = d["writer"].values
        pages = d["page"].values
        tr, te = page_partition(pages, writers)
        for feat, X in (("cls", cls_X), ("meanpatch", mp_X)):
            acc, chance, n_eval = writer_classifier_acc(X, writers, tr, te)
            rows.append({"group": group, "feat": feat, "acc": acc, "chance": chance,
                        "n_eval": n_eval, "n_train": int(tr.sum())})
            print(f"    [{group:<15}|{feat:<9}] fresh-probe acc={acc:6.2f}%  "
                  f"chance={chance:5.2f}%  n_eval={n_eval}")
        np.savez(f"{S.OUT}/embed_convnext_{group}.npz", cls=cls_X, meanpatch=mp_X,
                writer=writers, page=pages)

    R = pd.DataFrame(rows)
    R.to_csv(f"{S.OUT}/convnext_probe_results.csv", index=False)

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("-" * 78)
    retain_cls = R[(R.group == "retain_control") & (R.feat == "cls")]["acc"].iloc[0]
    forget_cls = R[(R.group == "forget") & (R.feat == "cls")]["acc"].iloc[0]
    test_cls = R[(R.group == "test_reference") & (R.feat == "cls")]["acc"].iloc[0]
    print(f"  Jointly-trained head : retain 5.00%  forget 0.00%")
    print(f"  Fresh probe (CLS)    : retain {retain_cls:.2f}%  forget {forget_cls:.2f}%  "
          f"test_reference (generic floor) {test_cls:.2f}%")
    if retain_cls > 3 * (100 / n_writers) and retain_cls > test_cls + 5:
        print("  -> Fresh probe finds MORE than the joint head on retain writers,")
        print("     clearly above the generic-capacity floor: ConvNeXt's representation")
        print("     DOES encode writer style -- the joint-training recipe failed to")
        print("     extract it, this is not architectural dissociation.")
    elif retain_cls <= test_cls + 5:
        print("  -> Fresh probe on retain writers is indistinguishable from the")
        print("     generic-capacity floor (any unseen writers): consistent with")
        print("     genuine architectural dissociation, not a training-recipe artifact.")
    else:
        print("  -> Ambiguous: fresh probe beats the joint head somewhat but the ")
        print("     margin over the generic floor is not clean -- read the full table.")
    print("=" * 78)
    print(f"\n  saved -> {S.OUT}/convnext_probe_results.csv")


if __name__ == "__main__":
    main()
