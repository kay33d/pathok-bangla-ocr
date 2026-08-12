"""
pathok_stage2_gate.py — remaining Stage II arms + membership-inference gate
==========================================================================
ONE CELL. Paste whole, edit the two paths in S, run. ~100 min on a T4.

WHAT THIS RUNS
--------------
  A. wic_unmasked, 800 steps, CHECKPOINTED EVERY 100. The earlier run saved
     only the final model, so the step-300 operating point (forget wACC 4.62%,
     just above the 1.33% chance rate) was unavailable. It is needed because
     the 800-step model reached forget wACC 0.00% -- BELOW chance, which is
     over-unlearning, not success.
  B. random_label, 800 steps, same schedule. The comparison baseline from the
     paper: assign each forget sample a uniformly random retain writer.
  C. Gate v3 across baseline / wic_masked / wic_unmasked@300 /
     wic_unmasked@800 / random_label@800.

wic_masked is reused from disk (auto-discovered). It is the documented broken
variant: forget logits get -inf, hence softmax prob 0, hence NO GRADIENT. Its
loss converged to the log(75)=4.317 floor while forget wACC stayed at 93%.
Kang et al. do not specify which formulation they use; one of the two is
vacuous, and that contrast is a result.

STAGE I is not used. It failed replication: real gap 8.50 at ffn 0.30 sits
inside a null band of 3.48 +/- 8.16 (null max 13.00 from a random half of
retain, i.e. no deletion at all). Median Cohen's d 0.153, only 0.08% of ~46k
neurons above d>0.8 -- writer identity is distributed, not prunable.

TARGETS -- UPDATED
------------------
Gate v2 reported a writer-level non-member rate of 16.67%, but that reused the
WORD-level Youden's J on writer means. Fitting the threshold at writer
granularity (v3) gives:
    word   level: forget 96.27% -> ~70.2%   (the non-member rate)
    writer level: forget 90.00% -> ~10.0%   (NOT 16.7%)
Landing far BELOW the non-member rate is over-unlearning: the forget set
becomes distinguishable in the other direction, which is still a leak.

THE ATTACKER NEVER READS THE WRITER HEAD. Stage II optimises that head
directly, so an attacker using it would measure the objective rather than the
leak. Features are 12 teacher-forced decoder-confidence statistics.

REQUIRES: pip install editdistance
"""

import os, glob, json, copy, time, random, warnings
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, editdistance
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, AutoImageProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore")


class S:
    # ---- EDIT THESE TWO ----------------------------------------------------
    CKPT_ROOT = "/kaggle/input/notebooks/kayed007/fork-of-retrain-pathok/pathok_clean"
    ROOT      = "/kaggle/input/datasets/kaidul007/bnhtrd/Dataset"
    # ------------------------------------------------------------------------
    TROCR_MODEL = "microsoft/trocr-base-handwritten"
    CKPT_FILE   = "baseline_clean_final.pt"
    SPLIT_FILE  = "split.json"
    VOCAB_FILE  = "vocab_clean.json"
    SEED = 42
    VAL_FRAC = 0.05
    BATCH = 16
    NUM_WORKERS = 2
    STRICT_W2I = True
    TARGET_H = 64
    TARGET_W = 256
    OUT = "/kaggle/working/stage2"


class W:                       # Stage II training
    STEPS      = 800
    EVAL_EVERY = 100           # also the checkpoint interval
    LR      = 1e-5
    WD      = 0.01
    WARMUP  = 50
    CLIP    = 1.0
    MICRO_F = 6                # forget imgs/step
    MICRO_R = 6                # retain imgs/step (12 per fused forward)
    LAMBDA_WIC = 1.0
    LAMBDA_WID = 1.0
    LAMBDA_REC = 1.0
    AMP = True
    EVAL_ACC_N = 800
    EVAL_CER_N = 200


class G:                       # gate
    WORDS_PER_WRITER = 80
    N_FOLDS = 5
    RANK_N = 400
    DECODE_N = 500
    OUT = "/kaggle/working/gate_v3"


class RUN:
    TRAIN_WIC_UNMASKED = True
    TRAIN_RANDOM_LABEL = True
    REUSE_EXISTING     = True      # skip training if the checkpoint is on disk
    GATE_LABELS = ["baseline", "wic_masked",
                   "wic_unmasked@300", "wic_unmasked@800",
                   "random_label@800"]


PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
MAX_LEN = 64
PUNCT = {'.', ':', '।', ',', ';', '!', '?', '-', '–', '—'}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(S.OUT, exist_ok=True); os.makedirs(G.OUT, exist_ok=True)
random.seed(S.SEED); np.random.seed(S.SEED); torch.manual_seed(S.SEED)


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED: graphemes / vocab / preprocessing  (identical to the retrain)
# ═══════════════════════════════════════════════════════════════════════════
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
        self.vocab_size = 4

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
        return v


def encode_label(vocab, text):
    seq = (vocab.encode(text) + [EOS_ID])[:MAX_LEN]
    return seq + [PAD_ID] * (MAX_LEN - len(seq))


def preprocess(gray, cfg=S):
    """height normalise -> contrast standardise (content only) -> pad.
    Over-wide words squeezed, never truncated."""
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


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED: data / split / model
# ═══════════════════════════════════════════════════════════════════════════
def load_with_writers(cfg=S):
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
    df["writer"] = df["image_path"].apply(
        lambda p: os.path.relpath(p, cfg.ROOT).split(os.sep)[0])
    print(f"  samples: {len(df):,} | writers: {df['writer'].nunique()}")
    return df


def reload_split(df, ck, cfg=S):
    """writer->index comes FROM THE CHECKPOINT; the rebuilt sorted() map is a
    cross-check only (string sort puts '10' before '2')."""
    spec = json.load(open(os.path.join(cfg.CKPT_ROOT, cfg.SPLIT_FILE)))
    test_w, forget_w, retain_w = (set(map(str, spec[k]))
                                  for k in ("test", "forget", "retain"))
    w2i_built = {w: i for i, w in enumerate(sorted(retain_w | forget_w))}
    w2i = ({str(k): int(v) for k, v in ck["w2i"].items()}
           if "w2i" in ck else w2i_built)
    if "w2i" in ck:
        if w2i == w2i_built:
            print(f"  [ok] w2i matches checkpoint ({len(w2i)} classes)")
        elif cfg.STRICT_W2I:
            raise ValueError("w2i mismatch -- set S.STRICT_W2I=False to proceed")

    test_df   = df[df["writer"].isin(test_w)].reset_index(drop=True)
    forget_df = df[df["writer"].isin(forget_w)].reset_index(drop=True)
    retain_all = df[df["writer"].isin(retain_w)].reset_index(drop=True)
    retain_df, val_df = train_test_split(retain_all, test_size=cfg.VAL_FRAC,
                                         random_state=cfg.SEED)
    retain_df = retain_df.reset_index(drop=True); val_df = val_df.reset_index(drop=True)
    for d in (retain_df, forget_df, val_df):
        d["wid"] = d["writer"].map(w2i).fillna(-1).astype(int)
    test_df["wid"] = -1
    print(f"  retain {len(retain_df):,}/{retain_df['writer'].nunique()}w (MEMBERS) | "
          f"forget {len(forget_df):,}/{forget_df['writer'].nunique()}w | "
          f"test {len(test_df):,}/{test_df['writer'].nunique()}w (NON-MEMBERS)")
    return retain_df, forget_df, val_df, test_df, w2i


class MultiHead(nn.Module):
    def __init__(self, ved, n_writers):
        super().__init__()
        self.ved = ved
        self.writer_head = nn.Linear(ved.config.encoder.hidden_size, n_writers)
        self.n_writers = n_writers


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


def load_into(model, sd, label=""):
    res = model.load_state_dict(sd, strict=False)
    crit = [k for k in res.missing_keys if not k.startswith("ved.encoder.pooler")]
    if crit:
        raise RuntimeError(f"[{label}] {len(crit)} critical keys missing, "
                           f"e.g. {crit[:5]} -- every number would be fiction")
    return model.to(DEVICE).eval()


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED: cheap evaluation
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def writer_acc(model, df, vocab, fe, cfg=S, n=800):
    d = df.sample(min(n, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    was = model.training; model.eval(); c = t = 0
    for b in loader:
        pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
        enc = model.ved.encoder(pixel_values=pv, return_dict=True)
        wl = model.writer_head(enc.last_hidden_state[:, 0, :])
        c += (wl.argmax(-1) == wid).sum().item(); t += len(wid)
    if was: model.train()
    return 100.0 * c / max(t, 1)


@torch.no_grad()
def rec_quality(model, df, vocab, fe, cfg=S, n=300):
    d = df.sample(min(n, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    was = model.training; model.eval(); preds = []
    for b in loader:
        gen = model.ved.generate(b["pixel_values"].to(DEVICE),
                                 max_new_tokens=MAX_LEN, num_beams=1)
        preds += [vocab.decode(r) for r in gen.cpu().tolist()]
    if was: model.train()
    gts = d["label"].astype(str).tolist()
    dist = np.array([editdistance.eval(list(p), list(g)) for p, g in zip(preds, gts)])
    gl = np.array([max(len(g), 1) for g in gts])
    keep = np.array([len(p) for p in preds]) <= 3 * gl + 5   # drop repetition loops
    return {"WER": 100 * np.mean([p != g for p, g in zip(preds, gts)]),
            "CER": (100 * dist[keep].sum() / gl[keep].sum()
                    if keep.any() else float("nan")),
            "runaway": int((~keep).sum())}


def setup():
    print("=" * 74); print(f"  SETUP  device={DEVICE}"); print("-" * 74)
    if DEVICE == "cpu":
        print("  [WARN] no GPU. Settings -> Accelerator -> GPU, then restart.")
    vocab = GraphemeVocab.load(os.path.join(S.CKPT_ROOT, S.VOCAB_FILE))
    print(f"  vocab_size={vocab.vocab_size}")
    base_ck = torch.load(os.path.join(S.CKPT_ROOT, S.CKPT_FILE),
                         map_location="cpu", weights_only=False)
    df = load_with_writers()
    retain_df, forget_df, val_df, test_df, w2i = reload_split(df, base_ck)
    fe = AutoImageProcessor.from_pretrained(S.TROCR_MODEL)
    model = build_model(vocab, len(w2i))
    load_into(model, base_ck["model"], "baseline")
    state = copy.deepcopy(model.state_dict())

    print("\n  BASELINE GATE -- unpruned clean baseline")
    print("-" * 74)
    base = {}
    for name, d in (("forget", forget_df), ("retain", retain_df),
                    ("val", val_df), ("test", test_df)):
        line = f"  [{name:<6}] "
        if name != "test":
            a = writer_acc(model, d, vocab, fe); base[f"ACC_{name}"] = a
            line += f"wACC {a:6.2f} %   "
        q = rec_quality(model, d, vocab, fe)
        base[f"WER_{name}"] = q["WER"]; base[f"CER_{name}"] = q["CER"]
        line += f"WER {q['WER']:6.2f} %  CER {q['CER']:5.2f} %"
        print(line)
    print("-" * 74)
    if min(base["ACC_forget"], base["ACC_retain"]) < 80.0:
        raise RuntimeError("baseline gate failed -- w2i permuted, wrong "
                           "checkpoint, or preprocessing mismatch")
    print("  [PASS] baseline reproduces the retrain")
    return {"model": model, "state": state, "base_sd": base_ck["model"],
            "vocab": vocab, "fe": fe, "retain_df": retain_df,
            "forget_df": forget_df, "val_df": val_df, "test_df": test_df,
            "w2i": w2i, "baseline": base}


def restore(model, state):
    model.load_state_dict(state, strict=True)
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE II
# ═══════════════════════════════════════════════════════════════════════════
def class_partition(ctx):
    w2i = ctx["w2i"]
    retain_w = set(ctx["retain_df"]["writer"]) | set(ctx["val_df"]["writer"])
    forget_w = set(ctx["forget_df"]["writer"])
    dead_w = set(w2i) - retain_w - forget_w
    ri = sorted(w2i[w] for w in retain_w)
    fi = sorted(w2i[w] for w in forget_w)
    di = sorted(w2i[w] for w in dead_w)
    print(f"  live retain {len(ri)} | forget {len(fi)} | dead {len(di)} "
          f"{sorted(dead_w) if dead_w else ''}")
    assert len(ri) + len(fi) + len(di) == len(w2i)
    return (torch.tensor(ri, device=DEVICE), torch.tensor(fi, device=DEVICE),
            torch.tensor(di, dtype=torch.long, device=DEVICE))


def confusion_loss(wlog_f, retain_idx, dead_idx, mask_forget=False):
    """CE of the forget-set writer posterior against uniform over live retain
    classes.

    mask_forget=False (CORRECT): the softmax spans all live classes, so placing
      uniform mass on retain REQUIRES pushing the forget logits down. They get
      real gradient. Floor is log(len(retain_idx)) = 4.317 for 75 classes.
    mask_forget=True (VACUOUS): forget logits are -inf -> softmax prob 0 ->
      gradient 0. The loss reaches the same floor while the forget logits are
      never touched, leaving forget wACC at ~93%.
    Always fp32: a -inf fill under fp16 makes log_softmax return NaN."""
    lg = wlog_f.float()
    if mask_forget:
        keep = torch.zeros(lg.shape[1], dtype=torch.bool, device=lg.device)
        keep[retain_idx] = True
        lg = lg.masked_fill(~keep, float("-inf"))
    elif dead_idx.numel():
        lg = lg.index_fill(1, dead_idx, float("-inf"))
    return -torch.log_softmax(lg, -1)[:, retain_idx].mean(1).mean()


def random_label_loss(wlog_f, retain_idx):
    """Paper's comparison baseline: each forget sample gets a uniformly random
    RETAIN writer as its target."""
    pick = retain_idx[torch.randint(len(retain_idx), (wlog_f.shape[0],),
                                    device=wlog_f.device)]
    return F.cross_entropy(wlog_f.float(), pick)


def infinite(loader):
    while True:
        for b in loader:
            yield b


def stage2_eval(ctx, tag=""):
    m, v, fe = ctx["model"], ctx["vocab"], ctx["fe"]
    r = {"ACC_forget": writer_acc(m, ctx["forget_df"], v, fe, n=W.EVAL_ACC_N),
         "ACC_retain": writer_acc(m, ctx["retain_df"], v, fe, n=W.EVAL_ACC_N),
         "ACC_val":    writer_acc(m, ctx["val_df"],    v, fe, n=W.EVAL_ACC_N)}
    for name, d in (("forget", ctx["forget_df"]), ("retain", ctx["retain_df"]),
                    ("val", ctx["val_df"]), ("test", ctx["test_df"])):
        q = rec_quality(m, d, v, fe, n=W.EVAL_CER_N)
        r[f"WER_{name}"] = q["WER"]; r[f"CER_{name}"] = q["CER"]
    print(f"  [{tag:>6}] wACC f {r['ACC_forget']:6.2f} r {r['ACC_retain']:6.2f} "
          f"v {r['ACC_val']:6.2f} | WER f {r['WER_forget']:5.2f} "
          f"r {r['WER_retain']:5.2f} v {r['WER_val']:5.2f} t {r['WER_test']:5.2f}")
    return r


def run_stage2(ctx, tag, method="wic", mask_forget=False):
    """Trains one arm and CHECKPOINTS AT EVERY EVAL, so every operating point
    on the trajectory is gateable without retraining."""
    model = ctx["model"]
    restore(model, ctx["state"])            # always from the clean baseline
    ri, fi, di = class_partition(ctx)
    chance = 100.0 / len(ri)
    print(f"\n{'=' * 74}\n  STAGE II ARM: {tag}   "
          f"(method={method}, mask_forget={mask_forget})")
    print(f"  chance wACC over live retain classes: {chance:.2f} %")
    print("-" * 74)

    mk = lambda d, bs: DataLoader(WDataset(d, ctx["vocab"], ctx["fe"]),
                                  batch_size=bs, shuffle=True, drop_last=True,
                                  num_workers=S.NUM_WORKERS)
    f_it = infinite(mk(ctx["forget_df"], W.MICRO_F))
    r_it = infinite(mk(ctx["retain_df"], W.MICRO_R))

    opt = torch.optim.AdamW(model.parameters(), lr=W.LR, weight_decay=W.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(W.WARMUP, 1)))
    scaler = torch.amp.GradScaler("cuda", enabled=W.AMP and DEVICE == "cuda")

    hist = [dict(step=0, **stage2_eval(ctx, "0"))]
    paths = {}
    model.train(); t0 = time.time()

    for step in range(1, W.STEPS + 1):
        bf, br = next(f_it), next(r_it)
        nf = bf["pixel_values"].shape[0]
        pv = torch.cat([bf["pixel_values"], br["pixel_values"]]).to(DEVICE)
        lab = torch.cat([bf["labels"], br["labels"]]).to(DEVICE)
        wid_r = br["wid"].to(DEVICE)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=W.AMP and DEVICE == "cuda"):
            enc = model.ved.encoder(pixel_values=pv, return_dict=True)
            wlog = model.writer_head(enc.last_hidden_state[:, 0, :])
            rec = model.ved(encoder_outputs=enc, labels=lab, return_dict=True)

        l_forget = (confusion_loss(wlog[:nf], ri, di, mask_forget)
                    if method == "wic" else random_label_loss(wlog[:nf], ri))
        l_wid = F.cross_entropy(wlog[nf:].float(), wid_r)
        l_rec = rec.loss.float()
        loss = (W.LAMBDA_WIC * l_forget + W.LAMBDA_WID * l_wid
                + W.LAMBDA_REC * l_rec)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), W.CLIP)
        scaler.step(opt); scaler.update(); sched.step()

        if step % 25 == 0:
            print(f"    step {step:4d}  loss {loss.item():6.3f}  "
                  f"forget {l_forget.item():6.3f}  wid {l_wid.item():6.3f}  "
                  f"rec {l_rec.item():6.3f}  ({time.time()-t0:5.0f}s)")

        if step % W.EVAL_EVERY == 0:
            hist.append(dict(step=step, **stage2_eval(ctx, str(step))))
            p = f"{S.OUT}/unlearned_{tag}_step{step}.pt"
            torch.save({"model": model.state_dict(), "w2i": ctx["w2i"],
                        "method": method, "mask_forget": mask_forget,
                        "step": step}, p)
            paths[f"{tag}@{step}"] = p
            model.train()

    h = pd.DataFrame(hist)
    h.to_csv(f"{S.OUT}/history_{tag}.csv", index=False)
    b = ctx["baseline"]
    fin = hist[-1]
    print(f"\n  {tag}: forget wACC {b['ACC_forget']:.2f} -> {fin['ACC_forget']:.2f} % "
          f"(chance {chance:.2f}) | retain {b['ACC_retain']:.2f} -> "
          f"{fin['ACC_retain']:.2f} % | test WER {b['WER_test']:.2f} -> "
          f"{fin['WER_test']:.2f} %")
    if fin["ACC_forget"] < chance:
        print("  [!] forget wACC is BELOW chance -- over-unlearning. The model is "
              "avoiding the deleted writers, not confused about them. An earlier "
              "checkpoint is likely the better operating point.")
    return h, paths


# ═══════════════════════════════════════════════════════════════════════════
#  GATE v3
# ═══════════════════════════════════════════════════════════════════════════
FEATNAMES = ["mean_lp", "min_lp", "max_lp", "std_lp", "sum_lp", "mean_ent",
             "mean_margin", "mean_top1p", "frac_correct", "mean_logrank",
             "n_tokens", "last_lp"]


@torch.no_grad()
def extract_features(model, df, vocab, fe, cfg=S):
    """One 12-vector per word from teacher-forced decoder confidence. The
    writer head is deliberately NOT used -- Stage II optimises it directly, so
    an attacker reading it would be measuring the objective, not the leak."""
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


def sample_per_writer(df, k, seed=S.SEED):
    return (df.groupby("writer", group_keys=False)
              .apply(lambda g: g.sample(min(k, len(g)), random_state=seed))
              .reset_index(drop=True))


@torch.no_grad()
def writer_rank_diag(model, df, vocab, fe, n_classes, cfg=S, n=G.RANK_N):
    """Mean rank of the TRUE writer's logit, 1-indexed. ~n_classes/2 is
    confusion (identity genuinely removed). ~n_classes is INVERSION: the model
    systematically ranks deleted writers last, which an adversary can read off
    just as easily as a correct prediction."""
    d = df.sample(min(n, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval(); ranks, acc = [], []
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
            "ACC": 100 * float(torch.cat(acc).mean())}


def run_attack(model, ctx, label):
    vocab, fe = ctx["vocab"], ctx["fe"]
    k = G.WORDS_PER_WRITER
    mem = sample_per_writer(ctx["retain_df"], k)
    non = sample_per_writer(ctx["test_df"], k)
    fgt = sample_per_writer(ctx["forget_df"], k)
    Xm, Xn, Xf = (extract_features(model, d, vocab, fe) for d in (mem, non, fgt))
    X = np.vstack([Xm, Xn])
    y = np.r_[np.ones(len(Xm)), np.zeros(len(Xn))]
    groups = np.r_[mem["writer"].values, non["writer"].values]

    # grouped CV -- the attacker never sees a writer in both train and test
    oof = np.zeros(len(X)); fold_f = []
    for tr, te in GroupKFold(n_splits=G.N_FOLDS).split(X, y, groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        fold_f.append(clf.predict_proba(Xf)[:, 1])
    sf = np.mean(fold_f, axis=0)
    sm, sn = oof[:len(Xm)], oof[len(Xm):]

    # WORD level: its own ROC, its own Youden's J
    auc_w = roc_auc_score(y, oof)
    fpr, tpr, thr = roc_curve(y, oof)
    t_word = float(thr[np.argmax(tpr - fpr)])

    # WRITER level: aggregate FIRST, then fit a SEPARATE threshold.
    # v2 reused the word threshold here; the writer-mean distribution is far
    # tighter, so the percentages were not properly calibrated.
    pw = lambda s, d: pd.Series(s, index=d["writer"].values).groupby(level=0).mean()
    wm, wn, wf = pw(sm, mem), pw(sn, non), pw(sf, fgt)
    wy = np.r_[np.ones(len(wm)), np.zeros(len(wn))]
    ws = np.r_[wm.values, wn.values]
    auc_r = roc_auc_score(wy, ws)
    fpr_r, tpr_r, thr_r = roc_curve(wy, ws)
    t_writer = float(thr_r[np.argmax(tpr_r - fpr_r)])

    seen = lambda s, t: 100.0 * float((s > t).mean())
    return ({"label": label, "auc_word": auc_w, "auc_writer": auc_r,
             "thr_word": t_word, "thr_writer": t_writer,
             "word_member": seen(sm, t_word), "word_nonmember": seen(sn, t_word),
             "word_forget": seen(sf, t_word),
             "writer_member": seen(wm.values, t_writer),
             "writer_nonmember": seen(wn.values, t_writer),
             "writer_forget": seen(wf.values, t_writer),
             "n_forget_writers": len(wf),
             "forget_above": int((wf.values > t_writer).sum())}, wf, wm, wn)


def report(res, wf, wm, wn, ctx, model, n_classes):
    vocab, fe = ctx["vocab"], ctx["fe"]
    print("\n" + "=" * 74)
    print(f"  MEMBERSHIP INFERENCE — {res['label']}")
    print("-" * 74)
    print(f"  PER-WORD   AUC {res['auc_word']:.3f}  (thr {res['thr_word']:.3f}, "
          f"fit on the WORD ROC)")
    print(f"  PER-WRITER AUC {res['auc_writer']:.3f}  (thr {res['thr_writer']:.3f}, "
          f"fit SEPARATELY on the WRITER ROC)")
    print(f"\n  word    members {res['word_member']:6.2f} %   "
          f"NON-MEMBERS {res['word_nonmember']:6.2f} %   "
          f"forget {res['word_forget']:6.2f} %")
    print(f"  writer  members {res['writer_member']:6.2f} %   "
          f"NON-MEMBERS {res['writer_nonmember']:6.2f} %   "
          f"forget {res['writer_forget']:6.2f} %  "
          f"({res['forget_above']}/{res['n_forget_writers']})")
    q = np.percentile(wf.values, [0, 25, 50, 75, 100])
    print(f"\n  n=20 forget writers -> 5-point granularity; the distribution:")
    print(f"    forget  min {q[0]:.3f}  q25 {q[1]:.3f}  med {q[2]:.3f}  "
          f"q75 {q[3]:.3f}  max {q[4]:.3f}")
    print(f"    member med {np.median(wm.values):.3f} | "
          f"non-member med {np.median(wn.values):.3f}")

    print("\n  TRUE-WRITER LOGIT RANK  (confusion vs inversion)")
    for nm, d in (("forget", ctx["forget_df"]), ("retain", ctx["retain_df"])):
        r = writer_rank_diag(model, d, vocab, fe, n_classes)
        v = ""
        if nm == "forget":
            if r["mean_rank"] < 5:
                v = "  <- still identified"
            elif r["mean_rank"] > 0.8 * n_classes:
                v = "  <- INVERSION: model avoids these writers. Still a leak."
            elif abs(r["mean_rank"] - n_classes / 2) < 0.25 * n_classes / 2:
                v = "  <- confusion, as intended"
        print(f"    [{nm:<6}] wACC {r['ACC']:6.2f} %  mean rank {r['mean_rank']:5.1f}"
              f"/{n_classes}  median {r['median_rank']:5.1f}  "
              f"frac at worst {r['frac_last']:.3f}{v}")

    print("\n  RECOGNITION")
    for nm, d in (("retain", ctx["retain_df"]), ("test", ctx["test_df"])):
        qd = rec_quality(model, d, vocab, fe, n=G.DECODE_N)
        print(f"    [{nm:<6}] WER {qd['WER']:6.2f} %  CER {qd['CER']:5.2f} %")


def discover_checkpoints():
    """Find every unlearned_*.pt on disk, in inputs or working."""
    hits = sorted(set(glob.glob("/kaggle/input/**/unlearned_*.pt", recursive=True)
                      + glob.glob("/kaggle/working/**/unlearned_*.pt", recursive=True)))
    reg = {}
    for p in hits:
        stem = os.path.basename(p)[len("unlearned_"):-3]
        if "_step" in stem:
            tag, step = stem.rsplit("_step", 1)
            reg[f"{tag}@{step}"] = p
        else:
            reg[stem] = p
    return reg


# ═══════════════════════════════════════════════════════════════════════════
#  DRIVER
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ctx = setup()
    n_classes = len(ctx["w2i"])
    reg = discover_checkpoints()
    print(f"\n  found on disk: {sorted(reg) if reg else '(none)'}")

    if RUN.TRAIN_WIC_UNMASKED and not (
            RUN.REUSE_EXISTING and "wic_unmasked@800" in reg):
        _, p = run_stage2(ctx, "wic_unmasked", "wic", mask_forget=False)
        reg.update(p)
    if RUN.TRAIN_RANDOM_LABEL and not (
            RUN.REUSE_EXISTING and "random_label@800" in reg):
        _, p = run_stage2(ctx, "random_label", "random_label", mask_forget=False)
        reg.update(p)

    rows, dists = [], {}
    for label in RUN.GATE_LABELS:
        if label == "baseline":
            sd = ctx["base_sd"]
        elif label in reg:
            sd = torch.load(reg[label], map_location="cpu",
                            weights_only=False)["model"]
        else:
            print(f"\n  [skip] {label}: not found. available -> {sorted(reg)}")
            continue
        load_into(ctx["model"], sd, label)
        res, wf, wm, wn = run_attack(ctx["model"], ctx, label)
        report(res, wf, wm, wn, ctx, ctx["model"], n_classes)
        rows.append(res); dists[label] = wf

    t = pd.DataFrame(rows)
    t.to_csv(f"{G.OUT}/gate_v3.csv", index=False)
    pd.DataFrame(dists).to_csv(f"{G.OUT}/forget_writer_scores.csv")

    print("\n" + "=" * 74)
    print("  COMPARISON  (target = the NON-MEMBER rate on the SAME row, not 50%)")
    print("-" * 74)
    print(f"  {'checkpoint':<20} {'AUCw':>6} {'AUCr':>6} | {'word_f':>7} "
          f"{'w_non':>7} | {'wr_f':>7} {'wr_non':>7} | {'verdict':<12}")
    for r in rows:
        dw = r["writer_forget"] - r["writer_nonmember"]
        verdict = ("still member" if dw > 25 else
                   "OVER-unlearn" if dw < -15 else "near target")
        print(f"  {r['label']:<20} {r['auc_word']:6.3f} {r['auc_writer']:6.3f} | "
              f"{r['word_forget']:6.2f}% {r['word_nonmember']:6.2f}% | "
              f"{r['writer_forget']:6.2f}% {r['writer_nonmember']:6.2f}% | "
              f"{verdict:<12}")
    print("\n  Read the rank diagnostic alongside this: a forget rate at or below")
    print("  the non-member rate is only a success if the true-writer rank sits")
    print("  mid-pack. Rank pinned at the bottom is inversion, which leaks.")
    print(f"\n  saved -> {G.OUT}/gate_v3.csv, {G.OUT}/forget_writer_scores.csv")
    return t, reg


if __name__ == "__main__":
    gate, registry = main()
