# ═══════════════════════════════════════════════════════════════════════════
#  stage1_pruning  (v2)
#  Kang et al. Stage I: prune writer-specific neurons, ranked by how
#  differently they respond to forget vs retain.
#
#  Three method changes from the first attempt:
#    1. Score the CLS token only (position 0 is what the writer head reads;
#       the decoder reads patch tokens via cross-attention, so CLS-selected
#       neurons are weakly coupled to recognition).
#    2. Welch t-statistic instead of the paper's raw ratio S_f/S_r (eq. 1),
#       which sits near 1.0 for nearly every neuron in a pretrained ViT.
#    3. Sweep from 2%, not 10-40%.
#
#  Correctness fixes: verified checkpoint load; w2i taken from the checkpoint;
#  unpruned baseline is a hard gate; baseline computed once; abs(t) ranking
#  option; float64 variance; strict restore.
# ═══════════════════════════════════════════════════════════════════════════

import os, glob, json, copy, random, warnings
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, editdistance
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, AutoImageProcessor
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


class S:
    # ---- EDIT THESE TWO -----------------------------------------------------
    CKPT_ROOT = "/kaggle/input/notebooks/kayed007/fork-of-retrain-pathok/pathok_clean"
    ROOT      = "/kaggle/input/datasets/kaidul007/bnhtrd/Dataset"
    # -------------------------------------------------------------------------
    TROCR_MODEL = "microsoft/trocr-base-handwritten"
    CKPT_FILE   = "baseline_clean_final.pt"
    SPLIT_FILE  = "split.json"
    VOCAB_FILE  = "vocab_clean.json"
    OUT         = "/kaggle/working/stage1"

    SEED = 42
    VAL_FRAC = 0.05
    BATCH = 16
    NUM_WORKERS = 2

    ACT_BATCHES = 60          # batches per set for activation statistics
    EVAL_N = 800              # samples for writer ACC
    CER_N  = 300              # samples for CER (generation is slow)

    SWEEP_RATES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]   # 0.0 is the gate
    SCORE = "welch"           # "welch" | "ratio"
    SCORE_ABS = True          # rank on |t| (both directions) vs signed t
    CLS_ONLY = True

    STRICT_W2I = True
    BASELINE_ACC_MIN = 80.0

    TARGET_H = 64
    TARGET_W = 256


RUN_GATE_NOW = True           # set False to only define, not execute

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = 4
MAX_LEN = 64
PUNCT = {'.', ':', '।', ',', ';', '!', '?', '-', '–', '—'}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(S.OUT, exist_ok=True)
random.seed(S.SEED); np.random.seed(S.SEED); torch.manual_seed(S.SEED)


# ─── graphemes / vocab / preprocessing  (identical to the retrain) ───────────
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
        return v


def encode_label(vocab, text):
    seq = (vocab.encode(text) + [EOS_ID])[:MAX_LEN]
    return seq + [PAD_ID] * (MAX_LEN - len(seq))


def preprocess(gray, cfg=S):
    """height normalise -> contrast standardise (content only) -> pad.
    Over-wide words are squeezed, never truncated."""
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


# ─── data / split / model ────────────────────────────────────────────────────
def writer_of(p, root):
    return os.path.relpath(p, root).split(os.sep)[0]


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
        raise FileNotFoundError(
            f"No label .xlsx under {cfg.ROOT}/*/*.xlsx -- check S.ROOT.")
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
    print(f"  samples: {len(df):,} | writers: {df['writer'].nunique()}")
    return df


def load_ckpt(cfg=S):
    p = os.path.join(cfg.CKPT_ROOT, cfg.CKPT_FILE)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Attach the retrain notebook's output and set "
            f"S.CKPT_ROOT to the directory holding {cfg.CKPT_FILE}, "
            f"{cfg.SPLIT_FILE}, {cfg.VOCAB_FILE}.")
    return torch.load(p, map_location="cpu", weights_only=False)


def reload_split(df, ck, cfg=S):
    """Rebuild the frozen split. writer->index comes FROM THE CHECKPOINT;
    the rebuilt sorted() map is only a cross-check (string sort: '10' < '2')."""
    spec = json.load(open(os.path.join(cfg.CKPT_ROOT, cfg.SPLIT_FILE)))
    test_w   = set(map(str, spec["test"]))
    forget_w = set(map(str, spec["forget"]))
    retain_w = set(map(str, spec["retain"]))

    w2i_built = {w: i for i, w in enumerate(sorted(retain_w | forget_w))}
    w2i_ck = ck.get("w2i")
    if w2i_ck is None:
        print("  [WARN] checkpoint has no 'w2i'; using the rebuilt sorted() map. "
              "If the retrain built it differently every writer label is permuted.")
        w2i = w2i_built
    else:
        w2i = {str(k): int(v) for k, v in w2i_ck.items()}
        if w2i == w2i_built:
            print(f"  [ok] w2i matches checkpoint ({len(w2i)} classes)")
        else:
            diff = [w for w in sorted(set(w2i) | set(w2i_built))
                    if w2i.get(w) != w2i_built.get(w)]
            msg = (f"  [MISMATCH] rebuilt w2i disagrees with checkpoint on "
                   f"{len(diff)} writers, e.g. "
                   f"{[(w, w2i_built.get(w), w2i.get(w)) for w in diff[:5]]} "
                   f"(writer, rebuilt, checkpoint)")
            print(msg)
            if cfg.STRICT_W2I:
                raise ValueError(msg + "\n  The checkpoint map is used and is "
                                 "almost certainly right -- set S.STRICT_W2I=False "
                                 "to proceed once confirmed.")

    test_df   = df[df["writer"].isin(test_w)].reset_index(drop=True)
    forget_df = df[df["writer"].isin(forget_w)].reset_index(drop=True)
    retain_all = df[df["writer"].isin(retain_w)].reset_index(drop=True)
    retain_df, val_df = train_test_split(retain_all, test_size=cfg.VAL_FRAC,
                                         random_state=cfg.SEED)
    retain_df = retain_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    for name, d in (("retain", retain_df), ("forget", forget_df), ("val", val_df)):
        d["wid"] = d["writer"].map(w2i)
        n_bad = int(d["wid"].isna().sum())
        if n_bad:
            print(f"  [WARN] {name}: {n_bad} rows whose writer is absent from w2i "
                  f"-- dropped")
            d.drop(d.index[d["wid"].isna()], inplace=True)
            d.reset_index(drop=True, inplace=True)
        d["wid"] = d["wid"].astype(int)
    test_df["wid"] = -1

    present = set(retain_df["writer"]) | set(forget_df["writer"]) | set(val_df["writer"])
    dead = sorted(set(w2i) - present)
    if dead:
        print(f"  [note] {len(dead)} writer class(es) with no surviving images "
              f"after the label dedupe: {dead} -- dead output units, harmless "
              f"but worth one line in the writeup")

    print(f"  retain {len(retain_df):,}/{retain_df['writer'].nunique()}w | "
          f"forget {len(forget_df):,}/{forget_df['writer'].nunique()}w | "
          f"test {len(test_df):,}/{test_df['writer'].nunique()}w | "
          f"{len(w2i)} writer classes")
    return retain_df, forget_df, val_df, test_df, w2i


class MultiHead(nn.Module):
    def __init__(self, ved, n_writers):
        super().__init__()
        self.ved = ved
        self.writer_head = nn.Linear(ved.config.encoder.hidden_size, n_writers)
        self.n_writers = n_writers

    @property
    def vit(self):
        return self.ved.encoder

    def forward(self, pv, labels=None):
        enc = self.ved.encoder(pixel_values=pv, return_dict=True)
        wlog = self.writer_head(enc.last_hidden_state[:, 0, :])
        rec = (self.ved(encoder_outputs=enc, labels=labels, return_dict=True)
               if labels is not None else None)
        return rec, wlog


def load_model(vocab, ck, cfg=S):
    """Build MultiHead and load the checkpoint WITH A VERIFIED REPORT.
    Returns (model, pristine_state_snapshot)."""
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

    res = model.load_state_dict(ck["model"], strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    print(f"\n  CHECKPOINT LOAD REPORT  (writers={n_writers})")
    print(f"    missing    : {len(missing)}")
    print(f"    unexpected : {len(unexpected)}")
    for k in missing[:8]:
        print(f"      MISSING    {k}")
    for k in unexpected[:8]:
        print(f"      UNEXPECTED {k}")

    critical = [k for k in missing if not k.startswith("ved.encoder.pooler")]
    if critical:
        raise RuntimeError(
            f"{len(critical)} critical keys missing, e.g. {critical[:5]}. "
            f"The model would be part stock-TrOCR / part random writer head "
            f"and every number below would be fiction.")

    model = model.to(DEVICE).eval()
    state = copy.deepcopy(model.state_dict())      # snapshot AFTER loading
    return model, state


# ─── prunable modules, grouped as in the paper ───────────────────────────────
def target_modules(model):
    """blocks 0-3 shallow, 4-7 middle, 8-11 deep -- maps 1:1 onto ViT-base."""
    vit = model.vit
    groups = {"embed": [("embed.proj", vit.embeddings.patch_embeddings.projection)],
              "attn_shallow": [], "attn_middle": [], "attn_deep": [], "ffn": []}
    for i, layer in enumerate(vit.encoder.layer):
        key = ("attn_shallow" if i <= 3 else
               "attn_middle" if i <= 7 else "attn_deep")
        groups[key].append((f"L{i}.attn_out", layer.attention.output.dense))
        groups["ffn"].append((f"L{i}.ffn", layer.intermediate.dense))
    return groups


GROUP_ORDER = ["embed", "attn_shallow", "attn_middle", "attn_deep", "ffn"]


# ─── activation statistics (CLS token, float64) ──────────────────────────────
@torch.no_grad()
def collect_stats(model, df, vocab, fe, cfg=S, tag=""):
    """{name: (mean, var, n)} per output neuron. CLS_ONLY takes position 0 --
    the token the writer head reads. The patch embedding is a conv applied
    before CLS exists, so it always uses the spatial mean. float64 because
    sum(x^2)/n - mean^2 cancels badly in float32 and var enters the Welch
    denominator."""
    groups = target_modules(model)
    flat = [(n, m) for g in groups.values() for (n, m) in g]
    s1 = {n: None for n, _ in flat}
    s2 = {n: None for n, _ in flat}
    cnt = {n: 0 for n, _ in flat}
    handles = []

    def mk(name):
        def hook(_m, _i, out):
            o = out.detach().double()
            if o.dim() == 4:                      # conv (B,C,H,W)
                v = o.mean(dim=(2, 3))
            elif cfg.CLS_ONLY:
                v = o[:, 0, :]                    # CLS
            else:
                v = o.mean(dim=1)
            s1[name] = v.sum(0) if s1[name] is None else s1[name] + v.sum(0)
            sq = (v ** 2).sum(0)
            s2[name] = sq if s2[name] is None else s2[name] + sq
            cnt[name] += v.shape[0]
        return hook

    for n, m in flat:
        handles.append(m.register_forward_hook(mk(n)))

    g = torch.Generator().manual_seed(cfg.SEED)
    loader = DataLoader(WDataset(df, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=True, num_workers=cfg.NUM_WORKERS, generator=g)
    try:
        for i, b in enumerate(tqdm(loader, total=min(cfg.ACT_BATCHES, len(loader)),
                                   desc=f"acts {tag}", leave=False)):
            if i >= cfg.ACT_BATCHES:
                break
            model.ved.encoder(pixel_values=b["pixel_values"].to(DEVICE))
    finally:
        for h in handles:
            h.remove()

    out = {}
    for n, _ in flat:
        c = max(cnt[n], 1)
        mu = s1[n] / c
        var = (s2[n] / c) - mu ** 2
        out[n] = (mu.float().cpu(), var.clamp(min=0).float().cpu(), c)
    return out


def neuron_scores(stat_f, stat_r, mode="welch", eps=1e-8):
    """welch: (mu_f-mu_r)/sqrt(var_f/n_f + var_r/n_r)  -- signed
       ratio: (|mu_f|+eps)/(|mu_r|+eps)                -- the paper's eq. 1"""
    scores = {}
    for n in stat_f:
        mf, vf, nf = stat_f[n]
        mr, vr, nr = stat_r[n]
        if mode == "welch":
            scores[n] = (mf - mr) / torch.sqrt(vf / nf + vr / nr + eps)
        elif mode == "ratio":
            scores[n] = (mf.abs() + eps) / (mr.abs() + eps)
        else:
            raise ValueError(mode)
    return scores


def rank_scores(scores, cfg=S):
    """What topk ranks on. abs(t) prunes neurons differing in EITHER direction;
    signed t only prunes forget-over-activating ones."""
    if not cfg.SCORE_ABS:
        return scores
    if cfg.SCORE == "welch":
        return {n: s.abs() for n, s in scores.items()}
    return {n: torch.log(s.clamp(min=1e-8)).abs() for n, s in scores.items()}


def score_diagnostic(scores_welch, scores_ratio, save=None):
    """A tight spike means nothing selective exists to find -- a reportable
    negative result, with this table as the evidence."""
    print("\n" + "=" * 72)
    print("  SCORE DIAGNOSTIC -- is there anything selective to rank?")
    print("-" * 72)
    print(f"  {'module':<14} {'n':>6} | {'welch |t| p50':>13} {'p95':>8} "
          f"{'frac|t|>2':>10} | {'ratio p50':>10} {'p95':>8}")
    print("-" * 72)
    rows = []
    for n in scores_welch:
        w = scores_welch[n].abs().numpy()
        r = scores_ratio[n].numpy()
        row = {"module": n, "n": len(w),
               "welch_p50": float(np.percentile(w, 50)),
               "welch_p95": float(np.percentile(w, 95)),
               "frac_gt2": float((w > 2).mean()),
               "frac_neg": float((scores_welch[n].numpy() < 0).mean()),
               "ratio_p50": float(np.percentile(r, 50)),
               "ratio_p95": float(np.percentile(r, 95))}
        rows.append(row)
        if n in ("embed.proj", "L0.attn_out", "L5.attn_out", "L11.attn_out",
                 "L0.ffn", "L5.ffn", "L11.ffn"):
            print(f"  {n:<14} {row['n']:>6} | {row['welch_p50']:13.2f} "
                  f"{row['welch_p95']:8.2f} {row['frac_gt2']:10.3f} | "
                  f"{row['ratio_p50']:10.3f} {row['ratio_p95']:8.3f}")
    t = pd.DataFrame(rows)
    print("-" * 72)
    print(f"  ALL MODULES: median |t| {t['welch_p50'].median():.2f}, "
          f"median frac|t|>2 {t['frac_gt2'].median():.3f}, "
          f"median ratio {t['ratio_p50'].median():.3f}")
    if t["frac_gt2"].median() < 0.02:
        print("\n  ** Very few neurons differ beyond sampling noise. Read with the")
        print("     ratio median: ~1.000 alongside low frac|t|>2 is the genuinely-")
        print("     no-signal signature. Report it -- the paper's eq. 1 premise")
        print("     does not transfer to a pretrained ViT encoder.")
    else:
        print(f"\n  {100*t['frac_gt2'].median():.1f}% of neurons differ beyond "
              "noise -> there is signal to rank on")
    top_sign = []
    for n, s in scores_welch.items():
        k = max(1, int(0.05 * s.numel()))
        top_sign.append(float((s[torch.topk(s.abs(), k).indices] < 0).float().mean()))
    print(f"  of the top-5% |t| neurons, {100*np.mean(top_sign):.1f}% have t<0 "
          f"(forget UNDER-activates); signed ranking would miss these. "
          f"S.SCORE_ABS={S.SCORE_ABS}")
    if save:
        t.to_csv(save, index=False)
        print(f"  saved -> {save}")
    return t


# ─── pruning ─────────────────────────────────────────────────────────────────
def apply_pruning(model, ranked, rates, verbose=True):
    groups = target_modules(model)
    masks, total, zeroed = {}, 0, 0
    for gname, entries in groups.items():
        k = rates.get(gname, 0.0)
        for name, mod in entries:
            sc = ranked[name]
            n_out = sc.numel()
            n_cut = int(round(k * n_out))
            mask = torch.ones(n_out)
            if n_cut > 0:
                mask[torch.topk(sc, n_cut).indices] = 0.0
            masks[name] = mask
            with torch.no_grad():
                w = mod.weight
                shape = [-1] + [1] * (w.dim() - 1)
                w.mul_(mask.to(w.device, w.dtype).view(*shape))
                if mod.bias is not None:
                    mod.bias.mul_(mask.to(w.device, w.dtype))
            total += n_out
            zeroed += int((mask == 0).sum())
    sp = 100.0 * zeroed / max(total, 1)
    if verbose:
        print(f"    pruned {zeroed:,}/{total:,} neurons ({sp:.2f}%)")
    return masks, sp


def reapply_masks(model, masks):
    """Call after every optimizer step in Stage II to hold pruned units at zero."""
    mods = {n: m for g in target_modules(model).values() for (n, m) in g}
    with torch.no_grad():
        for name, mask in masks.items():
            mod = mods[name]
            w = mod.weight
            shape = [-1] + [1] * (w.dim() - 1)
            w.mul_(mask.to(w.device, w.dtype).view(*shape))
            if mod.bias is not None:
                mod.bias.mul_(mask.to(w.device, w.dtype))


def restore(model, state):
    model.load_state_dict(state, strict=True)
    return model


# ─── evaluation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def writer_acc(model, df, vocab, fe, cfg=S, n=None):
    n = n or cfg.EVAL_N
    d = df.sample(min(n, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval(); c = t = 0
    for b in loader:
        pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
        enc = model.ved.encoder(pixel_values=pv, return_dict=True)
        wl = model.writer_head(enc.last_hidden_state[:, 0, :])
        c += (wl.argmax(-1) == wid).sum().item(); t += len(wid)
    return 100.0 * c / max(t, 1)


@torch.no_grad()
def rec_quality(model, df, vocab, fe, cfg=S, n=None):
    n = n or cfg.CER_N
    d = df.sample(min(n, len(df)), random_state=cfg.SEED).reset_index(drop=True)
    loader = DataLoader(WDataset(d, vocab, fe, cfg), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval(); preds = []
    for b in loader:
        gen = model.ved.generate(b["pixel_values"].to(DEVICE),
                                 max_new_tokens=MAX_LEN, num_beams=1)
        preds += [vocab.decode(r) for r in gen.cpu().tolist()]
    gts = d["label"].astype(str).tolist()
    dists = np.array([editdistance.eval(list(p), list(g)) for p, g in zip(preds, gts)])
    gl = np.array([max(len(g), 1) for g in gts])
    pl = np.array([len(p) for p in preds])
    keep = pl <= 3 * gl + 5                       # drop greedy repetition loops
    cer_raw = 100 * dists.sum() / gl.sum()
    cer_t = 100 * dists[keep].sum() / gl[keep].sum() if keep.any() else float("nan")
    wer = 100 * np.mean([p != g for p, g in zip(preds, gts)])
    return {"CER": cer_t, "CER_raw": cer_raw, "WER": wer,
            "runaway": int((~keep).sum())}


# ─── setup + gate ────────────────────────────────────────────────────────────
def quick_check(cfg=S, with_cer=True):
    """Load, verify, evaluate the UNPRUNED model. ~2 min. This is the go/no-go:
    if the unpruned model does not reproduce the retrain, nothing downstream
    means anything."""
    print("=" * 72)
    print(f"  SETUP   device={DEVICE}")
    print("-" * 72)
    vocab = GraphemeVocab.load(os.path.join(cfg.CKPT_ROOT, cfg.VOCAB_FILE))
    print(f"  vocab_size={vocab.vocab_size}")
    ck = load_ckpt(cfg)
    df = load_with_writers(cfg)
    retain_df, forget_df, val_df, test_df, w2i = reload_split(df, ck, cfg)
    model, state = load_model(vocab, ck, cfg)
    fe = AutoImageProcessor.from_pretrained(cfg.TROCR_MODEL)

    print("\n" + "=" * 72)
    print("  BASELINE GATE -- unpruned (height+contrast preprocessing)")
    print("-" * 72)
    base = {}
    for name, d in (("forget", forget_df), ("retain", retain_df), ("test", test_df)):
        line = f"  [{name:<6}] "
        if name != "test":
            a = writer_acc(model, d, vocab, fe, cfg)
            base[f"ACC_{name}"] = a
            line += f"writer ACC {a:6.2f} %   "
        if with_cer:
            q = rec_quality(model, d, vocab, fe, cfg)
            base[f"CER_{name}"] = q["CER"]; base[f"WER_{name}"] = q["WER"]
            line += (f"WER {q['WER']:6.2f} %  CER {q['CER']:5.2f} %  "
                     f"runaway {q['runaway']}")
        print(line)

    ok = (base.get("ACC_forget", 0) >= cfg.BASELINE_ACC_MIN
          and base.get("ACC_retain", 0) >= cfg.BASELINE_ACC_MIN)
    print("-" * 72)
    if not ok:
        print(f"  [FAIL] unpruned writer ACC below {cfg.BASELINE_ACC_MIN}%.")
        print("         Do NOT read the sweep. Likely causes, in order:")
        print("         1. w2i permuted (labels do not match training)")
        print("         2. wrong checkpoint / wrong CKPT_ROOT")
        print("         3. preprocessing mismatch vs the retrain")
        raise RuntimeError("baseline gate failed")
    print("  [PASS] unpruned model reproduces the retrain -> sweep is meaningful")

    return {"model": model, "state": state, "vocab": vocab, "fe": fe,
            "retain_df": retain_df, "forget_df": forget_df, "val_df": val_df,
            "test_df": test_df, "w2i": w2i, "baseline": base,
            "scores": None, "ranked": None, "cfg": cfg}


def compute_scores(ctx, cfg=S):
    """Activation stats + both scores + the diagnostic. ~5 min."""
    model, vocab, fe = ctx["model"], ctx["vocab"], ctx["fe"]
    restore(model, ctx["state"])
    print("\ncollecting activation statistics (CLS token)...")
    stat_f = collect_stats(model, ctx["forget_df"], vocab, fe, cfg, "forget")
    stat_r = collect_stats(model, ctx["retain_df"], vocab, fe, cfg, "retain")
    torch.save({"forget": stat_f, "retain": stat_r},
               os.path.join(cfg.OUT, "act_stats.pt"))
    sw = neuron_scores(stat_f, stat_r, "welch")
    sr = neuron_scores(stat_f, stat_r, "ratio")
    torch.save({"welch": sw, "ratio": sr}, os.path.join(cfg.OUT, "scores.pt"))
    diag = score_diagnostic(sw, sr, os.path.join(cfg.OUT, "score_diagnostic.csv"))
    ctx["scores"] = {"welch": sw, "ratio": sr}
    ctx["ranked"] = rank_scores(sw if cfg.SCORE == "welch" else sr, cfg)
    ctx["diag"] = diag
    return diag


# ─── sweep (the paper's Tables 4-7) ──────────────────────────────────────────
def sweep_group(group, ctx, rates=None, cfg=S, with_cer=False):
    model, state, ranked = ctx["model"], ctx["state"], ctx["ranked"]
    vocab, fe = ctx["vocab"], ctx["fe"]
    retain_df, forget_df, base = ctx["retain_df"], ctx["forget_df"], ctx["baseline"]
    rates = rates if rates is not None else cfg.SWEEP_RATES

    print(f"\n  --- {group} ---")
    hdr = f"    {'rate':>6} {'sparsity':>9} {'ACC_f':>8} {'ACC_r':>8} {'gap':>7}"
    if with_cer:
        hdr += f" {'CER_f':>7} {'CER_r':>7}"
    print(hdr)
    print(f"    {0.00:6.2f} {0.0:8.2f}% {base['ACC_forget']:7.2f}% "
          f"{base['ACC_retain']:7.2f}% {base['ACC_retain']-base['ACC_forget']:6.2f}"
          + (f" {base.get('CER_forget', float('nan')):6.2f}%"
             f" {base.get('CER_retain', float('nan')):6.2f}%" if with_cer else ""))

    rows = [{"group": group, "rate": 0.0, "sparsity": 0.0,
             "ACC_forget": base["ACC_forget"], "ACC_retain": base["ACC_retain"],
             "gap": base["ACC_retain"] - base["ACC_forget"],
             "CER_forget": base.get("CER_forget"),
             "CER_retain": base.get("CER_retain")}]
    for rate in rates:
        restore(model, state)
        _, sp = apply_pruning(model, ranked, {group: rate}, verbose=False)
        af = writer_acc(model, forget_df, vocab, fe, cfg)
        ar = writer_acc(model, retain_df, vocab, fe, cfg)
        row = {"group": group, "rate": rate, "sparsity": sp,
               "ACC_forget": af, "ACC_retain": ar, "gap": ar - af}
        line = f"    {rate:6.2f} {sp:8.2f}% {af:7.2f}% {ar:7.2f}% {ar-af:6.2f}"
        if with_cer:
            qf = rec_quality(model, forget_df, vocab, fe, cfg)
            qr = rec_quality(model, retain_df, vocab, fe, cfg)
            row.update({"CER_forget": qf["CER"], "CER_retain": qr["CER"],
                        "WER_forget": qf["WER"], "WER_retain": qr["WER"]})
            line += f" {qf['CER']:6.2f}% {qr['CER']:6.2f}%"
        print(line)
        rows.append(row)
    restore(model, state)
    return rows


def run_sweep(with_cer=False, ctx=None, cfg=S):
    """with_cer=False is ~10 min and answers the go/no-go. Use refine_rates()
    to add recognition cost for the shortlisted rates only."""
    if ctx is None:
        ctx = quick_check(cfg)
    if ctx.get("ranked") is None:
        compute_scores(ctx, cfg)

    print("\n" + "=" * 72)
    print(f"  RATE SWEEP  (score={cfg.SCORE}, abs={cfg.SCORE_ABS}, "
          f"CLS_ONLY={cfg.CLS_ONLY})")
    print("  looking for: ACC_forget DOWN while ACC_retain HOLDS")
    print("=" * 72)

    rows = []
    for g in GROUP_ORDER:
        rows += sweep_group(g, ctx, cfg=cfg, with_cer=with_cer)

    t = pd.DataFrame(rows)
    tag = f"{cfg.SCORE}{'_abs' if cfg.SCORE_ABS else '_signed'}"
    t.to_csv(os.path.join(cfg.OUT, f"sweep_{tag}.csv"), index=False)

    print("\n" + "=" * 72)
    print("  BEST SELECTIVITY PER GROUP (max ACC_retain - ACC_forget)")
    print("-" * 72)
    for g in GROUP_ORDER:
        sub = t[(t.group == g) & (t.rate > 0)]
        if sub.empty:
            continue
        b = sub.loc[sub["gap"].idxmax()]
        print(f"  {g:<14} rate {b['rate']:.2f} -> forget {b['ACC_forget']:6.2f}% "
              f"retain {b['ACC_retain']:6.2f}%  gap {b['gap']:6.2f}")
    print(f"\n  saved -> {cfg.OUT}/sweep_{tag}.csv")
    ctx["sweep"] = t
    return t, ctx.get("diag"), ctx


def refine_rates(groups, rates, ctx, cfg=S):
    """Re-run only shortlisted (group, rate) pairs WITH recognition cost."""
    rows = []
    for g in groups:
        rows += sweep_group(g, ctx, rates=rates, cfg=cfg, with_cer=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(cfg.OUT, "sweep_refined.csv"), index=False)
    print(f"\n  saved -> {cfg.OUT}/sweep_refined.csv")
    return t


def apply_best(rates, ctx, cfg=S):
    """rates e.g. {"embed":0.05,"attn_shallow":0.05,"attn_middle":0.10,
                   "attn_deep":0.02,"ffn":0.05}"""
    model, state, ranked = ctx["model"], ctx["state"], ctx["ranked"]
    vocab, fe = ctx["vocab"], ctx["fe"]
    restore(model, state)
    masks, sp = apply_pruning(model, ranked, rates)
    af = writer_acc(model, ctx["forget_df"], vocab, fe, cfg)
    ar = writer_acc(model, ctx["retain_df"], vocab, fe, cfg)
    qf = rec_quality(model, ctx["forget_df"], vocab, fe, cfg)
    qr = rec_quality(model, ctx["retain_df"], vocab, fe, cfg)
    b = ctx["baseline"]
    print(f"\n  PRUNED M*  sparsity {sp:.2f}%   rates={rates}")
    print(f"    forget ACC {af:6.2f}% (was {b['ACC_forget']:.2f}%)  "
          f"WER {qf['WER']:6.2f}%  CER {qf['CER']:5.2f}%")
    print(f"    retain ACC {ar:6.2f}% (was {b['ACC_retain']:.2f}%)  "
          f"WER {qr['WER']:6.2f}%  CER {qr['CER']:5.2f}%")
    torch.save({"model": model.state_dict(), "masks": masks, "rates": rates,
                "sparsity": sp, "w2i": ctx["w2i"],
                "post_prune": {"ACC_forget": af, "ACC_retain": ar,
                               "WER_forget": qf["WER"], "WER_retain": qr["WER"],
                               "CER_forget": qf["CER"], "CER_retain": qr["CER"]}},
               os.path.join(cfg.OUT, "pruned_M_star.pt"))
    print(f"  saved -> {cfg.OUT}/pruned_M_star.pt")
    return model, masks


print("defined: quick_check, compute_scores, run_sweep, refine_rates, apply_best")

if RUN_GATE_NOW:
    ctx = quick_check()
    print("\nnext:  sweep, diag, ctx = run_sweep(with_cer=False, ctx=ctx)")
