"""
pathok-exp9-crosslang.py
=========================
Cross-dataset / cross-script validation of the CNN-vs-ViT writer-ID
dissociation found on BnHTRD (Bangla): ConvNeXt-Base matches the TrOCR ViT
encoder on recognition but writer identification never rises above chance,
and a targeted tuning attempt (20x LR on the fresh CLS-token parameters,
5x WRITER_W) produced zero measurable change across 8 epochs there.

This tests whether that dissociation is BnHTRD/Bangla-specific or a more
general property, using Felix92/ibm-handwriting-campaign-word: English
words, 302 writers, IBM handwriting-campaign forms -- a different script,
different dataset, different writer population, same architecture question.

Both arms share the SAME pretrained TrOCR decoder and the SAME joint
recognition + writer-classification recipe; only the encoder differs. The
ConvNeXt arm uses the ALREADY-TUNED recipe from the Bangla follow-up
(separate high-LR param group for the CLS token/projection/writer head,
WRITER_W=10) from the start, so this run does not re-litigate "maybe it
just needs tuning" -- that was already tested and failed on Bangla.
Recognition is tracked via training/val loss only (no CER/WER decoding) to
keep this a fast, focused test of the one question that matters here:
does writer-ID accuracy rise above chance (1/302 = 0.33%) for each arm.

OUTPUT
------
Per-epoch printed table for both arms (loss + val writer-ID accuracy) and
a FINAL RESULT banner directly comparing the two.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers.modeling_outputs import BaseModelOutput
from datasets import load_dataset, concatenate_datasets
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

CROSSLANG_SMOKE_TEST = False  # flip to False for the real run once this passes clean
CLS_LR_MULT = 20             # ConvNeXt fresh-params LR multiplier (proven-tuned recipe)


class C:
    TROCR_MODEL = "microsoft/trocr-base-handwritten"
    OUT = "/kaggle/working/crosslang"
    HF_DATASET = "Felix92/ibm-handwriting-campaign-word"

    SEED = 42
    VAL_FRAC = 0.10
    BATCH = 16
    EPOCHS = 2 if CROSSLANG_SMOKE_TEST else 5
    LR = 5e-5
    WEIGHT_DECAY = 0.01
    WRITER_W = 10.0
    FP16 = True
    NUM_WORKERS = 2
    MAX_LEN = 24              # words are 1-20 chars per the dataset card
    EVAL_N = 40 if CROSSLANG_SMOKE_TEST else 1000
    TRAIN_SUBSAMPLE = 300 if CROSSLANG_SMOKE_TEST else None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(C.OUT, exist_ok=True)
np.random.seed(C.SEED); torch.manual_seed(C.SEED)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_pool(cfg=C):
    """Merge all provided splits into one pool, then re-split ourselves at the
    SAMPLE level (not writer-disjoint) -- this test needs every writer present
    in both train and val, matching how the Bangla ConvNeXt run tracked
    per-epoch writer accuracy on a random subsample of the same training
    writer pool, not a held-out writer set."""
    ds = load_dataset(cfg.HF_DATASET)
    pool = concatenate_datasets([ds[s] for s in ds.keys()])
    print(f"[data] {len(pool):,} rows pooled from splits {list(ds.keys())}")
    writers = sorted(set(int(w) for w in pool["writer_id"]))
    print(f"[data] {len(writers)} unique writers")
    return pool, writers


class WDataset(Dataset):
    def __init__(self, pool, indices, processor, w2i, cfg=C):
        self.pool = pool
        self.indices = indices
        self.proc = processor
        self.w2i = w2i
        self.cfg = cfg

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.pool[int(self.indices[i])]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        pv = self.proc.image_processor(img, return_tensors="pt").pixel_values.squeeze(0)
        enc = self.proc.tokenizer(str(row["text"]), padding="max_length", truncation=True,
                                  max_length=self.cfg.MAX_LEN, return_tensors="pt")
        labels = enc.input_ids.squeeze(0).clone()
        labels[labels == self.proc.tokenizer.pad_token_id] = -100
        wid = self.w2i[int(row["writer_id"])]
        return {"pixel_values": pv, "labels": labels, "wid": torch.tensor(wid, dtype=torch.long)}


# ══════════════════════════════════════════════════════════════════════════════
# MODELS -- ViT arm (native TrOCR encoder) and ConvNeXt arm (adapter, proven
# on the Bangla study -- same class, verbatim, architecture-agnostic wrt
# language/script)
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
    """Verbatim from the Bangla study's adapter (unlearning/exp5_convnext_base.py)."""

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


def _fix_special_tokens(ved, processor, tag):
    """microsoft/trocr-base-handwritten's top-level VisionEncoderDecoderConfig
    does not have pad_token_id / decoder_start_token_id populated (only its
    generation_config and tokenizer do) -- but forward()'s internal
    shift_tokens_right reads them from self.config directly, not
    self.generation_config, so training crashes with an AttributeError
    without this. Fall back generation_config -> tokenizer so this is
    correct regardless of which one the checkpoint actually populated."""
    gc, cfg_, tok = ved.generation_config, ved.config, processor.tokenizer
    cfg_.pad_token_id = gc.pad_token_id if gc.pad_token_id is not None else tok.pad_token_id
    cfg_.decoder_start_token_id = (gc.decoder_start_token_id if gc.decoder_start_token_id is not None
                                   else tok.eos_token_id)
    cfg_.eos_token_id = gc.eos_token_id if gc.eos_token_id is not None else tok.eos_token_id
    cfg_.bos_token_id = gc.bos_token_id if gc.bos_token_id is not None else tok.bos_token_id
    print(f"[{tag}] special tokens: pad={cfg_.pad_token_id} "
          f"decoder_start={cfg_.decoder_start_token_id} eos={cfg_.eos_token_id} "
          f"bos={cfg_.bos_token_id}")


def build_vit(n_writers, processor, cfg=C):
    ved = VisionEncoderDecoderModel.from_pretrained(cfg.TROCR_MODEL)
    _fix_special_tokens(ved, processor, "ViT")
    m = MultiHead(ved, n_writers).to(DEVICE)
    n_par = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"[model] ViT arm | {n_par:.1f}M params | writers={n_writers}")
    return m


def build_convnext(n_writers, processor, cfg=C):
    ved = VisionEncoderDecoderModel.from_pretrained(cfg.TROCR_MODEL)
    _fix_special_tokens(ved, processor, "ConvNeXt")
    hidden_size = ved.config.encoder.hidden_size
    # ved.encoder kept but never called -- see the Bangla study's note: HF's
    # forward() unconditionally checks self.encoder.config.hidden_size even
    # when encoder_outputs is already supplied.
    m = ConvNeXtMultiHead(ved, n_writers, hidden_size).to(DEVICE)
    n_par = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"[model] ConvNeXt arm | {n_par:.1f}M params | writers={n_writers}")
    return m


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
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
                              n_batches=None if not CROSSLANG_SMOKE_TEST else 5)
        row = {"tag": tag, "epoch": ep + 1, "train_rec_loss": agg_rec / n,
               "train_wid_loss": agg_wid / n, "train_wacc": 100 * agg_wacc / n,
               "val_wacc": val_wacc}
        rows.append(row)
        print(f"  [{tag}] epoch {ep+1}: rec_loss {row['train_rec_loss']:.3f}  "
              f"train_wacc {row['train_wacc']:6.2f}%  val_wacc {val_wacc:6.2f}%")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"device: {DEVICE}\n")
    pool, writers = load_pool()
    w2i = {w: i for i, w in enumerate(writers)}
    n_writers = len(writers)
    chance = 100.0 / n_writers
    print(f"[data] chance level = {chance:.2f}% ({n_writers} writers)\n")

    n = len(pool)
    perm = np.random.RandomState(C.SEED).permutation(n)
    n_val = max(1, int(n * C.VAL_FRAC))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if C.TRAIN_SUBSAMPLE is not None:
        train_idx = train_idx[:C.TRAIN_SUBSAMPLE]
    print(f"[split] train={len(train_idx):,}  val={len(val_idx):,}\n")

    processor = TrOCRProcessor.from_pretrained(C.TROCR_MODEL)

    train_ds = WDataset(pool, train_idx, processor, w2i)
    val_ds = WDataset(pool, val_idx, processor, w2i)
    train_loader = DataLoader(train_ds, batch_size=C.BATCH, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH, shuffle=False,
                            num_workers=C.NUM_WORKERS)

    all_rows = []

    vit = build_vit(n_writers, processor)
    all_rows += train_arm(vit, train_loader, val_loader, is_convnext=False, tag="ViT")
    vit_final = all_rows[-1]["val_wacc"]
    del vit; torch.cuda.empty_cache()

    convnext = build_convnext(n_writers, processor)
    all_rows += train_arm(convnext, train_loader, val_loader, is_convnext=True, tag="ConvNeXt")
    convnext_final = all_rows[-1]["val_wacc"]
    del convnext; torch.cuda.empty_cache()

    import pandas as pd
    pd.DataFrame(all_rows).to_csv(f"{C.OUT}/crosslang_curves.csv", index=False)

    smoke_tag = "SMOKE TEST" if CROSSLANG_SMOKE_TEST else "REAL RUN"
    print("\n" + "#" * 90)
    print(f"#  FINAL RESULT -- Cross-dataset validation on {C.HF_DATASET} ({smoke_tag})")
    print(f"#  English, {n_writers} writers, chance = {chance:.2f}%")
    print("#" * 90)
    print(f"#  ViT       final val writer-ID acc.: {vit_final:6.2f}%")
    print(f"#  ConvNeXt  final val writer-ID acc.: {convnext_final:6.2f}%")
    print("#" * 90)
    if vit_final > 5 * chance and convnext_final < 2 * chance:
        print("#  RESULT: dissociation REPLICATES on English/IBM-campaign data --")
        print("#  ViT learns writer identity, ConvNeXt does not. Consistent with a")
        print("#  general architectural property rather than a Bangla-specific one.")
    elif vit_final > 5 * chance and convnext_final > 5 * chance:
        print("#  RESULT: dissociation does NOT replicate -- ConvNeXt learns writer")
        print("#  identity here. The Bangla result may be dataset- or script-specific")
        print("#  after all (or an adapter issue specific to that run's conditions).")
    else:
        print("#  RESULT: inconclusive by this simple threshold -- read the per-epoch")
        print("#  table above/CSV directly.")
    print("#" * 90 + "\n")
    return all_rows


if __name__ == "__main__":
    main()
