"""Train a PATHOK model: any encoder x tokenizer combination.

Example (best configuration from the thesis):
    python scripts/train.py --encoder trocr --tokenizer grapheme \
        --data-root data/bnhtrd --epochs 30
"""

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathok.config import ModelConfig, TrainConfig
from pathok.dataset import WordImageDataset, collate, read_labels
from pathok.metrics import cer, word_accuracy
from pathok.models.pathok_model import PathokModel
from pathok.tokenizers import build_tokenizer


def evaluate(model, loader, tokenizer, device, max_len):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for imgs, _, labels in loader:
            out = model.generate(imgs.to(device), max_len=max_len)
            preds += [tokenizer.decode(s.tolist()) for s in out]
            refs += labels
    return cer(preds, refs), word_accuracy(preds, refs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="trocr", choices=["trocr", "resnet18"])
    ap.add_argument("--tokenizer", default="grapheme",
                    choices=["grapheme", "char", "bpe"])
    ap.add_argument("--data-root", default="data/bnhtrd")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init-checkpoint", default=None,
                    help="warm-start weights (e.g. handwritten ckpt for scene phases)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    mcfg = ModelConfig(encoder=args.encoder, tokenizer=args.tokenizer)
    tcfg = TrainConfig(data_root=args.data_root, epochs=args.epochs,
                       batch_size=args.batch_size, lr=args.lr)
    torch.manual_seed(tcfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir or f"runs/{args.encoder}_{args.tokenizer}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_labels = [lbl for _, lbl in read_labels(args.data_root, "train")]
    tokenizer = build_tokenizer(args.tokenizer, train_labels)
    tokenizer.save(out_dir / "tokenizer.json")
    print(f"vocab size: {len(tokenizer)}")

    size = tcfg.resolved_image_size(args.encoder)
    coll = partial(collate, pad_id=tokenizer.pad_id)
    loaders = {}
    for split in ("train", "val"):
        ds = WordImageDataset(args.data_root, split, tokenizer, size,
                              mcfg.max_seq_len)
        loaders[split] = DataLoader(
            ds, batch_size=tcfg.batch_size, shuffle=split == "train",
            num_workers=tcfg.num_workers, collate_fn=coll, pin_memory=True)

    model = PathokModel(mcfg, len(tokenizer), tokenizer.pad_id,
                        tokenizer.bos_id, tokenizer.eos_id).to(device)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"initialised from {args.init_checkpoint}")

    # Pretrained TrOCR encoder gets a smaller LR than the fresh decoder.
    enc_params = list(model.encoder.parameters())
    dec_params = [p for n, p in model.named_parameters()
                  if not n.startswith("encoder.")]
    optim = torch.optim.AdamW(
        [{"params": enc_params, "lr": tcfg.encoder_lr},
         {"params": dec_params, "lr": tcfg.lr}],
        weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=[tcfg.encoder_lr, tcfg.lr],
        total_steps=tcfg.epochs * len(loaders["train"]),
        pct_start=min(0.3, tcfg.warmup_steps / (tcfg.epochs * len(loaders["train"]))))
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id, label_smoothing=tcfg.label_smoothing)

    best_cer = float("inf")
    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        running = 0.0
        for imgs, seqs, _ in tqdm(loaders["train"], desc=f"epoch {epoch}"):
            imgs, seqs = imgs.to(device), seqs.to(device)
            logits = model(imgs, seqs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                           seqs[:, 1:].reshape(-1))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            optim.step()
            sched.step()
            running += loss.item()

        val_cer, val_wa = evaluate(model, loaders["val"], tokenizer, device,
                                   mcfg.max_seq_len)
        print(f"epoch {epoch}: loss={running / len(loaders['train']):.4f} "
              f"val_cer={val_cer:.4f} val_word_acc={val_wa:.4f}")

        if val_cer < best_cer:
            best_cer = val_cer
            torch.save({"model": model.state_dict(), "model_cfg": vars(mcfg),
                        "epoch": epoch, "val_cer": val_cer},
                       out_dir / "best.pt")
            print(f"  saved best checkpoint (CER {val_cer:.4f})")


if __name__ == "__main__":
    main()
