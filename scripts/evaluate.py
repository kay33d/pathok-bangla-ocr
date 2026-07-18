"""Evaluate a trained checkpoint on a dataset split.

    python scripts/evaluate.py --checkpoint runs/trocr_grapheme/best.pt \
        --data-root data/bnhtrd --split test
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
from pathok.dataset import WordImageDataset, collate
from pathok.metrics import cer, wer, word_accuracy
from pathok.models.pathok_model import PathokModel
from pathok.tokenizers import BaseTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="data/bnhtrd")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dump-errors", default=None,
                    help="optional TSV path for misrecognised samples")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    mcfg = ModelConfig(**ckpt["model_cfg"])
    tokenizer = BaseTokenizer.load(Path(args.checkpoint).parent / "tokenizer.json")

    model = PathokModel(mcfg, len(tokenizer), tokenizer.pad_id,
                        tokenizer.bos_id, tokenizer.eos_id).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    size = TrainConfig().resolved_image_size(mcfg.encoder)
    ds = WordImageDataset(args.data_root, args.split, tokenizer, size,
                          mcfg.max_seq_len, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=partial(collate, pad_id=tokenizer.pad_id))

    preds, refs = [], []
    with torch.no_grad():
        for imgs, _, labels in tqdm(loader, desc=args.split):
            out = model.generate(imgs.to(device), max_len=mcfg.max_seq_len)
            preds += [tokenizer.decode(s.tolist()) for s in out]
            refs += labels

    print(f"samples:       {len(refs)}")
    print(f"word accuracy: {word_accuracy(preds, refs) * 100:.2f}%")
    print(f"CER:           {cer(preds, refs) * 100:.2f}%")
    print(f"WER:           {wer(preds, refs) * 100:.2f}%")

    if args.dump_errors:
        with open(args.dump_errors, "w", encoding="utf-8") as f:
            f.write("reference\tprediction\n")
            for p, r in zip(preds, refs):
                if p != r:
                    f.write(f"{r}\t{p}\n")
        print(f"errors written to {args.dump_errors}")


if __name__ == "__main__":
    main()
