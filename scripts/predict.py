"""Recognize Bangla text in a single image or a folder of images.

    python scripts/predict.py --checkpoint runs/trocr_grapheme/best.pt --image sample.png
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathok.config import ModelConfig, TrainConfig
from pathok.dataset import build_transform
from pathok.models.pathok_model import PathokModel
from pathok.tokenizers import BaseTokenizer

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True, help="image file or directory")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    mcfg = ModelConfig(**ckpt["model_cfg"])
    tokenizer = BaseTokenizer.load(Path(args.checkpoint).parent / "tokenizer.json")

    model = PathokModel(mcfg, len(tokenizer), tokenizer.pad_id,
                        tokenizer.bos_id, tokenizer.eos_id).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = build_transform(
        TrainConfig().resolved_image_size(mcfg.encoder), train=False)

    path = Path(args.image)
    files = ([path] if path.is_file()
             else sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS))

    with torch.no_grad():
        for f in files:
            img = transform(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
            seq = model.generate(img, max_len=mcfg.max_seq_len)[0]
            print(f"{f.name}\t{tokenizer.decode(seq.tolist())}")


if __name__ == "__main__":
    main()
