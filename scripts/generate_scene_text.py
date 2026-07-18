"""Synthetic Bangla scene text generation for Phase-1 scene training.

Renders words from a corpus with random Bangla fonts, colours, backgrounds,
perspective distortion, blur, and noise to imitate signboards and posters.

    python scripts/generate_scene_text.py --corpus data/words.txt \
        --fonts-dir assets/fonts --out data/synth_scene --num-images 50000
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


def random_color(low=0, high=255):
    return tuple(random.randint(low, high) for _ in range(3))


def contrasting_colors():
    """Background/foreground pair with enough luminance gap to stay legible."""
    while True:
        bg, fg = random_color(), random_color()
        lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        if abs(lum(bg) - lum(fg)) > 90:
            return bg, fg


def render_word(word, font_path):
    font = ImageFont.truetype(str(font_path), random.randint(32, 72))
    bg, fg = contrasting_colors()

    l, t, r, b = font.getbbox(word)
    pad = random.randint(8, 30)
    img = Image.new("RGB", (r - l + 2 * pad, b - t + 2 * pad), bg)
    draw = ImageDraw.Draw(img)

    if random.random() < 0.3:  # outline, common on signboards
        draw.text((pad - l, pad - t), word, font=font, fill=fg,
                  stroke_width=random.randint(1, 3),
                  stroke_fill=random_color())
    else:
        draw.text((pad - l, pad - t), word, font=font, fill=fg)
    return img


def distort(img):
    w, h = img.size
    # random perspective
    if random.random() < 0.6:
        d = 0.12
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [(x + random.uniform(-d, d) * w, y + random.uniform(-d, d) * h)
               for x, y in src]
        coeffs = _perspective_coeffs(dst, src)
        img = img.transform((w, h), Image.PERSPECTIVE, coeffs,
                            Image.BICUBIC, fillcolor=random_color())
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-6, 6), expand=True,
                         fillcolor=random_color())
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
    if random.random() < 0.3:  # sensor noise
        arr = np.asarray(img, dtype=np.int16)
        arr = np.clip(arr + np.random.normal(0, 12, arr.shape), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
    return img


def _perspective_coeffs(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    B = np.array([c for pt in dst for c in pt], dtype=np.float64)
    return np.linalg.solve(np.array(A, dtype=np.float64), B).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="text file with one Bangla word per line")
    ap.add_argument("--fonts-dir", required=True,
                    help="directory of Bangla .ttf fonts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-images", type=int, default=50000)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    words = [w.strip() for w in
             Path(args.corpus).read_text(encoding="utf-8").splitlines()
             if w.strip()]
    fonts = sorted(Path(args.fonts_dir).glob("*.ttf")) \
        + sorted(Path(args.fonts_dir).glob("*.otf"))
    if not words or not fonts:
        raise SystemExit("empty corpus or no fonts found")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "split"])
        for i in range(args.num_images):
            word = random.choice(words)
            try:
                img = distort(render_word(word, random.choice(fonts)))
            except OSError:  # font missing glyphs for this word
                continue
            name = f"synth_{i:07d}.png"
            img.save(out / "images" / name)
            split = "val" if random.random() < args.val_fraction else "train"
            writer.writerow([name, word, split])
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{args.num_images}")

    print(f"done -> {out}")


if __name__ == "__main__":
    main()
