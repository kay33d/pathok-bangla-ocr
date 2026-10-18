import math
import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "analysis_out")
IMG_DIR = os.path.join(BASE, "grid_images")

BN_FONT_PATH = r"C:\Windows\Fonts\Nirmala.ttc"
font_gt = ImageFont.truetype(BN_FONT_PATH, 22, index=1)   # bold face in the ttc
font_pr = ImageFont.truetype(BN_FONT_PATH, 20)
font_sm = ImageFont.truetype(BN_FONT_PATH, 15)

rows = [
    # (local filename, ground_truth, bpe_pred, grapheme_pred)
    ("19_1_10_7.jpg",   "প্রতিদ্বন্দ্বিতা",  "প্রতিদ্ব্বন্দ্বিতা",  "প্রতিদ্বন্দ্বিতা"),
    ("103_2_10_2.JPG",  "অন্তর্ভূক্ত",        "অন্তজুক্ত",            "অন্তর্ভূক্ত"),
    ("19_1_16_7.jpg",   "ঐক্যফ্রন্ট",         "ঐক্যব্রষ্ট",           "ঐক্যফ্রন্ট"),
    ("1_3_11_4.JPG",    "দ্বারপ্রান্তে",       "দারপপান্তে",           "দ্বারপ্রান্তে"),
    ("79_3_10_8.JPG",   "অ্যান্টিসেপটিক",     "অ্যানন্টিমেপপপটিক",    "অ্যান্টিসেপটিক"),
    ("100_5_3_1.JPG",   "ইন্টারন্যাশনাল",     "ইন্টারনশনাল",         "ইন্টারন্যাশনাল"),
    ("38_3_20_3.JPG",   "পারফরম্যান্স",       "পারফর্স্যান",          "পারফরম্যান্স"),
    ("78_3_6_1.JPG",    "পদার্থবিদ্যার",      "পদার্থবিদার",          "পদার্থবিদ্যার"),
    ("108_3_11_1.JPG",  "স্বেচ্ছাসেবীদের",    "স্বেচ্চ্াসবীদের",       "স্বেচ্ছাসেবীদের"),
    ("43_5_11_8.JPG",   "রাষ্ট্রবিজ্ঞানীরা",   "রাষ্ট্রবিানীরা",       "রাষ্ট্রবিজ্ঞানীরা"),
    ("71_2_19_3.jpg",   "ছাত্রছাত্রীদের",     "ছাত্রদানীদের",         "ছাত্রছাত্রীদের"),
    ("14_3_12_1.JPG",   "বিশ্বব্যাংকের",      "বিশ্বব্যংকের",         "বিশ্বব্যাংকের"),
]

COLS = 3
CELL_W, IMG_H, TEXT_H = 380, 90, 100
CELL_H = IMG_H + TEXT_H
PAD = 10
BG = (255, 255, 255)
GT_COL = (30, 30, 30)
BPE_COL = (176, 46, 46)
GRA_COL = (33, 122, 74)


def make_cell(fname, gt, bpe, gra):
    cell = Image.new("RGB", (CELL_W, CELL_H), (250, 250, 248))
    draw = ImageDraw.Draw(cell)
    img_path = os.path.join(IMG_DIR, fname)
    raw = Image.open(img_path).convert("L")
    scale = min(IMG_H / raw.height, (CELL_W - 2 * PAD) / raw.width)
    rw, rh = int(raw.width * scale), int(raw.height * scale)
    raw = raw.resize((rw, rh)).convert("RGB")
    cell.paste(raw, ((CELL_W - rw) // 2, (IMG_H - rh) // 2 + 4))
    draw.line([(0, IMG_H + 6), (CELL_W, IMG_H + 6)], fill=(210, 210, 210), width=1)

    y = IMG_H + 12
    draw.text((PAD, y), "GT", font=font_sm, fill=(120, 120, 120))
    draw.text((PAD + 34, y - 3), gt, font=font_gt, fill=GT_COL)
    y += 27
    draw.text((PAD, y), "BPE", font=font_sm, fill=(120, 120, 120))
    draw.text((PAD + 34, y - 3), bpe, font=font_pr, fill=BPE_COL)
    y += 27
    draw.text((PAD, y), "GRA", font=font_sm, fill=(120, 120, 120))
    draw.text((PAD + 34, y - 3), gra, font=font_pr, fill=GRA_COL)

    draw.rectangle([0, 0, CELL_W - 1, CELL_H - 1], outline=(200, 200, 200), width=1)
    return cell


cells = [make_cell(*r) for r in rows]
ROWS = math.ceil(len(cells) / COLS)
grid = Image.new("RGB", (CELL_W * COLS + PAD, CELL_H * ROWS + PAD), (235, 235, 232))
for idx, cell in enumerate(cells):
    r, c = divmod(idx, COLS)
    grid.paste(cell, (c * CELL_W + PAD // 2, r * CELL_H + PAD // 2))

# header band
header = Image.new("RGB", (grid.width, 56), (245, 245, 242))
hd = ImageDraw.Draw(header)
hd.text((16, 8), "Grapheme correct, BPE wrong  --  handwritten input on top, three transcriptions below",
         font=ImageFont.truetype(BN_FONT_PATH, 18), fill=(20, 20, 20))
hd.text((16, 32), "black = ground truth   |   red = BPE prediction   |   green = Grapheme prediction",
         font=font_sm, fill=(90, 90, 90))

final = Image.new("RGB", (grid.width, grid.height + header.height), BG)
final.paste(header, (0, 0))
final.paste(grid, (0, header.height))
final.save(os.path.join(OUT, "fig_qualitative_grid.png"), dpi=(180, 180))
print("saved fig_qualitative_grid.png", final.size)
