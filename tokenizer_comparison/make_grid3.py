import math
import os

from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "analysis_out")
IMG_DIR = os.path.join(BASE, "grid_images3")

BN_FONT_PATH = r"C:\Windows\Fonts\Nirmala.ttc"
font_gt = ImageFont.truetype(BN_FONT_PATH, 21, index=1)
font_pr = ImageFont.truetype(BN_FONT_PATH, 19)
font_sm = ImageFont.truetype(BN_FONT_PATH, 14)

# (filename, ground_truth, bpe_pred, char_pred, grapheme_pred)
rows = [
    ("43_5_11_8.JPG", "রাষ্ট্রবিজ্ঞানীরা", "রাষ্ট্রবিানীরা", "রাষ্টবিডানীরা", "রাষ্ট্রবিজ্ঞানীরা"),
    ("78_8_19_4.JPG", "সত্যেন্দ্রনাথ", "সত্য্যপ্র্রনা", "সত্যেন্দ্রণাথ", "সত্যেন্দ্রনাথ"),
    ("67_1_6_4.JPG", "গুরুত্বপূর্ণ", "হুরুবূর্ণ", "পুরুত্বর্ণ", "গুরুত্বপূর্ণ"),
    ("9_2_10_5.JPG", "অস্ত্রশক্তি", "অস্তক্রক্তি", "অস্তশক্তি", "অস্ত্রশক্তি"),
    ("81_3_16_3.JPG", "চ্যালেঞ্জের", "চ্যালেঞ্ঙ্গার", "চ্যালেঞ্জোর", "চ্যালেঞ্জের"),
    ("46_6_1_5.JPG", "সেপ্টেম্বর", "সেপটেম্বর", "সেস্টেম্বর", "সেপ্টেম্বর"),
    ("19_2_15_7.jpg", "কর্মকান্ড", "কর্মকাপ্ত", "কর্মকাণ্ড", "কর্মকান্ড"),
    ("50_1_4_1.JPG", "ইংল্যান্ডে", "ইংল্যান্ড", "ইংল্যাভে", "ইংল্যান্ডে"),
    ("107_2_8_4.jpg", "সিদ্ধান্তে", "সিদ্ধানেে", "মিদ্ধান্তে", "সিদ্ধান্তে"),
]

COLS = 3
CELL_W, IMG_H, TEXT_H = 380, 90, 122
CELL_H = IMG_H + TEXT_H
PAD = 10
GT_COL = (30, 30, 30)
BPE_COL = (176, 46, 46)
CHAR_COL = (40, 75, 130)
GRA_COL = (33, 122, 74)


def make_cell(fname, gt, bpe, ch, gra):
    cell = Image.new("RGB", (CELL_W, CELL_H), (250, 250, 248))
    draw = ImageDraw.Draw(cell)
    raw = Image.open(os.path.join(IMG_DIR, fname)).convert("L")
    scale = min(IMG_H / raw.height, (CELL_W - 2 * PAD) / raw.width)
    rw, rh = int(raw.width * scale), int(raw.height * scale)
    raw = raw.resize((rw, rh)).convert("RGB")
    cell.paste(raw, ((CELL_W - rw) // 2, (IMG_H - rh) // 2 + 4))
    draw.line([(0, IMG_H + 6), (CELL_W, IMG_H + 6)], fill=(210, 210, 210), width=1)

    y = IMG_H + 11
    for tag, val, col in (("GT", gt, GT_COL), ("BPE", bpe, BPE_COL),
                           ("CHR", ch, CHAR_COL), ("GRA", gra, GRA_COL)):
        draw.text((PAD, y), tag, font=font_sm, fill=(120, 120, 120))
        draw.text((PAD + 36, y - 3), val, font=(font_gt if tag == "GT" else font_pr), fill=col)
        y += 25

    draw.rectangle([0, 0, CELL_W - 1, CELL_H - 1], outline=(200, 200, 200), width=1)
    return cell


cells = [make_cell(*r) for r in rows]
ROWS = math.ceil(len(cells) / COLS)
grid = Image.new("RGB", (CELL_W * COLS + PAD, CELL_H * ROWS + PAD), (235, 235, 232))
for idx, cell in enumerate(cells):
    r, c = divmod(idx, COLS)
    grid.paste(cell, (c * CELL_W + PAD // 2, r * CELL_H + PAD // 2))

header = Image.new("RGB", (grid.width, 60), (245, 245, 242))
hd = ImageDraw.Draw(header)
hd.text((16, 8), "Triple divergence: Grapheme correct, BOTH BPE and Char wrong",
         font=ImageFont.truetype(BN_FONT_PATH, 19), fill=(20, 20, 20))
hd.text((16, 34), "black = ground truth  |  red = BPE  |  blue = Char  |  green = Grapheme",
         font=font_sm, fill=(90, 90, 90))

final = Image.new("RGB", (grid.width, grid.height + header.height), (255, 255, 255))
final.paste(header, (0, 0))
final.paste(grid, (0, header.height))
final.save(os.path.join(OUT, "fig_triple_divergence_grid.png"), dpi=(180, 180))
print("saved fig_triple_divergence_grid.png", final.size)
