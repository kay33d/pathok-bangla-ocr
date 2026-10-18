# ══════════════════════════════════════════════════════════════════════════════
#  Bangla Handwritten Word Recognition — EVALUATION ONLY
#  Loads from a saved checkpoint; no training.
#
#  Usage:
#    1. Set CHECKPOINT_DIR to your best checkpoint folder, e.g.
#         /kaggle/input/notebooks/kayed007/vitocr-char/trocr_char/checkpoint-9924
#    2. Set VOCAB_PATH to char_vocab.json
#    3. Run.  Everything else (dataset path, splits) is identical to training.
# ══════════════════════════════════════════════════════════════════════════════

# !pip install transformers>=4.35 editdistance -q

import os, glob, json, random, math, warnings
from collections import Counter
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.manifold import TSNE
import editdistance
from tqdm import tqdm
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, AutoImageProcessor
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════════════════
#  ▶▶  CONFIGURE THESE PATHS  ◀◀
# ══════════════════════════════════════════════════════════════════════════════
# Pick whichever checkpoint you want to evaluate.
# checkpoint-9924 is the later one; use checkpoint-8270 if that was the best.
CHECKPOINT_DIR = (
    "/kaggle/input/notebooks/kayed007/vitocr-char/trocr_char/checkpoint-9924"
)
VOCAB_PATH = (
    "/kaggle/input/notebooks/kayed007/vitocr-char/trocr_char/char_vocab.json"
)
# ══════════════════════════════════════════════════════════════════════════════

ROOT        = "/kaggle/input/datasets/kaidul007/bnhtrd/Dataset"
PUNCT       = {'.', ':', '।', ',', ';', '!', '?', '-', '–', '—'}
RESULTS_CSV = '/kaggle/working/test_results_trocr_char.csv'
PLOT_DIR    = '/kaggle/working/plots_trocr_char'

# ── Special token ids (must match training) ───────────────────────────────────
PAD_ID         = 0
UNK_ID         = 1
BOS_ID         = 2
EOS_ID         = 3
SPECIAL_TOKENS = 4
MAX_LEN        = 64


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    TROCR_MODEL = 'microsoft/trocr-base-handwritten'
    NUM_WORKERS = 1
    TEST_SPLIT  = 0.10
    VAL_SPLIT   = 0.10
    SEED        = 42
    NUM_BEAMS   = 2
    INFER_BATCH = 32

random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
os.makedirs(PLOT_DIR, exist_ok=True)

device = (torch.accelerator.current_accelerator().type
          if torch.accelerator.is_available() else 'cpu')
print(f"Using device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (identical to training script)
# ══════════════════════════════════════════════════════════════════════════════
def is_bangla_word(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    bangla = sum(1 for c in chars if '\u0980' <= c <= '\u09FF')
    return bangla / len(chars) > 0.8


def load_dataset(root: str) -> pd.DataFrame:
    label_dfs = []
    for file in glob.glob(root + "/*/*.xlsx"):
        try:
            sheets = pd.read_excel(file, sheet_name=None, engine="openpyxl")
            for df in sheets.values():
                if df.shape[1] < 2:
                    continue
                df = df.iloc[:, :2].copy()
                df.columns = ["Id", "Word"]
                label_dfs.append(df)
        except Exception:
            continue

    labels_df = pd.concat(label_dfs, ignore_index=True)
    images    = glob.glob(root + "/*/Words/*/*.jpg")
    images   += glob.glob(root + "/*/Words/*/*.JPG")

    image_df = pd.DataFrame({
        "Id":         [os.path.basename(p).split('.')[0] for p in images],
        "image_path": images,
    })

    df = (image_df
          .merge(labels_df, on="Id", how="inner")
          .rename(columns={"Word": "label"})[["image_path", "label"]]
          .dropna()
          .reset_index(drop=True))

    df['label'] = df['label'].astype(str).str.strip()
    df = df[df['label'] != '']
    df = df[~df['label'].isin(PUNCT)]
    df = df[df['label'].apply(is_bangla_word)]
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"Total usable samples: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CHARACTER VOCABULARY
# ══════════════════════════════════════════════════════════════════════════════
class CharVocab:
    def __init__(self):
        self.c2i: dict = {}
        self.i2c: dict = {
            PAD_ID: '<pad>', UNK_ID: '<unk>',
            BOS_ID: '<s>',   EOS_ID: '</s>',
        }
        self.vocab_size: int = SPECIAL_TOKENS

    @classmethod
    def load(cls, path: str) -> 'CharVocab':
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        v = cls()
        v.c2i        = d['c2i']
        v.i2c        = {int(k): val for k, val in d['i2c'].items()}
        v.vocab_size = d['vocab_size']
        print(f"Loaded vocab: {v.vocab_size} tokens  "
              f"({v.vocab_size - SPECIAL_TOKENS} characters)")
        return v

    def encode(self, word: str) -> list:
        return [self.c2i.get(ch, UNK_ID) for ch in str(word)]

    def decode(self, ids) -> str:
        out = []
        for t in ids:
            t = int(t)
            if t == EOS_ID:
                break
            if t in (PAD_ID, BOS_ID, UNK_ID):
                continue
            out.append(self.i2c.get(t, ''))
        return ''.join(out)


def encode_label(vocab: CharVocab, text: str) -> list:
    ids = vocab.encode(text)
    seq = ([BOS_ID] + ids + [EOS_ID])[:MAX_LEN]
    seq += [PAD_ID] * (MAX_LEN - len(seq))
    return seq


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class TrOCRCharDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: CharVocab, feature_extractor):
        self.df                = df.reset_index(drop=True)
        self.vocab             = vocab
        self.feature_extractor = feature_extractor

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.df.iloc[idx, 0])
        if img_bgr is None:
            img_rgb = Image.new('RGB', (384, 384), color=255)
        else:
            img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pixel_values = self.feature_extractor(
            img_rgb, return_tensors='pt'
        ).pixel_values.squeeze(0)
        label_ids = encode_label(self.vocab, str(self.df.iloc[idx, 1]))
        labels    = torch.tensor(label_ids, dtype=torch.long)
        labels[labels == PAD_ID] = -100
        return {'pixel_values': pixel_values, 'labels': labels}

    def get_path(self, idx):  return self.df.iloc[idx, 0]
    def get_label(self, idx): return str(self.df.iloc[idx, 1])


def infer_collate_fn(batch):
    return torch.stack([item['pixel_values'] for item in batch])


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def cer_single(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return editdistance.eval(list(pred), list(gt)) / len(gt)


def corpus_cer_wer(preds, gts):
    total_dist = total_chars = total_word_err = 0
    for p, g in zip(preds, gts):
        total_dist     += editdistance.eval(list(p), list(g))
        total_chars    += max(len(g), 1)
        total_word_err += (0 if p == g else 1)
    return (total_dist / max(total_chars, 1),
            total_word_err / max(len(gts), 1))


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_test_set(model, test_dataset: TrOCRCharDataset,
                      vocab: CharVocab) -> pd.DataFrame:
    model.eval()
    model.to(device)

    infer_loader = DataLoader(
        test_dataset,
        batch_size  = Config.INFER_BATCH,
        shuffle     = False,
        collate_fn  = infer_collate_fn,
        num_workers = Config.NUM_WORKERS,
        pin_memory  = (device != 'cpu'),
    )

    all_preds, all_gts, all_paths = [], [], []
    sample_idx = 0

    print("\nRunning batched inference on test set …")
    for pixel_values in tqdm(infer_loader, dynamic_ncols=True):
        pixel_values = pixel_values.to(device)
        batch_size   = pixel_values.size(0)
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens = MAX_LEN,
                num_beams      = Config.NUM_BEAMS,
                early_stopping = True,
            )
        for i in range(batch_size):
            pred = vocab.decode(generated_ids[i].cpu().tolist())
            gt   = test_dataset.get_label(sample_idx)
            path = test_dataset.get_path(sample_idx)
            all_preds.append(pred)
            all_gts.append(gt)
            all_paths.append(path)
            sample_idx += 1

    cer, wer = corpus_cer_wer(all_preds, all_gts)

    print(f"\n{'═'*55}")
    print(f"  EXPERIMENT  : TrOCR ViT  +  Character Tokenizer")
    print(f"  Checkpoint  : {CHECKPOINT_DIR}")
    print(f"  Test samples: {len(all_gts):,}")
    print(f"  CER         : {cer  * 100:.2f} %")
    print(f"  WER         : {wer  * 100:.2f} %")
    print(f"  Word Acc    : {(1 - wer) * 100:.2f} %")
    print(f"{'═'*55}")

    results_df = pd.DataFrame({
        'image_path':   all_paths,
        'ground_truth': all_gts,
        'prediction':   all_preds,
        'cer':          [cer_single(p, g) for p, g in zip(all_preds, all_gts)],
        'correct':      [p == g           for p, g in zip(all_preds, all_gts)],
    })
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Results saved → {RESULTS_CSV}\n")

    print(f"{'Ground Truth':<30} {'Prediction':<30} {'CER':>7}  OK")
    print("─" * 75)
    for _, row in results_df.head(20).iterrows():
        mark = "✓" if row['correct'] else "✗"
        print(f"{row['ground_truth']:<30} {row['prediction']:<30}"
              f" {row['cer']*100:>6.1f}%  {mark}")

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS  (all unchanged from training script)
# ══════════════════════════════════════════════════════════════════════════════
def plot_cer_distribution(results_df: pd.DataFrame, save_dir=PLOT_DIR):
    cer_vals = results_df['cer'].values * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f0f14')

    ax = axes[0]
    ax.set_facecolor('#1a1a24')
    bins = np.linspace(0, 100, 30)
    counts, edges, patches = ax.hist(cer_vals, bins=bins, edgecolor='none')
    for patch, left_edge in zip(patches, edges[:-1]):
        if left_edge == 0:   patch.set_facecolor('#2ecc71')
        elif left_edge < 20: patch.set_facecolor('#f39c12')
        else:                patch.set_facecolor('#e74c3c')
    ax.axvline(np.mean(cer_vals), color='white', linestyle='--',
               linewidth=1.5, label=f'Mean {np.mean(cer_vals):.1f}%')
    ax.axvline(np.median(cer_vals), color='#00d4ff', linestyle=':',
               linewidth=1.5, label=f'Median {np.median(cer_vals):.1f}%')
    ax.legend(facecolor='#1a1a24', labelcolor='white', fontsize=9)
    ax.set_xlabel('CER (%)', color='#aaaaaa')
    ax.set_ylabel('Count',   color='#aaaaaa')
    ax.set_title('CER Distribution', color='white', fontweight='bold')
    ax.tick_params(colors='#aaaaaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#333344')

    ax2 = axes[1]
    ax2.set_facecolor('#1a1a24')
    sorted_cer = np.sort(cer_vals)
    cumulative = np.arange(1, len(sorted_cer) + 1) / len(sorted_cer) * 100
    ax2.plot(sorted_cer, cumulative, color='#00d4ff', linewidth=2.5)
    ax2.fill_between(sorted_cer, cumulative, alpha=0.1, color='#00d4ff')
    for threshold, clr in [(0, '#2ecc71'), (10, '#f39c12'), (25, '#e74c3c')]:
        pct = np.interp(threshold, sorted_cer, cumulative)
        ax2.axvline(threshold, color=clr, linestyle='--', alpha=0.6, linewidth=1)
        ax2.text(threshold + 1, 5, f'{pct:.1f}%\n≤{threshold}%CER',
                 color=clr, fontsize=8)
    ax2.set_xlabel('CER (%)',                   color='#aaaaaa')
    ax2.set_ylabel('Cumulative % of samples',   color='#aaaaaa')
    ax2.set_title('Cumulative Accuracy vs CER', color='white', fontweight='bold')
    ax2.tick_params(colors='#aaaaaa')
    for sp in ax2.spines.values(): sp.set_edgecolor('#333344')
    ax2.grid(color='#333344', linewidth=0.5, alpha=0.6)

    fig.suptitle('TrOCR ViT + Character Tokenizer — CER Distribution',
                 color='white', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(pad=2)
    path = f"{save_dir}/cer_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close()
    print(f"Saved → {path}")


def plot_error_analysis(results_df: pd.DataFrame, save_dir=PLOT_DIR):
    df = results_df.copy()
    df['gt_len'] = df['ground_truth'].apply(lambda w: len(str(w)))

    buckets = [(1, 3), (4, 6), (7, 9), (10, 13), (14, 99)]
    labels_b, accs, counts = [], [], []
    for lo, hi in buckets:
        sub = df[(df['gt_len'] >= lo) & (df['gt_len'] <= hi)]
        if len(sub) == 0: continue
        lab = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
        labels_b.append(lab)
        accs.append(sub['correct'].mean() * 100)
        counts.append(len(sub))

    char_pairs = Counter()
    for _, row in df[~df['correct']].iterrows():
        for g, p in zip(list(str(row['ground_truth'])),
                        list(str(row['prediction']))):
            if g != p:
                char_pairs[(g, p)] += 1

    fig = plt.figure(figsize=(16, 6), facecolor='#0f0f14')
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1a1a24')
    bar_colors = ['#2ecc71' if a >= 70 else '#f39c12' if a >= 40
                  else '#e74c3c' for a in accs]
    bars = ax1.bar(labels_b, accs, color=bar_colors, edgecolor='none', width=0.6)
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'n={cnt:,}', ha='center', va='bottom',
                 color='#aaaaaa', fontsize=8)
    ax1.set_xlabel('Word length (characters)', color='#aaaaaa')
    ax1.set_ylabel('Word Accuracy (%)',         color='#aaaaaa')
    ax1.set_title('Accuracy by Character Count', color='white', fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.tick_params(colors='#aaaaaa')
    for sp in ax1.spines.values(): sp.set_edgecolor('#333344')
    ax1.grid(axis='y', color='#333344', linewidth=0.5, alpha=0.6)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#1a1a24')
    top_pairs = char_pairs.most_common(15)
    if top_pairs:
        pair_labels = [f"{g}→{p}" for (g, p), _ in top_pairs]
        pair_counts = [c for _, c in top_pairs]
        y_pos = range(len(pair_labels))
        ax2.barh(list(y_pos), pair_counts, color='#ff6b6b',
                 edgecolor='none', height=0.6)
        ax2.set_yticks(list(y_pos))
        ax2.set_yticklabels(pair_labels, color='white', fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('Frequency', color='#aaaaaa')
        ax2.set_title('Top 15 Confused Character Pairs\n(GT → Prediction)',
                      color='white', fontweight='bold')
        ax2.tick_params(colors='#aaaaaa')
        for sp in ax2.spines.values(): sp.set_edgecolor('#333344')
        ax2.grid(axis='x', color='#333344', linewidth=0.5, alpha=0.6)

    fig.suptitle('TrOCR ViT + Character — Error Analysis',
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    path = f"{save_dir}/error_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close()
    print(f"Saved → {path}")


def plot_char_heatmap(results_df: pd.DataFrame, top_n=30, save_dir=PLOT_DIR):
    gt_chars, pr_chars = [], []
    for _, row in results_df.iterrows():
        for g, p in zip(str(row['ground_truth']), str(row['prediction'])):
            gt_chars.append(g); pr_chars.append(p)

    counter = Counter(gt_chars)
    top_chs = [c for c, _ in counter.most_common(top_n)]
    mask    = [c in top_chs for c in gt_chars]
    gt_f    = [g for g, m in zip(gt_chars, mask) if m]
    pr_f    = [p for p, m in zip(pr_chars, mask) if m]

    from sklearn.metrics import confusion_matrix
    label_set = sorted(set(top_chs))
    l2i   = {l: i for i, l in enumerate(label_set)}
    gt_idx = [l2i[g] for g in gt_f]
    pr_idx = [l2i.get(p, -1) for p in pr_f]
    valid  = [(g, p) for g, p in zip(gt_idx, pr_idx) if p >= 0]
    if not valid:
        print("Insufficient data for confusion heatmap — skipping"); return

    gt_v, pr_v = zip(*valid)
    cm     = confusion_matrix(gt_v, pr_v, labels=list(range(len(label_set))))
    row_s  = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_s, where=row_s != 0,
                        out=np.zeros_like(cm, dtype=float))

    fig, ax = plt.subplots(figsize=(14, 12), facecolor='#0f0f14')
    ax.set_facecolor('#0f0f14')
    sns.heatmap(cm_norm, ax=ax, xticklabels=label_set, yticklabels=label_set,
                cmap='Blues', annot=False, linewidths=0.3, linecolor='#222233',
                cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted Character',    color='#aaaaaa', fontsize=11)
    ax.set_ylabel('Ground Truth Character', color='#aaaaaa', fontsize=11)
    ax.set_title(f'Character Confusion Matrix  (top {top_n} chars, row-normalised)',
                 color='white', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='white', labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.figure.axes[-1].tick_params(colors='white')

    path = f"{save_dir}/char_confusion_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close()
    print(f"Saved → {path}")


def plot_tsne(model, test_dataset: TrOCRCharDataset,
              feature_extractor, n_samples=600, save_dir=PLOT_DIR):
    model.eval()
    model.to(device)
    idxs = random.sample(range(len(test_dataset)),
                         min(n_samples, len(test_dataset)))
    feats, lengths = [], []

    with torch.no_grad():
        for idx in tqdm(idxs, desc="Extracting ViT features for t-SNE",
                        dynamic_ncols=True):
            path    = test_dataset.get_path(idx)
            gt      = test_dataset.get_label(idx)
            img_bgr = cv2.imread(path)
            if img_bgr is None: continue
            img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            pv      = feature_extractor(img_rgb, return_tensors='pt').pixel_values.to(device)
            enc_out = model.encoder(pixel_values=pv)
            cls_feat = enc_out.last_hidden_state[:, 0, :].squeeze(0)
            feats.append(cls_feat.cpu().float().numpy())
            lengths.append(len(str(gt)))

    if not feats:
        print("No features extracted — skipping t-SNE"); return

    feats   = np.array(feats)
    lengths = np.array(lengths)
    print("Running t-SNE …")
    tsne  = TSNE(n_components=2, perplexity=30, n_iter=300,
                 random_state=Config.SEED)
    emb2d = tsne.fit_transform(feats)

    def bucket(l):
        if l <= 3:  return '1-3'
        if l <= 6:  return '4-6'
        if l <= 9:  return '7-9'
        if l <= 13: return '10-13'
        return '14+'

    buckets = [bucket(l) for l in lengths]
    palette = {'1-3': '#2ecc71', '4-6': '#3498db', '7-9': '#9b59b6',
               '10-13': '#f39c12', '14+': '#e74c3c'}

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f0f14')
    ax.set_facecolor('#1a1a24')
    for label, color in palette.items():
        mask = np.array(buckets) == label
        if mask.sum() == 0: continue
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   c=color, label=f'{label} chars',
                   alpha=0.65, s=20, edgecolors='none')
    ax.legend(facecolor='#1a1a24', labelcolor='white',
              title='Word length (chars)', title_fontsize=9,
              fontsize=9, loc='upper right')
    ax.set_title('TrOCR ViT + Character — t-SNE of ViT CLS features',
                 color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#aaaaaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#333344')

    path = f"{save_dir}/tsne_features.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close()
    print(f"Saved → {path}")


def plot_error_grid(results_df: pd.DataFrame, max_errors=40, save_dir=PLOT_DIR):
    import subprocess
    subprocess.run(["apt-get", "install", "-y", "fonts-beng", "-qq"],
                   capture_output=True)

    def find_bangla_font():
        for pat in ["/usr/share/fonts/**/*Lohit*Bengali*",
                    "/usr/share/fonts/**/*Lohit*Beng*",
                    "/usr/share/fonts/**/*Bengali*"]:
            hits = glob.glob(pat, recursive=True)
            if hits: return hits[0]
        return None

    font_path = find_bangla_font()
    if font_path is None:
        print("Bengali font not found — skipping error grid"); return

    COLS   = 4; CELL_W = 320; IMG_H = 80; TEXT_H = 70
    CELL_H = IMG_H + TEXT_H
    font   = ImageFont.truetype(font_path, 18)
    sfont  = ImageFont.truetype(font_path, 13)

    errors = (results_df[~results_df['correct']]
              .head(max_errors).reset_index(drop=True))
    print(f"\nError grid: {len(errors)} samples")

    def make_cell(i, row):
        cell = Image.new("RGB", (CELL_W, CELL_H), (245, 245, 245))
        draw = ImageDraw.Draw(cell)
        raw  = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            scale  = min(IMG_H / raw.shape[0], (CELL_W - 10) / raw.shape[1])
            rh, rw = int(raw.shape[0] * scale), int(raw.shape[1] * scale)
            raw    = cv2.resize(raw, (rw, rh))
            pil    = Image.fromarray(raw).convert("RGB")
            cell.paste(pil, ((CELL_W - rw) // 2, (IMG_H - rh) // 2))
        draw.rectangle([0, 0, 26, 20], fill=(50, 50, 50))
        draw.text((4, 2), f"#{i+1}", font=sfont, fill=(255, 255, 255))
        border = ((220, 50, 50) if row['cer'] > 0.5
                  else (255, 165, 0) if row['cer'] > 0.2
                  else (50, 180, 50))
        draw.rectangle([0, 0, CELL_W - 1, CELL_H - 1],
                       outline=border, width=3)
        draw.line([(0, IMG_H), (CELL_W, IMG_H)], fill=(180, 180, 180), width=1)
        draw.text((6,  IMG_H + 4),  "GT:", font=sfont, fill=(30,  120, 30))
        draw.text((36, IMG_H + 2),  row['ground_truth'], font=font, fill=(20, 100, 20))
        draw.text((6,  IMG_H + 30), "PR:", font=sfont, fill=(160, 40, 40))
        draw.text((36, IMG_H + 28), row['prediction'],   font=font, fill=(140, 20, 20))
        draw.text((CELL_W - 44, IMG_H + 4),
                  f"{row['cer']*100:.0f}%", font=sfont, fill=(80, 80, 80))
        return cell

    cells = [make_cell(i, row) for i, row in errors.iterrows()]
    ROWS  = math.ceil(len(cells) / COLS)
    grid  = Image.new("RGB", (CELL_W * COLS, CELL_H * ROWS), (210, 210, 210))
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, COLS)
        grid.paste(cell, (c * CELL_W, r * CELL_H))
    path = f"{save_dir}/error_grid.png"
    grid.save(path, dpi=(150, 150))
    fig, ax = plt.subplots(figsize=(COLS * 3.8, ROWS * 2.4))
    ax.imshow(grid); ax.axis('off')
    ax.set_title("Erroneous Predictions  "
                 "(green <20% CER | orange 20-50% | red >50%)",
                 fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_grid_plot.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


def print_summary(results_df: pd.DataFrame, vocab: CharVocab):
    cer_vals = results_df['cer'].values
    wer      = 1 - results_df['correct'].mean()
    print(f"\n{'═'*60}")
    print(f"  EXPERIMENT  : TrOCR ViT  +  Character-level Tokenizer")
    print(f"  Checkpoint  : {os.path.basename(CHECKPOINT_DIR)}")
    print(f"{'─'*60}")
    print(f"  Character vocab size  : {vocab.vocab_size}")
    print(f"  Test samples          : {len(results_df):,}")
    print(f"  Word Accuracy         : {results_df['correct'].mean()*100:.2f} %")
    print(f"  Corpus CER            : {cer_vals.mean()*100:.2f} %")
    print(f"  Corpus WER            : {wer*100:.2f} %")
    print(f"  Median CER            : {np.median(cer_vals)*100:.2f} %")
    print(f"  CER = 0  (perfect)    : {(cer_vals==0).sum()/len(cer_vals)*100:.1f} %")
    print(f"  CER ≤ 0.1 (near-perf) : {(cer_vals<=0.1).sum()/len(cer_vals)*100:.1f} %")
    print(f"  CER > 0.5 (bad)       : {(cer_vals>0.5).sum()/len(cer_vals)*100:.1f} %")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── 1. Load vocab ─────────────────────────────────────────────────────────
    vocab = CharVocab.load(VOCAB_PATH)

    # ── 2. Load & split dataset (same seed/splits as training) ───────────────
    df = load_dataset(ROOT)
    train_val_df, test_df = train_test_split(
        df, test_size=Config.TEST_SPLIT, random_state=Config.SEED)
    val_frac = Config.VAL_SPLIT / (1.0 - Config.TEST_SPLIT)
    train_df, _ = train_test_split(
        train_val_df, test_size=val_frac, random_state=Config.SEED)
    test_df  = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    print(f"Test set: {len(test_df):,} samples")

    # ── 3. Image processor ───────────────────────────────────────────────────
    feature_extractor = AutoImageProcessor.from_pretrained(Config.TROCR_MODEL)

    # ── 4. Load model from checkpoint ────────────────────────────────────────
    print(f"\nLoading model from {CHECKPOINT_DIR} …")
    model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT_DIR)
    model.generation_config.decoder_start_token_id = BOS_ID
    model.generation_config.pad_token_id           = PAD_ID
    model.generation_config.eos_token_id           = EOS_ID
    model.generation_config.max_new_tokens          = MAX_LEN
    model.to(device)
    print("Model loaded ✓")

    # ── 5. Build test dataset ─────────────────────────────────────────────────
    test_dataset = TrOCRCharDataset(test_df, vocab, feature_extractor)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    results_df = evaluate_test_set(model, test_dataset, vocab)
    print_summary(results_df, vocab)

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print(f"\nGenerating analysis plots → {PLOT_DIR}/\n")
    plot_cer_distribution(results_df)
    plot_error_analysis(results_df)
    plot_char_heatmap(results_df, top_n=30)
    plot_tsne(model, test_dataset, feature_extractor, n_samples=600)
    plot_error_grid(results_df, max_errors=40)

    print("\nAll done.")