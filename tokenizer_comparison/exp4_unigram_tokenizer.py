# ══════════════════════════════════════════════════════════════════════════════
#  Bangla Handwritten Word Recognition — Experiment 4 (tokenizer variants)
#  Full TrOCR fine-tune (microsoft/trocr-base-handwritten)
#  + SentencePiece UNIGRAM tokenizer trained on Bangla labels
#  Identical to the original grapheme/BPE/char comparison in every respect
#  (same row-level split protocol, same hyperparameters, same vocab size)
#  except model_type='unigram' instead of 'bpe' -- isolates the merge
#  algorithm as the single changed variable, per the reviewer's request to
#  test whether BPE's specific weaknesses are about greedy merging or about
#  subword tokenization in general.
#  Full pipeline: raw Excel/images → tokenizer → training → evaluation → plots
# ══════════════════════════════════════════════════════════════════════════════

# ── Install extra deps if needed ──────────────────────────────────────────────
# !pip install sentencepiece transformers>=4.35 editdistance -q

# ── Imports ───────────────────────────────────────────────────────────────────
import os, glob, json, random, math, warnings, time
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
from sklearn.metrics import confusion_matrix
import editdistance
import sentencepiece as spm
from tqdm import tqdm
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────
# SentencePiece special token ids (set during training below)
PAD_ID  = 0
UNK_ID  = 1
BOS_ID  = 2
EOS_ID  = 3
MAX_LEN = 64     # generous — TrOCR handles longer sequences well

ROOT  = "/kaggle/input/datasets/kaidul007/bnhtrd/Dataset"
PUNCT = {'.', ':', '।', ',', ';', '!', '?', '-', '–', '—'}

SPM_PREFIX  = '/kaggle/working/bangla_spm_unigram'
SPM_MODEL   = SPM_PREFIX + '.model'
LABELS_FILE = '/kaggle/working/bangla_labels.txt'
CKPT_DIR    = '/kaggle/working/trocr_bangla_unigram'
RESULTS_CSV = '/kaggle/working/test_results_unigram.csv'
PLOT_DIR    = '/kaggle/working/plots_unigram'


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
UNIGRAM_SMOKE_TEST = False  # flip to False for the real run once this passes clean


class Config:
    # TrOCR base-handwritten uses 384×384 internally — the feature extractor
    # handles resizing, so we just pass PIL images.
    TROCR_MODEL     = 'microsoft/trocr-base-handwritten'
    SPM_VOCAB_SIZE  = 2000
    BATCH_SIZE      = 16
    EVAL_BATCH_SIZE = 16
    EPOCHS          = 1 if UNIGRAM_SMOKE_TEST else 5
    LR              = 5e-5
    WARMUP_STEPS    = 500
    WEIGHT_DECAY    = 0.01
    TEST_SPLIT      = 0.10
    VAL_SPLIT       = 0.10
    SEED            = 42
    FP16            = True   # set False if not on GPU

random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

device = (torch.accelerator.current_accelerator().type
          if torch.accelerator.is_available() else 'cpu')
print(f"Using device: {device}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def is_bangla_word(text: str) -> bool:
    """True if >80% of non-space chars are in the Bangla Unicode block."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    bangla = sum(1 for c in chars if '\u0980' <= c <= '\u09FF')
    return bangla / len(chars) > 0.8


def cer_single(pred: str, gt: str) -> float:
    if not gt: return 0.0 if not pred else 1.0
    return editdistance.eval(list(pred), list(gt)) / len(gt)


def corpus_cer_wer(preds, gts):
    total_dist, total_chars, total_word_err = 0, 0, 0
    for p, g in zip(preds, gts):
        total_dist     += editdistance.eval(list(p), list(g))
        total_chars    += max(len(g), 1)
        total_word_err += (0 if p == g else 1)
    return (total_dist / max(total_chars, 1),
            total_word_err / max(len(gts), 1))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
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

    # Clean labels
    df['label'] = df['label'].astype(str).str.strip()
    df = df[df['label'] != '']
    df = df[~df['label'].isin(PUNCT)]
    df = df[df['label'].apply(is_bangla_word)]   # drop mixed-script rows
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"Total usable samples: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SENTENCEPIECE TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════
def train_spm_tokenizer(train_labels: pd.Series) -> spm.SentencePieceProcessor:
    """Train a Unigram-LM tokenizer on the training labels and return a loaded
    processor. Only line changed from the BPE arm: model_type='unigram'
    instead of 'bpe' -- everything else (vocab size, character coverage,
    special-token ids, input labels, split) is held identical."""
    print("Training SentencePiece UNIGRAM tokenizer …")
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        for label in train_labels.astype(str):
            f.write(label.strip() + '\n')

    spm.SentencePieceTrainer.train(
        input            = LABELS_FILE,
        model_prefix     = SPM_PREFIX,
        vocab_size       = Config.SPM_VOCAB_SIZE,
        character_coverage = 1.0,     # must be 1.0 for Bangla Unicode coverage
        model_type       = 'unigram',
        pad_id           = PAD_ID,
        unk_id           = UNK_ID,
        bos_id           = BOS_ID,
        eos_id           = EOS_ID,
        pad_piece        = '<pad>',
        unk_piece        = '<unk>',
        bos_piece        = '<s>',
        eos_piece        = '</s>',
    )

    sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)
    print(f"SPM vocab size: {sp.vocab_size()}  |  saved → {SPM_MODEL}")
    return sp


def encode_label(sp: spm.SentencePieceProcessor, text: str) -> list:
    """Encode a label to token ids with BOS/EOS, padded to MAX_LEN."""
    ids = sp.encode(str(text).strip(), out_type=int)
    ids = ([BOS_ID] + ids + [EOS_ID])[:MAX_LEN]
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return ids


def decode_ids(sp: spm.SentencePieceProcessor, ids) -> str:
    """Decode token ids back to a string, stopping at EOS."""
    ids = [int(i) for i in ids]
    if EOS_ID in ids:
        ids = ids[:ids.index(EOS_ID)]
    ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID, UNK_ID)]
    return sp.decode(ids)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class TrOCRBanglaDataset(Dataset):
    """
    Wraps image paths + labels for TrOCR.

    The HuggingFace feature extractor (ViTFeatureExtractor / AutoFeatureExtractor)
    handles:
      - Converting BGR→RGB
      - Resizing to 384×384 (trocr-base-handwritten's native resolution)
      - Normalising with ImageNet mean/std
    So we only need to pass a PIL RGB image.

    Labels are encoded with the custom SentencePiece tokenizer.
    Padding positions are set to -100 so CrossEntropyLoss ignores them
    (standard HuggingFace convention).
    """

    def __init__(self, df: pd.DataFrame,
                 sp: spm.SentencePieceProcessor,
                 feature_extractor,
                 is_train: bool = False):
        self.df               = df.reset_index(drop=True)
        self.sp               = sp
        self.feature_extractor = feature_extractor
        self.is_train          = is_train

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        # ── Image ─────────────────────────────────────────────────────────────
        img_bgr = cv2.imread(self.df.iloc[idx, 0])
        if img_bgr is None:
            # Return a blank image rather than crashing the DataLoader
            img_rgb = Image.new('RGB', (384, 384), color=255)
        else:
            img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        pixel_values = self.feature_extractor(
            img_rgb, return_tensors='pt'
        ).pixel_values.squeeze(0)   # (3, H, W)

        # ── Label ─────────────────────────────────────────────────────────────
        label_str = str(self.df.iloc[idx, 1])
        label_ids = encode_label(self.sp, label_str)
        labels    = torch.tensor(label_ids, dtype=torch.long)
        # Mask padding so it is ignored by the loss
        labels[labels == PAD_ID] = -100

        return {'pixel_values': pixel_values, 'labels': labels}

    def get_path(self, idx):  return self.df.iloc[idx, 0]
    def get_label(self, idx): return str(self.df.iloc[idx, 1])


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════
def build_model(sp: spm.SentencePieceProcessor) -> VisionEncoderDecoderModel:
    """
    Load TrOCR and swap its decoder vocabulary for the custom SPM tokenizer.

    Key config fields that MUST be set:
      - vocab_size             : new BPE vocab size
      - decoder_start_token_id : BOS (used to seed generation)
      - pad_token_id           : PAD (masked in loss)
      - eos_token_id           : EOS (stops generation)
      - bos_token_id           : BOS
    """
    print(f"Loading {Config.TROCR_MODEL} …")
    model = VisionEncoderDecoderModel.from_pretrained(Config.TROCR_MODEL)

    new_vocab = sp.vocab_size()

    # Resize decoder embedding matrix and language-model head
    model.decoder.resize_token_embeddings(new_vocab)

    # Update all config slots that reference vocabulary size
    model.config.vocab_size             = new_vocab
    model.config.decoder.vocab_size     = new_vocab
    model.config.decoder_start_token_id = BOS_ID
    model.config.pad_token_id           = PAD_ID
    model.config.eos_token_id           = EOS_ID
    model.config.bos_token_id           = BOS_ID

    # Generation config (controls model.generate() behaviour)
    model.generation_config.decoder_start_token_id = BOS_ID
    model.generation_config.pad_token_id           = PAD_ID
    model.generation_config.eos_token_id           = EOS_ID
    model.generation_config.bos_token_id           = BOS_ID
    model.generation_config.max_new_tokens          = MAX_LEN

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 7.  METRICS FOR TRAINER
# ══════════════════════════════════════════════════════════════════════════════
def make_compute_metrics(sp: spm.SentencePieceProcessor):
    """
    Returns a compute_metrics function closed over the SPM processor.
    Called by Seq2SeqTrainer after each eval epoch.

    pred.predictions : (N, seq_len) int array — generated token ids
    pred.label_ids   : (N, seq_len) int array — ground truth ids (-100 for pad)
    """
    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids.copy()

        # Replace -100 (ignored positions) with PAD so decode doesn't crash
        label_ids[label_ids == -100] = PAD_ID

        # Clip any out-of-range ids that can appear from generate()
        vocab = sp.vocab_size()
        pred_ids  = np.clip(pred_ids,  0, vocab - 1)
        label_ids = np.clip(label_ids, 0, vocab - 1)

        preds = [decode_ids(sp, row) for row in pred_ids]
        gts   = [decode_ids(sp, row) for row in label_ids]

        cer, wer = corpus_cer_wer(preds, gts)
        return {'cer': round(cer, 4), 'wer': round(wer, 4)}

    return compute_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_model(model, train_dataset, val_dataset,
                sp: spm.SentencePieceProcessor) -> Seq2SeqTrainer:

    args = Seq2SeqTrainingArguments(
        output_dir                  = CKPT_DIR,
        num_train_epochs            = Config.EPOCHS,
        per_device_train_batch_size = Config.BATCH_SIZE,
        per_device_eval_batch_size  = Config.EVAL_BATCH_SIZE,
        learning_rate               = Config.LR,
        warmup_steps                = Config.WARMUP_STEPS,
        weight_decay                = Config.WEIGHT_DECAY,
        predict_with_generate       = True,     # use model.generate() at eval
        generation_max_length       = MAX_LEN,
        fp16                        = Config.FP16 and (device != 'cpu'),
        eval_strategy               = 'epoch',
        save_strategy               = 'epoch',
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = 'cer',
        greater_is_better           = False,
        logging_steps               = 100,
        logging_dir                 = f'{CKPT_DIR}/logs',
        report_to                   = 'none',   # disable wandb/tensorboard
        dataloader_num_workers      = 2,
        remove_unused_columns       = False,    # keep 'pixel_values' in batch
        seed                        = Config.SEED,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = make_compute_metrics(sp),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=4)],
    )

    print(f"\n{'═'*60}")
    print(f"  Training TrOCR for up to {Config.EPOCHS} epochs on {device}")
    print(f"{'═'*60}\n")
    trainer.train()
    return trainer


# ══════════════════════════════════════════════════════════════════════════════
# 9.  TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_test_set(model, test_dataset: TrOCRBanglaDataset,
                      sp: spm.SentencePieceProcessor,
                      feature_extractor) -> pd.DataFrame:
    """
    Run greedy generation on every test sample individually.
    Uses model.generate() so it respects the generation config
    (BOS seed, EOS stopping, max length).
    """
    model.eval()
    model.to(device)

    all_preds, all_gts, all_paths = [], [], []

    print("\nRunning inference on test set …")
    for idx in tqdm(range(len(test_dataset)), dynamic_ncols=True):
        path   = test_dataset.get_path(idx)
        gt_str = test_dataset.get_label(idx)

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        pixel_values = feature_extractor(
            img_rgb, return_tensors='pt'
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens = MAX_LEN,
                num_beams      = 4,       # beam search at test time
                early_stopping = True,
            )

        pred = decode_ids(sp, generated_ids[0].cpu().tolist())
        all_preds.append(pred)
        all_gts.append(gt_str)
        all_paths.append(path)

    cer, wer = corpus_cer_wer(all_preds, all_gts)

    print(f"\n{'═'*50}")
    print(f"  Test samples  : {len(all_gts):,}")
    print(f"  CER           : {cer  * 100:.2f} %")
    print(f"  WER           : {wer  * 100:.2f} %")
    print(f"  Word Accuracy : {(1 - wer) * 100:.2f} %")
    print(f"{'═'*50}")

    results_df = pd.DataFrame({
        'image_path':   all_paths,
        'ground_truth': all_gts,
        'prediction':   all_preds,
        'cer':          [cer_single(p, g) for p, g in zip(all_preds, all_gts)],
        'correct':      [p == g           for p, g in zip(all_preds, all_gts)],
    })
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Results saved → {RESULTS_CSV}\n")

    print(f"{'Ground Truth':<30} {'Prediction':<30} {'CER':>7}  {'OK'}")
    print("─" * 75)
    for _, row in results_df.head(20).iterrows():
        mark = "✓" if row['correct'] else "✗"
        print(f"{row['ground_truth']:<30} {row['prediction']:<30}"
              f" {row['cer']*100:>6.1f}%  {mark}")

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# 10. ANALYSIS PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _read_trainer_history(trainer: Seq2SeqTrainer) -> dict:
    """Extract per-epoch train loss, val CER, val WER from trainer log history."""
    log = trainer.state.log_history
    history = {'train_loss': [], 'val_cer': [], 'val_wer': []}

    epoch_losses = {}
    for entry in log:
        if 'loss' in entry and 'epoch' in entry:
            ep = int(entry['epoch'])
            epoch_losses.setdefault(ep, []).append(entry['loss'])

    eval_entries = [e for e in log if 'eval_cer' in e]
    for i, e in enumerate(eval_entries):
        ep = int(e.get('epoch', i + 1))
        avg_loss = float(np.mean(epoch_losses.get(ep, [float('nan')])))
        history['train_loss'].append(avg_loss)
        history['val_cer'].append(e['eval_cer'])
        history['val_wer'].append(e['eval_wer'])

    return history


def plot_training_curves(history: dict, save_dir=PLOT_DIR):
    if not history['val_cer']:
        print("No training history to plot — skipping")
        return
    epochs = range(1, len(history['val_cer']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#0f0f14')
    colors = {'loss': '#00d4ff', 'cer': '#ff6b6b', 'wer': '#ffd93d'}

    for ax, key, title, color in zip(
        axes,
        ['train_loss', 'val_cer', 'val_wer'],
        ['Training Loss', 'Validation CER', 'Validation WER'],
        [colors['loss'], colors['cer'], colors['wer']]
    ):
        vals = history[key]
        ax.set_facecolor('#1a1a24')
        ax.plot(epochs, vals, color=color, linewidth=2.5, zorder=3)
        ax.fill_between(epochs, vals, alpha=0.15, color=color, zorder=2)
        if vals and not all(v != v for v in vals):  # skip if all-NaN (can
            best_idx = int(np.nanargmin(vals))       # happen with very few
            ax.scatter(best_idx + 1, vals[best_idx],  # logged eval steps)
                       color=color, s=80, zorder=5,
                       edgecolors='white', linewidths=1.5)
            ax.annotate(f"best: {vals[best_idx]:.4f}",
                        xy=(best_idx + 1, vals[best_idx]),
                        xytext=(8, 8), textcoords='offset points',
                        color='white', fontsize=8)
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', color='#aaaaaa', fontsize=9)
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333344')
        ax.grid(color='#333344', linewidth=0.5, alpha=0.6)

    plt.tight_layout(pad=2)
    path = f"{save_dir}/training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close()
    print(f"Saved → {path}")


def plot_cer_distribution(results_df: pd.DataFrame, save_dir=PLOT_DIR):
    cer_vals = results_df['cer'].values * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f0f14')

    ax = axes[0]
    ax.set_facecolor('#1a1a24')
    bins = np.linspace(0, 100, 30)
    counts, edges, patches = ax.hist(cer_vals, bins=bins, edgecolor='none')
    for patch, left_edge in zip(patches, edges[:-1]):
        if left_edge == 0:    patch.set_facecolor('#2ecc71')
        elif left_edge < 20:  patch.set_facecolor('#f39c12')
        else:                 patch.set_facecolor('#e74c3c')
    ax.axvline(np.mean(cer_vals), color='white', linestyle='--',
               linewidth=1.5, label=f'Mean {np.mean(cer_vals):.1f}%')
    ax.axvline(np.median(cer_vals), color='#00d4ff', linestyle=':',
               linewidth=1.5, label=f'Median {np.median(cer_vals):.1f}%')
    ax.legend(facecolor='#1a1a24', labelcolor='white', fontsize=9)
    ax.set_xlabel('CER (%)', color='#aaaaaa')
    ax.set_ylabel('Count', color='#aaaaaa')
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
    ax2.set_xlabel('CER (%)', color='#aaaaaa')
    ax2.set_ylabel('Cumulative % of samples', color='#aaaaaa')
    ax2.set_title('Cumulative Accuracy vs CER Threshold', color='white', fontweight='bold')
    ax2.tick_params(colors='#aaaaaa')
    for sp2 in ax2.spines.values(): sp2.set_edgecolor('#333344')
    ax2.grid(color='#333344', linewidth=0.5, alpha=0.6)

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

    # Top confused character pairs (character-level, not grapheme)
    char_pairs = Counter()
    for _, row in df[~df['correct']].iterrows():
        gt_chars   = list(str(row['ground_truth']))
        pred_chars = list(str(row['prediction']))
        for g, p in zip(gt_chars, pred_chars):
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
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f'n={cnt:,}', ha='center', va='bottom',
                 color='#aaaaaa', fontsize=8)
    ax1.set_xlabel('Word length (chars)', color='#aaaaaa')
    ax1.set_ylabel('Word Accuracy (%)', color='#aaaaaa')
    ax1.set_title('Word Accuracy by Character Count', color='white', fontweight='bold')
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
        ax2.set_title('Top 15 Confused Character Pairs\n(ground truth → prediction)',
                      color='white', fontweight='bold')
        ax2.tick_params(colors='#aaaaaa')
        for sp in ax2.spines.values(): sp.set_edgecolor('#333344')
        ax2.grid(axis='x', color='#333344', linewidth=0.5, alpha=0.6)

    plt.suptitle('Error Analysis', color='white', fontsize=14,
                 fontweight='bold', y=1.02)
    path = f"{save_dir}/error_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close()
    print(f"Saved → {path}")


def plot_tsne(model, test_dataset: TrOCRBanglaDataset,
              feature_extractor, n_samples=600, save_dir=PLOT_DIR):
    """
    Extract encoder (ViT) CLS-token features and project with t-SNE.
    Coloured by word length bucket.
    """
    model.eval()
    model.to(device)

    idxs = random.sample(range(len(test_dataset)),
                         min(n_samples, len(test_dataset)))
    feats, lengths = [], []

    with torch.no_grad():
        for idx in tqdm(idxs, desc="Extracting ViT features for t-SNE",
                        dynamic_ncols=True):
            path = test_dataset.get_path(idx)
            gt   = test_dataset.get_label(idx)
            img_bgr = cv2.imread(path)
            if img_bgr is None: continue
            img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            pv = feature_extractor(img_rgb, return_tensors='pt').pixel_values.to(device)

            # encoder_outputs.last_hidden_state: (1, 577, 768) for trocr-base
            enc_out = model.encoder(pixel_values=pv)
            # CLS token is index 0
            cls_feat = enc_out.last_hidden_state[:, 0, :].squeeze(0)
            feats.append(cls_feat.cpu().float().numpy())
            lengths.append(len(str(gt)))

    feats   = np.array(feats)     # (n, 768) — always 2-D
    lengths = np.array(lengths)

    print("Running t-SNE …")
    tsne  = TSNE(n_components=2, perplexity=30, n_iter=1000,
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
              title='Word length', title_fontsize=9,
              fontsize=9, loc='upper right')
    ax.set_title('t-SNE of ViT CLS-token Features (coloured by word length)',
                 color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#aaaaaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#333344')

    path = f"{save_dir}/tsne_features.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close()
    print(f"Saved → {path}")


def plot_error_grid(results_df: pd.DataFrame, max_errors=40,
                    save_dir=PLOT_DIR):
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
        print("Bengali font not found — skipping error grid")
        return

    COLS = 4; CELL_W = 320; IMG_H = 80; TEXT_H = 70
    CELL_H = IMG_H + TEXT_H
    font  = ImageFont.truetype(font_path, 18)
    sfont = ImageFont.truetype(font_path, 13)

    errors = (results_df[~results_df['correct']]
              .head(max_errors).reset_index(drop=True))
    print(f"\nError grid: {len(errors)} samples")

    def make_cell(i, row):
        cell = Image.new("RGB", (CELL_W, CELL_H), (245, 245, 245))
        draw = ImageDraw.Draw(cell)
        raw  = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            scale  = min(IMG_H / raw.shape[0], (CELL_W-10) / raw.shape[1])
            rh, rw = int(raw.shape[0]*scale), int(raw.shape[1]*scale)
            raw    = cv2.resize(raw, (rw, rh))
            pil    = Image.fromarray(raw).convert("RGB")
            cell.paste(pil, ((CELL_W-rw)//2, (IMG_H-rh)//2))
        draw.rectangle([0,0,26,20], fill=(50,50,50))
        draw.text((4,2), f"#{i+1}", font=sfont, fill=(255,255,255))
        border = ((220,50,50) if row['cer']>0.5 else
                  (255,165,0) if row['cer']>0.2 else (50,180,50))
        draw.rectangle([0,0,CELL_W-1,CELL_H-1], outline=border, width=3)
        draw.line([(0,IMG_H),(CELL_W,IMG_H)], fill=(180,180,180), width=1)
        draw.text((6, IMG_H+4),  "GT:", font=sfont, fill=(30,120,30))
        draw.text((36,IMG_H+2),  row['ground_truth'], font=font, fill=(20,100,20))
        draw.text((6, IMG_H+30), "PR:", font=sfont, fill=(160,40,40))
        draw.text((36,IMG_H+28), row['prediction'],   font=font, fill=(140,20,20))
        draw.text((CELL_W-44, IMG_H+4),
                  f"{row['cer']*100:.0f}%", font=sfont, fill=(80,80,80))
        return cell

    cells = [make_cell(i, row) for i, row in errors.iterrows()]
    ROWS  = math.ceil(len(cells) / COLS)
    grid  = Image.new("RGB", (CELL_W*COLS, CELL_H*ROWS), (210,210,210))
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, COLS)
        grid.paste(cell, (c*CELL_W, r*CELL_H))

    path = f"{save_dir}/error_grid.png"
    grid.save(path, dpi=(150, 150))
    fig, ax = plt.subplots(figsize=(COLS*3.8, ROWS*2.4))
    ax.imshow(grid); ax.axis('off')
    ax.set_title("Erroneous Predictions  "
                 "(green <20% CER | orange 20-50% | red >50%)",
                 fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_grid_plot.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


def print_summary(results_df: pd.DataFrame, history: dict):
    cer_vals = results_df['cer'].values
    wer      = 1 - results_df['correct'].mean()

    print(f"\n{'═'*55}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'─'*55}")
    print(f"  Test samples          : {len(results_df):,}")
    print(f"  Word Accuracy         : {results_df['correct'].mean()*100:.2f} %")
    print(f"  Corpus CER            : {cer_vals.mean()*100:.2f} %")
    print(f"  Corpus WER            : {wer*100:.2f} %")
    print(f"  Median CER            : {np.median(cer_vals)*100:.2f} %")
    print(f"  CER = 0 (perfect)     : "
          f"{(cer_vals==0).sum()/len(cer_vals)*100:.1f} %")
    print(f"  CER ≤ 0.1 (near-perf) : "
          f"{(cer_vals<=0.1).sum()/len(cer_vals)*100:.1f} %")
    print(f"  CER > 0.5 (bad)       : "
          f"{(cer_vals>0.5).sum()/len(cer_vals)*100:.1f} %")
    if history and history['val_cer']:
        print(f"  Best Val CER (train)  : {min(history['val_cer'])*100:.2f} %")
    print(f"{'═'*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── Load & split data ──────────────────────────────────────────────────────
    df = load_dataset(ROOT)
    if UNIGRAM_SMOKE_TEST:
        # must stay well above SPM_VOCAB_SIZE=2000 or SentencePiece can't find
        # enough distinct pieces to reach the target vocabulary
        df = df.sample(min(8000, len(df)), random_state=Config.SEED).reset_index(drop=True)
        print(f"[smoke test] subsampled to {len(df):,} rows")

    train_val_df, test_df = train_test_split(
        df, test_size=Config.TEST_SPLIT, random_state=Config.SEED
    )
    val_frac = Config.VAL_SPLIT / (1.0 - Config.TEST_SPLIT)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_frac, random_state=Config.SEED
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | "
          f"Test: {len(test_df):,}")

    # ── Train SentencePiece tokenizer on training labels only ─────────────────
    sp = train_spm_tokenizer(train_df['label'].dropna())

    # ── Load TrOCR feature extractor (handles image preprocessing) ─────────────
    feature_extractor = AutoImageProcessor.from_pretrained(Config.TROCR_MODEL)

    # ── Build model with resized vocab ────────────────────────────────────────
    model = build_model(sp)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = TrOCRBanglaDataset(train_df, sp, feature_extractor, is_train=True)
    val_dataset   = TrOCRBanglaDataset(val_df,   sp, feature_extractor, is_train=False)
    test_dataset  = TrOCRBanglaDataset(test_df,  sp, feature_extractor, is_train=False)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = train_model(model, train_dataset, val_dataset, sp)

    # ── Load best checkpoint saved by Trainer ─────────────────────────────────
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        print(f"\nLoading best checkpoint from {best_ckpt} …")
        model = VisionEncoderDecoderModel.from_pretrained(best_ckpt).to(device)
        # Re-apply generation config after loading
        model.generation_config.decoder_start_token_id = BOS_ID
        model.generation_config.pad_token_id           = PAD_ID
        model.generation_config.eos_token_id           = EOS_ID
        model.generation_config.max_new_tokens          = MAX_LEN
    else:
        print("No best checkpoint found — using final model weights")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    results_df = evaluate_test_set(model, test_dataset, sp, feature_extractor)

    # ── History from trainer ──────────────────────────────────────────────────
    history = _read_trainer_history(trainer)
    print_summary(results_df, history)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\nGenerating analysis plots → {PLOT_DIR}/\n")
    plot_training_curves(history)
    plot_cer_distribution(results_df)
    plot_error_analysis(results_df)
    plot_tsne(model, test_dataset, feature_extractor, n_samples=600)
    plot_error_grid(results_df, max_errors=40)

    print("\nAll done.")