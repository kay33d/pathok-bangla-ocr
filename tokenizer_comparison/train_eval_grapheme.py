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
import editdistance
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
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.model_selection import train_test_split

# ── Special token ids ─────────────────────────────────────────────────────────
# We build a custom vocabulary from graphemes.
# These 4 ids are reserved and must match what the decoder config expects.
PAD_ID  = 0
UNK_ID  = 1
BOS_ID  = 2
EOS_ID  = 3
SPECIAL_TOKENS = 4     # number of reserved slots before real graphemes start
MAX_LEN = 64           # max decoder sequence length (generous for long words)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = "/kaggle/input/datasets/kaidul007/bnhtrd/Dataset"
PUNCT       = {'.', ':', '।', ',', ';', '!', '?', '-', '–', '—'}
CKPT_DIR      = '/kaggle/working/trocr_grapheme'
# Read-only input path that contains checkpoints from the previous Kaggle run.
# epoch_* folders are expected at  PREV_CKPT_DIR/trocr_grapheme/epoch_XX/
PREV_CKPT_DIR = '/kaggle/input/notebooks/irtisam/trocr-vit-grapheme/trocr_grapheme'
RESULTS_CSV = '/kaggle/working/test_results_trocr_grapheme.csv'
PLOT_DIR    = '/kaggle/working/plots_trocr_grapheme'

# ══════════════════════════════════════════════════════════════════════════════
# 1.  SHARED BASELINE CONFIG  (identical across all 5 experiments)
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    # ── encoder ───────────────────────────────────────────────────────────────
    TROCR_MODEL     = 'microsoft/trocr-base-handwritten'

    # ── training (FIXED – do not change across experiments) ───────────────────
    BATCH_SIZE      = 16
    EVAL_BATCH_SIZE = 16
    EPOCHS          = 5         # ← shared baseline
    LR              = 5e-5
    WARMUP_STEPS    = 500
    WEIGHT_DECAY    = 0.01
    TEST_SPLIT      = 0.10        # ← shared baseline
    VAL_SPLIT       = 0.10        # ← shared baseline
    SEED            = 42          # ← shared baseline
    FP16            = True        # set False if CPU-only

random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

device = (torch.accelerator.current_accelerator().type
          if torch.accelerator.is_available() else 'cpu')
print(f"Using device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING  (identical across all 5 experiments)
# ══════════════════════════════════════════════════════════════════════════════
def is_bangla_word(text: str) -> bool:
    """True when >80 % of non-space chars are in the Bangla Unicode block."""
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
    df = df[df['label'].apply(is_bangla_word)]   # same filter as TrOCR+BPE run
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"Total usable samples: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BANGLA GRAPHEME SEGMENTER
#     Identical to the ResNet+Grapheme experiments (Doc 2/3).
#     A grapheme = one visual unit: consonant cluster + optional vowel sign.
# ══════════════════════════════════════════════════════════════════════════════
HASANTA          = '\u09CD'
CHANDRABINDU     = '\u0981'
ANUSVARA         = '\u0982'
VISARGA          = '\u0983'
NUKTA            = '\u09BC'

VOWEL_SIGNS = set(
    '\u09BE\u09BF\u09C0\u09C1\u09C2\u09C3\u09C4'
    '\u09C7\u09C8\u09CB\u09CC\u09D7'
)
INDEPENDENT_VOWELS = set(
    '\u0985\u0986\u0987\u0988\u0989\u098A\u098B'
    '\u098F\u0990\u0993\u0994'
)
CONSONANTS = set(chr(c) for c in range(0x0995, 0x09BA)) | {
    '\u09DC', '\u09DD', '\u09DF', '\u09CE', '\u09B0'
}


def segment_graphemes(word: str) -> list:
    """
    Segment a Bangla word into a list of grapheme strings.
    Each grapheme is one of:
      • independent vowel  (+ optional anusvara/visarga/chandrabindu)
      • consonant cluster  (C (hasanta C)*)  + optional vowel sign
      • any other character treated as a single grapheme
    """
    graphemes = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in INDEPENDENT_VOWELS:
            g = ch; i += 1
            while i < n and word[i] in {CHANDRABINDU, ANUSVARA, VISARGA}:
                g += word[i]; i += 1
            graphemes.append(g)
        elif ch in CONSONANTS:
            g = ch; i += 1
            if i < n and word[i] == NUKTA:
                g += word[i]; i += 1
            # absorb hasanta + following consonant  (conjunct consonants)
            while i + 1 < n and word[i] == HASANTA and word[i + 1] in CONSONANTS:
                g += word[i] + word[i + 1]; i += 2
            # absorb trailing vowel sign
            if i < n and word[i] in VOWEL_SIGNS:
                g += word[i]; i += 1
            graphemes.append(g)
        else:
            graphemes.append(ch); i += 1
    return graphemes

# ══════════════════════════════════════════════════════════════════════════════
# 4.  GRAPHEME VOCABULARY  (this file's key difference vs BPE)
# ══════════════════════════════════════════════════════════════════════════════
class GraphemeVocab:
    """
    Minimal vocabulary built from the training set's grapheme inventory.
    Maps every unique grapheme → integer id, keeping 4 special slots.

    Compared with SentencePiece BPE:
      + Linguistically motivated — each token = one visual unit
      + Fully interpretable vocabulary
      + No hyper-parameter (vocab size) to tune
      - Larger effective vocab than BPE for the same coverage
      - Longer sequences on average (no subword merging)
    """

    def __init__(self):
        self.g2i: dict = {}
        self.i2g: dict = {
            PAD_ID: '<pad>',
            UNK_ID: '<unk>',
            BOS_ID: '<s>',
            EOS_ID: '</s>',
        }
        self.vocab_size: int = SPECIAL_TOKENS

    # ── build from training labels ────────────────────────────────────────────
    def build(self, labels) -> 'GraphemeVocab':
        all_g = sorted({
            g
            for word in labels
            for g in segment_graphemes(str(word))
        })
        for idx, g in enumerate(all_g):
            tid = idx + SPECIAL_TOKENS
            self.g2i[g]   = tid
            self.i2g[tid] = g
        self.vocab_size = SPECIAL_TOKENS + len(all_g)
        print(f"Grapheme vocab: {len(all_g)} unique graphemes  "
              f"(vocab_size = {self.vocab_size})")
        return self

    # ── serialise / deserialise ───────────────────────────────────────────────
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'g2i': self.g2i,
                       'i2g': {str(k): v for k, v in self.i2g.items()},
                       'vocab_size': self.vocab_size}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'GraphemeVocab':
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        v = cls()
        v.g2i        = d['g2i']
        v.i2g        = {int(k): val for k, val in d['i2g'].items()}
        v.vocab_size = d['vocab_size']
        return v

    # ── encode / decode ───────────────────────────────────────────────────────
    def encode(self, word: str) -> list:
        """
        word → list of int ids, WITHOUT BOS/EOS.
        BOS/EOS are added by encode_label() below.
        """
        return [self.g2i.get(g, UNK_ID) for g in segment_graphemes(str(word))]

    def decode(self, ids) -> str:
        """ids (list/tensor) → string, stopping at EOS, skipping specials."""
        out = []
        for t in ids:
            t = int(t)
            if t == EOS_ID:
                break
            if t in (PAD_ID, BOS_ID, UNK_ID):
                continue
            out.append(self.i2g.get(t, ''))
        return ''.join(out)


def encode_label(vocab: GraphemeVocab, text: str) -> list:
    """
    Encode a word to a padded id sequence:
        [BOS, g1, g2, …, EOS, PAD, PAD, …]   length = MAX_LEN
    Used to build the labels tensor fed to the decoder.
    """
    ids = vocab.encode(text)
    seq = ([BOS_ID] + ids + [EOS_ID])[:MAX_LEN]
    seq += [PAD_ID] * (MAX_LEN - len(seq))
    return seq

# ══════════════════════════════════════════════════════════════════════════════
# 5.  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class TrOCRGraphemeDataset(Dataset):
    """
    PyTorch Dataset for TrOCR with grapheme labels.

    Image preprocessing is delegated entirely to HuggingFace's
    AutoImageProcessor (ViTImageProcessor under the hood), which:
      • resizes to 384×384
      • normalises with ImageNet mean/std
    So we only need to hand it a PIL RGB image.

    Labels are encoded with GraphemeVocab.
    Positions that correspond to PAD_ID are set to -100 so
    CrossEntropyLoss ignores them (standard HF convention).
    """

    def __init__(self, df: pd.DataFrame,
                 vocab: GraphemeVocab,
                 feature_extractor):
        self.df               = df.reset_index(drop=True)
        self.vocab            = vocab
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # ── load image ────────────────────────────────────────────────────────
        img_bgr = cv2.imread(self.df.iloc[idx, 0])
        if img_bgr is None:
            img_rgb = Image.new('RGB', (384, 384), color=255)
        else:
            img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        pixel_values = self.feature_extractor(
            img_rgb, return_tensors='pt'
        ).pixel_values.squeeze(0)                    # (3, 384, 384)

        # ── encode label ──────────────────────────────────────────────────────
        label_str = str(self.df.iloc[idx, 1])
        label_ids = encode_label(self.vocab, label_str)
        labels    = torch.tensor(label_ids, dtype=torch.long)
        labels[labels == PAD_ID] = -100              # mask padding for loss

        return {'pixel_values': pixel_values, 'labels': labels}

    def get_path(self, idx):  return self.df.iloc[idx, 0]
    def get_label(self, idx): return str(self.df.iloc[idx, 1])

# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL SETUP  (encoder identical to TrOCR+BPE; only vocab size differs)
# ══════════════════════════════════════════════════════════════════════════════
def build_model(vocab: GraphemeVocab) -> VisionEncoderDecoderModel:
    """
    Load pretrained TrOCR and swap the decoder vocabulary for our
    grapheme vocabulary.

    The ViT encoder weights are kept exactly as in the BPE experiment.
    Only the decoder embedding matrix and LM head are re-initialised
    to the new vocab size — giving us a clean tokenizer-only comparison.
    """
    print(f"Loading {Config.TROCR_MODEL} …")
    model = VisionEncoderDecoderModel.from_pretrained(Config.TROCR_MODEL)

    new_vocab = vocab.vocab_size

    # resize decoder embedding + LM head
    model.decoder.resize_token_embeddings(new_vocab)

    # update every config slot that references vocab size
    model.config.vocab_size             = new_vocab
    model.config.decoder.vocab_size     = new_vocab
    model.config.decoder_start_token_id = BOS_ID
    model.config.pad_token_id           = PAD_ID
    model.config.eos_token_id           = EOS_ID
    model.config.bos_token_id           = BOS_ID

    # generation config
    model.generation_config.decoder_start_token_id = BOS_ID
    model.generation_config.pad_token_id           = PAD_ID
    model.generation_config.eos_token_id           = EOS_ID
    model.generation_config.bos_token_id           = BOS_ID
    model.generation_config.max_new_tokens          = MAX_LEN

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total/1e6:.1f}M total  |  {trainable/1e6:.1f}M trainable")
    print(f"Decoder vocab size: {new_vocab} graphemes")
    return model

# ══════════════════════════════════════════════════════════════════════════════
# 7.  METRICS
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
# 8.  COMPUTE_METRICS FOR TRAINER
# ══════════════════════════════════════════════════════════════════════════════
def make_compute_metrics(vocab: GraphemeVocab):
    """
    Returns a compute_metrics closure for Seq2SeqTrainer.
    Called after each eval epoch with:
      pred.predictions : (N, seq_len)  int array  – generated token ids
      pred.label_ids   : (N, seq_len)  int array  – ground truth (-100 for pad)
    """
    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids.copy()

        # restore masked positions so decode doesn't crash
        label_ids[label_ids == -100] = PAD_ID

        # clip any out-of-range ids from generate()
        pred_ids  = np.clip(pred_ids,  0, vocab.vocab_size - 1)
        label_ids = np.clip(label_ids, 0, vocab.vocab_size - 1)

        preds = [vocab.decode(row) for row in pred_ids]
        gts   = [vocab.decode(row) for row in label_ids]

        cer, wer = corpus_cer_wer(preds, gts)
        return {'cer': round(cer, 4), 'wer': round(wer, 4)}

    return compute_metrics

# ══════════════════════════════════════════════════════════════════════════════
# 9.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_model(model, train_dataset, val_dataset,
                vocab: GraphemeVocab,
                start_epoch: int = 0) -> 'Seq2SeqTrainer':
    """
    start_epoch : number of epochs already completed before this call.
                  EpochCheckpointCallback adds this offset so the saved
                  folders are epoch_(start_epoch+1), epoch_(start_epoch+2) …
                  Pass 0 (default) for a brand-new run.
    """

    args = Seq2SeqTrainingArguments(
        output_dir                  = CKPT_DIR,
        num_train_epochs            = Config.EPOCHS,
        per_device_train_batch_size = Config.BATCH_SIZE,
        per_device_eval_batch_size  = Config.EVAL_BATCH_SIZE,
        learning_rate               = Config.LR,
        warmup_steps                = Config.WARMUP_STEPS,
        weight_decay                = Config.WEIGHT_DECAY,
        predict_with_generate       = True,
        fp16                        = Config.FP16 and (device != 'cpu'),
        eval_strategy               = 'epoch',
        save_strategy               = 'epoch',
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = 'cer',
        greater_is_better           = False,
        logging_steps               = 100,
        logging_dir                 = f'{CKPT_DIR}/logs',
        report_to                   = 'none',
        dataloader_num_workers      = 2,
        remove_unused_columns       = False,
        seed                        = Config.SEED,
    )

    # ── EpochCheckpointCallback: saves model weights keyed by epoch number ──
    class EpochCheckpointCallback(TrainerCallback):
        """
        Saves model weights after every epoch into a clearly named sub-folder:
            <CKPT_DIR>/epoch_<N>/
        This is separate from the HF Trainer's own save_strategy='epoch'
        checkpoints so that every epoch's weights are always preserved,
        regardless of save_total_limit.
        """
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            # state.epoch always resets to 1,2,… each trainer.train() call.
            # Adding start_epoch gives the true global epoch number.
            global_epoch = int(round(state.epoch)) + start_epoch
            save_path = os.path.join(CKPT_DIR, f"epoch_{global_epoch:02d}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            print(f"  [EpochCheckpoint] Saved epoch {global_epoch} weights → {save_path}")

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = make_compute_metrics(vocab),
        callbacks       = [
            EarlyStoppingCallback(early_stopping_patience=3),
            EpochCheckpointCallback(),
        ],
    )

    print(f"\n{'═'*60}")
    print(f"  TrOCR ViT  +  Grapheme Tokenizer")
    if start_epoch > 0:
        print(f"  Resuming from epoch {start_epoch}  →  "
              f"saving as epoch_{start_epoch+1:02d} … "
              f"epoch_{start_epoch+Config.EPOCHS:02d}")
    else:
        print(f"  Training for up to {Config.EPOCHS} epochs on {device}")
    print(f"  Grapheme vocab size : {vocab.vocab_size}")
    print(f"{'═'*60}\n")

    trainer.train()
    return trainer

# ══════════════════════════════════════════════════════════════════════════════
# 9b.  CHECKPOINT LOADER
#      Load a previously saved epoch checkpoint for further training or
#      evaluation.  Works with both the per-epoch folders written by
#      EpochCheckpointCallback and the HF Trainer's own checkpoint folders.
# ══════════════════════════════════════════════════════════════════════════════
def load_checkpoint(
    epoch: int | None = None,
    checkpoint_dir: str = CKPT_DIR,
    prev_checkpoint_dir: str = PREV_CKPT_DIR,
    vocab: GraphemeVocab | None = None,
) -> tuple[VisionEncoderDecoderModel, int]:
    """
    Load model weights saved by EpochCheckpointCallback.

    Search order (when epoch=None):
      1. checkpoint_dir  (CKPT_DIR  — writable working output)
      2. prev_checkpoint_dir  (PREV_CKPT_DIR — read-only Kaggle input)
    The folder with the highest epoch number across both locations wins.

    Parameters
    ----------
    epoch : int or None
        If given, loads from  <checkpoint_dir>/epoch_<N>/  (working dir first,
        then prev_checkpoint_dir as fallback).
        If None, auto-selects the latest epoch across both directories.
    checkpoint_dir : str
        Writable output folder (default: CKPT_DIR).
    prev_checkpoint_dir : str
        Read-only input folder from a previous Kaggle run (default: PREV_CKPT_DIR).
    vocab : GraphemeVocab or None
        When provided, validates vocab size and restores generation config.

    Returns
    -------
    (model, loaded_epoch) — model on `device`, loaded_epoch as int.
    loaded_epoch is used as start_epoch for the next train_model() call.

    Example — resume from latest checkpoint across both dirs
    ----------------------------------------------------------
    vocab  = GraphemeVocab.load(f'{PREV_CKPT_DIR}/grapheme_vocab.json')
    model, start_epoch = load_checkpoint(vocab=vocab)
    trainer = train_model(model, train_dataset, val_dataset, vocab,
                          start_epoch=start_epoch)
    """
    if epoch is None:
        # Collect epoch_* folders from BOTH dirs, pick the highest epoch number.
        def _collect(d):
            return [f for f in glob.glob(os.path.join(d, "epoch_*"))
                    if os.path.isdir(f)]
        all_folders = _collect(checkpoint_dir) + _collect(prev_checkpoint_dir)
        all_folders = sorted(
            all_folders,
            key=lambda p: int(os.path.basename(p).split("_")[-1]),
        )
        if not all_folders:
            raise FileNotFoundError(
                f"No epoch_* folders found in:\n"
                f"  {checkpoint_dir}\n"
                f"  {prev_checkpoint_dir}"
            )
        ckpt_path = all_folders[-1]
        loaded_epoch = int(os.path.basename(ckpt_path).split("_")[-1])
        print(f"[load_checkpoint] Auto-selected latest checkpoint: "
              f"epoch {loaded_epoch} ({ckpt_path})")
    else:
        # Try writable dir first, then fall back to read-only input dir.
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:02d}")
        if not os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(prev_checkpoint_dir, f"epoch_{epoch:02d}")
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(
                f"epoch_{epoch:02d} not found in either:\n"
                f"  {checkpoint_dir}\n"
                f"  {prev_checkpoint_dir}"
            )
        loaded_epoch = epoch
        print(f"[load_checkpoint] Loading epoch {loaded_epoch} from {ckpt_path}")

    model = VisionEncoderDecoderModel.from_pretrained(ckpt_path).to(device)

    # Restore generation config so the model works out-of-the-box
    model.generation_config.decoder_start_token_id = BOS_ID
    model.generation_config.pad_token_id           = PAD_ID
    model.generation_config.eos_token_id           = EOS_ID
    model.generation_config.bos_token_id           = BOS_ID
    model.generation_config.max_new_tokens          = MAX_LEN

    if vocab is not None:
        # Sanity check: loaded vocab size must match the model's embedding dim
        expected = vocab.vocab_size
        actual   = model.config.decoder.vocab_size
        if expected != actual:
            print(
                f"  ⚠  vocab mismatch: vocab says {expected}, "
                f"model config says {actual}. "
                "Pass the correct GraphemeVocab or re-build the vocabulary."
            )
        else:
            print(f"  ✓  Vocab size matches: {actual}")

    print(f"[load_checkpoint] Model ready on {device}. (epoch {loaded_epoch})")
    return model, loaded_epoch

# ══════════════════════════════════════════════════════════════════════════════
# 10. TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_test_set(model, test_dataset: TrOCRGraphemeDataset,
                      vocab: GraphemeVocab,
                      feature_extractor) -> pd.DataFrame:
    """
    Beam-search decoding (beam=4) on every test sample.
    Returns a DataFrame with columns:
        image_path | ground_truth | prediction | cer | correct
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
                num_beams      = 4,
                early_stopping = True,
            )

        pred = vocab.decode(generated_ids[0].cpu().tolist())
        all_preds.append(pred)
        all_gts.append(gt_str)
        all_paths.append(path)

    cer, wer = corpus_cer_wer(all_preds, all_gts)

    print(f"\n{'═'*55}")
    print(f"  EXPERIMENT  : TrOCR ViT  +  Grapheme Tokenizer")
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

    # print first 20 samples
    print(f"{'Ground Truth':<30} {'Prediction':<30} {'CER':>7}  OK")
    print("─" * 75)
    for _, row in results_df.head(20).iterrows():
        mark = "✓" if row['correct'] else "✗"
        print(f"{row['ground_truth']:<30} {row['prediction']:<30}"
              f" {row['cer']*100:>6.1f}%  {mark}")

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# 11. ANALYSIS PLOTS
#     Identical plotting code across all 5 experiments so figures are
#     directly comparable.  Only titles/filenames differ.
# ══════════════════════════════════════════════════════════════════════════════

def _read_trainer_history(trainer) -> dict:
    log = trainer.state.log_history
    history = {'train_loss': [], 'val_cer': [], 'val_wer': []}
    epoch_losses: dict = {}
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
        print("No training history — skipping curve plot")
        return
    epochs = range(1, len(history['val_cer']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#0f0f14')
    specs = [
        ('train_loss', 'Training Loss',    '#00d4ff'),
        ('val_cer',    'Validation CER',   '#ff6b6b'),
        ('val_wer',    'Validation WER',   '#ffd93d'),
    ]
    for ax, (key, title, color) in zip(axes, specs):
        vals = history[key]
        ax.set_facecolor('#1a1a24')
        ax.plot(epochs, vals, color=color, linewidth=2.5, zorder=3)
        ax.fill_between(epochs, vals, alpha=0.15, color=color, zorder=2)
        best_idx = int(np.nanargmin(vals))
        ax.scatter(best_idx + 1, vals[best_idx],
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

    fig.suptitle('TrOCR ViT + Grapheme Tokenizer — Training Curves',
                 color='white', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(pad=2)
    path = f"{save_dir}/training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close()
    print(f"Saved → {path}")


def plot_cer_distribution(results_df: pd.DataFrame, save_dir=PLOT_DIR):
    cer_vals = results_df['cer'].values * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f0f14')

    # ── histogram ─────────────────────────────────────────────────────────────
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
    for sp in ax.spines.values():
        sp.set_edgecolor('#333344')

    # ── cumulative ────────────────────────────────────────────────────────────
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
    ax2.set_xlabel('CER (%)',                        color='#aaaaaa')
    ax2.set_ylabel('Cumulative % of samples',        color='#aaaaaa')
    ax2.set_title('Cumulative Accuracy vs CER',      color='white', fontweight='bold')
    ax2.tick_params(colors='#aaaaaa')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#333344')
    ax2.grid(color='#333344', linewidth=0.5, alpha=0.6)

    fig.suptitle('TrOCR ViT + Grapheme Tokenizer — CER Distribution',
                 color='white', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(pad=2)
    path = f"{save_dir}/cer_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close()
    print(f"Saved → {path}")


def plot_error_analysis(results_df: pd.DataFrame, save_dir=PLOT_DIR):
    df = results_df.copy()
    # word length in graphemes (linguistically meaningful for Bangla)
    df['gt_len'] = df['ground_truth'].apply(
        lambda w: len(segment_graphemes(str(w)))
    )
    buckets = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 99)]
    labels_b, accs, counts = [], [], []
    for lo, hi in buckets:
        sub = df[(df['gt_len'] >= lo) & (df['gt_len'] <= hi)]
        if len(sub) == 0:
            continue
        lab = (f"{lo}" if lo == hi
               else (f"{lo}-{hi}" if hi < 99 else f"{lo}+"))
        labels_b.append(lab)
        accs.append(sub['correct'].mean() * 100)
        counts.append(len(sub))

    # confused grapheme pairs
    grapheme_pairs = Counter()
    for _, row in df[~df['correct']].iterrows():
        gt_gs   = segment_graphemes(str(row['ground_truth']))
        pred_gs = segment_graphemes(str(row['prediction']))
        for g, p in zip(gt_gs, pred_gs):
            if g != p:
                grapheme_pairs[(g, p)] += 1

    fig = plt.figure(figsize=(16, 6), facecolor='#0f0f14')
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── accuracy by word length ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1a1a24')
    bar_colors = ['#2ecc71' if a >= 70 else '#f39c12' if a >= 40
                  else '#e74c3c' for a in accs]
    bars = ax1.bar(labels_b, accs, color=bar_colors, edgecolor='none', width=0.6)
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f'n={cnt:,}', ha='center', va='bottom',
                 color='#aaaaaa', fontsize=8)
    ax1.set_xlabel('Word length (graphemes)', color='#aaaaaa')
    ax1.set_ylabel('Word Accuracy (%)',        color='#aaaaaa')
    ax1.set_title('Accuracy by Grapheme Count', color='white', fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.tick_params(colors='#aaaaaa')
    for sp in ax1.spines.values():
        sp.set_edgecolor('#333344')
    ax1.grid(axis='y', color='#333344', linewidth=0.5, alpha=0.6)

    # ── top confused grapheme pairs ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#1a1a24')
    top_pairs = grapheme_pairs.most_common(15)
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
        ax2.set_title('Top 15 Confused Grapheme Pairs',
                      color='white', fontweight='bold')
        ax2.tick_params(colors='#aaaaaa')
        for sp in ax2.spines.values():
            sp.set_edgecolor('#333344')
        ax2.grid(axis='x', color='#333344', linewidth=0.5, alpha=0.6)

    fig.suptitle('TrOCR ViT + Grapheme — Error Analysis',
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    path = f"{save_dir}/error_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close()
    print(f"Saved → {path}")


def plot_tsne(model, test_dataset: TrOCRGraphemeDataset,
              feature_extractor, n_samples=600, save_dir=PLOT_DIR):
    """
    Extract ViT CLS-token features and project with t-SNE,
    coloured by word-length bucket (in graphemes).
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
            if img_bgr is None:
                continue
            img_rgb = Image.fromarray(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            )
            pv = feature_extractor(
                img_rgb, return_tensors='pt'
            ).pixel_values.to(device)

            # ViT encoder: last_hidden_state shape (1, 577, 768) for base
            enc_out  = model.encoder(pixel_values=pv)
            cls_feat = enc_out.last_hidden_state[:, 0, :].squeeze(0)
            feats.append(cls_feat.cpu().float().numpy())
            lengths.append(len(segment_graphemes(gt)))

    feats   = np.array(feats)
    lengths = np.array(lengths)

    print("Running t-SNE …")
    tsne  = TSNE(n_components=2, perplexity=30, n_iter=1000,
                 random_state=Config.SEED)
    emb2d = tsne.fit_transform(feats)

    def bucket(l):
        if l == 1: return '1'
        if l == 2: return '2'
        if l == 3: return '3'
        if l <= 5: return '4-5'
        if l <= 8: return '6-8'
        return '9+'

    buckets = [bucket(l) for l in lengths]
    palette = {'1': '#2ecc71', '2': '#3498db', '3': '#9b59b6',
               '4-5': '#f39c12', '6-8': '#e74c3c', '9+': '#95a5a6'}

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f0f14')
    ax.set_facecolor('#1a1a24')
    for label, color in palette.items():
        mask = np.array(buckets) == label
        if mask.sum() == 0:
            continue
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   c=color, label=f'{label} grapheme(s)',
                   alpha=0.65, s=20, edgecolors='none')
    ax.legend(facecolor='#1a1a24', labelcolor='white',
              title='Word length (graphemes)', title_fontsize=9,
              fontsize=9, loc='upper right')
    ax.set_title(
        'TrOCR ViT + Grapheme — t-SNE of ViT CLS features',
        color='white', fontsize=13, fontweight='bold'
    )
    ax.tick_params(colors='#aaaaaa')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333344')

    path = f"{save_dir}/tsne_features.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close()
    print(f"Saved → {path}")


def plot_error_grid(results_df: pd.DataFrame, max_errors=40,
                    save_dir=PLOT_DIR):
    """Render erroneous predictions as an image grid (requires Bengali font)."""
    import subprocess
    subprocess.run(["apt-get", "install", "-y", "fonts-beng", "-qq"],
                   capture_output=True)

    def find_bangla_font():
        for pat in ["/usr/share/fonts/**/*Lohit*Bengali*",
                    "/usr/share/fonts/**/*Lohit*Beng*",
                    "/usr/share/fonts/**/*Bengali*"]:
            hits = glob.glob(pat, recursive=True)
            if hits:
                return hits[0]
        return None

    font_path = find_bangla_font()
    if font_path is None:
        print("Bengali font not found — skipping error grid")
        return

    COLS   = 4
    CELL_W = 320
    IMG_H  = 80
    TEXT_H = 70
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
        draw.line([(0, IMG_H), (CELL_W, IMG_H)],
                  fill=(180, 180, 180), width=1)
        draw.text((6,  IMG_H + 4),  "GT:", font=sfont, fill=(30,  120, 30))
        draw.text((36, IMG_H + 2),  row['ground_truth'],
                  font=font, fill=(20, 100, 20))
        draw.text((6,  IMG_H + 30), "PR:", font=sfont, fill=(160, 40,  40))
        draw.text((36, IMG_H + 28), row['prediction'],
                  font=font, fill=(140, 20, 20))
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
    ax.imshow(grid)
    ax.axis('off')
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

    print(f"\n{'═'*60}")
    print(f"  EXPERIMENT  : TrOCR ViT  +  Grapheme Tokenizer")
    print(f"{'─'*60}")
    print(f"  Test samples          : {len(results_df):,}")
    print(f"  Word Accuracy         : {results_df['correct'].mean()*100:.2f} %")
    print(f"  Corpus CER            : {cer_vals.mean()*100:.2f} %")
    print(f"  Corpus WER            : {wer*100:.2f} %")
    print(f"  Median CER            : {np.median(cer_vals)*100:.2f} %")
    print(f"  CER = 0  (perfect)    : {(cer_vals==0).sum()/len(cer_vals)*100:.1f} %")
    print(f"  CER ≤ 0.1 (near-perf) : {(cer_vals<=0.1).sum()/len(cer_vals)*100:.1f} %")
    print(f"  CER > 0.5 (bad)       : {(cer_vals>0.5).sum()/len(cer_vals)*100:.1f} %")
    if history and history['val_cer']:
        print(f"  Best Val CER (train)  : {min(history['val_cer'])*100:.2f} %")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 12. MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── 1. load & split  (identical split to all other experiments) ───────────
    df = load_dataset(ROOT)

    train_val_df, test_df = train_test_split(
        df,
        test_size    = Config.TEST_SPLIT,
        random_state = Config.SEED,
    )
    val_frac = Config.VAL_SPLIT / (1.0 - Config.TEST_SPLIT)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size    = val_frac,
        random_state = Config.SEED,
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | "
          f"Test: {len(test_df):,}")

    # ── 2. build grapheme vocabulary from TRAINING labels only ────────────────
    vocab = GraphemeVocab().build(train_df['label'].dropna())
    vocab.save(f'{CKPT_DIR}/grapheme_vocab.json')

    # ── 3. load TrOCR image processor ─────────────────────────────────────────
    feature_extractor = AutoImageProcessor.from_pretrained(Config.TROCR_MODEL)


    # ── 4. build or resume model ──────────────────────────────────────────────
    # Scan both the writable working dir and the read-only input dir for any
    # epoch_* checkpoint folders.  If found, resume from the latest one;
    # otherwise build a fresh model from the pretrained TrOCR weights.
    def _has_epoch_folders(d):
        return any(os.path.isdir(f)
                   for f in glob.glob(os.path.join(d, "epoch_*")))

    if _has_epoch_folders(CKPT_DIR) or _has_epoch_folders(PREV_CKPT_DIR):
        # Load vocab from the read-only input (it was built on the training set
        # and must stay identical across runs).
        vocab = GraphemeVocab.load(
            os.path.join(PREV_CKPT_DIR, "grapheme_vocab.json")
        )
        model, start_epoch = load_checkpoint(vocab=vocab)
        print(f"Resuming: loaded epoch {start_epoch}, "
              f"next checkpoint → epoch_{start_epoch+1:02d}")
    else:
        start_epoch = 0
        model = build_model(vocab)
        print("No prior checkpoints found — starting fresh from epoch 0")

    # ── 5. create datasets ────────────────────────────────────────────────────
    train_dataset = TrOCRGraphemeDataset(train_df, vocab, feature_extractor)
    val_dataset   = TrOCRGraphemeDataset(val_df,   vocab, feature_extractor)
    test_dataset  = TrOCRGraphemeDataset(test_df,  vocab, feature_extractor)

    # ── 6. train (start_epoch offsets the checkpoint filenames) ───────────────
    trainer = train_model(model, train_dataset, val_dataset, vocab,
                          start_epoch=start_epoch)

    # ── 7. load best checkpoint ───────────────────────────────────────────────
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        print(f"\nLoading best checkpoint from {best_ckpt} …")
        model = VisionEncoderDecoderModel.from_pretrained(best_ckpt).to(device)
        # re-apply generation config after loading from disk
        model.generation_config.decoder_start_token_id = BOS_ID
        model.generation_config.pad_token_id           = PAD_ID
        model.generation_config.eos_token_id           = EOS_ID
        model.generation_config.max_new_tokens          = MAX_LEN
    else:
        print("No best checkpoint found — using final weights")

    # ── 8. evaluate ───────────────────────────────────────────────────────────
    results_df = evaluate_test_set(model, test_dataset, vocab, feature_extractor)

    # ── 9. history & summary ──────────────────────────────────────────────────
    history = _read_trainer_history(trainer)
    print_summary(results_df, history)

    # ── 10. plots ─────────────────────────────────────────────────────────────
    print(f"\nGenerating analysis plots → {PLOT_DIR}/\n")
    plot_training_curves(history)
    plot_cer_distribution(results_df)
    plot_error_analysis(results_df)
    plot_tsne(model, test_dataset, feature_extractor, n_samples=600)
    plot_error_grid(results_df, max_errors=40)

    print("\nAll done.")

