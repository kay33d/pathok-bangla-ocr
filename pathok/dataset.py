"""Word-image dataset loading for BnHTRD, IIIT-Indic, and synthetic scene text.

Expected layout for each split (train/val/test):

    <data_root>/
        labels.csv          # columns: image, label, split   (utf-8)
        images/<image>      # word-level crops

BnHTRD word crops use a 70/15/15 split in the thesis (46,297 train /
9,923 val / 6,614 test after filtering).
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_labels(data_root, split):
    rows = []
    with open(Path(data_root) / "labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split", "train") == split:
                rows.append((row["image"], row["label"]))
    return rows


def build_transform(image_size, train):
    h, w = image_size
    aug = []
    if train:
        aug = [
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=2, translate=(0.02, 0.02),
                                         scale=(0.9, 1.1), fill=255)], p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.3),
            transforms.RandomApply(
                [transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
        ]
    return transforms.Compose([
        transforms.Resize((h, w)),
        *aug,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class WordImageDataset(Dataset):
    def __init__(self, data_root, split, tokenizer, image_size, max_seq_len=40,
                 train=None):
        self.data_root = Path(data_root)
        self.samples = read_labels(data_root, split)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = build_transform(
            image_size, train if train is not None else split == "train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        img = Image.open(self.data_root / "images" / name).convert("RGB")
        ids = self.tokenizer.encode(label)[: self.max_seq_len]
        return self.transform(img), torch.tensor(ids, dtype=torch.long), label


def collate(batch, pad_id):
    imgs, seqs, labels = zip(*batch)
    max_len = max(s.numel() for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : s.numel()] = s
    return torch.stack(imgs), padded, list(labels)
