"""Autoregressive Transformer decoder shared by all encoder/tokenizer variants."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids, memory):
        """tgt_ids: (B, T) token ids; memory: (B, N, d_model) encoder features."""
        T = tgt_ids.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(
            T, device=tgt_ids.device)
        pad_mask = tgt_ids.eq(self.pad_id)
        x = self.pos(self.embed(tgt_ids))
        x = self.decoder(x, memory, tgt_mask=causal,
                         tgt_key_padding_mask=pad_mask)
        return self.head(x)                            # (B, T, vocab)
