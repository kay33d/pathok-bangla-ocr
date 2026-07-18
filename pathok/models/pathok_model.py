"""Full PATHOK encoder-decoder model."""

import torch
import torch.nn as nn

from .decoder import TextDecoder
from .encoders import build_encoder


class PathokModel(nn.Module):
    def __init__(self, cfg, vocab_size, pad_id, bos_id, eos_id):
        super().__init__()
        self.cfg = cfg
        self.bos_id, self.eos_id, self.pad_id = bos_id, eos_id, pad_id
        self.encoder = build_encoder(cfg.encoder, cfg.d_model, cfg.trocr_name)
        self.decoder = TextDecoder(
            vocab_size, cfg.d_model, cfg.nhead, cfg.num_decoder_layers,
            cfg.dim_feedforward, cfg.dropout, pad_id=pad_id)

    def forward(self, images, tgt_ids):
        """Teacher-forced logits for tgt_ids[:, :-1] predicting tgt_ids[:, 1:]."""
        memory = self.encoder(images)
        return self.decoder(tgt_ids[:, :-1], memory)

    @torch.no_grad()
    def generate(self, images, max_len=40):
        """Greedy autoregressive decoding."""
        memory = self.encoder(images)
        B = images.size(0)
        seq = torch.full((B, 1), self.bos_id, dtype=torch.long,
                         device=images.device)
        done = torch.zeros(B, dtype=torch.bool, device=images.device)
        for _ in range(max_len - 1):
            logits = self.decoder(seq, memory)
            nxt = logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            done |= nxt.eq(self.eos_id)
            if done.all():
                break
        return seq
