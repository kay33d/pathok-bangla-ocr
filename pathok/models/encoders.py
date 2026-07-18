"""Visual encoders evaluated in the thesis: ResNet-18 and the TrOCR ViT encoder.

Both produce a sequence of d_model-dimensional feature vectors that the
Transformer decoder cross-attends to.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Encoder(nn.Module):
    """ResNet-18 backbone; the spatial feature map is flattened to a sequence."""

    def __init__(self, d_model=256, pretrained=True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(*list(backbone.children())[:-2])  # -> (B, 512, H/32, W/32)
        self.proj = nn.Linear(512, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, images):
        feat = self.stem(images)                       # (B, 512, h, w)
        feat = feat.flatten(2).transpose(1, 2)         # (B, h*w, 512)
        feat = self.proj(feat)                         # (B, h*w, d_model)
        return feat + self.pos[:, : feat.size(1)]


class TrOCREncoder(nn.Module):
    """Pretrained TrOCR ViT encoder (best-performing encoder in the thesis)."""

    def __init__(self, d_model=256, model_name="microsoft/trocr-base-stage1"):
        super().__init__()
        from transformers import VisionEncoderDecoderModel

        self.vit = VisionEncoderDecoderModel.from_pretrained(model_name).encoder
        self.proj = nn.Linear(self.vit.config.hidden_size, d_model)

    def forward(self, images):
        hidden = self.vit(pixel_values=images).last_hidden_state  # (B, N, 768)
        return self.proj(hidden)


def build_encoder(kind: str, d_model: int, trocr_name=None) -> nn.Module:
    kind = kind.lower()
    if kind == "resnet18":
        return ResNet18Encoder(d_model)
    if kind == "trocr":
        return TrOCREncoder(d_model, trocr_name or "microsoft/trocr-base-stage1")
    raise ValueError(f"unknown encoder: {kind}")
