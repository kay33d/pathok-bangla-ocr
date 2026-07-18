"""Central configuration for PATHOK experiments."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    encoder: str = "trocr"          # "trocr" | "resnet18"
    tokenizer: str = "grapheme"     # "grapheme" | "char" | "bpe"
    d_model: int = 256
    nhead: int = 8
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 40           # max tokens per word (incl. BOS/EOS)
    trocr_name: str = "microsoft/trocr-base-stage1"


@dataclass
class TrainConfig:
    data_root: str = "data/bnhtrd"
    out_dir: str = "runs"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    encoder_lr: float = 3e-5        # lower LR when fine-tuning pretrained TrOCR encoder
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    num_workers: int = 4
    seed: int = 42
    # Image size: TrOCR expects 384x384; ResNet uses a wide word strip.
    image_size: tuple = field(default=None)

    def resolved_image_size(self, encoder: str):
        if self.image_size is not None:
            return self.image_size
        return (384, 384) if encoder == "trocr" else (64, 256)


@dataclass
class SceneTrainConfig(TrainConfig):
    """Three-phase synthetic-to-real curriculum for scene text.

    Phase 1: synthetic scene text only.
    Phase 2: synthetic + real scene text mixed.
    Phase 3: real scene text only (fine-tune).
    The model is initialised from the best handwritten checkpoint.
    """

    init_checkpoint: str = "runs/trocr_grapheme/best.pt"
    phase: int = 1
    synth_root: str = "data/synth_scene"
    real_root: str = "data/iiit_indic_bn"
