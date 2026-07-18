"""Tokenizers compared in the thesis: grapheme, character, and BPE.

Bangla words are composed of graphemes — a vowel, consonant, or consonant
conjunct (grapheme root), optionally combined with vowel/consonant diacritics.
Grapheme tokenization splits on extended grapheme clusters (Unicode UAX #29),
so a conjunct like ক্রা stays a single token instead of being fragmented into
ক + ্ + র + া.
"""

import json
from pathlib import Path

import regex  # supports \X (extended grapheme cluster)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


class BaseTokenizer:
    """Vocabulary built from the training labels; encodes text <-> id sequences."""

    def __init__(self, vocab=None):
        self.itos = list(vocab) if vocab else []
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    # -- to be implemented by subclasses -------------------------------------
    def split(self, text: str) -> list:
        raise NotImplementedError

    # -- shared API ----------------------------------------------------------
    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def bos_id(self):
        return self.stoi[BOS]

    @property
    def eos_id(self):
        return self.stoi[EOS]

    def __len__(self):
        return len(self.itos)

    def build(self, texts):
        units = set()
        for t in texts:
            units.update(self.split(t))
        self.itos = SPECIALS + sorted(units)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        return self

    def encode(self, text: str) -> list:
        unk = self.stoi[UNK]
        return (
            [self.bos_id]
            + [self.stoi.get(u, unk) for u in self.split(text)]
            + [self.eos_id]
        )

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            out.append(self.itos[i])
        return "".join(out)

    def save(self, path):
        Path(path).write_text(
            json.dumps({"type": type(self).__name__, "vocab": self.itos},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @staticmethod
    def load(path):
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        cls = {c.__name__: c for c in (GraphemeTokenizer, CharTokenizer, BPETokenizer)}
        return cls[blob["type"]](vocab=blob["vocab"])


class GraphemeTokenizer(BaseTokenizer):
    """Splits into extended grapheme clusters (best-performing in the thesis)."""

    def split(self, text):
        return regex.findall(r"\X", text)


class CharTokenizer(BaseTokenizer):
    """Splits into individual Unicode code points."""

    def split(self, text):
        return list(text)


class BPETokenizer(BaseTokenizer):
    """Byte Pair Encoding learned from the training labels."""

    def __init__(self, vocab=None, vocab_size=1000):
        super().__init__(vocab)
        self.vocab_size = vocab_size
        self._bpe = None

    def build(self, texts):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = Tokenizer(BPE(unk_token=UNK))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=SPECIALS)
        tok.train_from_iterator(texts, trainer)
        self._bpe = tok
        self.itos = [tok.id_to_token(i) for i in range(tok.get_vocab_size())]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        return self

    def split(self, text):
        if self._bpe is None:  # loaded from vocab only: greedy longest-match fallback
            return self._greedy_match(text)
        return self._bpe.encode(text).tokens

    def _greedy_match(self, text):
        units, i = [], 0
        max_len = max((len(t) for t in self.itos), default=1)
        while i < len(text):
            for j in range(min(len(text), i + max_len), i, -1):
                if text[i:j] in self.stoi:
                    units.append(text[i:j])
                    i = j
                    break
            else:
                units.append(text[i])
                i += 1
        return units


def build_tokenizer(kind: str, texts, **kwargs) -> BaseTokenizer:
    kind = kind.lower()
    if kind == "grapheme":
        return GraphemeTokenizer().build(texts)
    if kind == "char":
        return CharTokenizer().build(texts)
    if kind == "bpe":
        return BPETokenizer(**kwargs).build(texts)
    raise ValueError(f"unknown tokenizer: {kind}")
