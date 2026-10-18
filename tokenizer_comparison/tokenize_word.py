"""Run every tokenizer arm from the PATHOK ablation on one Bangla word."""
import json
import sys

import regex
import sentencepiece as spm
from tokenizers import Tokenizer

BASE = r"D:/claude files/tokenizer_models"
WORD = sys.argv[1] if len(sys.argv) > 1 else "শিক্ষার্থী"


def visible(t):
    """Strip subword bookkeeping markers so boxes show script, not markers."""
    if t.startswith("\u2581"):
        t = t[1:]
    if t.startswith("##"):
        t = t[2:]
    return t


rows = [
    ("Character", list(WORD)),
    ("Grapheme", regex.findall(r"\X", WORD)),
]

SPM_ARMS = [
    ("BPE-500", "bangla_spm_vocab500.model"),
    ("BPE-2000", "bangla_spm.model"),
    ("BPE-4000", "bangla_spm_vocab4000.model"),
    ("Unigram-500", "bangla_spm_unigram500.model"),
    ("Unigram-2000", "bangla_spm_unigram.model"),
]
for name, fn in SPM_ARMS:
    sp = spm.SentencePieceProcessor(model_file=f"{BASE}/{fn}")
    rows.append((name, sp.encode(WORD, out_type=str)))

wp = Tokenizer.from_file(f"{BASE}/bangla_wordpiece.json")
rows.append(("WordPiece-2000", wp.encode(WORD).tokens))

ncp = len(WORD)
print(f"WORD = {WORD}   ({ncp} codepoints)\n")
out = []
for name, toks in rows:
    vis = [visible(t) for t in toks]
    cover = sum(len(v) for v in vis)
    flag = "" if cover == ncp else f"   <-- covers {cover}/{ncp} codepoints"
    print(f"{name:<16} {len(toks):>2} tokens   {' | '.join(vis)}{flag}")
    print(f"{'':<16}    raw:      {' | '.join(repr(t)[1:-1] for t in toks)}")
    out.append({"name": name, "tokens": toks, "visible": vis})

with open(f"{BASE}/tokenization_{WORD}.json", "w", encoding="utf-8") as f:
    json.dump({"word": WORD, "rows": out}, f, ensure_ascii=False, indent=1)
print(f"\nsaved -> {BASE}/tokenization_{WORD}.json")
