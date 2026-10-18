"""Emit a XeLaTeX document visualising how each PATHOK tokenizer arm carves a word."""
import unicodedata

import regex
import sentencepiece as spm
from tokenizers import Tokenizer

BASE = r"D:/claude files/tokenizer_models"
WORDS = ["সংস্কৃতি", "শিক্ষার্থী"]

SPM_ARMS = [
    ("BPE-500", "bangla_spm_vocab500.model"),
    ("BPE-2000", "bangla_spm.model"),
    ("BPE-4000", "bangla_spm_vocab4000.model"),
    ("Unigram-500", "bangla_spm_unigram500.model"),
    ("Unigram-2000", "bangla_spm_unigram.model"),
]

# layout, centimetres
TOTAL_W, GAP, H, VGAP, MINW = 11.0, 0.07, 0.54, 0.25, 0.34


def visible(t):
    if t.startswith("\u2581"):
        t = t[1:]
    if t.startswith("##"):
        t = t[2:]
    return t


def orphaned(vis):
    """True if the piece opens with a combining mark, so it cannot stand alone."""
    return bool(vis) and unicodedata.category(vis[0]) in ("Mn", "Mc")


def tokenize_all(word):
    rows = [("Character", list(word)), ("Grapheme", regex.findall(r"\X", word))]
    for name, fn in SPM_ARMS:
        sp = spm.SentencePieceProcessor(model_file=f"{BASE}/{fn}")
        rows.append((name, sp.encode(word, out_type=str)))
    wp = Tokenizer.from_file(f"{BASE}/bangla_wordpiece.json")
    rows.append(("WordPiece-2000", wp.encode(word).tokens))
    return rows


def emit_panel(word):
    rows = tokenize_all(word)
    ncp = len(word)
    out = [r"\begin{tikzpicture}[font=\footnotesize]"]
    for i, (name, toks) in enumerate(rows):
        y = -i * (H + VGAP)
        vis = [visible(t) for t in toks]
        vis = [v for v in vis if v]                       # drop bare '▁' pieces
        avail = TOTAL_W - GAP * (len(vis) - 1)
        bold = r"\bfseries " if name == "Grapheme" else ""
        out.append(rf"  \node[anchor=east,align=right] at (-0.18,{y:.2f}) "
                   rf"{{{bold}{name}\;({len(vis)})}};")
        x = 0.0
        for v in vis:
            w = max(MINW, len(v) / ncp * avail)
            fill = "orphan" if orphaned(v) else ("graph" if name == "Grapheme" else "white")
            out.append(
                rf"  \node[draw=gray!60,fill={fill},rounded corners=1.5pt,"
                rf"minimum height={H}cm,text width={max(0.12, w - 0.10):.2f}cm,"
                rf"align=center,anchor=west,inner sep=1pt] "
                rf"at ({x:.2f},{y:.2f}) {{\bn {v}}};")
            x += w + GAP
    out.append(r"\end{tikzpicture}")
    return "\n".join(out), rows


PRE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1.5cm]{geometry}
\usepackage{fontspec}
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{caption}
\setmainfont{Latin Modern Roman}
\newfontfamily\bn{Nirmala UI}[Script=Bengali]
\definecolor{orphan}{RGB}{250,231,205}
\definecolor{graph}{RGB}{223,238,246}
\setlength{\parindent}{0pt}
\pagestyle{empty}
\begin{document}

{\Large\bfseries Tokenization strategies on Bangla words}\\[2pt]
{\footnotesize Each row shows how one tokenizer arm segments the same word; box width is
proportional to the share of the word's codepoints the piece covers.
\colorbox{graph}{\strut\ Grapheme\ } is the paper's proposed unit.
\colorbox{orphan}{\strut\ Shaded\ } pieces begin with a combining mark (virama or dependent
vowel sign) and therefore cannot stand alone as renderable text.}
\vspace{3mm}

"""

POST = r"""
\vspace{-3mm}
{\large\bfseries Reading the figure}\\[3pt]
{\footnotesize
\textbf{1. Grapheme is the only arm that both segments and stays renderable.}
Across the two words it produces 6 pieces and \emph{zero} that begin with a combining
mark. Every other segmenting arm orphans at least one diacritic.

\textbf{2. Zero orphans is not automatically good.} BPE-4000 and Unigram-2000 also score
zero --- but only because each emits the whole word as a single token. That is vocabulary
coverage, not segmentation: both words happen to be frequent enough to have been memorised
outright. On an out-of-vocabulary word these arms fall back to the same fragmentation their
smaller-vocabulary siblings show here.

\textbf{3. Subword arms split conjuncts mid-cluster.} BPE-500 cuts
{\bn সংস্কৃতি} after the virama, leaving {\bn স্} and a bare {\bn ৃ};
WordPiece-2000 produces {\bn ্ক}, which opens with a virama. These are byte- or
frequency-motivated boundaries that do not correspond to any unit a reader perceives.

\textbf{4. Character tokenisation orphans every diacritic} --- 9 of its 18 pieces across the
two words are combining marks with no base consonant.

\vspace{2mm}
\textbf{Orphaned pieces (both words combined):}
Character 9 \textbullet\ BPE-500 5 \textbullet\ Unigram-500 5 \textbullet\ WordPiece-2000 3
\textbullet\ BPE-2000 2 \textbullet\ Grapheme 0 \textbullet\ BPE-4000 0$^\dagger$
\textbullet\ Unigram-2000 0$^\dagger$.\quad
{\itshape $^\dagger$single-token memorisation, see note 2.}

\vspace{2mm}
{\scriptsize\itshape Grapheme segmentation via \texttt{regex} \texttt{\textbackslash X}
(Unicode UAX\,\#29), matching \texttt{GraphemeTokenizer.split()}. BPE/Unigram are the
SentencePiece models trained in
\texttt{kaidul007/trocr-new-claude-trocr-bpe} and \texttt{pathok-exp4-\{vocab500, vocab4000,
unigram500, unigram-tokenizer\}}; WordPiece from \texttt{pathok-exp4-wordpiece}.
SentencePiece \texttt{U+2581} and WordPiece \texttt{\#\#} markers stripped for display.}
}
\end{document}
"""

if __name__ == "__main__":
    body = []
    for w in WORDS:
        panel, rows = emit_panel(w)
        body.append(rf"{{\bfseries\bn {w}}}\quad{{\footnotesize({len(w)} codepoints)}}"
                    "\n\n\\vspace{2mm}\n" + panel + "\n\n\\vspace{9mm}\n")
    with open(f"{BASE}/tokenization_figure.tex", "w", encoding="utf-8") as f:
        f.write(PRE + "\n".join(body) + POST)
    print("wrote tokenization_figure.tex")
