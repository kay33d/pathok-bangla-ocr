"""Evaluation metrics: Character Error Rate, Word Error Rate, Word Accuracy.

CER is computed over Unicode code points; WER treats each word image's full
label as one word (word-level dataset), so WER equals the fraction of words
with at least one error unless multi-word labels are present.
"""


def _edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(preds, refs):
    errors = sum(_edit_distance(list(p), list(r)) for p, r in zip(preds, refs))
    total = sum(len(r) for r in refs)
    return errors / max(total, 1)


def wer(preds, refs):
    errors = sum(_edit_distance(p.split(), r.split())
                 for p, r in zip(preds, refs))
    total = sum(len(r.split()) for r in refs)
    return errors / max(total, 1)


def word_accuracy(preds, refs):
    return sum(p == r for p, r in zip(preds, refs)) / max(len(refs), 1)
