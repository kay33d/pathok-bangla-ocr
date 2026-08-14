# Machine unlearning experiments on the PATHOK baseline

Can a writer's biometric identity be *deleted* from a handwriting-OCR model after
training, or only *hidden*? These scripts test the two-stage unlearning recipe of
Kang et al. against a TrOCR + grapheme model trained on BN-HTRd, with an exact-retrain
oracle as the reference for every claim.

All scripts are written for Kaggle (paths under `/kaggle/input`, `/kaggle/working`) and
are self-contained cells except the `exp*` files, which import `pathok_common.py`.
Edit `S.CKPT_ROOT` and `S.ROOT` at the top of each file before running.

## Setup

The baseline is retrained on a **writer-disjoint** split — the original PATHOK split
was random over rows, so all 125 writers appeared in train, val *and* test, which makes
test accuracy partly writer memorisation and invalidates the membership-inference
"non-member" class.

| Group | Writers | Words | Role |
|---|---|---|---|
| retain | 74 | 38,275 | members, kept |
| forget | 20 | 7,739 | members, to be deleted |
| test | 30 | 18,067 | non-members, never seen |

The writer head is a 95-way linear classifier on the encoder CLS token, trained jointly
with recognition.

## Files

| File | Purpose |
|---|---|
| `pathok_retrain_clean.py` | Retrain the baseline on the frozen writer-disjoint split; writes `split.json`, `vocab_clean.json`, `baseline_clean_final.pt` |
| `retrain_preprocessing_override.py` | Height+contrast preprocessing; paste after the retrain script, before `main()` |
| `mia_gate_v2.py` | Membership-inference gate on the baseline (writer-level CV, calibrated threshold) |
| `stage1_pruning.py` | Stage I: rank writer-specific neurons (Welch *t* on CLS activations) and sweep prune rates |
| `stage1_null_control.py` | Effect sizes, logit-margin metrics, and the retain-vs-retain null control — run after `stage1_pruning.py` in the same session |
| `pathok_stage2_gate.py` | Stage II unlearning arms (`wic_unmasked`, `random_label`) + gate v3 |
| `pathok_common.py` | Shared machinery for `exp1`–`exp5`: data, split, model, metrics, writer-level bootstrap |
| `exp1_retrain_oracle.py` | **The oracle.** Retrain with the 20 forget writers absent — exact deletion, the reference for everything else |
| `exp2_reprobe.py` | Freeze the encoder, discard the writer head, train a fresh 20-way probe on forget-writer embeddings |
| `exp3_relearn.py` | Few-shot relearning attack; steps-to-half recovery vs the oracle's from-scratch rate |
| `exp4_null_distribution.py` | Permutation test over 20 pseudo-forget draws + random-ranking arm for Stage I |
| `exp5_gate_v4.py` | Final MIA table, everything referenced to the oracle, with writer-cluster bootstrap CIs |
| `exp2b_probe_repair.py` | Replaces `exp2_reprobe.py`. Sample-efficiency few-shot curves + cosine verification (AUC/EER), fixed decoder-state probing. See "2026-08-14 additions" below. |
| `exp3b_relearn_repair.py` | Replaces `exp3_relearn.py`. Adds retain replay so the relearning attack can't hide behind head collapse; see below. |
| `prune_unlearn_5methods.py` | Self-contained Stage I+II pipeline covering all five of the paper's Stage II methods (WIC, Random Labeling, Fisher Forgetting, Amnesiac Unlearning, DELETE), not just the two `pathok_stage2_gate.py` covers. See below. |

Run order: `pathok_retrain_clean` → `mia_gate_v2` → `stage1_pruning` → `stage1_null_control`
→ `pathok_stage2_gate` → `exp1` (long) → `exp2`–`exp5`.

## Findings so far

**Stage I (neuron pruning) does not replicate.** The real forget/retain accuracy gap of
8.50 points at `ffn` 0.30 sits inside the null band produced by pruning on a *pseudo*-forget
set of 20 retain writers with nobody deleted (null mean −1.77, p95 12.20, max 16.00;
permutation p = 0.286). Median Cohen's *d* across ~46k neurons is 0.153, and only 0.23%
exceed *d* > 0.8 — the 73% of neurons with |t| > 2 is an artefact of n ≈ 960 per group,
not evidence of structure. Writer identity is distributed, not localised to prunable units.

**Stage II drives the writer head to 0% but the encoder still knows.** Both arms reach
forget wACC ≈ 0–1% (chance 1.33%), yet under the decoder-confidence MIA the forget writers
remain fully member-like (writer-level 95%, vs a 10–17% non-member rate).

**The oracle reframes "over-unlearning".** Exact retraining *also* gives forget wACC 0.00%
and a mean true-writer rank of 66.1/95, so a low rank is not by itself evidence of
over-unlearning — only a deviation from the oracle's rank distribution is. Under gate v4
the baseline scores U-gap 100 and the oracle 0, at near-identical test WER (18.00% vs 18.50%).

**Identity survives the encoder.** A frozen-encoder probe on forget writers recovers 99.93%
(baseline ceiling) against an 85.50% oracle floor — and the oracle floor is itself far above
the 5% chance rate, because a ViT trained on other handwriting is generically
style-discriminative. Probe splits are by *page* so this is not page recall.

**Relearning is fast even from exact deletion.** Fine-tuning the oracle on 5/10/25 words per
writer recovers 54%/73%/84% of baseline forget accuracy within 200 steps — but at the cost of
collapsing retain accuracy by >90 points, so it is not a clean recovery.

Numbers above come from single runs at seed 42; `exp1` supports multiple seeds via `O.SEEDS`.

## 2026-08-14 additions

A separate Kaggle account (`mohammedkai`) was independently running a *different*
Stage II attempt (`unlearning-privacy` kernel) in parallel with `pathok_stage2_gate.py`.
That one never worked: `UConfig.UNLEARN_BATCH_SIZE` was referenced but never defined,
so every Stage II call crashed with `AttributeError` before touching the model. Only
its Stage 0 checkpoint (`multihead_baseline/`) ever made it to disk — confirmed from
the run's own surviving output, not from reading the code. Whether `pathok_stage2_gate.py`
itself was ever actually executed to completion, or whether this README's numbers above
were transcribed from a run whose checkpoints are no longer attached anywhere, wasn't
established this session either way — the `wic_unmasked`/`random_label` checkpoints were
not present as an input to `pathok-common-revised` (the Kaggle notebook built around
`exp2b_probe_repair.py`/`exp3b_relearn_repair.py`) when this session started.

Given that, this session:

1. **Fixed `pathok-common-revised`'s own crash** — one forget writer (of 20) has a single
   labeled sample, so the page-disjoint split gives it zero training rows; the few-shot
   probe aborted its *entire* result set the moment any one class had an empty pool instead
   of excluding just that writer. `exp2b_probe_repair.py`/`exp3b_relearn_repair.py` above are
   the fixed versions (previously living only in the Kaggle notebook, not in this repo).
2. **Built `prune_unlearn_5methods.py`**, a from-scratch Stage I+II pipeline against the same
   canonical `split.json` / `baseline_clean_final.pt`, and ran it end-to-end (smoke-tested at
   tiny scale first, then the real 800-iteration run) — producing the first unlearning
   checkpoints this session could confirm actually exist and evaluate. Prune rates follow
   `stage1_pruning.py`'s groups (shallow/middle/deep) but **the null-distribution control from
   `exp4_null_distribution.py` was not re-run against this pass** — so, consistent with this
   README's own "Stage I does not replicate" finding above, the "pruning only" damage reported
   below should not be read as evidence pruning is targeting forget-specific structure. Every
   Stage II arm starts from the same pruned checkpoint regardless.
3. **Implemented Fisher Forgetting, Amnesiac Unlearning, and DELETE** — genuinely new; none of
   the three existed anywhere in this project before. Fisher's diagonal-update formula
   (`θ - α·g_f/(F+μ)`, full 334M parameters, `μ=1e-8`) proved numerically unstable at this
   scale: the model never recovered (CER stayed >180% through 760 fine-tuning steps). Reported
   as-is rather than re-tuned — a real negative result, not a bug to paper over.
4. **Extended `exp2b_probe_repair.py`/`exp3b_relearn_repair.py`'s arms** to cover all five
   methods and re-ran both audits.

**Corroboration.** Independent of `pathok_stage2_gate.py`'s "decoder-confidence MIA ... writer-level
95%" finding above, this session's own recognition-head MIA on WIC/Random-Labeling landed at
92–93% forget-seen-as-member — the same phenomenon, measured a different way, on a different run.

**New finding — the relearning attack works on the paper's own proposed defense, cleanly.**
`exp3_relearn.py`'s finding above (fast recovery from the oracle, but "at the cost of collapsing
retain accuracy by >90 points") is exactly the flaw `exp3b_relearn_repair.py`'s retain-replay
fix targets. Run against WIC/Random-Labeling/Amnesiac/DELETE (no oracle checkpoint was available
this session), the repaired attack recovers 87–90% of baseline forget-writer identifiability from
just 5 real examples per writer and 150 steps — **while retain accuracy stays flat** (all rows
`valid=True`, retain wACC moves by <3 points). That rules out the collapse confound: the recovery
isn't an artifact of the attack quietly retraining the whole model, it's the forget-writer
information resurfacing on its own. A second, independent measurement — 1-shot linear probes
directly on the frozen ViT encoder embeddings, no attack involved — agrees: forget writers sit
5–19 points above genuinely-unseen writers in separability, smallest for WIC, largest for
Amnesiac, the same ranking the attack produces. Two independently-built diagnostics converging
on "the head's output was suppressed, the representation wasn't" is the headline result of this
session's work.

One caveat this session did *not* resolve: this README's own "identity survives the encoder"
finding above shows even the **oracle** (zero training exposure) sits at 85.50% forget-writer
separability against a 5% chance rate — a ViT is generically style-discriminative for Bangla
handwriting regardless of memorisation. This session's few-shot probe compared forget writers
against a `test_reference` group (writers never in the training set at all) rather than against
a true oracle, since no oracle checkpoint existed to attach this session. The reported residual
gaps (WIC +11.2pt, Amnesiac +18.7pt, etc.) are real, but a rerun against `exp1_retrain_oracle.py`'s
output would be the more rigorous version of the same comparison.

## Deviations from the pasted originals

These files are transcriptions of Kaggle notebook cells. Two edits were made for the repo:

- `pathok_stage2_gate.py`: the trailing `gate, registry = main()` is wrapped in
  `if __name__ == "__main__":` so importing the module does not start a ~100-minute run.
- `stage1_null_control.py` was split out of the `stage1_pruning.py` cell. It reuses `ctx`,
  `S`, and helpers from that script and must be run after it in the same session.

`stage1_pruning.py` still executes `quick_check()` on import via `RUN_GATE_NOW = True`;
set that to `False` to only define the functions.
