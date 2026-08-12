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

## Deviations from the pasted originals

These files are transcriptions of Kaggle notebook cells. Two edits were made for the repo:

- `pathok_stage2_gate.py`: the trailing `gate, registry = main()` is wrapped in
  `if __name__ == "__main__":` so importing the module does not start a ~100-minute run.
- `stage1_null_control.py` was split out of the `stage1_pruning.py` cell. It reuses `ctx`,
  `S`, and helpers from that script and must be run after it in the same session.

`stage1_pruning.py` still executes `quick_check()` on import via `RUN_GATE_NOW = True`;
set that to `False` to only define the functions.
