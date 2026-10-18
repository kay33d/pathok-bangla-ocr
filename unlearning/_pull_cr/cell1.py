"""
exp0_diagnose_baseline.py
=========================
~3 min. RUN THIS BEFORE ANYTHING ELSE.

THE SYMPTOM
-----------
The baseline reports 48.25% writer accuracy on its own training members, when
the original retrain logged 100.00%. A 1-epoch oracle beats it by 40 points.
Nothing downstream is interpretable until this is explained.

THE DECISION TREE THIS RUNS
---------------------------
  A. Does the checkpoint's own stored gate dict say 100%?
       yes -> the checkpoint is fine, the CURRENT pipeline is broken. Go to C-F.
       no  -> the checkpoint is not the one you think it is. Go to B.
  B. Key-level load report, un-suppressed, plus writer-head weight statistics.
     A head that was reinitialised or shape-mismatched shows up here.
  C. Are the split sizes 38,275 / 7,739 / 2,015 / 18,067? If not, the dataframe
     changed (glob order, cache, or the label dedupe) and the split is not the
     frozen one.
  D. Are any wid values -1 in retain/forget? A silent map failure sends every
     row to -1 and accuracy collapses without an error.
  E. PER-WRITER accuracy histogram. THIS IS THE KEY TEST:
       bimodal (writers at ~100% and ~0%) -> index/permutation problem on a
         subset of classes
       uniformly degraded (most writers 30-70%) -> input pipeline drift, i.e.
         the images the model sees now differ from the ones it trained on
  F. Confusion offset: is predicted_wid - true_wid a constant, or does it map
     through a consistent permutation? Reveals an off-by-k or a sorted() vs
     checkpoint w2i mismatch.
  G. Fast vs slow image processor, same batch, same weights. The transformers
     default flipped at some point; if this moves the number, that is the cause.

FREE RESULT WHILE YOU ARE HERE
------------------------------
Baseline forget CER, paired with the oracle's forget CER of 15.13%. Same 20
writers, same images, membership as the only manipulated variable. That
difference is the cleanest causal estimate of memorisation in the project and
it costs two extra minutes.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor


EXPECTED = {"retain": 38275, "forget": 7739, "val": 2015, "test": 18067,
            "total": 66096, "writers": 125, "classes": 95}


def hr(t=""):
    print("\n" + "=" * 78); print(f"  {t}"); print("-" * 78)


def main(cfg=S):
    hr("A. CHECKPOINT SELF-REPORT")
    p = os.path.join(cfg.CKPT_ROOT, cfg.CKPT_FILE)
    print(f"  file: {p}")
    print(f"  size: {os.path.getsize(p)/1e6:.1f} MB")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print(f"  top-level keys: {list(ck.keys())}")
    if "gate" in ck:
        print("  STORED GATE FROM THE ORIGINAL RUN:")
        print(json.dumps(ck["gate"], indent=4, default=float))
        print("  --> If this says retain/forget ACC ~100 and you now measure 48,")
        print("      the checkpoint is fine and the CURRENT pipeline is at fault.")
    else:
        print("  [!] no 'gate' key -- this may not be baseline_clean_final.pt")

    hr("B. LOAD REPORT (un-suppressed) + WRITER HEAD STATISTICS")
    sd = ck["model"]
    w2i = {str(k): int(v) for k, v in ck["w2i"].items()}
    vocab = GraphemeVocab.load(os.path.join(cfg.CKPT_ROOT, cfg.VOCAB_FILE))
    model = build_model(vocab, len(w2i))
    res = model.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys]
    unexp = list(res.unexpected_keys)
    print(f"  missing    ({len(miss)}): {miss[:10]}")
    print(f"  unexpected ({len(unexp)}): {unexp[:10]}")
    crit = [k for k in miss if not k.startswith("ved.encoder.pooler")]
    print(f"  CRITICAL missing: {len(crit)} {crit[:10]}")

    W = sd.get("writer_head.weight"); B = sd.get("writer_head.bias")
    if W is None:
        print("  [!!] writer_head.weight ABSENT from the checkpoint -- the head is "
              "randomly initialised and 48% would be impossible; something else "
              "is going on")
    else:
        rn = W.float().norm(dim=1)
        print(f"  writer_head.weight  shape {tuple(W.shape)}  "
              f"|W| mean {W.float().abs().mean():.4f}  std {W.float().std():.4f}")
        print(f"  per-class row norms: min {rn.min():.3f}  med {rn.median():.3f}  "
              f"max {rn.max():.3f}  (a trained head has tight, similar norms; a "
              f"reinitialised one is near-uniform and small)")
        print(f"  bias: mean {B.float().mean():.4f}  std {B.float().std():.4f}")
        dead = int((rn < 0.1 * rn.median()).sum())
        print(f"  near-zero rows (possible never-supervised classes): {dead}")

    model = model.to(DEVICE).eval()

    hr("C. DATA + SPLIT RECONSTRUCTION")
    df = load_with_writers(cfg, use_cache=False)   # bypass cache deliberately
    print(f"  total {len(df):,} (expected {EXPECTED['total']:,})  "
          f"writers {df['writer'].nunique()} (expected {EXPECTED['writers']})")
    retain_df, forget_df, val_df, test_df, w2i2 = reload_split(df, cfg, w2i=w2i)
    got = {"retain": len(retain_df), "forget": len(forget_df),
           "val": len(val_df), "test": len(test_df)}
    for k, v in got.items():
        flag = "OK " if v == EXPECTED[k] else "!! "
        print(f"  {flag}{k:<7} {v:>7,}  expected {EXPECTED[k]:>7,}")
    if any(got[k] != EXPECTED[k] for k in got):
        print("  --> The dataframe is not the one the split was frozen against.")
        print("      Delete /kaggle/working/df_cache.parquet and check the label "
              "dedupe and the glob order.")

    hr("D. wid MAPPING")
    for nm, d in (("retain", retain_df), ("forget", forget_df), ("val", val_df)):
        bad = int((d["wid"] < 0).sum())
        print(f"  {nm:<7} rows {len(d):>7,} | wid==-1: {bad:>6,} "
              f"| unique wid {d['wid'].nunique():>3} | range "
              f"[{d['wid'].min()}, {d['wid'].max()}]")
        if bad:
            print(f"    [!!] {bad} rows failed the writer->index map. Check dtype: "
                  f"df['writer'] is {d['writer'].dtype}, w2i keys are str.")

    built = {w: i for i, w in enumerate(sorted(set(w2i)))}
    print(f"  checkpoint w2i == rebuilt sorted(): {w2i == built}")

    hr("E. PER-WRITER ACCURACY  (the key test)")
    fe = AutoImageProcessor.from_pretrained(cfg.TROCR_MODEL)
    per = {}
    for nm, d in (("retain", retain_df), ("forget", forget_df)):
        sub = d.sample(min(3000, len(d)), random_state=cfg.SEED).reset_index(drop=True)
        loader = DataLoader(WDataset(sub, vocab, fe, cfg), batch_size=cfg.BATCH,
                            shuffle=False, num_workers=cfg.NUM_WORKERS)
        hits, preds, trues = [], [], []
        with torch.no_grad():
            for b in loader:
                pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
                enc = model.ved.encoder(pixel_values=pv, return_dict=True)
                wl = model.writer_head(enc.last_hidden_state[:, 0, :])
                pr = wl.argmax(-1)
                hits.append((pr == wid).float().cpu())
                preds.append(pr.cpu()); trues.append(wid.cpu())
        hits = torch.cat(hits).numpy()
        preds = torch.cat(preds).numpy(); trues = torch.cat(trues).numpy()
        acc = 100 * hits.mean()
        t = pd.DataFrame({"writer": sub["writer"].values, "hit": hits})
        pw = t.groupby("writer")["hit"].mean() * 100
        per[nm] = (acc, pw, preds, trues)
        print(f"\n  [{nm}] overall {acc:.2f} %  over {len(pw)} writers")
        print(f"    per-writer acc: min {pw.min():.1f}  q25 {pw.quantile(.25):.1f}  "
              f"med {pw.median():.1f}  q75 {pw.quantile(.75):.1f}  max {pw.max():.1f}")
        hi = int((pw > 90).sum()); lo = int((pw < 10).sum())
        print(f"    writers >90%: {hi}   writers <10%: {lo}   in between: "
              f"{len(pw)-hi-lo}")
        if hi and lo and (hi + lo) > 0.7 * len(pw):
            print("    ==> BIMODAL. A subset of classes is systematically wrong.")
            print("        Cause is an index/permutation problem, not the images.")
            print(f"        broken writers: {sorted(pw[pw<10].index)[:15]}")
        elif pw.between(20, 80).mean() > 0.5:
            print("    ==> UNIFORMLY DEGRADED. Every writer is partly wrong.")
            print("        Cause is the input pipeline: the images the model sees "
                  "now differ from the ones it trained on. Go to section G.")
        else:
            print("    ==> Looks healthy.")

    hr("F. CONFUSION STRUCTURE (retain)")
    _, _, preds, trues = per["retain"]
    wrong = preds != trues
    if wrong.any():
        off = (preds[wrong] - trues[wrong])
        c = Counter(off.tolist()).most_common(5)
        print(f"  errors {wrong.sum():,}/{len(preds):,}")
        print(f"  most common (pred - true) offsets: {c}")
        if c and c[0][1] > 0.3 * wrong.sum():
            print(f"  ==> A single dominant offset of {c[0][0]} means the labels "
                  f"are shifted, not the model.")
        i2w = {v: k for k, v in w2i.items()}
        mm = Counter((i2w.get(int(t), t), i2w.get(int(p), p))
                     for t, p in zip(trues[wrong], preds[wrong])).most_common(8)
        print(f"  most common (true_writer -> pred_writer): {mm}")
    else:
        print("  no errors")

    hr("G. FAST vs SLOW IMAGE PROCESSOR")
    sub = retain_df.sample(400, random_state=cfg.SEED).reset_index(drop=True)
    for use_fast in (True, False):
        try:
            fe2 = AutoImageProcessor.from_pretrained(cfg.TROCR_MODEL,
                                                     use_fast=use_fast)
        except Exception as e:
            print(f"  use_fast={use_fast}: unavailable ({e})"); continue
        loader = DataLoader(WDataset(sub, vocab, fe2, cfg), batch_size=cfg.BATCH,
                            shuffle=False, num_workers=cfg.NUM_WORKERS)
        c = t = 0
        with torch.no_grad():
            for b in loader:
                pv = b["pixel_values"].to(DEVICE); wid = b["wid"].to(DEVICE)
                enc = model.ved.encoder(pixel_values=pv, return_dict=True)
                wl = model.writer_head(enc.last_hidden_state[:, 0, :])
                c += (wl.argmax(-1) == wid).sum().item(); t += len(wid)
        print(f"  use_fast={str(use_fast):<5}  retain wACC {100*c/max(t,1):6.2f} %")
    print("  ==> If these differ by more than a point, the processor default "
          "changed between your retrain and now, and that is the whole story.")

    hr("H. BASELINE RECOGNITION -- including the forget CER you need")
    out = {}
    for nm, d in (("retain", retain_df), ("forget", forget_df),
                  ("val", val_df), ("test", test_df)):
        q = rec_quality(model, d, vocab, fe, cfg, n=1000)
        out[nm] = q
        print(f"  [{nm:<6}] WER {q['WER']:6.2f} %  CER {q['CER']:5.2f} %  "
              f"runaway {q['runaway']}")
    print("\n  MEMORISATION ESTIMATE (fill in the oracle numbers from exp1):")
    print(f"    baseline forget CER : {out['forget']['CER']:.2f} %   (members)")
    print(f"    oracle   forget CER : 15.13 %   (same writers, non-members)")
    print(f"    difference          : {15.13 - out['forget']['CER']:.2f} points")
    print("    Same 20 writers, same images, membership as the only manipulated")
    print("    variable. This is the cleanest causal memorisation estimate you")
    print("    have -- much stronger than the retain-vs-test contrast, which")
    print("    confounds membership with writer identity.")

    return {"gate": ck.get("gate"), "per_writer": {k: v[1] for k, v in per.items()},
            "recognition": out}


if __name__ == "__main__":
    main()