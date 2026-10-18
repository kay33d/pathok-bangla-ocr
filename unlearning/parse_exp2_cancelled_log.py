import json, re
import numpy as np
import pandas as pd

LOG = r"D:\claude files\kaggle_downloads\exp2_cancelled_log.txt"
OUT_DIR = r"D:\claude files\kaggle_downloads"

with open(LOG, encoding="utf-8") as f:
    raw = f.read()

# The file is a JSON-array-per-line dump; each stdout line has a "data" field.
lines = []
for m in re.finditer(r'"data":"((?:[^"\\]|\\.)*)"', raw):
    text = m.group(1)
    text = text.encode().decode("unicode_escape")
    lines.append(text)

header_re = re.compile(r"--- (\S+) \| k=(\d+)/writer \| (FORGET|UNSEEN CONTROL)")
step_forget_re = re.compile(r"step\s+(\d+)\s+f95\s+([\d.]+)\s+f20\s+([\d.]+)\s+retain\s+([\d.]+)")
step_unseen_re = re.compile(r"step\s+(\d+)\s+unseen_acc\s+([\d.]+)\s+retain\s+([\d.]+)")

rows = []
cur_arm = cur_k = cur_group = None
for line in lines:
    hm = header_re.search(line)
    if hm:
        cur_arm, cur_k, grp = hm.group(1), int(hm.group(2)), hm.group(3)
        cur_group = "forget" if grp == "FORGET" else "unseen_control"
        continue
    if cur_group == "forget":
        sm = step_forget_re.search(line)
        if sm:
            rows.append({"arm": cur_arm, "k": cur_k, "group": cur_group,
                        "step": int(sm.group(1)), "forget_full95": float(sm.group(2)),
                        "forget_restrict20": float(sm.group(3)), "retain_wACC": float(sm.group(4))})
    elif cur_group == "unseen_control":
        sm = step_unseen_re.search(line)
        if sm:
            rows.append({"arm": cur_arm, "k": cur_k, "group": cur_group,
                        "step": int(sm.group(1)), "unseen_acc": float(sm.group(2)),
                        "retain_wACC": float(sm.group(3))})

df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}\\exp2_recovered_curves.csv", index=False)
print(f"parsed {len(df)} step rows")
print(df.groupby(["arm", "group"]).size())

# Only keep (arm,k,group) triples that reached step 150 (fully completed)
complete = (df[df.step == 150][["arm", "k", "group"]]
            .drop_duplicates())
complete_set = set(map(tuple, complete.values))
print(f"\n{len(complete_set)} complete (arm,k,group) combinations out of "
      f"{df[['arm','k','group']].drop_duplicates().shape[0]} attempted")

def auc(sub, col):
    x = sub["step"].values.astype(float)
    y = sub[col].values.astype(float)
    return float(np.trapz(y, x) / max(x.max() - x.min(), 1e-9))

summ = []
for (arm, k, grp), sub in df.groupby(["arm", "k", "group"]):
    if (arm, k, grp) not in complete_set:
        continue
    sub = sub.sort_values("step")
    if grp == "forget":
        summ.append({"arm": arm, "k": k, "group": grp,
                     "start_acc": sub["forget_full95"].iloc[0],
                     "final_acc": sub["forget_full95"].iloc[-1],
                     "final_restrict": sub["forget_restrict20"].iloc[-1],
                     "auc": auc(sub, "forget_full95"),
                     "retain_start": sub["retain_wACC"].iloc[0],
                     "retain_final": sub["retain_wACC"].iloc[-1]})
    else:
        summ.append({"arm": arm, "k": k, "group": grp,
                     "start_acc": sub["unseen_acc"].iloc[0],
                     "final_acc": sub["unseen_acc"].iloc[-1],
                     "final_restrict": sub["unseen_acc"].iloc[-1],
                     "auc": auc(sub, "unseen_acc"),
                     "retain_start": sub["retain_wACC"].iloc[0],
                     "retain_final": sub["retain_wACC"].iloc[-1]})

Sm = pd.DataFrame(summ)
Sm.to_csv(f"{OUT_DIR}\\exp2_recovered_summary.csv", index=False)

piv = Sm.pivot_table(index=["arm", "k"], columns="group", values=["final_acc", "auc"])
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv.dropna()  # only rows with both forget and unseen_control present
piv["final_acc_gap"] = piv["final_acc_forget"] - piv["final_acc_unseen_control"]
piv["auc_gap"] = piv["auc_forget"] - piv["auc_unseen_control"]
piv = piv.reset_index().sort_values(["arm", "k"])
piv.to_csv(f"{OUT_DIR}\\exp2_recovered_speed_gap.csv", index=False)

print("\n" + "=" * 100)
print("RECOVERED EXPERIMENT 2 RESULTS (forget-writer vs. unseen-writer relearning speed)")
print("=" * 100)
for _, r in piv.iterrows():
    print(f"  {r['arm']:<20} k={int(r['k'])}  forget={r['final_acc_forget']:6.2f}%  "
          f"unseen={r['final_acc_unseen_control']:6.2f}%  gap={r['final_acc_gap']:+6.2f}pt  "
          f"AUCgap={r['auc_gap']:+6.2f}")

missing = df[["arm", "k", "group"]].drop_duplicates()
missing_incomplete = missing[~missing.apply(tuple, axis=1).isin(complete_set)]
print(f"\nIncomplete (never reached step 150), needs rerun: \n{missing_incomplete.to_string(index=False)}")
