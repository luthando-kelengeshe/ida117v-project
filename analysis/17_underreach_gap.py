"""Does the model actually produce the stakeholder decision the brief asks for?
The brief wants 'identify need / improve reach of a service'. Our model predicts
RECEIPT. The decision needs the GAP: children whose profile matches recipients
but who report no grant. Quantify that cell."""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding="utf-8")
d = np.load("out/val_test_probs_csg_takeup.npz")
cal = joblib.load("out/models/calibrators_csg_takeup.joblib")["calibrators"]["MLP (PyTorch)"]
p_raw, y = d["MLP (PyTorch)|test"], d["y_test"]
p = np.clip(cal.predict(p_raw), 0, 1)

panel = pd.read_parquet("out/analysis_panel.parquet")
t = panel[panel["age"].between(0,17)].dropna(subset=["grant_any"])
t = t[t["split"]=="test"].reset_index(drop=True)
assert len(t)==len(y), (len(t), len(y))
t["p"] = p; t["y"] = y

print(f"test children: {len(t):,}   observed grant rate: {t['y'].mean():.3f}")
print()
print("THE DECISION CELL: profile looks like a recipient, but no grant reported")
print("-"*78)
for thr in [0.7, 0.8, 0.9]:
    m = (t["p"]>=thr) & (t["y"]==0)
    wsum = t.loc[m,"weight"].sum()
    print(f"  p>={thr:.1f} & no grant : {m.sum():>6,} sampled children "
          f"-> {wsum/1e3:>7.0f}k children nationally (weighted)")
print()
m = (t["p"]>=0.8) & (t["y"]==0)
print("Who are they? (p>=0.8, no grant reported)  vs all test children")
for col in ["prov","geotype","race","medaid"]:
    a = t.loc[m].groupby(col)["weight"].sum() / t.loc[m,"weight"].sum()*100
    b = t.groupby(col)["weight"].sum() / t["weight"].sum()*100
    cmp = pd.DataFrame({"under-reached %": a.round(1), "all children %": b.round(1)})
    cmp["lift"] = (cmp["under-reached %"]/cmp["all children %"]).round(2)
    print(f"\n-- {col} --"); print(cmp.sort_values("under-reached %", ascending=False).head(5).to_string())
print()
print("Contrast: the OTHER error direction (p<0.2 but IS receiving a grant)")
m2 = (t["p"]<0.2) & (t["y"]==1)
print(f"  {m2.sum():,} sampled -> {t.loc[m2,'weight'].sum()/1e3:.0f}k nationally")
