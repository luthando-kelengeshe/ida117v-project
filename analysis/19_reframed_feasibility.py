"""
Step 19: Feasibility of the REFRAMED task.

  population : children 0-17 in households with no employed member and no medical aid
               (near-certainly income-eligible for the Child Support Grant)
  target     : 1 = NOT receiving a grant  (the reach failure the stakeholder wants to find)
               0 = receiving a grant

If the model has no signal here the redesign is pointless, so test it before committing.
"""
import numpy as np, pandas as pd, sys
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import ghs_dataset as gd
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

p = pd.read_parquet("out/analysis_panel.parquet")
k = p[p["age"].between(0, 17)].dropna(subset=["grant_any"]).copy()
elig = k["hh_no_earner"].eq(1) & k["hh_any_medaid"].eq(0)
d = k[elig].copy()
d["y"] = (1 - d["grant_any"]).astype(int)     # positive class = reach failure
for c in d.columns:
    if d[c].dtype == object:
        d[c] = d[c].astype(object).where(d[c].notna(), np.nan)

print("CORRECTED weighted population (per survey year, not pooled)")
print("-" * 86)
w = d.groupby("year").apply(
    lambda s: pd.Series({
        "eligible_children_M": s["weight"].sum() / 1e6,
        "not_receiving_M": s.loc[s["y"].eq(1), "weight"].sum() / 1e6,
        "gap_rate_%": s.loc[s["y"].eq(1), "weight"].sum() / s["weight"].sum() * 100}),
    include_groups=False)
print(w.round(2).to_string())
print(f"\nmost recent year (2022): {w.loc[2022,'not_receiving_M']:.2f} million eligible-by-proxy "
      f"children not receiving a grant ({w.loc[2022,'gap_rate_%']:.1f}% of that group)")

NUM = gd.PERSON_NUM + gd.HH_NUM
CAT = gd.PERSON_CAT
tr, va, te = d[d.split == "train"], d[d.split == "val"], d[d.split == "test"]
print(f"\ntrain {len(tr):,} (pos {tr.y.mean():.3f}) | val {len(va):,} (pos {va.y.mean():.3f}) "
      f"| test {len(te):,} (pos {te.y.mean():.3f})")

def pre():
    return ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
        ("c", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="NA")),
                        ("o", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                            sparse_output=False))]), CAT)])

print("\n" + "=" * 86)
print("FEASIBILITY on the reframed target (validation, survey-weighted)")
print("=" * 86)
rows = []
for name, clf in [
    ("Dummy (prior)", DummyClassifier(strategy="prior", random_state=42)),
    ("Logistic Regression", LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)),
    ("Gradient Boosting", HistGradientBoostingClassifier(max_iter=300, random_state=42)),
]:
    m = Pipeline([("pre", pre()), ("clf", clf)])
    m.fit(tr[NUM + CAT], tr["y"])
    pv = m.predict_proba(va[NUM + CAT])[:, 1]
    yh = (pv >= .5).astype(int)
    rows.append({"model": name,
                 "roc_auc": roc_auc_score(va["y"], pv, sample_weight=va["weight"]),
                 "pr_auc": average_precision_score(va["y"], pv, sample_weight=va["weight"]),
                 "macro_f1": f1_score(va["y"], yh, average="macro", sample_weight=va["weight"]),
                 "precision_pos": precision_score(va["y"], yh, zero_division=0, sample_weight=va["weight"]),
                 "recall_pos": recall_score(va["y"], yh, zero_division=0, sample_weight=va["weight"])})
r = pd.DataFrame(rows).set_index("model")
print(r.round(4).to_string())
print(f"\nbase rate (val) = {va['y'].mean():.3f}  -> PR-AUC above this means real signal")

# operational framing: top-decile targeting
best = Pipeline([("pre", pre()),
                 ("clf", HistGradientBoostingClassifier(max_iter=300, random_state=42))])
best.fit(tr[NUM + CAT], tr["y"])
pv = best.predict_proba(va[NUM + CAT])[:, 1]
print("\n" + "=" * 86)
print("OPERATIONAL VIEW: if outreach capacity is limited, who do we visit?")
print("=" * 86)
o = pd.DataFrame({"p": pv, "y": va["y"].values, "w": va["weight"].values})
o = o.sort_values("p", ascending=False)
base = np.average(o["y"], weights=o["w"])
print(f"{'targeting':<26}{'children reached':>18}{'hit rate':>11}{'lift':>8}{'of all gaps found':>20}")
tot_gap = o.loc[o.y.eq(1), "w"].sum()
for frac in [0.05, 0.10, 0.20, 0.30, 0.50]:
    nsel = int(len(o) * frac)
    s = o.iloc[:nsel]
    hit = np.average(s["y"], weights=s["w"])
    cov = s.loc[s.y.eq(1), "w"].sum() / tot_gap
    print(f"top {frac*100:>4.0f}% by risk score{'':<6}{s['w'].sum()/1e3:>14.0f}k{hit:>11.3f}"
          f"{hit/base:>8.2f}{cov*100:>19.1f}%")
print(f"\nrandom targeting hit rate = {base:.3f} (the baseline any model must beat)")
