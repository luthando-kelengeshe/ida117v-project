"""
Step 9: Quantify how much of 'social grant receipt' is explained by age alone,
by legitimate predictors, and by leaked columns. Then test the better-posed
framing: predicting TAKE-UP within an eligibility stratum.

Validation uses a year-based split (train <=2019, test 2021-2022) because the
records are a repeated cross-section and PSUs repeat within a year.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import ghs_harmonise as gh

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

feats = ["grant", "grant_oag", "grant_csg", "grant_dis", "age", "sex", "race",
         "prov", "geotype", "educ", "attend", "medaid", "disab", "wage", "bus",
         "vol", "ret", "look", "sto", "weight"]

frames = []
for y in gh.YEARS:
    d = gh.load_year(y, feats)
    d["lab_status"] = gh.derive_labour_status(d)
    frames.append(d)
p = pd.concat(frames, ignore_index=True)
p["target"] = p["grant"].map({"Yes": 1, "No": 0})
p = p.dropna(subset=["target"]).reset_index(drop=True)
p["target"] = p["target"].astype(int)

# sklearn's imputers cannot evaluate pd.NA in object columns; normalise to np.nan
for c in p.columns:
    if p[c].dtype == object:
        p[c] = p[c].astype(object).where(p[c].notna(), np.nan)
print(f"panel: {len(p):,}   overall grant rate (unweighted): {p['target'].mean():.3f}\n")

CATS = ["sex", "race", "prov", "geotype", "educ_band", "attend", "medaid",
        "disab", "lab_status"]
NUMS = ["age"]


def build(num, cat, model="gb"):
    blocks = []
    if num:
        blocks.append(("n", Pipeline([("i", SimpleImputer(strategy="median")),
                                      ("s", StandardScaler())]), num))
    if cat:
        blocks.append(("c", Pipeline([
            ("i", SimpleImputer(strategy="constant", fill_value="__NA__")),
            ("o", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                sparse_output=False))]), cat))
    pre = ColumnTransformer(blocks)
    clf = (HistGradientBoostingClassifier(max_iter=250, learning_rate=.1, random_state=0)
           if model == "gb" else
           LogisticRegression(max_iter=2000, n_jobs=-1))
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(name, num, cat, data=None, model="gb"):
    d = p if data is None else data
    tr = d[d["year"] <= 2019]
    te = d[d["year"] >= 2021]
    if len(tr) < 200 or len(te) < 200 or te["target"].nunique() < 2:
        print(f"{name:<46} (insufficient data)")
        return
    pipe = build(num, cat, model)
    pipe.fit(tr[num + cat], tr["target"])
    pr = pipe.predict_proba(te[num + cat])[:, 1]
    auc = roc_auc_score(te["target"], pr, sample_weight=te["weight"])
    acc = accuracy_score(te["target"], (pr > .5).astype(int), sample_weight=te["weight"])
    base = max(te["target"].mean(), 1 - te["target"].mean())
    print(f"{name:<46} AUC={auc:.3f}  acc={acc:.3f}  (majority-class={base:.3f})  "
          f"n_tr={len(tr):,} n_te={len(te):,}")
    return auc


print("=" * 108)
print("A. FULL POPULATION — how much is age alone?")
print("=" * 108)
evaluate("age only", ["age"], [])
evaluate("age + sex", ["age"], ["sex"])
evaluate("everything legitimate", NUMS, CATS)
evaluate("legitimate MINUS age", [], CATS)
print("\n  (leakage demo — do NOT use these)")
evaluate("+ grant_oag/csg/dis  [LEAKED]", NUMS, CATS + ["grant_oag", "grant_csg", "grant_dis"])

print("\n" + "=" * 108)
print("B. BETTER-POSED: take-up WITHIN an eligibility stratum")
print("=" * 108)
print("Here age no longer carries the answer, so the socioeconomic columns do the work.\n")

strata = {
    "Children 0-17 (child support grant)": p[p["age"].between(0, 17)],
    "Older persons 60+ (old-age grant)":   p[p["age"] >= 60],
    "Working age 18-59 (disability/SRD)":  p[p["age"].between(18, 59)],
}
for label, d in strata.items():
    print(f"--- {label}   n={len(d):,}  grant rate={d['target'].mean():.3f}")
    evaluate("    age only", ["age"], [], data=d)
    evaluate("    age + socioeconomic", NUMS, CATS, data=d)
    evaluate("    socioeconomic, no age", [], CATS, data=d)
    print()

print("=" * 108)
print("C. PERMUTATION-STYLE CHECK: drop one feature at a time (children stratum)")
print("=" * 108)
d = strata["Children 0-17 (child support grant)"]
full = evaluate("    all features", NUMS, CATS, data=d)
for drop in CATS:
    rest = [c for c in CATS if c != drop]
    a = evaluate(f"    without {drop}", NUMS, rest, data=d)
    if a and full:
        print(f"        -> dropping {drop} changes AUC by {a - full:+.4f}")
