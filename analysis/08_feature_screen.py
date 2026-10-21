"""
Step 8: Feature screening for predicting SOCIAL GRANT RECEIPT.

Three questions:
  1. Which columns are LEAKAGE (only observable because the person gets a grant)?
  2. How much of the target is mechanically determined by eligibility age thresholds?
  3. Among legitimate predictors, which carry real signal?
"""

import os
import warnings

import numpy as np
import pandas as pd

import ghs_harmonise as gh

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)

OUT = gh.OUT_DIR

# ---------------------------------------------------------------- 1. LEAKAGE
print("=" * 108)
print("1. LEAKAGE CHECK — are the grant-type columns only populated for grant recipients?")
print("=" * 108)
print("If a grant-type column is ~100% missing among non-recipients, it encodes the answer.\n")

for y in [2018, 2022]:
    df = gh.load_year(y, ["grant", "grant_oag", "grant_csg", "grant_dis", "grant_srd",
                          "age", "weight"])
    print(f"--- {y} ---")
    for c in ["grant_oag", "grant_csg", "grant_dis", "grant_srd"]:
        if c not in df.columns:
            continue
        got = df[df["grant"].eq("Yes")][c].notna().mean() * 100
        notgot = df[df["grant"].eq("No")][c].notna().mean() * 100
        print(f"  {c:<12} non-missing among recipients: {got:5.1f}%   "
              f"among non-recipients: {notgot:5.1f}%")
    print()

# ---------------------------------------------------------------- 2. build panel
feats = ["grant", "age", "sex", "race", "prov", "geotype", "educ", "attend",
         "medaid", "disab", "sevdisab", "undisab", "relation", "marital",
         "wage", "bus", "vol", "ret", "look", "sto", "salary", "weight"]

frames = []
for y in gh.YEARS:
    d = gh.load_year(y, feats)
    d["lab_status"] = gh.derive_labour_status(d)
    frames.append(d)
panel = pd.concat(frames, ignore_index=True)
panel["target"] = panel["grant"].map({"Yes": 1, "No": 0})
panel = panel.dropna(subset=["target"])
print(f"Analysis panel: {len(panel):,} records with a usable grant outcome\n")

# ---------------------------------------------------------------- 3. age structure
print("=" * 108)
print("2. HOW MUCH IS MECHANICAL? Grant receipt by single year of age (weighted %)")
print("=" * 108)
ag = (panel[panel["target"].eq(1)].groupby("age")["weight"].sum()
      / panel.groupby("age")["weight"].sum() * 100).round(1)
band = []
for lo, hi in [(0, 6), (7, 12), (13, 17), (18, 24), (25, 34), (35, 44),
               (45, 54), (55, 59), (60, 64), (65, 74), (75, 110)]:
    sel = panel[panel["age"].between(lo, hi)]
    r = sel[sel["target"].eq(1)]["weight"].sum() / sel["weight"].sum() * 100
    band.append({"age_band": f"{lo}-{hi}", "grant_rate_%": round(r, 1),
                 "n": len(sel)})
print(pd.DataFrame(band).to_string(index=False))
print("\nSingle-year detail around the old-age threshold (age 55-68):")
print(ag.loc[55:68].to_string())

# ---------------------------------------------------------------- 4. signal screen
print("\n" + "=" * 108)
print("3. SIGNAL SCREEN — weighted grant rate by category, plus spread")
print("=" * 108)


def screen(col):
    d = panel.dropna(subset=[col])
    num = d[d["target"].eq(1)].groupby(col)["weight"].sum()
    den = d.groupby(col)["weight"].sum()
    rate = (num / den * 100).round(1).sort_values(ascending=False)
    cov = d[col].notna().mean() * 100
    return rate, cov, rate.max() - rate.min()


for col in ["lab_status", "educ_band", "geotype", "prov", "race", "sex",
            "disab", "medaid", "relation", "marital", "attend"]:
    if col not in panel.columns:
        continue
    rate, cov, spread = screen(col)
    print(f"\n--- {col}   (coverage {cov:.1f}% of panel, spread {spread:.1f} pp) ---")
    print(rate.head(12).to_string())

# ---------------------------------------------------------------- 5. missingness
print("\n" + "=" * 108)
print("4. CANDIDATE PREDICTOR COVERAGE across the pooled panel")
print("=" * 108)
cand = ["age", "sex", "race", "prov", "geotype", "educ_band", "attend", "medaid",
        "disab", "sevdisab", "undisab", "relation", "marital", "lab_status", "salary"]
cov = pd.DataFrame({
    "non_null_%": (panel[cand].notna().mean() * 100).round(1),
    "n_unique": [panel[c].nunique() for c in cand],
}).sort_values("non_null_%", ascending=False)
print(cov.to_string())

# per-year availability of each candidate
print("\nPer-year non-null % (spot the variables that vanish in some years):")
peryear = panel.groupby("year")[cand].apply(lambda d: (d.notna().mean() * 100).round(0))
print(peryear.to_string())
