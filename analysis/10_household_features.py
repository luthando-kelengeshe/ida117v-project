"""
Step 10: Test whether HOUSEHOLD-CONTEXT features (built by self-joining the person
file on uqnr) beat raw person-level columns.

Rationale: social grants are means-tested at household level, and the child support
grant is driven by the CAREGIVER's circumstances, not the child's. A person-level
row for a 4-year-old carries almost no predictive information on its own.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import ghs_harmonise as gh

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

feats = ["uqnr", "grant", "age", "sex", "race", "prov", "geotype", "educ", "attend",
         "medaid", "disab", "wage", "bus", "vol", "ret", "look", "sto",
         "salary", "weight"]

frames = []
for y in gh.YEARS:
    d = gh.load_year(y, feats)
    d["lab_status"] = gh.derive_labour_status(d)
    frames.append(d)
p = pd.concat(frames, ignore_index=True)
p["target"] = p["grant"].map({"Yes": 1, "No": 0})
p = p.dropna(subset=["target"]).reset_index(drop=True)
p["target"] = p["target"].astype(int)
for c in p.columns:
    if p[c].dtype == object:
        p[c] = p[c].astype(object).where(p[c].notna(), np.nan)

# ---------------------------------------------------------- household features
print("Building household-context features by self-join on uqnr ...")
p["hh"] = p["year"].astype(str) + "_" + p["uqnr"].astype(str)

g = p.groupby("hh")
hh = pd.DataFrame({
    "hh_size": g.size(),
    "hh_n_child_0_17": g["age"].apply(lambda s: (s <= 17).sum()),
    "hh_n_elderly_60p": g["age"].apply(lambda s: (s >= 60).sum()),
    "hh_mean_age": g["age"].mean(),
    "hh_max_age": g["age"].max(),
    "hh_n_employed": g["lab_status"].apply(lambda s: (s == "Employed").sum()),
    "hh_n_unemployed": g["lab_status"].apply(lambda s: (s == "Unemployed").sum()),
    "hh_any_medaid": g["medaid"].apply(lambda s: (s == "Yes").any().astype(int)),
    "hh_n_disabled": g["disab"].apply(lambda s: (s == "Disabled").sum()),
    "hh_total_salary": g["salary"].sum(min_count=1),
    "hh_max_educ": g["educ_band"].apply(
        lambda s: max((gh.EDUC_ORDER.index(v) for v in s.dropna()), default=np.nan)),
})
hh["hh_dependency_ratio"] = ((hh["hh_n_child_0_17"] + hh["hh_n_elderly_60p"])
                             / hh["hh_size"].clip(lower=1))
hh["hh_workers_per_capita"] = hh["hh_n_employed"] / hh["hh_size"].clip(lower=1)
hh["hh_salary_per_capita"] = hh["hh_total_salary"] / hh["hh_size"].clip(lower=1)
hh["hh_no_earner"] = (hh["hh_n_employed"] == 0).astype(int)

p = p.merge(hh, left_on="hh", right_index=True, how="left")
print(f"  {len(hh):,} households; median size {hh['hh_size'].median():.0f}\n")

PERSON_CAT = ["sex", "race", "prov", "geotype", "educ_band", "attend", "medaid",
              "disab", "lab_status"]
PERSON_NUM = ["age"]
HH_NUM = ["hh_size", "hh_n_child_0_17", "hh_n_elderly_60p", "hh_mean_age",
          "hh_max_age", "hh_n_employed", "hh_n_unemployed", "hh_n_disabled",
          "hh_max_educ", "hh_dependency_ratio", "hh_workers_per_capita",
          "hh_salary_per_capita", "hh_no_earner", "hh_any_medaid"]


def run(name, num, cat, data):
    tr, te = data[data["year"] <= 2019], data[data["year"] >= 2021]
    blocks = []
    if num:
        blocks.append(("n", Pipeline([("i", SimpleImputer(strategy="median")),
                                      ("s", StandardScaler())]), num))
    if cat:
        blocks.append(("c", Pipeline([
            ("i", SimpleImputer(strategy="constant", fill_value="__NA__")),
            ("o", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                sparse_output=False))]), cat))
    pipe = Pipeline([("pre", ColumnTransformer(blocks)),
                     ("clf", HistGradientBoostingClassifier(max_iter=250, random_state=0))])
    pipe.fit(tr[num + cat], tr["target"])
    pr = pipe.predict_proba(te[num + cat])[:, 1]
    auc = roc_auc_score(te["target"], pr, sample_weight=te["weight"])
    acc = accuracy_score(te["target"], (pr > .5).astype(int), sample_weight=te["weight"])
    print(f"  {name:<44} AUC={auc:.3f}  acc={acc:.3f}")
    return auc


for label, d in [
    ("CHILDREN 0-17  (child support grant)", p[p["age"].between(0, 17)]),
    ("OLDER PERSONS 60+  (old-age grant)",   p[p["age"] >= 60]),
    ("WORKING AGE 18-59  (disability / SRD)", p[p["age"].between(18, 59)]),
    ("FULL POPULATION",                       p),
]:
    print("=" * 92)
    print(f"{label}   n={len(d):,}   grant rate={d['target'].mean():.3f}")
    print("=" * 92)
    a = run("person-level columns only", PERSON_NUM, PERSON_CAT, d)
    b = run("+ household-context features", PERSON_NUM + HH_NUM, PERSON_CAT, d)
    c = run("household features only (no person cols)", HH_NUM, [], d)
    print(f"  {'':<44} household features add {b - a:+.3f} AUC\n")
