"""
Step 11: Viability screen for candidate SERVICE-DELIVERY targets in the GHS person files.

For each candidate: year coverage, in-scope population, class balance, and a quick
predictability test using only legitimate (non-leaking) features.
"""
import warnings, glob, json, os
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import ghs_harmonise as gh
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

DATA = gh.DATA_DIR
SCHEMA = gh.SCHEMA

# candidate target -> per-year source column aliases
CAND = {
 "nsnp_provided":  ["q122afood", "edu_food"],
 "nsnp_eaten":     ["q122befood", "edu_eatfood"],
 "nofee_school":   ["q117rnfe", "edu_nofees"],
 "bursary":        ["q118burs", "edu_burs"],
 "attending":      ["q110atte", "edu_attend"],
 "ecd_attending":  ["q17atechild", "q16atechild", "ecd_chldatt"],
 "hlt_consulted":  ["q127acon", "q24acon", "hlt_consult"],
}
FEATS = ["uqnr","age","sex","race","prov","geotype","educ","medaid","disab","weight",
         "wage","bus","vol","ret","look","sto"]

def src(y, aliases):
    actual = {c.strip().strip('"').lower(): c.strip().strip('"') for c in SCHEMA[y]["columns"]}
    for a in aliases:
        if a in actual: return actual[a]
    return None

print("=" * 110)
print("A. YEAR COVERAGE OF CANDIDATE TARGETS")
print("=" * 110)
print(f"{'candidate':<16}" + "".join(f"{y:>6}" for y in gh.YEARS) + "   years")
for k, al in CAND.items():
    row = ["Y" if src(y, al) else "-" for y in gh.YEARS]
    print(f"{k:<16}" + "".join(f"{v:>6}" for v in row) + f"   {row.count('Y')}/11")

frames = []
for y in gh.YEARS:
    d = gh.load_year(y, FEATS)
    d["lab_status"] = gh.derive_labour_status(d)
    extra = {k: src(y, al) for k, al in CAND.items()}
    extra = {k: v for k, v in extra.items() if v}
    if extra:
        raw = pd.read_csv(glob.glob(os.path.join(DATA, f"ghs-{y}-person-*.csv"))[0],
                          encoding=SCHEMA[y]["encoding"],
                          usecols=list(dict.fromkeys(extra.values())), dtype=str, low_memory=False)
        for k, col in extra.items():
            d[k] = gh._to_yesno(raw[col], y).values
    for k in CAND:
        if k not in d.columns: d[k] = np.nan
    frames.append(d)
p = pd.concat(frames, ignore_index=True)
for c in p.columns:
    if p[c].dtype == object:
        p[c] = p[c].astype(object).where(p[c].notna(), np.nan)
print(f"\npooled: {len(p):,} records")

CATS = ["sex","race","prov","geotype","educ_band","medaid","disab","lab_status"]
NUMS = ["age"]

def auc_for(d, tcol):
    d = d.dropna(subset=[tcol])
    d = d[d[tcol].isin(["Yes","No"])]
    if d[tcol].nunique() < 2: return np.nan, 0, np.nan
    d = d.copy(); d["t"] = (d[tcol] == "Yes").astype(int)
    tr, te = d[d["year"] <= 2019], d[d["year"] >= 2021]
    prior = d["t"].mean()
    if len(tr) < 500 or len(te) < 300 or te["t"].nunique() < 2:
        return np.nan, len(d), prior
    pipe = Pipeline([("pre", ColumnTransformer([
        ("n", Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]), NUMS),
        ("c", Pipeline([("i",SimpleImputer(strategy="constant",fill_value="NA")),
                        ("o",OneHotEncoder(handle_unknown="ignore",min_frequency=50,sparse_output=False))]), CATS)])),
        ("clf", HistGradientBoostingClassifier(max_iter=200, random_state=0))])
    pipe.fit(tr[NUMS+CATS], tr["t"])
    pr = pipe.predict_proba(te[NUMS+CATS])[:,1]
    return roc_auc_score(te["t"], pr, sample_weight=te["weight"]), len(d), prior

print("\n" + "=" * 110)
print("B. VIABILITY: in-scope N, class balance, and held-out AUC from demographics only")
print("=" * 110)
rows = []
scopes = {
 "nsnp_provided": ("learners 6-18", p[p["age"].between(6,18)]),
 "nsnp_eaten":    ("learners 6-18", p[p["age"].between(6,18)]),
 "nofee_school":  ("learners 6-18", p[p["age"].between(6,18)]),
 "bursary":       ("learners 6-24", p[p["age"].between(6,24)]),
 "attending":     ("children 7-17", p[p["age"].between(7,17)]),
 "ecd_attending": ("children 0-6",  p[p["age"].between(0,6)]),
 "hlt_consulted": ("all ages",      p),
}
for k,(label,d) in scopes.items():
    auc, n, prior = auc_for(d, k)
    yrs = sum(1 for y in gh.YEARS if src(y, CAND[k]))
    rows.append({"target":k, "scope":label, "years":f"{yrs}/11", "n_labelled":n,
                 "pct_Yes":round(prior*100,1) if prior==prior else np.nan,
                 "AUC_demog_only":round(auc,3) if auc==auc else np.nan})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 110)
print("C. NSNP (school feeding) DETAIL - weighted % of learners 6-18 whose school provides food")
print("=" * 110)
L = p[p["age"].between(6,18)]
for col in ["nsnp_provided","nsnp_eaten","nofee_school"]:
    d = L.dropna(subset=[col]); d = d[d[col].isin(["Yes","No"])]
    if len(d)==0: continue
    g = (d[d[col].eq("Yes")].groupby("year")["weight"].sum() / d.groupby("year")["weight"].sum()*100).round(1)
    print(f"\n{col}:"); print(g.to_string())

print("\n" + "=" * 110)
print("D. NSNP provision by province and settlement type (weighted %, pooled)")
print("=" * 110)
d = L.dropna(subset=["nsnp_provided"]); d = d[d["nsnp_provided"].isin(["Yes","No"])]
for by in ["prov","geotype","race"]:
    g = (d[d["nsnp_provided"].eq("Yes")].groupby(by)["weight"].sum()
         / d.groupby(by)["weight"].sum()*100).round(1).sort_values(ascending=False)
    print(f"\nby {by}:"); print(g.to_string())
