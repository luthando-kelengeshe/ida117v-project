"""
Step 6: Build the harmonised 11-year panel and compute weighted national indicators.
This is the proof of what the dataset can actually deliver.
"""

import os
import warnings

import pandas as pd

import ghs_harmonise as gh

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

OUT = gh.OUT_DIR

print("=" * 110)
print("VARIABLE COVERAGE MATRIX  (Y = present in that year's file)")
print("=" * 110)
cov = gh.coverage_table()
print(cov.to_string())
cov.to_csv(os.path.join(OUT, "coverage_matrix.csv"))

frames = []
for y in gh.YEARS:
    df = gh.load_year(y)
    df["lab_status"] = gh.derive_labour_status(df)
    frames.append(df)
    print(f"loaded {y}: {len(df):,} rows")

panel = pd.concat(frames, ignore_index=True)
print(f"\nPooled panel: {len(panel):,} person-records x {panel.shape[1]} harmonised columns")

rows = []
for y, d in panel.groupby("year"):
    adults = d[d["age"] >= 20]
    youth = d[d["age"].between(15, 24)]
    wa = d[d["age"].between(15, 64)]
    kids = d[d["age"].between(7, 17)]
    emp = gh.wdist(wa, "lab_status")

    unemployed = emp.get("Unemployed", float("nan"))
    employed = emp.get("Employed", float("nan"))
    nea = emp.get("Not economically active", float("nan"))
    # official-style unemployment rate = unemployed / labour force
    lf = employed + unemployed
    rows.append({
        "year": y,
        "n": len(d),
        "pop_M": round(d["weight"].sum() / 1e6, 2),
        "median_age": gh.wmedian(d, "age"),
        "pct_female": gh.wshare(d, "sex", "Female"),
        "school_attend_7_17": gh.wshare(kids, "attend", "Yes"),
        "matric_plus_20up": round(
            gh.wdist(adults, "educ_band").reindex(["Matric (Grade 12)", "Post-school"]).sum(), 2),
        "no_schooling_20up": gh.wdist(adults, "educ_band").get("No schooling", float("nan")),
        "medical_aid": gh.wshare(d, "medaid", "Yes"),
        "social_grant": gh.wshare(d, "grant", "Yes"),
        "disability": gh.wshare(d[d["age"] >= 5], "disab", "Disabled"),
        "employed_rate_15_64": round(employed, 2),
        "unemp_rate_strict": round(unemployed / lf * 100, 2) if lf == lf else float("nan"),
        "nea_15_64": round(nea, 2),
        "median_salary_R": gh.wmedian(d[d["salary"].notna()], "salary") if d["salary"].notna().any() else float("nan"),
    })

ind = pd.DataFrame(rows).set_index("year")
print("\n" + "=" * 110)
print("NATIONAL WEIGHTED INDICATOR SERIES, 2012-2022")
print("=" * 110)
print(ind.to_string())
ind.to_csv(os.path.join(OUT, "indicators_national.csv"))

print("\n" + "=" * 110)
print("EDUCATIONAL ATTAINMENT, adults 20+, weighted % (shows the real structural shift)")
print("=" * 110)
att = gh.wdist(panel[panel["age"] >= 20], "educ_band", by="year").reindex(columns=gh.EDUC_ORDER)
print(att.to_string())
att.to_csv(os.path.join(OUT, "attainment_20plus.csv"))

print("\n" + "=" * 110)
print("UNEMPLOYMENT RATE (strict) BY PROVINCE, weighted %")
print("=" * 110)
wa = panel[panel["age"].between(15, 64)].copy()
wa["is_unemp"] = wa["lab_status"].eq("Unemployed")
wa["in_lf"] = wa["lab_status"].isin(["Employed", "Unemployed"])
lf = wa[wa["in_lf"]]
prov_un = (lf[lf["is_unemp"]].groupby(["year", "prov"])["weight"].sum()
           / lf.groupby(["year", "prov"])["weight"].sum() * 100).round(1).unstack()
print(prov_un.to_string())
prov_un.to_csv(os.path.join(OUT, "unemployment_by_province.csv"))

print("\n" + "=" * 110)
print("MEDICAL AID COVERAGE BY POPULATION GROUP, weighted % (inequality signal)")
print("=" * 110)
med = (panel[panel["medaid"].eq("Yes")].groupby(["year", "race"])["weight"].sum()
       / panel[panel["medaid"].notna()].groupby(["year", "race"])["weight"].sum() * 100).round(1).unstack()
print(med.to_string())
med.to_csv(os.path.join(OUT, "medaid_by_race.csv"))

print("\n" + "=" * 110)
print("SOCIAL GRANT RECEIPT BY PROVINCE, weighted %")
print("=" * 110)
gr = (panel[panel["grant"].eq("Yes")].groupby(["year", "prov"])["weight"].sum()
      / panel[panel["grant"].notna()].groupby(["year", "prov"])["weight"].sum() * 100).round(1).unstack()
print(gr.to_string())
gr.to_csv(os.path.join(OUT, "grants_by_province.csv"))

print("\n" + "=" * 110)
print("GEOTYPE x MATRIC+ ATTAINMENT (adults 20+), weighted % — urban/rural divide")
print("=" * 110)
a20 = panel[panel["age"] >= 20].copy()
a20["matric_plus"] = a20["educ_band"].isin(["Matric (Grade 12)", "Post-school"])
geo = (a20[a20["matric_plus"]].groupby(["year", "geotype"])["weight"].sum()
       / a20[a20["educ_band"].notna()].groupby(["year", "geotype"])["weight"].sum() * 100).round(1).unstack()
print(geo.to_string())
geo.to_csv(os.path.join(OUT, "matric_by_geotype.csv"))

panel_slim = panel[["year", "uqnr", "personnr", "psu", "stratum", "weight", "prov",
                    "geotype", "sex", "age", "race", "educ_band", "attend", "medaid",
                    "grant", "disab", "lab_status", "salary"]]
p = os.path.join(OUT, "ghs_panel_2012_2022.parquet")
try:
    panel_slim.to_parquet(p, index=False)
    print(f"\nWrote harmonised panel -> {p}")
except Exception as e:
    p = os.path.join(OUT, "ghs_panel_2012_2022.csv.gz")
    panel_slim.to_csv(p, index=False, compression="gzip")
    print(f"\n(parquet unavailable: {e})\nWrote harmonised panel -> {p}")
