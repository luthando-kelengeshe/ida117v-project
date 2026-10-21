"""
Step 4: Empirically verify the numeric code -> label mapping for the 2014-2018 files.

Method: compute WEIGHTED shares of each category in the numeric years and in the
adjacent labelled years (2013, 2019). Because population composition changes only
slowly year-on-year, matching shares reveal the code meanings without a codebook.
"""

import glob
import json
import os

import pandas as pd

DATA_DIR = r"c:\Users\USER-PC\TUT\IDA\Project\ida117v-project\dataset"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

with open(os.path.join(OUT_DIR, "schema.json"), encoding="utf-8") as fh:
    SCHEMA = {r["year"]: r for r in json.load(fh)}

VARS = {
    "sex":     ["gender", "sex"],
    "race":    ["race", "population"],
    "prov":    ["prov"],
    "geotype": ["geotype"],
    "disab":   ["disab"],
    "attend":  ["q110atte", "edu_attend"],
    "medaid":  ["q125medi", "q21medi", "hlt_medi"],
    "grant":   ["q132asg", "q31asg", "soc_grant"],
}


def resolve(year, canon):
    actual = {c.strip().strip('"').lower(): c.strip().strip('"') for c in SCHEMA[year]["columns"]}
    for cand in VARS[canon]:
        if cand in actual:
            return actual[cand]
    return None


def weighted_shares(year, canon):
    col = resolve(year, canon)
    if col is None:
        return None
    path = glob.glob(os.path.join(DATA_DIR, f"ghs-{year}-person-*.csv"))[0]
    df = pd.read_csv(path, encoding=SCHEMA[year]["encoding"],
                     usecols=[col, "person_wgt"] if col != "person_wgt" else [col],
                     dtype={col: str}, low_memory=False)
    wcol = [c for c in df.columns if c.lower() == "person_wgt"][0]
    df[wcol] = pd.to_numeric(df[wcol], errors="coerce")
    df[col] = df[col].astype(str).str.strip()
    g = df.groupby(col)[wcol].sum()
    return (g / g.sum() * 100).sort_values(ascending=False)


def show(canon):
    print("\n" + "=" * 104)
    print(f"WEIGHTED % DISTRIBUTION OF '{canon}'  (labelled years 2013/2019/2022 vs numeric years 2014-2018)")
    print("=" * 104)
    for y in [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022]:
        s = weighted_shares(y, canon)
        if s is None:
            print(f"{y}: (not present)")
            continue
        regime = "NUM " if y in (2014, 2015, 2016, 2017, 2018) else "LBL "
        parts = "  ".join(f"{k}={v:.1f}%" for k, v in s.head(12).items())
        print(f"{y} {regime}| {parts}")


if __name__ == "__main__":
    for v in ["sex", "race", "prov", "geotype", "disab", "attend", "medaid", "grant"]:
        show(v)
