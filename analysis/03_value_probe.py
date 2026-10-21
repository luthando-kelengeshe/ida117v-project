"""
Step 3: Probe actual VALUES in the core variables that exist in all 11 years.

Checks:
  - label-coded vs numeric-coded years
  - survey weight sanity (sum of person_wgt should approximate SA population)
  - missing / 'not applicable' sentinel codes (8, 88, 9, 99, 999...)
  - age range validity
  - duplicate person keys
"""

import glob
import json
import os
import re

import pandas as pd

DATA_DIR = r"c:\Users\USER-PC\TUT\IDA\Project\ida117v-project\dataset"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

with open(os.path.join(OUT_DIR, "schema.json"), encoding="utf-8") as fh:
    SCHEMA = {r["year"]: r for r in json.load(fh)}

# Core variables, mapped per naming regime.
# 2012-2018 use Q-codes; 2019-2022 use descriptive prefixes.
CORE = {
    "sex":      ["gender", "sex"],
    "age":      ["age"],
    "race":     ["race", "population"],
    "prov":     ["prov"],
    "geotype":  ["geotype"],
    "metro":    ["metro"],
    "educ":     ["q16hiedu", "q15hiedu", "education"],
    "attend":   ["q110atte", "edu_attend"],
    "disab":    ["disab"],
    "sevdisab": ["sevdisab"],
    "grant":    ["q132asg", "q31asg", "soc_grant"],
    "medaid":   ["q125medi", "q21medi", "hlt_medi"],
    "wage":     ["q21awge", "q41awge", "lab_wge"],
    "weight":   ["person_wgt"],
    "uqnr":     ["uqnr"],
    "personnr": ["personnr"],
}


def resolve(year):
    """Map canonical name -> actual column name in that year's file."""
    actual = {c.strip().strip('"').lower(): c.strip().strip('"') for c in SCHEMA[year]["columns"]}
    out = {}
    for canon, candidates in CORE.items():
        for cand in candidates:
            if cand in actual:
                out[canon] = actual[cand]
                break
    return out


def load(year):
    path = glob.glob(os.path.join(DATA_DIR, f"ghs-{year}-person-*.csv"))[0]
    mapping = resolve(year)
    df = pd.read_csv(
        path,
        encoding=SCHEMA[year]["encoding"],
        usecols=list(mapping.values()),
        dtype=str,
        low_memory=False,
    )
    inv = {v: k for k, v in mapping.items()}
    df = df.rename(columns=inv)
    return df, mapping


SENTINEL = {"8", "88", "9", "99", "888", "999", "98", "97"}


def main():
    years = sorted(SCHEMA)
    print("=" * 100)
    print("A. CORE VARIABLE AVAILABILITY  (canonical name -> source column per year)")
    print("=" * 100)
    maps = {y: resolve(y) for y in years}
    print(f"{'canonical':<11}" + "".join(f"{y:>20}" for y in years[:5]))
    for canon in CORE:
        print(f"{canon:<11}" + "".join(f"{maps[y].get(canon, '-- MISSING --'):>20}" for y in years[:5]))
    print()
    print(f"{'canonical':<11}" + "".join(f"{y:>20}" for y in years[5:]))
    for canon in CORE:
        print(f"{canon:<11}" + "".join(f"{maps[y].get(canon, '-- MISSING --'):>20}" for y in years[5:]))

    summary = []
    print("\n" + "=" * 100)
    print("B. VALUE CODING + WEIGHT + QUALITY CHECKS PER YEAR")
    print("=" * 100)

    for y in years:
        df, mapping = load(y)
        n = len(df)

        sex_vals = df["sex"].dropna().unique()[:6]
        coding = "LABEL" if any(not v.strip().isdigit() for v in map(str, sex_vals)) else "NUMERIC"

        w = pd.to_numeric(df["weight"], errors="coerce")
        age = pd.to_numeric(df["age"], errors="coerce")

        # duplicate person key check
        key = df["uqnr"].astype(str) + "|" + df["personnr"].astype(str)
        dups = int(key.duplicated().sum())

        # sentinel prevalence in a categorical variable
        sent_disab = df["disab"].astype(str).str.strip().isin(SENTINEL).mean() * 100

        print(f"\n--- {y} ---  n={n:,}  coding={coding}")
        print(f"  sex values        : {list(sex_vals)}")
        print(f"  prov values (3)   : {list(df['prov'].dropna().unique()[:3])}")
        print(f"  race values (3)   : {list(df['race'].dropna().unique()[:3])}")
        print(f"  person_wgt        : sum={w.sum():,.0f}  min={w.min():,.2f}  max={w.max():,.2f}  nulls={w.isna().sum()}")
        print(f"  age               : min={age.min()}  max={age.max()}  nulls={age.isna().sum()}  >110={int((age > 110).sum())}")
        print(f"  duplicate uqnr+personnr : {dups}")
        print(f"  'disab' sentinel-coded (8/88/9/99) share : {sent_disab:.1f}%")
        print(f"  disab value counts: {df['disab'].astype(str).str.strip().value_counts().head(5).to_dict()}")

        summary.append({
            "year": y, "n": n, "coding": coding,
            "weight_sum": float(w.sum()), "dups": dups,
            "age_min": float(age.min()), "age_max": float(age.max()),
        })

    print("\n" + "=" * 100)
    print("C. WEIGHTED POPULATION TOTAL vs ROW COUNT")
    print("=" * 100)
    print(f"{'year':<6}{'rows':>10}{'sum(person_wgt)':>20}{'implied pop (M)':>18}  coding")
    print("-" * 70)
    for s in summary:
        print(f"{s['year']:<6}{s['n']:>10,}{s['weight_sum']:>20,.0f}{s['weight_sum']/1e6:>18.1f}  {s['coding']}")

    pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "year_summary.csv"), index=False)
    print(f"\nWrote {os.path.join(OUT_DIR, 'year_summary.csv')}")


if __name__ == "__main__":
    main()
