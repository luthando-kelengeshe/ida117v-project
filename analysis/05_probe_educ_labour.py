"""
Step 5: Inspect the category sets for education, employment status and grant type.
These are the highest-value variables but have the messiest label drift.
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


def col_of(year, candidates):
    actual = {c.strip().strip('"').lower(): c.strip().strip('"') for c in SCHEMA[year]["columns"]}
    for cand in candidates:
        if cand in actual:
            return actual[cand]
    return None


def dump(year, candidates, label, topn=40):
    col = col_of(year, candidates)
    if col is None:
        print(f"  {year}: -- not present --")
        return
    path = glob.glob(os.path.join(DATA_DIR, f"ghs-{year}-person-*.csv"))[0]
    df = pd.read_csv(path, encoding=SCHEMA[year]["encoding"], usecols=[col],
                     dtype=str, low_memory=False)
    vc = df[col].astype(str).str.strip().value_counts()
    print(f"\n  {year}  [{col}]  {len(vc)} distinct")
    for k, v in vc.head(topn).items():
        print(f"      {k[:95]:<97} {v:>8,}")


print("#" * 110)
print("EDUCATION / HIGHEST LEVEL ATTAINED")
print("#" * 110)
for y in [2012, 2014, 2019, 2022]:
    dump(y, ["q16hiedu", "q15hiedu", "education"], "educ", topn=40)

print("\n\n" + "#" * 110)
print("EMPLOYMENT STATUS (derived variable, only some years)")
print("#" * 110)
for y in [2014, 2015, 2016, 2017, 2018, 2021, 2022]:
    dump(y, ["employ_status1"], "emp1", topn=15)
for y in [2014, 2021, 2022]:
    dump(y, ["employ_status2"], "emp2", topn=15)

print("\n\n" + "#" * 110)
print("LABOUR: wage work / business / job seeking")
print("#" * 110)
for y in [2012, 2014, 2019, 2022]:
    dump(y, ["q21awge", "q41awge", "lab_wge"], "wage", topn=10)
    dump(y, ["q26alook", "q46alook", "lab_look"], "look", topn=10)

print("\n\n" + "#" * 110)
print("GRANT TYPE")
print("#" * 110)
for y in [2019, 2022]:
    dump(y, ["soc_grant_type"], "gtype", topn=15)
for y in [2020, 2022]:
    dump(y, ["soc_grant_srd"], "srd", topn=8)

print("\n\n" + "#" * 110)
print("SALARY / INCOME BRACKET (lab_salary / q42msal)")
print("#" * 110)
for y in [2014, 2019, 2022]:
    dump(y, ["q42msal", "q22msal", "lab_salary"], "sal", topn=20)
