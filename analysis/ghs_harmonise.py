"""
GHS 2012-2022 harmonisation layer.

Resolves the three obstacles that stop these 11 files from being pooled:
  1. Column names drift (Q-codes 2012-18 -> descriptive names 2019-22, plus case drift).
  2. Value coding differs (2014-2018 are numeric-coded; other years are label-coded).
  3. Sentinel codes (8/88/9/99/98/888888888) encode "not applicable"/"unspecified",
     not real values, and will silently corrupt any mean if treated as numbers.

All code->label mappings below were verified empirically in 04_verify_codes.py by
matching weighted category shares against the adjacent label-coded years.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

DATA_DIR = r"c:\Users\USER-PC\TUT\IDA\Project\ida117v-project\dataset"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "schema.json"), encoding="utf-8") as fh:
    SCHEMA = {r["year"]: r for r in json.load(fh)}

YEARS = sorted(SCHEMA)
NUMERIC_YEARS = {2014, 2015, 2016, 2017, 2018}

# ---------------------------------------------------------------- column aliases
ALIASES = {
    "uqnr":      ["uqnr"],
    "personnr":  ["personnr"],
    "psu":       ["psu"],
    "stratum":   ["stratum"],
    "weight":    ["person_wgt"],
    "prov":      ["prov"],
    "geotype":   ["geotype"],
    "metro":     ["metro"],
    "sex":       ["gender", "sex"],
    "age":       ["age"],
    "race":      ["race", "population"],
    "relation":  ["q11relsh", "hhc_relationship"],
    "marital":   ["q12amarst", "hhc_marital"],
    "educ":      ["q16hiedu", "q15hiedu", "education"],
    "attend":    ["q110atte", "edu_attend"],
    "disab":     ["disab"],
    "sevdisab":  ["sevdisab"],
    "undisab":   ["undisab"],
    "medaid":    ["q125medi", "q21medi", "hlt_medi"],
    "grant":     ["q132asg", "q31asg", "soc_grant"],
    "grant_oag": ["q132boag", "q31boag", "soc_grant_oag"],
    "grant_csg": ["q132bcsg", "q31bcsg", "soc_grant_csg"],
    "grant_dis": ["q132bdis", "q31bdis", "soc_grant_dis"],
    "grant_srd": ["soc_grant_srd"],
    "wage":      ["q21awge", "q41awge", "lab_wge"],
    "bus":       ["q21bbus", "q41bbus", "lab_bus"],
    "vol":       ["q21cvol", "q41cvol", "lab_vol"],
    "ret":       ["q21dret", "q41dret", "lab_ret"],
    "sto":       ["q22asto", "q42asto", "lab_sto"],
    "look":      ["q26alook", "q46alook", "lab_look"],
    "salary":    ["q22msal", "q42msal", "lab_salary"],
    "hlt_hbp":   ["q128ahbp", "q26ahbp", "q26hbp", "hlt_chronic_hbp"],
    "hlt_dbt":   ["q128adiab", "q26adbt", "q26dbt", "hlt_chronic_dbt"],
    "hlt_tb":    ["q26atb", "q26tb", "hlt_chronic_tb"],
    "hlt_asm":   ["q128aasm", "q26aasm", "q26asm", "hlt_chronic_asm"],
    "dsb_see":   ["q130see", "q28see", "q29see", "dsb_see"],
    "dsb_hea":   ["q130hea", "q28hea", "q29hea", "dsb_hea"],
    "dsb_wal":   ["q130wal", "q28wal", "q29wal", "dsb_wal"],
    "dsb_rem":   ["q130rem", "q28rem", "q29rem", "dsb_rem"],
    "dsb_sel":   ["q130sel", "q28sel", "q29sel", "dsb_sel"],
    "dsb_com":   ["q130com", "q28com", "q29com", "dsb_com"],
}

# ---------------------------------------------------------------- value mappings
PROV = {"1": "Western Cape", "2": "Eastern Cape", "3": "Northern Cape",
        "4": "Free State", "5": "KwaZulu-Natal", "6": "North West",
        "7": "Gauteng", "8": "Mpumalanga", "9": "Limpopo"}
SEX = {"1": "Male", "2": "Female"}
RACE = {"1": "African/Black", "2": "Coloured", "3": "Indian/Asian", "4": "White"}
DISAB = {"0": "Not disabled", "1": "Disabled", "9": "Unspecified"}
# geotype: 2012-2014 used a 4-category scheme; 2015+ collapsed urban formal+informal
# into a single "Urban". To compare across the break we map everything down to the
# coarser 3-category scheme (Urban / Traditional / Rural formal).
GEOTYPE_OLD = {"1": "Urban", "2": "Urban", "4": "Traditional", "5": "Rural formal"}
GEOTYPE_NEW = {"1": "Urban", "2": "Traditional", "3": "Rural formal"}
# 2012 and 2013 store geotype as text, not codes
GEOTYPE_TEXT = {
    "urban formal": "Urban",
    "urban informal": "Urban",
    "tribal areas": "Traditional",
    "traditional areas": "Traditional",
    "tribal area": "Traditional",
    "rural formal": "Rural formal",
    "farms": "Rural formal",
}

YESNO = {"1": "Yes", "2": "No", "3": "Do not know",
         "8": "Not applicable", "9": "Unspecified", "88": "Not applicable",
         "99": "Unspecified"}

MISSING_LABELS = {"not applicable", "unspecified", "do not know", "dont know",
                  "do not know ", "", "nan", "none", "88", "99", "888888888"}
MISSING_NUMERIC = {"8", "88", "9", "99", "98", "888", "999", "888888888", "999999999"}


def _norm_cols(year):
    return {c.strip().strip('"').lower(): c.strip().strip('"') for c in SCHEMA[year]["columns"]}


def resolve(year, canon):
    actual = _norm_cols(year)
    for cand in ALIASES[canon]:
        if cand in actual:
            return actual[cand]
    return None


def coverage_table():
    rows = []
    for canon in ALIASES:
        rows.append({"variable": canon, **{y: ("Y" if resolve(y, canon) else "") for y in YEARS}})
    return pd.DataFrame(rows).set_index("variable")


def _clean_text(s):
    return (s.astype(str).str.strip().str.strip('"').str.replace(r"\s+", " ", regex=True))


def _to_yesno(s, year):
    """Return a clean Yes/No/NaN series regardless of coding regime."""
    t = _clean_text(s)
    if year in NUMERIC_YEARS:
        t = t.map(YESNO).fillna(pd.NA)
    low = t.str.lower()
    out = pd.Series(pd.NA, index=t.index, dtype="object")
    out[low.eq("yes")] = "Yes"
    out[low.eq("no")] = "No"
    return out


def load_year(year, variables=None):
    """Load one GHS year, harmonised. Missing/not-applicable become NaN."""
    variables = list(ALIASES) if variables is None else list(variables)
    mapping = {c: resolve(year, c) for c in variables}
    mapping = {k: v for k, v in mapping.items() if v}
    path = glob.glob(os.path.join(DATA_DIR, f"ghs-{year}-person-*.csv"))[0]

    df = pd.read_csv(path, encoding=SCHEMA[year]["encoding"],
                     usecols=list(dict.fromkeys(mapping.values())),
                     dtype=str, low_memory=False)
    # rename source -> canonical (first canonical wins if two alias the same col)
    rename, seen = {}, set()
    for canon, src in mapping.items():
        if src not in seen:
            rename[src] = canon
            seen.add(src)
    df = df.rename(columns=rename)
    for canon, src in mapping.items():
        if canon not in df.columns and rename.get(src) in df.columns:
            df[canon] = df[rename[src]]

    df["year"] = year
    num = year in NUMERIC_YEARS

    # --- numeric fields
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 0) | (df["age"] > 110), "age"] = np.nan

    # --- categorical fields
    if "sex" in df:
        df["sex"] = _clean_text(df["sex"]).map(SEX) if num else _clean_text(df["sex"])
    if "race" in df:
        df["race"] = _clean_text(df["race"]).map(RACE) if num else _clean_text(df["race"])
    if "prov" in df:
        df["prov"] = _clean_text(df["prov"]).map(PROV) if num else _clean_text(df["prov"])
    if "geotype" in df:
        gt = _clean_text(df["geotype"])
        code_map = GEOTYPE_OLD if year <= 2014 else GEOTYPE_NEW
        # try numeric codes first, then fall back to the text labels used in 2012/13
        mapped = gt.map(code_map)
        mapped = mapped.fillna(gt.str.lower().map(GEOTYPE_TEXT))
        df["geotype"] = mapped.where(mapped.isin(["Urban", "Traditional", "Rural formal"]))
    if "disab" in df:
        d = _clean_text(df["disab"])
        d = d.map(DISAB).fillna(d) if num else d
        df["disab"] = d.where(d.isin(["Disabled", "Not disabled"]))

    # --- yes/no battery
    for c in ["attend", "medaid", "grant", "grant_oag", "grant_csg", "grant_dis",
              "grant_srd", "wage", "bus", "vol", "ret", "sto", "look",
              "hlt_hbp", "hlt_dbt", "hlt_tb", "hlt_asm"]:
        if c in df.columns:
            df[c] = _to_yesno(df[c], year)

    # --- salary: strip sentinels, keep real rand amounts
    if "salary" in df.columns:
        s = _clean_text(df["salary"])
        s = s.where(~s.str.lower().isin(MISSING_LABELS))
        sal = pd.to_numeric(s, errors="coerce")
        sal = sal.where((sal > 0) & (sal < 5_000_000))
        df["salary"] = sal

    # --- education: harmonise to ordered attainment bands
    if "educ" in df.columns:
        df["educ_band"] = _educ_band(df["educ"], num)

    return df


def _educ_band(series, numeric):
    """Collapse the 32-34 education categories into 5 comparable bands."""
    t = _clean_text(series)
    out = pd.Series(pd.NA, index=t.index, dtype="object")

    if numeric:
        v = pd.to_numeric(t, errors="coerce")
        out[v.eq(98)] = "No schooling"
        out[v.between(0, 7)] = "Primary or less"
        out[v.between(8, 11)] = "Some secondary"
        out[v.eq(12)] = "Matric (Grade 12)"
        out[v.between(13, 31)] = "Post-school"
        # 99 and anything else stay NaN
        return out

    low = t.str.lower()
    out[low.str.contains("no schooling", na=False)] = "No schooling"
    grade = low.str.extract(r"grade\s*(r|0|\d{1,2})", expand=False)
    gnum = grade.replace({"r": "0"}).pipe(pd.to_numeric, errors="coerce")
    out[gnum.between(0, 7)] = "Primary or less"
    out[gnum.between(8, 11)] = "Some secondary"
    out[gnum.eq(12)] = "Matric (Grade 12)"
    post = low.str.contains(
        r"diploma|degree|certificate|ntc|^n[1-6]|honours|master|doctor|b-tech|bachelor|postgrad",
        na=False, regex=True)
    out[post & out.isna()] = "Post-school"
    # diplomas/degrees mentioning Grade 12 must outrank the Matric match
    out[low.str.contains(r"diploma|degree|higher|honours|master|doctor|b-tech|bachelor", na=False)] = "Post-school"
    out[low.isin(["do not know", "unspecified", "other", "nan"])] = pd.NA
    return out


EDUC_ORDER = ["No schooling", "Primary or less", "Some secondary",
              "Matric (Grade 12)", "Post-school"]


def derive_labour_status(df):
    """
    Build a consistent employment status for ALL years from the raw activity
    questions, because employ_status1/2 only exist in 2014-18, 2021, 2022.

    Follows the standard Stats SA logic:
      employed            = did wage work OR ran a business OR helped in a
                            household business OR did work for own profit
      unemployed          = not employed, aged 15-64, looked for work or tried
                            to start a business
      not economically active = not employed, aged 15-64, did neither
      NaN                 = under 15 / over 64 (outside working-age scope)
    """
    work_cols = [c for c in ["wage", "bus", "vol", "ret"] if c in df.columns]
    seek_cols = [c for c in ["look", "sto"] if c in df.columns]

    employed = pd.Series(False, index=df.index)
    for c in work_cols:
        employed |= df[c].eq("Yes")

    seeking = pd.Series(False, index=df.index)
    for c in seek_cols:
        seeking |= df[c].eq("Yes")

    wa = df["age"].between(15, 64)
    status = pd.Series(pd.NA, index=df.index, dtype="object")
    status[wa & employed] = "Employed"
    status[wa & ~employed & seeking] = "Unemployed"
    status[wa & ~employed & ~seeking] = "Not economically active"
    return status


# ---------------------------------------------------------------- weighted stats
def wshare(df, col, value=None, by=None):
    """Weighted share (%) of `value` in `col`, optionally grouped by `by`."""
    d = df.dropna(subset=[col, "weight"])
    if by:
        num = d[d[col].eq(value)].groupby(by)["weight"].sum()
        den = d.groupby(by)["weight"].sum()
        return (num / den * 100).round(2)
    return round(d.loc[d[col].eq(value), "weight"].sum() / d["weight"].sum() * 100, 2)


def wdist(df, col, by=None):
    """Weighted percentage distribution of `col`."""
    d = df.dropna(subset=[col, "weight"])
    if by:
        g = d.groupby([by, col])["weight"].sum()
        return (g / g.groupby(level=0).sum() * 100).round(2).unstack()
    g = d.groupby(col)["weight"].sum()
    return (g / g.sum() * 100).round(2)


def wmean(df, col, by=None):
    d = df.dropna(subset=[col, "weight"])
    if by:
        return d.groupby(by).apply(
            lambda x: np.average(x[col], weights=x["weight"]), include_groups=False).round(2)
    return round(np.average(d[col], weights=d["weight"]), 2)


def wmedian(df, col, by=None):
    def _med(x):
        x = x.sort_values(col)
        c = x["weight"].cumsum()
        return x.loc[c >= x["weight"].sum() / 2, col].iloc[0]
    d = df.dropna(subset=[col, "weight"])
    if by:
        return d.groupby(by).apply(_med, include_groups=False)
    return _med(d)
