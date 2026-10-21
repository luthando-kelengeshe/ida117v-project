"""
Shared analysis-panel builder for the social-grant prediction task.

Guarantees that EDA and modelling see the IDENTICAL panel, and that every
leakage column is excluded structurally rather than by convention.

Two prediction tasks are defined:
  grant_any  : does this person receive any social grant?  (whole population)
  csg_takeup : among children 0-17, is a grant received?   (child-support-grant take-up)

Split is TIME-BASED because the GHS is a repeated cross-section and PSU ids
repeat within a survey year, so a random split would place geographic
neighbours on both sides of the boundary and inflate held-out scores.
  train 2012-2018   validation 2019-2020   test 2021-2022
"""
import os
import numpy as np
import pandas as pd
import ghs_harmonise as gh

OUT_DIR = gh.OUT_DIR
SEED = 42

# Columns asked ONLY of grant recipients. Including any of these lifts AUC
# 0.906 -> 0.993 and is unusable at prediction time. Never load them as features.
LEAKAGE_COLUMNS = [
    "soc_grant_oag", "soc_grant_csg", "soc_grant_dis", "soc_grant_srd",
    "soc_grant_type", "soc_disb_perm",
    "Q31boag", "Q31bdis", "Q31bcsg", "Q31bcar", "Q31bfos", "Q31bwvt",
    "Q31bgrn", "Q31bsoc", "Q32disg", "Q33csgw",
    "Q132boag", "Q132bdis", "Q132bcsg", "Q132bcar", "Q132bfos",
    "Q132bwvt", "Q132bgrn", "Q132bsoc", "Q133disg", "Q134csgw",
]

PERSON_LOAD = [
    "uqnr", "personnr", "psu", "stratum", "weight",
    "grant", "age", "sex", "race", "prov", "geotype",
    "educ", "attend", "medaid", "disab", "sevdisab",
    "wage", "bus", "vol", "ret", "look", "sto", "salary",
]

PERSON_CAT = ["sex", "race", "prov", "geotype", "educ_band", "attend",
              "medaid", "disab", "lab_status"]
PERSON_NUM = ["age"]
HH_NUM = ["hh_size", "hh_n_child_0_17", "hh_n_elderly_60p", "hh_mean_age",
          "hh_max_age", "hh_n_employed", "hh_n_unemployed", "hh_n_disabled",
          "hh_max_educ", "hh_dependency_ratio", "hh_workers_per_capita",
          "hh_salary_per_capita", "hh_no_earner", "hh_any_medaid"]

TRAIN_YEARS = list(range(2012, 2019))
VAL_YEARS = [2019, 2020]
TEST_YEARS = [2021, 2022]


def _nan_normalise(df):
    """sklearn imputers cannot evaluate pd.NA in object columns."""
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


def build_panel(verbose=True):
    frames = []
    for y in gh.YEARS:
        d = gh.load_year(y, PERSON_LOAD)
        d["lab_status"] = gh.derive_labour_status(d)
        frames.append(d)
        if verbose:
            print(f"  loaded {y}: {len(d):,}")
    p = pd.concat(frames, ignore_index=True)

    # ---- household context, aggregated within (year, uqnr)
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
        "hh_n_disabled": g["disab"].apply(lambda s: (s == "Disabled").sum()),
        "hh_any_medaid": g["medaid"].apply(lambda s: int((s == "Yes").any())),
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

    # ---- targets
    p["grant_any"] = p["grant"].map({"Yes": 1, "No": 0})
    p["split"] = np.select(
        [p["year"].isin(TRAIN_YEARS), p["year"].isin(VAL_YEARS), p["year"].isin(TEST_YEARS)],
        ["train", "val", "test"], default="unused")
    return _nan_normalise(p)


# Eligibility proxy for the Child Support Grant. The CSG means test keys on caregiver
# income, which the person files do not carry reliably (salary is ~20% populated and
# partly bracket-midpoint imputed). Instead we RESTRICT to households where income
# eligibility is near-certain: no employed member and nobody on medical aid. Within
# that group the grant rate is 85.7%, so non-receipt cannot be explained by affluence
# and is interpretable as a reach failure rather than ineligibility.
def eligibility_proxy(d):
    return d["hh_no_earner"].eq(1) & d["hh_any_medaid"].eq(0)


def get_task(panel, task):
    """
    Return (frame, target_column, feature_lists) for a named task.

    grant_any      : whole population, 1 = receives any social grant
    csg_takeup     : children 0-17,    1 = receives a grant
    csg_reach_gap  : children 0-17 in near-certainly-eligible households,
                     1 = NOT receiving a grant  <- the reach failure to be found

    csg_reach_gap is the project's primary task. The positive class is the outcome
    the stakeholder wants to locate, and restricting the population removes the
    eligibility/take-up conflation by design instead of inferring it from the model.
    """
    if task == "grant_any":
        d = panel.dropna(subset=["grant_any"]).copy()
        d["y"] = d["grant_any"].astype(int)
    elif task == "csg_takeup":
        d = panel[panel["age"].between(0, 17)].dropna(subset=["grant_any"]).copy()
        d["y"] = d["grant_any"].astype(int)
    elif task == "csg_reach_gap":
        d = panel[panel["age"].between(0, 17)].dropna(subset=["grant_any"]).copy()
        d = d[eligibility_proxy(d)].copy()
        d["y"] = (1 - d["grant_any"]).astype(int)
    else:
        raise ValueError(f"unknown task: {task}")
    return d, "y", (PERSON_NUM + HH_NUM, PERSON_CAT)


POSITIVE_LABEL = {
    "grant_any": "receives a social grant",
    "csg_takeup": "receives a social grant",
    "csg_reach_gap": "eligible but NOT receiving a grant",
}


def split_xy(d, num, cat, target="y"):
    out = {}
    for name in ["train", "val", "test"]:
        s = d[d["split"] == name]
        out[name] = (s[num + cat], s[target].values, s["weight"].values, s)
    return out

