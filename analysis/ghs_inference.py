"""
Inference layer for the deployed interface.

Responsibilities
  1. load SAVED artefacts only - preprocessor, model weights, calibrator. Never refit.
  2. turn a household roster into the feature row used in training
  3. validate input against ghs_schema and refuse out-of-scope households
  4. return a calibrated probability, an operational risk band and a local explanation

Each model carries its OWN feature contract (self.num / self.cat) because the
optimised champion was selected with a reduced 19-feature set and a different
one-hot min_frequency. Relying on module-level feature lists would silently feed
it the wrong design matrix.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

import ghs_schema as S
from ghs_models import MLP, TabTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MOD = os.path.join(OUT, "models")

# Full feature set produced by build_row. Individual models consume a subset.
NUM = ["age", "hh_size", "hh_n_child_0_17", "hh_n_elderly_60p", "hh_mean_age",
       "hh_max_age", "hh_n_employed", "hh_n_unemployed", "hh_n_disabled",
       "hh_max_educ", "hh_dependency_ratio", "hh_workers_per_capita",
       "hh_salary_per_capita", "hh_no_earner", "hh_any_medaid"]
CAT = ["sex", "race", "prov", "geotype", "educ_band", "attend", "medaid",
       "disab", "lab_status"]

CHAMPION = "Optimised MLP"
MODEL_CHOICES = [CHAMPION, "MLP (PyTorch)", "Random Forest",
                 "Logistic Regression", "TabTransformer (PyTorch)"]

NEUTRAL = {"sex": "Female", "race": "African/Black", "prov": "Gauteng",
           "geotype": "Urban", "educ_band": "Primary or less", "attend": "Yes",
           "medaid": "No", "disab": "Not disabled",
           "lab_status": "Not economically active"}

# Zero-variance inside the csg_reach_gap population: these ARE the eligibility
# filter, so the model has no evidence about what happens when they change.
# Excluded from explanations. The optimised champion drops them outright.
CONSTANT_FEATURES = {
    "csg_reach_gap": ["hh_n_employed", "hh_workers_per_capita", "hh_no_earner",
                      "hh_any_medaid", "medaid"],
    "csg_takeup": [],
    "grant_any": [],
}

FRIENDLY = {"medaid": "medical aid cover", "race": "population group",
            "prov": "province", "geotype": "settlement type",
            "educ_band": "education level", "attend": "school attendance",
            "disab": "disability status", "sex": "sex", "lab_status": "labour status",
            "age": "age", "hh_size": "household size",
            "hh_n_child_0_17": "number of children in the household",
            "hh_n_elderly_60p": "number of older persons in the household",
            "hh_n_employed": "number of employed members",
            "hh_n_unemployed": "number of unemployed members",
            "hh_n_disabled": "number of disabled members",
            "hh_no_earner": "household has no earner",
            "hh_any_medaid": "anyone in household on medical aid",
            "hh_salary_per_capita": "household income per person",
            "hh_dependency_ratio": "dependency ratio",
            "hh_workers_per_capita": "workers per person",
            "hh_max_educ": "highest education in household",
            "hh_mean_age": "average household age",
            "hh_max_age": "oldest household member"}

# Survey skip patterns. Verified against the panel: lab_status is 0% populated for
# ages 0-14, attend 0% for ages 0-4. The interface must reproduce the same pattern.
APPLICABLE_FROM = {"lab_status": 15, "attend": 5}


def applicable(field, age):
    return age >= APPLICABLE_FROM.get(field, 0)


class InputError(ValueError):
    """Raised with a message that is safe and useful to show a user."""


# --------------------------------------------------------------------- loading
class Champion:
    def __init__(self, task="csg_reach_gap", which=None):
        self.task = task
        self.name = which or CHAMPION
        self._ref = None
        hp_path = os.path.join(OUT, f"hyperparameters_{task}.json")
        if not os.path.exists(hp_path):
            raise FileNotFoundError(
                f"Missing {hp_path}. Run 13_models.py first.")
        with open(hp_path, encoding="utf-8") as f:
            self.hp = json.load(f)

        if self.name == CHAMPION:
            self._load_optimised()
        else:
            self._load_baseline()

        cpath = os.path.join(MOD, f"calibrators_{task}.joblib")
        self.calibrator = None
        if os.path.exists(cpath):
            self.calibrator = joblib.load(cpath)["calibrators"].get(self.name)

    def _load_optimised(self):
        """The Q5 search winner: own preprocessor, own 19-feature contract."""
        bpath = os.path.join(MOD, f"optimised_bundle_{self.task}.joblib")
        wpath = os.path.join(MOD, f"mlp_optimised_{self.task}.pt")
        if not (os.path.exists(bpath) and os.path.exists(wpath)):
            raise FileNotFoundError(
                f"Optimised champion not found. Run 22_optimise.py, then "
                f"25_calibrate_optimised.py. Looked for:\n  {bpath}\n  {wpath}")
        b = joblib.load(bpath)
        self.pre, self.num, self.cat, self.cfg = b["pre"], b["num"], b["cat"], b["cfg"]
        d_in = self.pre.transform(
            pd.DataFrame([{c: np.nan for c in self.num + self.cat}])).shape[1]
        self.model = MLP(d_in, hidden=tuple(self.cfg["hidden"]),
                         p_drop=self.cfg["p_drop"])
        self.model.load_state_dict(torch.load(wpath, map_location="cpu"))
        self.model.eval()
        self.kind, self.encoding = "torch", "bundle"

    def _load_baseline(self):
        """The pre-optimisation models from 13_models.py, kept for comparison."""
        pre = joblib.load(os.path.join(MOD, f"preprocessors_{self.task}.joblib"))
        self.onehot, self.ordinal = pre["onehot"], pre["ordinal"]
        self.cards = pre["cards"]
        self.num, self.cat = NUM, CAT
        self.kind = "torch"
        if self.name.startswith("MLP"):
            self.model = MLP(self.hp["MLP (PyTorch)"]["input_dim"],
                             hidden=tuple(self.hp["MLP (PyTorch)"]["hidden"]),
                             p_drop=self.hp["MLP (PyTorch)"]["dropout"])
            self.model.load_state_dict(torch.load(
                os.path.join(MOD, f"mlp_{self.task}.pt"), map_location="cpu"))
            self.encoding = "onehot"
        elif self.name.startswith("TabTransformer"):
            c = self.hp["TabTransformer (PyTorch)"]
            self.model = TabTransformer(self.cards, len(NUM), d_model=c["d_model"],
                                        n_heads=c["n_heads"], n_blocks=c["n_blocks"],
                                        p_drop=c["dropout"])
            self.model.load_state_dict(torch.load(
                os.path.join(MOD, f"tabtransformer_{self.task}.pt"), map_location="cpu"))
            self.encoding = "ordinal"
        else:
            fn = "logreg" if self.name.startswith("Logistic") else "randomforest"
            self.model = joblib.load(os.path.join(MOD, f"{fn}_{self.task}.joblib"))
            self.kind, self.encoding = "sklearn", "pipeline"
        if self.kind == "torch":
            self.model.eval()

    # ------------------------------------------------------------- prediction
    def predict_proba(self, df, calibrated=True):
        raw = self._raw_proba(df)
        if calibrated and self.calibrator is not None:
            return np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return raw

    def _raw_proba(self, df):
        X = df[self.num + self.cat]
        if self.kind == "sklearn":
            return self.model.predict_proba(X)[:, 1]
        with torch.no_grad():
            if self.encoding == "bundle":
                A = torch.tensor(self.pre.transform(X), dtype=torch.float32)
                if len(A) == 1:                  # BatchNorm needs >1 row
                    return torch.sigmoid(self.model(A.repeat(2, 1), None)).numpy()[:1]
                return torch.sigmoid(self.model(A, None)).numpy()
            if self.encoding == "onehot":
                A = torch.tensor(self.onehot.transform(X), dtype=torch.float32)
                if len(A) == 1:
                    return torch.sigmoid(self.model(A.repeat(2, 1), None)).numpy()[:1]
                return torch.sigmoid(self.model(A, None)).numpy()
            Z = self.ordinal.transform(X)
            xn = torch.tensor(Z[:, :len(self.num)], dtype=torch.float32)
            xc = torch.tensor(Z[:, len(self.num):] + 1, dtype=torch.long).clamp_(min=0)
            return torch.sigmoid(self.model(xn, xc)).numpy()

    # ------------------------------------------------------------- reference
    def reference(self):
        """Calibrated validation score distribution, for percentile ranking."""
        if self._ref is not None:
            return self._ref
        if self.name == CHAMPION:
            path = os.path.join(OUT, f"optimised_calibrated_{self.task}.npz")
            if os.path.exists(path):
                z = np.load(path)
                self._ref = {"scores": np.sort(z["val"]),
                             "base_rate": float(z["y_val"].mean())}
                return self._ref
        path = os.path.join(OUT, f"val_test_probs_{self.task}.npz")
        if not os.path.exists(path):
            return None
        z = np.load(path)
        key = f"{self.name}|val"
        if key not in z:
            return None
        raw = z[key]
        p = (np.clip(self.calibrator.predict(raw), 0, 1)
             if self.calibrator is not None else raw)
        self._ref = {"scores": np.sort(p), "base_rate": float(z["y_val"].mean())}
        return self._ref

    def percentile(self, prob):
        r = self.reference()
        if r is None:
            return None
        return float(np.searchsorted(r["scores"], prob) / len(r["scores"]) * 100)

    def base_rate(self):
        r = self.reference()
        return None if r is None else r["base_rate"]

    def metrics(self):
        if self.name == CHAMPION:
            p = os.path.join(OUT, f"final_locked_{self.task}.csv")
            if os.path.exists(p):
                t = pd.read_csv(p, index_col=0)
                row = [i for i in t.index if "validation" in str(i).lower()]
                if row:
                    return t.loc[row[0]].to_dict()
            return {}
        p = os.path.join(OUT, f"results_val_{self.task}.csv")
        if not os.path.exists(p):
            return {}
        t = pd.read_csv(p, index_col=0)
        return t.loc[self.name].to_dict() if self.name in t.index else {}

    def n_features(self):
        return len(self.num) + len(self.cat)


# ------------------------------------------------------- roster -> feature row
def derive_household(roster):
    """Replicate the training-time aggregation over (year, uqnr)."""
    r = pd.DataFrame(roster)
    if r.empty:
        raise InputError("The household roster is empty. Add at least the focal child.")
    age = pd.to_numeric(r["age"], errors="coerce")
    if age.isna().any():
        raise InputError("Every household member needs a numeric age.")
    sal = pd.to_numeric(r["salary"], errors="coerce") if "salary" in r else pd.Series(dtype=float)
    emp = int((r["lab_status"] == "Employed").sum())
    une = int((r["lab_status"] == "Unemployed").sum())
    n = len(r)
    educ_idx = [S.EDUC_BANDS.index(v) for v in r["educ_band"] if v in S.EDUC_BANDS]
    tot_sal = float(sal.sum()) if len(sal) and sal.notna().any() else np.nan
    return {
        "hh_size": n,
        "hh_n_child_0_17": int((age <= 17).sum()),
        "hh_n_elderly_60p": int((age >= 60).sum()),
        "hh_mean_age": float(age.mean()),
        "hh_max_age": float(age.max()),
        "hh_n_employed": emp,
        "hh_n_unemployed": une,
        "hh_n_disabled": int((r["disab"] == "Disabled").sum()),
        "hh_any_medaid": int((r["medaid"] == "Yes").any()),
        "hh_max_educ": float(max(educ_idx)) if educ_idx else np.nan,
        "hh_dependency_ratio": float((int((age <= 17).sum()) + int((age >= 60).sum())) / max(n, 1)),
        "hh_workers_per_capita": float(emp / max(n, 1)),
        "hh_salary_per_capita": (tot_sal / max(n, 1)) if tot_sal == tot_sal else np.nan,
        "hh_no_earner": int(emp == 0),
    }


def validate_person(person):
    """Raise InputError naming the offending field and the accepted values."""
    try:
        a = int(person.get("age"))
    except (TypeError, ValueError):
        spec = S.PERSON_FIELDS["age"]
        raise InputError(f"Age must be a whole number between {spec['min']} and "
                         f"{spec['max']}. Received: {person.get('age')!r}.")
    spec = S.PERSON_FIELDS["age"]
    if not spec["min"] <= a <= spec["max"]:
        raise InputError(
            f"Age {a} is outside this model's supported range. This model covers "
            f"children {spec['min']}-{spec['max']} only (Child Support Grant age "
            f"range). For ages 18-59 or 60+ a separate model is required.")
    required = [f for f in S.PERSON_FIELDS if applicable(f, a)]
    missing = [f for f in required
               if person.get(f) is None or person.get(f) == ""]
    if missing:
        raise InputError("Missing required field(s): "
                         + ", ".join(FRIENDLY.get(m, m) for m in missing) + ".")
    for f, sp in S.PERSON_FIELDS.items():
        if sp["type"] != "cat" or not applicable(f, a):
            continue
        if person.get(f) not in sp["levels"]:
            raise InputError(f"'{FRIENDLY.get(f, f)}' must be one of: "
                             f"{', '.join(sp['levels'])}. Received: {person.get(f)!r}.")
    return True


def check_in_scope(hh, person):
    """
    The reach-gap model is valid only for near-certainly income-eligible households
    (no employed member, nobody on medical aid). Refuse rather than mislead.
    """
    if hh["hh_n_employed"] > 0:
        raise InputError(
            f"This household has {hh['hh_n_employed']} employed member(s), so it falls "
            "outside the model's scope. The model only covers households with no "
            "employed member, because that is what makes 'no grant' interpretable as a "
            "reach failure rather than as being above the means test. Set every "
            "member's labour status to 'Unemployed' or 'Not economically active'.")
    if hh["hh_any_medaid"] == 1 or person.get("medaid") == "Yes":
        raise InputError(
            "Someone in this household is covered by a medical aid scheme, so it falls "
            "outside the model's scope. Medical aid cover indicates income above the "
            "Child Support Grant means test, so such households are excluded by design.")
    return True


def build_row(person, roster=None, enforce_scope=True):
    validate_person(person)
    roster = roster or []
    age = int(person["age"])
    focal = {"age": age, "sex": person["sex"], "educ_band": person["educ_band"],
             "lab_status": person.get("lab_status") if applicable("lab_status", age) else np.nan,
             "medaid": person["medaid"], "disab": person["disab"], "salary": np.nan}
    members = [focal] + [dict(m, salary=m.get("salary", np.nan)) for m in roster]
    # A child under 15 cannot be the sole member of a household. Scoring one
    # produces a feature combination that does not exist in the survey (hh_size=1
    # with a toddler) and the model extrapolates to absurd probabilities.
    if age < 15 and len(members) == 1:
        raise InputError(
            f"A {age}-year-old cannot be the only member of a household. Add at least "
            "one other member (the caregiver) so the household-level features can be "
            "derived. In batch uploads, supply the caregiver_age column.")
    hh = derive_household(members)
    if enforce_scope:
        check_in_scope(hh, person)
    row = {f: (person[f] if applicable(f, age) else np.nan) for f in S.PERSON_FIELDS}
    row["age"] = age
    row.update(hh)
    return pd.DataFrame([row])


# --------------------------------------------------------------- explanation
def explain(champ, df, top=5):
    """
    One-at-a-time perturbation against a reference profile. Only features the model
    actually consumes AND that vary in its training population are perturbed.
    """
    dead = set(CONSTANT_FEATURES.get(champ.task, []))
    used_cat = [c for c in champ.cat if c in NEUTRAL and c not in dead]
    used_num = [c for c in ["hh_salary_per_capita", "hh_size", "hh_n_child_0_17",
                            "hh_n_elderly_60p", "hh_n_unemployed",
                            "hh_dependency_ratio"]
                if c in champ.num and c not in dead]
    base = float(champ.predict_proba(df)[0])
    deltas = []
    for f in used_cat:
        cur = df[f].iloc[0]
        if pd.isna(cur) or cur == NEUTRAL[f]:
            continue
        alt = df.copy(); alt[f] = NEUTRAL[f]
        deltas.append((f, base - float(champ.predict_proba(alt)[0])))
    for f in used_num:
        alt = df.copy()
        alt[f] = np.nan if f == "hh_salary_per_capita" else 1
        try:
            deltas.append((f, base - float(champ.predict_proba(alt)[0])))
        except Exception:
            pass
    deltas.sort(key=lambda t: -abs(t[1]))
    return base, deltas[:top]


BANDS = [(3.0, "Very high", "visit first"),
         (2.0, "High", "include in this outreach round"),
         (1.25, "Elevated", "include if capacity allows"),
         (0.0, "Typical or low", "no targeted action indicated")]


def risk_band(prob, base_rate):
    lift = prob / base_rate if base_rate else float("nan")
    for thr, name, action in BANDS:
        if lift >= thr:
            return name, action, lift
    return BANDS[-1][1], BANDS[-1][2], lift


def plain_language(prob, deltas, base_rate=None, pct=None):
    head = [f"Estimated probability this eligible child is **not** receiving a grant: "
            f"**{prob*100:.1f}%**."]
    if base_rate:
        name, action, lift = risk_band(prob, base_rate)
        head += ["", f"**Risk band: {name}** ({lift:.1f}x the population rate of "
                     f"{base_rate*100:.1f}%) - _{action}_."]
    if pct is not None:
        head.append(f"This household ranks in the **top {100-pct:.0f}%** of "
                    f"outreach priority.")
    lines = head + ["", "Factors moving this estimate most:"]
    for f, d in deltas:
        lines.append(f"- **{FRIENDLY.get(f, f)}** {'raises' if d > 0 else 'lowers'} it "
                     f"by {abs(d)*100:.1f} percentage points")
    lines += ["", "_A statistical priority score from survey patterns. Not a "
              "determination of eligibility, which only SASSA can make._"]
    return "\n".join(lines)


def predict_batch(champ, frame):
    """Validate and rank an uploaded table; returns (results, errors)."""
    needed = list(S.PERSON_FIELDS)
    miss = [c for c in needed if c not in frame.columns]
    if miss:
        raise InputError(f"Uploaded file is missing column(s): {', '.join(miss)}. "
                         f"Required columns are: {', '.join(needed)}.")
    rows, errs = [], []
    for i, rec in frame.iterrows():
        d = rec.to_dict()
        # Optional caregiver columns let a batch row describe a real household
        # rather than a lone child. Anything absent falls back to a sensible default.
        roster = []
        ca = d.get("caregiver_age")
        if ca is not None and str(ca) not in ("", "nan", "None"):
            roster.append({
                "age": ca,
                "sex": d.get("caregiver_sex") or "Female",
                "educ_band": d.get("caregiver_educ_band") or "Some secondary",
                "lab_status": d.get("caregiver_lab_status") or "Not economically active",
                "medaid": "No",
                "disab": d.get("caregiver_disab") or "Not disabled",
                "salary": np.nan})
        try:
            rows.append(build_row({k: v for k, v in d.items()
                                   if not str(k).startswith("caregiver_")}, roster))
        except InputError as e:
            errs.append({"row": int(i) + 2, "error": str(e)})
    if not rows:
        return pd.DataFrame(), errs
    X = pd.concat(rows, ignore_index=True)
    pr = champ.predict_proba(X).astype("float64")
    br = champ.base_rate() or float("nan")
    X["probability_%"] = (pr * 100).round(1)
    X["lift_vs_population"] = (pr / br).round(2) if br == br else np.nan
    X["risk_band"] = [risk_band(float(v), br)[0] for v in pr]
    X["outreach_percentile"] = [
        (None if champ.percentile(float(v)) is None
         else round(100 - champ.percentile(float(v)), 1)) for v in pr]
    return X.sort_values("probability_%", ascending=False), errs
