# Child Support Grant Reach-Gap Prioritiser — deployment guide

Stakeholder-facing interface for the IDA117V project (brief Q7), built on the
Statistics South Africa General Household Survey 2012–2022 person files.

## The task

**Stakeholder:** SASSA / Department of Social Development, whose own service
standard is "the right social grant, to the right person, at the right time and place".

**Problem:** roughly **750,000 children (2022)** who are near-certainly eligible for
the Child Support Grant are not receiving it. Outreach capacity is finite, so the
question is which households to visit first.

**Prediction unit:** one child aged 0–17.

**Population:** children in households with **no employed member and nobody on
medical aid** — near-certainly income-eligible. Restricting to this group is what
makes "no grant" mean *reach failure* rather than *ineligibility*. Within the group
85.7% do receive a grant, so the 14.3% who do not are the population of interest.

**Target:** `1` = eligible but NOT receiving a grant (the outcome to be found).

**Primary metric:** precision@k / lift, because the decision is capacity-constrained.
At 5% coverage the champion reaches **3.0× more** missed children than random outreach.

### Why not simply predict grant receipt?

An earlier version did. It scored much higher (ROC-AUC 0.857 vs 0.673) and was far
less useful, because grant receipt is largely determined by statutory age rules —
receipt jumps from 13.4% at age 59 to 51.3% at age 60. The model was rediscovering
the Social Assistance Act. Worse, "low probability of receipt" conflates *ineligible*
with *unreached*, so it could not support the decision at all. The headline metric
fell because the easy signal was removed; what remains is the genuinely hard part.

## 1. Environment

```bash
python -m pip install -r requirements.txt
```

Verified on Python 3.14.3 with pandas 2.3.3, scikit-learn 1.8.0,
matplotlib 3.10.7, torch 2.14.1+cpu, streamlit 1.52.1.

## 2. Reproduce from raw data (execution order)

Run from the `analysis/` directory. Place the 11 GHS person CSVs in `../dataset/`.

| Step | Script | Purpose |
|---|---|---|
| 1 | `01_schema_scan.py` | Scan all 11 files, write `out/schema.json` |
| 2 | `02_column_map.py` | Map columns to thematic domains |
| 3 | `03_value_probe.py` | Probe core variables, validate survey weights |
| 4 | `04_verify_codes.py` | Verify numeric→label code maps empirically |
| 5 | `12_eda.py` | Data dictionary, audit, figures F1–F7, cache panel |
| 6 | `18_eligibility_design.py` | Justify the eligibility-proxy restriction |
| 7 | `19_reframed_feasibility.py` | Confirm the reframed target has signal |
| 8 | `13_models.py` | Train baseline + 4 models, figures F8–F11, F13 |
| 9 | `15_calibrate.py` | Fit isotonic calibrators, figure F12 |
| 10 | `20_targeting_figure.py` | Targeting table + figure F13 (random tie-breaking) |
| 11 | `21_ablation_mcnemar.py` | Q6 ablations and pre-optimisation McNemar |
| 12 | `22_optimise.py` | Q5 random search, leave-one-year-out CV, locked test |
| 13 | `23_complete_optimisation.py` | Optimisation log and figure F15 |
| 14 | `25_calibrate_optimised.py` | Calibrate optimised champion, definitive McNemar |
| 15 | `16_ui_tests.py` | Acceptance tests (4 valid + 5 invalid/out-of-scope) |

Task is selected by environment variable:

```bash
# PRIMARY: reach gap among eligible children 0-17
GHS_TASK=csg_reach_gap python 13_models.py

# comparison: grant take-up among all children 0-17
GHS_TASK=csg_takeup    python 13_models.py

# comparison: any social grant, whole population
GHS_TASK=grant_any     python 13_models.py
```

Every script sets `SEED = 42`. Paths resolve relative to the script, so there
are no hard-coded absolute paths.

## 3. Launch the interface

```bash
cd analysis
streamlit run app.py
```

Opens on <http://localhost:8501>. Headless (for a server or demo recording):

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

Health check: `curl http://localhost:8501/_stcore/health` → `ok`

## 4. What the interface loads

It serves the **final optimised champion** (brief Q7.1) and **loads saved artefacts
only** — nothing is re-fitted at inference:

- `out/models/optimised_bundle_csg_reach_gap.joblib` — the fitted preprocessor and the
  19-feature contract selected by the Q5 random search
- `out/models/mlp_optimised_csg_reach_gap.pt` — optimised champion weights
  (512-256-128, dropout 0.1, pos_weight 3.0)
- `out/models/calibrators_csg_reach_gap.joblib` — isotonic calibrators, including the
  `Optimised MLP` entry fitted in `25_calibrate_optimised.py`
- `out/optimised_calibrated_csg_reach_gap.npz` — calibrated validation scores, used to
  express a single household as an outreach percentile

The four pre-optimisation models remain selectable in the sidebar for the comparison
the brief requires. Each model carries its **own** feature contract, because the
optimised champion consumes 19 features where the others consume 24; relying on a
shared feature list would silently feed it the wrong design matrix.

Serving the optimised champion rather than the pre-optimisation model is worth about
**24,000 additional children found** per outreach round (PR-AUC 0.2337 to 0.2567,
lift@10% 2.59x to 2.77x on the locked test set).

## 5. Features

- **Single household tab** — enter the child's details, then list other household
  members in an editable table. The 14 household-level features are derived from
  that roster exactly as during training, so the user never computes an aggregate.
- **Batch upload tab** — CSV upload with a downloadable template; invalid rows are
  reported individually with row numbers and the file still scores the valid ones.
- **Input guide tab** — accepted values and guidance per field, limitations,
  responsible-use notice.
- Survey skip patterns are respected: labour status is hidden below age 15 and
  school attendance below age 5, matching the training data (verified 0% populated).
- Model selector allows comparing champion against the other three models.

## 6. Seven corrections worth knowing

Each of these was found by checking a result that looked wrong, and each changed either a number the interface shows or a conclusion in the report.

**Probability calibration.** All models were trained with class rebalancing
(`class_weight="balanced"`, BCE `pos_weight=0.454`), so raw output is calibrated
to a 50/50 prior and understates the true likelihood by 16–18 percentage points.
An isotonic calibrator fitted on validation year 2019 and checked on 2020 reduces
mean bias from −0.180 to −0.025 and expected calibration error from 0.180 to 0.028,
leaving ROC-AUC unchanged. The interface shows the calibrated figure and discloses
the raw one.

**Skip-pattern inputs.** An early version sent `lab_status="Not economically active"`
for young children, a combination that never occurs in training (verified: 0%
populated for ages 0–14). Honouring the skip pattern moved one test case by 20
percentage points. The interface now hides fields the survey would not have asked.

**Risk bands instead of a 0.5 threshold.** With an 11% positive rate, a 0.5 cut-off
labels a household at 3× population risk as "already reached". The interface now
reports a lift-based band (Very high / High / Elevated / Typical) and an outreach
percentile, which is what a capacity-constrained decision actually needs.

**Random tie-breaking in the targeting curves.** `DummyClassifier` assigns an
identical score to every record, so sorting left ties in dataset order and made a
constant predictor appear to have lift. Ties are now broken with a fixed-seed random
permutation, and the baseline sits on the diagonal as it must.

**Constant features excluded from explanations.** Five features are constant inside
the eligible population because they *are* the eligibility filter (`medaid`,
`hh_any_medaid`, `hh_n_employed`, `hh_no_earner`, `hh_workers_per_capita`). The model
never saw them vary, so perturbing them measured extrapolation rather than a learned
effect. They are excluded from explanations, and the optimised champion drops them
from the feature set entirely — ablation A2 and the random search reached that
conclusion independently.

**An uncalibrated model is not comparable at a fixed threshold.** The first McNemar
run compared a calibrated Random Forest against the uncalibrated optimised champion
and reported a 6.9 pp gap at p ~ 1e-107. That measured the missing calibration stage,
not model quality. The rank-based test was unaffected (p = 0.68), which is the
diagnostic: monotone calibration cannot change a ranking. After calibrating both,
the models are statistically indistinguishable (p = 0.057 at 0.5, p = 0.82 on the
outreach rule).

**A child under 15 cannot live alone.** Batch upload originally built no household
roster, so every row was scored as a single-person household. A lone 1-year-old
produced a 100% probability against 39.8% for the same child with a caregiver, because
`hh_size=1` with a toddler never occurs in the survey. Such rows are now refused, and
the batch format accepts optional `caregiver_*` columns so a row can describe a real
household.

## 6a. Git LFS warning

`.gitattributes` routes `*.csv` and `*.joblib` through Git LFS. If the LFS smudge
filter does not run, those files appear in the working tree as 129-byte pointer stubs
and every script that loads a model will fail. Recover with:

```bash
git lfs install
git lfs checkout
```

Consider removing `*.csv` from LFS tracking — these are kilobyte-scale files and LFS
adds fragility without benefit.

## 7. Security and privacy

- No credentials, secrets or API keys. Nothing is read from a private endpoint.
- No personal records are bundled. GHS microdata is distributed de-identified and
  `uqnr` is a survey household identifier, not a person identifier.
- All computation is local; no user input leaves the machine.

## 8. Responsible use

Output supports **population-level prioritisation and outreach planning**. It must
not be used to adjudicate, approve, reduce or deny an individual's grant. It is not
a SASSA eligibility decision and carries no legal standing.

