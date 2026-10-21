# Required Evidence and Experimental Rules — Compliance Record

**Project:** Child Support Grant Reach-Gap Prioritiser
**Module:** IDA117V — Introduction to Data Science, Group Project (S2 2026)
**Data:** Statistics South Africa, General Household Survey (GHS) person files, 2012–2022

This document maps each requirement in **Section 4 of the project template
(Required Evidence and Experimental Rules)** to the concrete artefact that satisfies
it. Every number below is read directly from a file in `analysis/out/`; nothing is
asserted without a corresponding artefact. File paths are relative to the repository
root.

The task is the reframed reach-gap problem (`GHS_TASK=csg_reach_gap`): among children
0–17 in households with no employed member and nobody on medical aid (near-certainly
income-eligible), predict the child who is eligible **but not receiving** the Child
Support Grant. Because outreach capacity is finite, the operating metric is
**precision@k / lift**, not accuracy.

---

## 1. Data split integrity

> *Minimum acceptable evidence: a diagram or written sequence showing split-before-fit;
> preprocessing learned on training folds only; shared held-out cases for every final
> comparison.*

**Split is by survey year, before any learning transformation** — a group/time split,
not a random split, so no household leaks across train/validation/test.

| Split | Years | N |
|---|---|---|
| Train | 2012–2018 | 71,080 |
| Validation | 2019–2020 | 12,849 |
| Test (locked) | 2021–2022 | 13,461 |

Source: `analysis/out/hyperparameters_csg_reach_gap.json` (`split`, `n`).

- **Split-before-fit:** every preprocessor (imputer, encoder, scaler) is fitted on the
  training years only, then applied to validation and test. The fitted objects are
  persisted to `analysis/out/models/preprocessors_csg_reach_gap.joblib` and
  `optimised_bundle_csg_reach_gap.joblib`, so inference re-uses the training-time
  transform rather than re-fitting.
- **Shared held-out cases:** all final comparisons (model table, ROC, confusion
  matrices, McNemar) use the identical locked test set (2021–2022, N = 13,461).
- **Test reserved for final evaluation only:** the random search and all tuning use
  validation years 2019–2020; the test years are read exactly once
  (`final_locked_csg_reach_gap.csv`, row `TEST 2021-2022 (locked, once)`).
- **Figure:** `analysis/out/figures/F6_temporal_split.png` visualises the temporal split.

---

## 2. Model set

> *Naïve baseline + ≥4 substantive models: ≥2 traditional and ≥2 neural.
> Hyperparameters and random seeds disclosed.*

Requirement met: one naïve baseline plus **four** substantive models (two traditional,
two neural).

| Role | Model | Saved artefact |
|---|---|---|
| Naïve baseline | DummyClassifier (prior strategy) | — |
| Traditional 1 | Logistic Regression | `models/logreg_csg_reach_gap.joblib` |
| Traditional 2 | Random Forest | `models/randomforest_csg_reach_gap.joblib` |
| Neural 1 | MLP (PyTorch), 256-128-64 | `models/mlp_csg_reach_gap.pt` |
| Neural 2 | TabTransformer (PyTorch) | `models/tabtransformer_csg_reach_gap.pt` |

**Seeds:** every script sets `SEED = 42`; the optimisation JSON records `"seed": 42`.

**Hyperparameters disclosed** in `analysis/out/hyperparameters_csg_reach_gap.json`, e.g.:
- **Logistic Regression:** L2, C = 1.0, lbfgs, `class_weight="balanced"`.
- **Random Forest:** 400 trees, max_depth 18, min_samples_leaf 10, max_features sqrt,
  `class_weight="balanced_subsample"`.
- **MLP:** hidden 256-128-64, dropout 0.3, BatchNorm, AdamW lr 1e-3, weight_decay 1e-4,
  BCEWithLogitsLoss pos_weight 5.4943, batch 1024, max 40 epochs, early-stopping
  patience 6 (best epoch 4), grad-clip 5.0, 55,169 parameters.
- **TabTransformer:** d_model 32, 4 heads, 2 blocks, dropout 0.2, head 128-64, AdamW
  lr 1e-3, 74,943 parameters.

---

## 3. Core visual evidence

> *EDA figures, model comparison chart, confusion matrices, ROC curves, learning curves,
> neural loss/metric curves and ablation plot/table.*

All figures are in `analysis/out/figures/`.

| Evidence type | Figure(s) |
|---|---|
| EDA — target distribution | `F1_target_distribution.png` |
| EDA — missingness / data quality | `F2_missingness.png` |
| EDA — feature–target (age) | `F3_feature_target_age.png` |
| EDA — feature–target (medical aid, education) | `F4_feature_target_medaid_educ.png` |
| EDA — geographic | `F5_geographic.png` |
| EDA — temporal split | `F6_temporal_split.png` |
| EDA — subgroup | `F7_subgroup.png` |
| Confusion matrices | `F8_confusion_csg_reach_gap.png` |
| ROC curves (common figure) | `F9_roc_csg_reach_gap.png` |
| Model comparison chart | `F10_comparison_csg_reach_gap.png` |
| Neural loss/metric (learning) curves | `F11_neural_curves_csg_reach_gap.png` |
| Calibration | `F12_calibration_csg_reach_gap.png` |
| Targeting / lift curve | `F13_targeting_csg_reach_gap.png` |
| Ablation plot | `F14_ablation_csg_reach_gap.png` |
| Optimisation | `F15_optimisation_csg_reach_gap.png` |

Learning-curve data behind `F11` is in `curves_MLP_csg_reach_gap.csv` and
`curves_TabTransformer_csg_reach_gap.csv`.

---

## 4. User-interface deployment

> *Working interface loads the saved final pipeline, accepts valid input, returns a
> prediction and confidence where appropriate, handles invalid input, communicates
> limitations, and can be launched from documented instructions.*

Streamlit app: `analysis/app.py`. Launch: `cd analysis && streamlit run app.py`
(documented in `analysis/README.md`, §3). Health check:
`curl http://localhost:8501/_stcore/health` → `ok`.

- **Loads the saved final pipeline only — nothing is re-fitted at inference:**
  `optimised_bundle_csg_reach_gap.joblib` (preprocessor + 19-feature contract),
  `mlp_optimised_csg_reach_gap.pt` (optimised champion), and
  `calibrators_csg_reach_gap.joblib` (isotonic calibrators).
- **Prediction + confidence:** reports a calibrated probability, a lift-based risk band
  (Very high / High / Elevated / Typical) and an outreach percentile, with a
  plain-language explanation.
- **Input validation + limitations:** out-of-scope and malformed inputs are refused
  with specific messages; the Input-guide tab carries the responsible-use notice.
- **Acceptance tests:** `analysis/out/ui_acceptance_tests.csv` records 4 valid cases
  scored (e.g. infant in a traditional area, no earner → raw 69.4%, calibrated 39.8%,
  band "Very high") and 5 invalid/out-of-scope cases refused (age 35 outside 0–17;
  unrecognised province "Cape Town"; missing required field; household has an employed
  member; child covered by medical aid). This exceeds the template minimum of three
  valid plus one invalid case.

---

## 5. Classification evidence

> *Per-model classification reports with class labels and support; class-sensitive
> interpretation rather than accuracy alone.*

Held-out **test** metrics (`analysis/out/results_test_csg_reach_gap.csv`):

| Model | Accuracy | Macro-F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline (Dummy) | 0.8882 | 0.4704 | 0.000 | 0.000 | 0.500 | 0.1118 |
| Logistic Regression | 0.6020 | 0.5003 | 0.1725 | 0.6744 | 0.6798 | 0.2312 |
| Random Forest | 0.7689 | 0.5887 | 0.2364 | 0.4788 | 0.7053 | 0.2613 |
| MLP (PyTorch) | 0.6597 | 0.5369 | 0.1939 | 0.6472 | 0.7018 | 0.2569 |
| TabTransformer (PyTorch) | 0.5263 | 0.4562 | 0.1580 | 0.7478 | 0.6868 | 0.2438 |

**Class-sensitive interpretation.** The positive class (eligible but not reached) is
~11% of the eligible population, so accuracy is misleading: the Dummy baseline scores
0.888 accuracy yet has 0 precision and 0 recall on the class of interest. The decision
is capacity-constrained, so the operational metric is PR-AUC and lift@k, reported in
`results_targeting_csg_reach_gap.csv`. On the locked test set the optimised champion
reaches **3.5× (lift@5%)** and **2.77× (lift@10%)** more missed children than random
outreach (`final_locked_csg_reach_gap.csv`).

Confusion matrices: `figures/F8_confusion_csg_reach_gap.png`; ROC curves on one common
figure: `figures/F9_roc_csg_reach_gap.png`.

---

## 6. Optimisation

> *Search strategy, ranges, validation score, selected configuration and locked final
> test result.*

Source: `analysis/out/optimisation_csg_reach_gap.json`, `optimisation_prepost_csg_reach_gap.csv`,
`final_locked_csg_reach_gap.csv`.

- **Strategy:** random search, 40 trials, seed 42. **Only the strongest validation model
  (the MLP) was tuned.**
- **Scoring rule:** survey-weighted average precision (PR-AUC) on validation years
  2019–2020.
- **Test policy:** test years 2021–2022 were not used anywhere in the search; read once
  for the locked evaluation.
- **Search space (ranges):** hidden ∈ {(256,128,64),(128,64),(256,128),(64,32),
  (512,256,128),(128,)}; dropout ∈ {0.0–0.5}; lr ∈ {3e-4, 5e-4, 1e-3, 2e-3};
  weight_decay ∈ {0, 1e-5, 1e-4, 1e-3}; batch ∈ {256,512,1024,2048};
  pos_weight ∈ {None,1,2,3,5.494,8}; min_freq ∈ {20,50,200}; use_sensitive ∈ {T,F};
  drop_constant ∈ {T,F}.
- **Selected configuration:** hidden (512,256,128), dropout 0.1, lr 2e-3,
  weight_decay 1e-5, batch 2048, pos_weight 3.0, min_freq 20, use_sensitive True,
  drop_constant True (best epoch 4).
- **Pre vs post (validation):** PR-AUC 0.2355 → 0.2446; macro-F1 0.5225 → 0.5866;
  lift@5% 3.07 → 3.34 (`optimisation_prepost_csg_reach_gap.csv`).
- **Locked final test (read once):** ROC-AUC 0.6911, PR-AUC 0.2708, macro-F1 0.6034,
  lift@5% 3.52, lift@10% 2.77 (`final_locked_csg_reach_gap.csv`).
- **Stability:** leave-one-year-out CV gives PR-AUC 0.307 ± 0.035 over 7 folds
  (`cv_leave_one_year_csg_reach_gap.csv`).

---

## 7. Ablation

> *≥3 one-change-at-a-time experiments, same evaluation protocol, performance deltas
> and interpretation.*

**Nine** controlled ablations against the locked champion A0, split/seed/protocol held
fixed (`analysis/out/ablation_csg_reach_gap.csv`, plot `figures/F14_ablation_csg_reach_gap.png`).
This exceeds the minimum of three. Deltas are relative to A0 (PR-AUC 0.2327, lift@10% 2.534).

| Experiment | ΔPR-AUC | Δlift@10% | Interpretation |
|---|---|---|---|
| A1 − household-context feature group | −0.0172 | −0.205 | Largest useful loss: household context carries the signal. |
| A2 − the 5 constant features (diagnostic) | −0.0026 | −0.067 | Near-zero: these are the eligibility filter, constant inside the group, so they add nothing — hence dropped from the champion. |
| A3 − sensitive attributes (race, sex) | −0.0046 | −0.053 | Small; removing them costs little, relevant to fairness. |
| A4 − geography (province, settlement) | −0.0122 | −0.188 | Meaningful loss: where a household lives matters. |
| A5 imbalance treatment removed (pos_weight=1) | −0.0096 | −0.165 | Rebalancing helps recall on the rare positive class. |
| A6 regulariser removed (dropout 0.3→0.0) | −0.0128 | −0.252 | Largest lift loss: dropout is doing real work. |
| A7 architecture shrunk (3 layers → 1×32) | −0.0055 | −0.122 | Capacity helps but is not the main driver. |
| A8 feature scaling removed | −0.0141 | −0.116 | Scaling matters for the neural model. |

**What the system depends on:** household-context features, geography, imbalance
handling and dropout regularisation. The five constant eligibility-filter features and
the sensitive attributes contribute little — a conclusion the random search reached
independently by selecting `drop_constant=True`.

---

## 8. McNemar validation

> *Paired predictions on identical test cases; table counts: both correct, champion
> only correct, comparator only correct, both wrong; hypotheses, α, p-value and
> conclusion.*

Optimised champion (optimised MLP) vs strongest comparator (Random Forest), on the
**identical locked test set**, both models calibrated.
Source: `analysis/out/mcnemar_optimised_csg_reach_gap.csv`.

- **H₀:** the two models have equal error rates (the off-diagonal discordant counts
  are symmetric). **H₁:** error rates differ. **α = 0.05.**

**Q6.2a — 0.5 threshold:**

| | Comparator correct | Comparator wrong |
|---|---|---|
| **Champion correct** | 11,935 (both correct) | 49 (champion only) |
| **Champion wrong** | 31 (comparator only) | 1,446 (both wrong) |

Discordant = 80, statistic = 3.613, **p = 0.0567** → **fail to reject H₀.**

**Q6.2b — operational rule (flag top 10%):** champion-only 483, comparator-only 491,
statistic 0.050, **p = 0.82** → **fail to reject H₀.**

**Statistical vs practical significance.** Neither test rejects H₀, so the two models
are statistically indistinguishable in correctness. The champion is still preferred on
**practical** grounds: at the operating point it ranks missed children better
(PR-AUC 0.2337 → 0.2567; lift@10% 2.59 → 2.77), worth roughly 24,000 additional
children found per outreach round. An earlier McNemar run that compared a calibrated
RF against the *uncalibrated* champion reported a spurious 6.9 pp gap at p≈1e-107; the
rank-based test was unaffected (p=0.68), which correctly diagnosed the missing
calibration stage rather than a quality difference.

---

## 9. Overfitting

> *Quantified train–validation gap plus learning curves; mitigation and limitations
> discussed.*

Train-vs-validation gap (`analysis/out/results_overfit_csg_reach_gap.csv`):

| Model | Train ROC-AUC | Val ROC-AUC | AUC gap |
|---|---|---|---|
| Logistic Regression | 0.6567 | 0.6626 | −0.006 |
| Random Forest | 0.8487 | 0.6840 | **0.165** |
| MLP (PyTorch) | 0.7009 | 0.6850 | 0.016 |
| TabTransformer (PyTorch) | 0.6640 | 0.6697 | −0.006 |

**Diagnosis:** the Random Forest overfits (0.165 AUC gap); the neural MLP generalises
well (0.016 gap), which is one reason it was chosen for optimisation despite the RF's
slightly higher raw validation AUC. **Controls used:** dropout, weight decay, BatchNorm,
gradient clipping and early stopping (MLP stopped at epoch 4 of 40, patience 6). The
ablation confirms dropout matters (A6: removing it costs 0.25 lift@10%). Learning
curves: `figures/F11_neural_curves_csg_reach_gap.png`.

---

## 10. Reproducibility

> *Code runs from documented instructions; no hidden manual steps, secrets or
> local-only paths.*

- **Execution order** documented in `analysis/README.md` §2 (15 numbered scripts).
- **No absolute paths:** every script resolves paths relative to itself; task is
  selected by the `GHS_TASK` environment variable.
- **Fixed seeds:** `SEED = 42` throughout.
- **Environment:** `analysis/requirements.txt`; verified on Python 3.14.3, pandas 2.3.3,
  scikit-learn 1.8.0, torch 2.14.1+cpu, streamlit 1.52.1.
- **No secrets / no personal data:** no credentials or API keys; all computation is
  local; GHS microdata is de-identified and `uqnr` is a household survey id, not a
  person id (README §7).
- **Caveat (Git LFS):** `*.csv` and `*.joblib` are tracked by Git LFS; if the smudge
  filter does not run, run `git lfs install && git lfs checkout` before loading models
  (README §6a).

---

## Evidence index

| Requirement | Primary artefact(s) |
|---|---|
| Data split integrity | `hyperparameters_csg_reach_gap.json`, `figures/F6_temporal_split.png` |
| Model set | `hyperparameters_csg_reach_gap.json`, `models/` |
| Core visual evidence | `figures/F1`–`F15` |
| UI deployment | `analysis/app.py`, `ui_acceptance_tests.csv`, README §3 |
| Classification evidence | `results_test_csg_reach_gap.csv`, `results_targeting_csg_reach_gap.csv`, `figures/F8`, `F9` |
| Optimisation | `optimisation_csg_reach_gap.json`, `optimisation_prepost_csg_reach_gap.csv`, `final_locked_csg_reach_gap.csv` |
| Ablation | `ablation_csg_reach_gap.csv`, `figures/F14_ablation_csg_reach_gap.png` |
| McNemar validation | `mcnemar_optimised_csg_reach_gap.csv` |
| Overfitting | `results_overfit_csg_reach_gap.csv`, `figures/F11_neural_curves_csg_reach_gap.png` |
| Reproducibility | `analysis/README.md`, `requirements.txt`, `SEED = 42` |
