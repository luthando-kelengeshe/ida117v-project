# Requirements Document

## Introduction

This specification covers the IDA117V "Introduction to Data Science" Semester 2 2026 group project: an end-to-end, reproducible classification solution built on Stats SA General Household Survey (GHS) person-level microdata for 2012-2022, delivered with a stakeholder-facing user interface, a technical report, and a signed contribution statement.

The assessment brief (`analysis/out/brief.txt`) awards 100 marks across eight questions and eight rubric criteria. Marks are awarded per question, so every requirement below names the brief question and the mark allocation it serves. Requirements are ordered so that gating work (data approval, target definition) precedes modelling work, matching the brief's own sequencing rule that the Data Approval Checkpoint is satisfied before modelling begins.

### Established Baseline (verified, not to be re-derived)

Exploratory work already completed in `analysis/` (scripts 01-10 and `ghs_harmonise.py`) established the following, which this specification treats as ground truth:

- 773,311 person records across 11 annual files; 70,000-93,000 records per year except 2020 (34,370) and 2021 (35,265), both halved and collected telephonically under COVID-19 lockdown conditions.
- `person_wgt` exists in all 11 files and reproduces the South African population total from 52.3 million (2012) to 61.4 million (2022).
- Three harmonisation obstacles, solved in `analysis/ghs_harmonise.py`: name drift (only 12 of approximately 250 column names stable across all 11 years; questionnaire codes 2012-2018, descriptive names 2019-2022, plus case drift), coding regime split (2014-2018 numeric-coded, all other years label-coded, column counts 81 to 214), and sentinel codes 8/88/9/99/98/888888888 meaning "not applicable" or "unspecified" rather than real values (67,315 salary rows carry 888888888 in 2014 alone).
- Two structural breaks that are measurement artefacts, not real change: the disability measure changed between 2013 and 2014 (16.1% to 10.5% weighted prevalence, with a further anomalous 14.3% spike in 2019), and `geotype` collapsed from four categories to three between 2014 and 2015.
- Leakage: the grant-type follow-up columns (`soc_grant_oag`, `soc_grant_csg`, `soc_grant_dis`, `soc_grant_srd`, `soc_grant_type`, `soc_disb_perm` and the `Q31b*` / `Q132b*` equivalents) are asked only of grant recipients, are approximately 100% populated among recipients and approximately 0% among non-recipients, and lift held-out AUC from 0.906 to 0.993 when included.
- Target-choice problem: whole-population grant receipt is largely determined by age-based eligibility rules (age alone reaches AUC 0.824 against a 0.580 majority-class baseline; grant rate 62-72% for ages 0-17, 4-13% for ages 18-59, 51-76% for ages 60 and above, with a discontinuity from 13.4% at age 59 to 51.3% at age 60). Stratifying by eligibility group and predicting take-up within the stratum removes the mechanical signal (within-stratum age AUC 0.49-0.59) and yields AUC 0.824 for children 0-17, 0.923 for older persons 60 and above, and 0.713 for working age 18-59.
- Measured leave-one-out AUC cost in the children stratum: `medaid` -0.069, `prov` -0.014, `race` -0.012, `geotype` -0.006, `educ_band` / `attend` / `disab` / `lab_status` near zero, `sex` exactly 0.000.
- Household-context features built by self-joining the person file on `uqnr` add +0.039 AUC for children (0.824 to 0.863) across 218,980 households with a median size of 3.
- Available geography is province (9), metro, metro code, `geotype`, stratum and PSU. No district or local-municipality identifier exists, so the brief's "classify municipalities or wards" option is infeasible.
- Verified environment: Python 3.14.3, pandas 2.3.3, scikit-learn 1.8.0, matplotlib 3.10.7, pypdf 6.1.1. No deep-learning framework is installed.

### Mark Allocation and Traceability

| Requirement | Brief question / rubric criterion | Marks |
| --- | --- | --- |
| 1 Data Approval Checkpoint | Section 2 gating checkpoint (prerequisite to criteria 1-2) | Gate |
| 2 Scope Decision Gate | Section 2 scope and data; criterion 1 | Gate |
| 3 Problem and Task Definition | Q1.1 | 4 |
| 4 Metrics and Error Cost | Q1.2 | 2 |
| 5 Provenance, POPIA and Fairness | Q1.3 | 4 |
| 6 Data Dictionary and Audit | Q2.1 | 4 |
| 7 Harmonisation Correctness | Q2.1 / Q2.4 (enabling) | shared |
| 8 Visualisation Set | Q2.2 | 6 |
| 9 Split Protocol | Q2.3 | 3 |
| 10 Preprocessing Pipeline | Q2.4 | 5 |
| 11 Naive Baseline | Q3.1 | 2 |
| 12 Substantive Model Set | Q3.2, Q3.3 | 13 |
| 13 Training Documentation and Rationale | Q3.4, Q3.5 | 7 |
| 14 Held-out Metric Set | Q4.1 | 4 |
| 15 Per-model Diagnostic Evidence | Q4.2 | 5 |
| 16 Model Comparison and Champion Selection | Q4.3 | 3 |
| 17 Overfitting Diagnosis | Q4.4 | 4 |
| 18 Champion Optimisation | Q5.1, Q5.2, Q5.3 | 10 |
| 19 Ablation Study | Q6.1 | 5 |
| 20 McNemar Validation | Q6.2 | 5 |
| 21 User Interface Deployment | Q7.1-Q7.4 | 10 (rubric 7) |
| 22 Stakeholder Recommendation | Q8.1 | 2 |
| 23 Reproducible Repository | Q8.2 | 3 |
| 24 Presentation and Contribution Statement | Q8.3 | 2 |

### Open Decisions

Four decisions remain unresolved and are specified as explicit decision gates (Requirement 2) rather than silently assumed: acquisition of the GHS household files, final target definition, the role of survey weights, and the neural framework and second tabular architecture. Requirement 2 now carries a dated default for each of these open decisions, so that an unmade decision resolves deterministically rather than blocking work, except the neural-framework decision, which escalates to the lecturer because scikit-learn alone cannot satisfy the brief's two-neural-architecture rule.

## Glossary

- **Ablation_Study**: A set of controlled experiments in which exactly one component (feature group, preprocessing step, imbalance treatment, regulariser or architecture element) is removed or altered while split, seed and evaluation protocol are held fixed.
- **Approval_Package**: The three artefacts required by the brief's Data Approval Checkpoint: a one-page concept note, a dataset card and a feasibility check.
- **Baseline_Model**: A naive reference classifier (for example scikit-learn `DummyClassifier` with a majority or stratified strategy) that establishes the floor performance any substantive model must beat.
- **CSG**: Child Support Grant, a South African social grant paid to the caregiver of a qualifying child.
- **Data_Audit**: The documented examination of class balance, duplicate records, missingness, invalid values, outliers and leakage across the analysis panel.
- **Data_Dictionary**: A machine-readable and human-readable table listing every retained variable with source column name per year, canonical name, data type, value domain, coding regime, missingness rate and retention decision.
- **Decision_Log**: A dated, append-only record in the Technical_Report holding every scope and design decision, its named owner, its supporting evidence and the downstream requirements it constrains; superseded entries are retained rather than edited.
- **EARS**: Easy Approach to Requirements Syntax, the constrained requirement-writing patterns used in this document.
- **Eligibility_Stratum**: One of the three age-defined groups used to condition the prediction task: children aged 0-17, working-age persons aged 18-59, and older persons aged 60 and above.
- **Evaluation_Report**: The collection of held-out metric tables, classification reports, confusion matrices, ROC curves and comparison charts produced for every model.
- **GHS**: General Household Survey, an annual multi-stage stratified household survey conducted by Statistics South Africa.
- **geotype**: The GHS settlement-type classification of a dwelling, harmonised in this project to three categories: Urban, Traditional and Rural formal.
- **Harmonisation_Layer**: The code module (`analysis/ghs_harmonise.py` and successors) that resolves column-name drift, coding-regime differences and sentinel codes so that the 11 annual files can be pooled.
- **Leakage**: The presence in model inputs of information that is only observable because the outcome has already occurred, which inflates held-out performance and cannot be reproduced at prediction time.
- **McNemar_Test**: A paired statistical test on the 2x2 table of correct and incorrect predictions made by two classifiers on identical held-out cases, used to decide whether a difference in accuracy is attributable to chance.
- **Model_Suite**: The set of at least four substantive classifiers required by the brief, comprising at least two traditional models and at least two neural models, all trained on identical splits.
- **OAG**: Old Age Grant, a South African social grant paid to qualifying persons aged 60 and above.
- **Optimisation_Study**: The documented hyperparameter search applied to the single strongest validation model, including search space, scoring rule, search strategy and selected configuration.
- **person_wgt**: The GHS person-level survey weight, which scales a sampled record to the number of persons in the population it represents.
- **Placeholder**: An empty value or an unresolved marker such as TBD or TODO, which does not satisfy a requirement that an element be populated.
- **POPIA**: The Protection of Personal Information Act 4 of 2013, the South African data-protection statute governing processing of personal information.
- **Preprocessing_Pipeline**: A single serialisable pipeline object that performs cleaning, encoding, scaling, imputation, feature engineering and imbalance treatment, fitted only on training data.
- **Project_Team**: The group of up to six students accountable for producing every artefact in this specification.
- **PSU**: Primary Sampling Unit, the first-stage cluster of the GHS sample design; records sharing a PSU are geographic neighbours and are statistically dependent.
- **Repository**: The submitted code package containing README, environment specification, scripts, saved pipeline, user-interface code and data dictionary.
- **Split_Protocol**: The documented rule that assigns records to training, validation and test partitions, including any stratification, grouping or time constraint.
- **SRD**: Social Relief of Distress grant, introduced in 2020 as a COVID-19 income-support measure.
- **stratum**: The GHS sample-design stratum, a grouping of PSUs used for explicit stratification before first-stage selection.
- **Technical_Report**: The submitted PDF report answering all eight brief questions.
- **uqnr**: The GHS unique household identifier, which joins person records belonging to the same household and joins the person file to the household file.
- **User_Interface**: The deployed stakeholder-facing application that loads the saved final pipeline and returns predictions.
- **Visualisation_Suite**: The set of at least five purposeful figures required by the brief, each with a title, units, readable labels and a written interpretation.

## Requirements

### Requirement 1: Data Approval Checkpoint (gating, prerequisite to all modelling)

**User Story:** As the Project_Team, I want the brief's Data Approval Checkpoint satisfied before any model is trained, so that modelling effort is not invested in a dataset or target the lecturer rejects.

#### Acceptance Criteria

1. THE Approval_Package SHALL contain a concept note of at most one A4 page that states, in at least one complete sentence each, the service-delivery problem, the named stakeholder, the unit of prediction as one person record, the target variable, the decision the named stakeholder takes on the basis of a prediction, and the anticipated public value.
2. THE Approval_Package SHALL contain a dataset card that states the Stats SA source URL from which the 11 GHS person files were obtained, the data owner, the licence or terms of use, the collection period as survey years 2012 to 2022 inclusive, the unit of analysis as one person record, the retained feature list, the target, the record count, the missingness of every retained feature expressed as a percentage of records per survey year, the sensitive attributes race, sex, age and disability status, and at least three known limitations including the halved 2020 and 2021 samples, the column-name and coding-regime drift across years, and the absence of any district or local-municipality identifier.
3. THE Approval_Package SHALL contain a feasibility check that states (a) the target definition and the class prior of the target in each Eligibility_Stratum, (b) the record count of every target class in every Eligibility_Stratum together with confirmation that the smallest such class holds at least 1,000 records, (c) confirmation that the GHS person files carry no direct personal identifier, specifically no name, identity number, contact detail or street address, and that `uqnr` is a survey household identifier, with the POPIA basis for that confirmation, (d) the compute environment as Python 3.14.3, pandas 2.3.3, scikit-learn 1.8.0 and matplotlib 3.10.7 on the Project_Team hardware, together with confirmation that no deep-learning framework is yet installed, and (e) the evaluation plan naming the primary technical metric, the Split_Protocol and the Baseline_Model.
4. THE Approval_Package SHALL record the total record count 773,311, the record count of each of the 11 survey years 2012 to 2022 individually such that the 11 per-year counts sum to 773,311, the reduced counts 34,370 for 2020 and 35,265 for 2021, and the statement that the 2020 and 2021 counts are reduced because those samples were halved and collected telephonically under COVID-19 lockdown conditions.
5. WHEN the Approval_Package is recorded as complete, THE Project_Team SHALL submit the Approval_Package to the lecturer and record the submission date in the Technical_Report decision log.
6. IF the lecturer records the approval outcome as withheld or conditional, THEN THE Project_Team SHALL resubmit a revised Approval_Package within 5 working days of receiving that outcome, with every element the lecturer identified as deficient revised and with the outcome, its date and the revisions recorded in the Technical_Report decision log.
7. WHILE the Technical_Report decision log holds no entry recording lecturer approval of the Approval_Package as granted, THE Project_Team SHALL NOT train the Baseline_Model of Requirement 11 or any model of the Model_Suite of Requirement 12.
8. WHEN every element enumerated in criteria 1 to 4 is present in the Approval_Package and populated with content other than a placeholder, where a placeholder is an empty value or a marker such as TBD or TODO, THE Project_Team SHALL record the Approval_Package as complete in the Technical_Report decision log together with the recording date.
9. IF any element enumerated in criteria 1 to 4 is absent from the Approval_Package or populated only with a placeholder, THEN THE Project_Team SHALL record the Approval_Package as incomplete, list every absent element in the Technical_Report decision log, retain the elements already populated, and withhold submission until every listed element is populated.

### Requirement 2: Scope and Design Decision Gate

**User Story:** As the Project_Team, I want the four open scope decisions recorded with evidence and a date, so that downstream requirements have a single authoritative answer and the Technical_Report can defend each choice.

#### Acceptance Criteria

1. THE Project_Team SHALL record in the Technical_Report decision log, no later than 2 working days before the Approval_Package is submitted under Requirement 1, an acquisition decision for the GHS household files that states exactly one of the two values "acquired" or "not acquired", names exactly one Project_Team member as decision owner, and cites as evidence that the water, sanitation, electricity, refuse-removal and dwelling-type variables are absent from all 11 person files and are joinable from the household files on `uqnr`.
2. IF, by the deadline stated in acceptance criterion 1, GHS household files are present in the Repository workspace in loadable form for fewer than 8 of the 11 survey years 2012 to 2022, or the `uqnr` join matches fewer than 95% of person records in any survey year intended for use, THEN THE Project_Team SHALL record the acquisition decision as "not acquired", SHALL apply the target defined in acceptance criterion 4 with the Eligibility_Stratum of children aged 0-17, and SHALL record this default with its calendar date in the decision log.
3. WHERE the acquisition decision is recorded as "acquired", THE Project_Team SHALL define the target as a service-access classification derived from the household water, sanitation, electricity and refuse-removal variables, SHALL state the class count and the rule that assigns a record to each class, SHALL state the class prior of each class, and SHALL restrict the analysis panel to the survey years that satisfy the 8-year and 95% join-coverage conditions of acceptance criterion 2.
4. WHERE the acquisition decision is recorded as "not acquired", THE Project_Team SHALL define the target as grant take-up within exactly one named Eligibility_Stratum, SHALL state the class prior of that target in that stratum, and SHALL document in the Technical_Report that the service-delivery framing rests on social-grant reach rather than on water, sanitation, electricity or refuse-removal variables.
5. THE Project_Team SHALL record in the Technical_Report decision log, no later than the date on which the Preprocessing_Pipeline is first fitted under Requirement 10, a `person_wgt` decision that states separately whether `person_wgt` is applied in descriptive and population statistics, in Visualisation_Suite population figures, in the service-delivery success indicator of Requirement 4, in model training, and in held-out metric computation, and that names exactly one Project_Team member as decision owner.
6. IF the `person_wgt` decision is not recorded by the deadline stated in acceptance criterion 5, THEN THE Project_Team SHALL apply `person_wgt` to descriptive and population statistics, to Visualisation_Suite population figures and to the service-delivery success indicator, SHALL compute model training and all held-out metrics unweighted, SHALL exclude `person_wgt` from the model input feature set, and SHALL record this default with its calendar date in the decision log.
7. THE Project_Team SHALL record in the Technical_Report decision log, no later than 2 working days before the first neural model training run under Requirement 12, a neural-model decision that names the neural framework, the framework version verified as installed in the verified environment, the date of that installation check, the multi-layer perceptron as the first neural model, the second tabular neural architecture, and exactly one Project_Team member as decision owner.
8. IF the named neural framework and version are not verified as installed in the verified environment by the deadline stated in acceptance criterion 7, THEN THE Project_Team SHALL escalate the installation failure to the lecturer, SHALL record the failure with its calendar date in the decision log, and SHALL NOT proceed to the first neural model training run until a framework supporting two distinct tabular neural architectures is verified as installed, because scikit-learn 1.8.0 supplies only a single neural estimator and therefore cannot satisfy the two-architecture rule of Requirement 12.
9. WHEN a decision required by this requirement is recorded, THE Project_Team SHALL include in the same decision-log entry the decision statement, the named decision owner, the calendar date of the decision, a reference to the supporting evidence, and the list of downstream requirement numbers that the decision constrains.
10. IF a recorded decision is reversed, THEN THE Project_Team SHALL add a superseding decision-log entry carrying its own calendar date and named owner within 2 working days of the reversal, SHALL retain the superseded entry unchanged in the decision log, SHALL rerun every experiment in Requirements 11 to 20 affected by the reversal before the Technical_Report is submitted, and SHALL record the reason for any repeated test-partition evaluation as required by Requirement 9.

### Requirement 3: Problem and Classification Task Definition (Q1.1, 4 marks)

**User Story:** As a public-service decision-maker, I want the problem and the prediction task defined precisely, so that model output maps onto a decision I can act on.

#### Acceptance Criteria

1. THE Technical_Report SHALL define the South African service-delivery problem, the affected population, the named decision-maker and the intended benefit.
2. THE Technical_Report SHALL specify the unit of prediction as either a person record or a household record, and SHALL state that ward-level and municipality-level prediction is infeasible because the GHS person files expose no district or local-municipality identifier.
3. THE Technical_Report SHALL specify the input feature set, the target classes and the prediction horizon.
4. THE Technical_Report SHALL state the Eligibility_Stratum definitions as ages 0-17, 18-59, and 60 and above.
5. THE Technical_Report SHALL justify the stratified framing by reporting that whole-population age-only prediction reaches AUC 0.824 against a majority-class baseline of 0.580, that the grant rate rises from 13.4% at age 59 to 51.3% at age 60, and that within-stratum age-only AUC falls to the range 0.49 to 0.59.
6. THE Technical_Report SHALL state the class prior of the selected target for each Eligibility_Stratum.

### Requirement 4: Success Metrics and Error Cost (Q1.2, 2 marks)

**User Story:** As a public-service decision-maker, I want one primary technical metric and a service-delivery indicator, so that model quality and programme value are both measurable.

#### Acceptance Criteria

1. THE Technical_Report SHALL name exactly one primary technical metric and SHALL state the numeric decision threshold applied when the metric requires a hard class label.
2. THE Technical_Report SHALL name at least one service-delivery success indicator expressed in population terms using `person_wgt`.
3. THE Technical_Report SHALL state the operational consequence of a false positive and the operational consequence of a false negative for the named decision-maker.
4. THE Technical_Report SHALL state which error type carries the higher cost and SHALL justify the asymmetry with reference to the intended decision.
5. WHERE the class prior of the target is below 0.20 or above 0.80 in any Eligibility_Stratum, THE Evaluation_Report SHALL report PR-AUC alongside the primary technical metric.

### Requirement 5: Provenance, Licence, POPIA and Fairness (Q1.3, 4 marks)

**User Story:** As a responsible data-science practitioner, I want provenance and risk safeguards documented, so that the project is lawful, fair and defensible.

#### Acceptance Criteria

1. THE Technical_Report SHALL state the dataset relevance, provenance, licence or terms of use, temporal and geographic coverage, and known limitations for all 11 GHS person files.
2. THE Technical_Report SHALL state that GHS microdata is distributed in de-identified form and SHALL state that `uqnr` is a survey household identifier rather than a direct personal identifier.
3. THE Technical_Report SHALL identify POPIA, privacy, fairness and misuse risks and SHALL state one safeguard for each identified risk.
4. WHERE `race` is retained as a model input, THE Technical_Report SHALL state the justification for retention, SHALL report per-group performance of the champion model, and SHALL report the performance change measured when `race` is removed.
5. THE Technical_Report SHALL report the measured leave-one-out AUC cost of `race` as -0.012 and of `sex` as 0.000 in the children stratum, and SHALL state the retention decision for each attribute.
6. THE Technical_Report SHALL state the conditions under which the model output would constitute misuse, including use for individual benefit adjudication.
7. IF the GHS licence prohibits redistribution of the source files, THEN THE Repository SHALL contain download instructions and the dataset card in place of the data files.

### Requirement 6: Data Dictionary and Data Audit (Q2.1, 4 marks)

**User Story:** As an assessor, I want a complete data dictionary and audit, so that I can verify that the analysis panel is understood and leakage-free.

#### Acceptance Criteria

1. THE Data_Dictionary SHALL list every retained variable with the per-year source column name, the canonical name, the data type, the value domain, the coding regime and the retention decision.
2. THE Data_Audit SHALL report class balance of the target overall and per Eligibility_Stratum.
3. THE Data_Audit SHALL report duplicate-record counts using the combination of `uqnr`, person number and survey year as the record key.
4. THE Data_Audit SHALL report per-variable and per-year missingness as a percentage of records.
5. THE Data_Audit SHALL report the count of sentinel-coded values converted to missing for each affected variable, including the 67,315 salary records coded 888888888 in 2014.
6. THE Data_Audit SHALL report outlier treatment rules for `age` and for salary, including the retained valid ranges.
7. THE Data_Audit SHALL list every excluded leakage column by name, SHALL report populated-rate among recipients and among non-recipients for each listed column, and SHALL report the AUC inflation from 0.906 to 0.993 observed when the leakage columns are included.
8. THE Data_Audit SHALL report the summed `person_wgt` per year and SHALL state the reproduced population totals of 52.3 million for 2012 and 61.4 million for 2022.

### Requirement 7: Harmonisation Layer Correctness (Q2.1 and Q2.4, enabling)

**User Story:** As an analyst pooling 11 survey years, I want harmonisation verified rather than assumed, so that pooled results are not corrupted by name drift, coding regimes or structural breaks.

#### Acceptance Criteria

1. THE Harmonisation_Layer SHALL resolve each canonical variable to the correct source column for every survey year in which the variable exists.
2. WHEN a survey year in the set 2014 to 2018 is loaded, THE Harmonisation_Layer SHALL translate numeric category codes to the labels used by label-coded years.
3. THE Harmonisation_Layer SHALL convert the sentinel codes 8, 88, 9, 99, 98 and 888888888 to missing values for every variable in which the sentinel codes denote "not applicable" or "unspecified".
4. THE Harmonisation_Layer SHALL map `geotype` for all survey years onto the three-category scheme Urban, Traditional and Rural formal.
5. FOR ALL harmonised categorical variables, the weighted category shares of a numeric-coded year SHALL fall within a documented tolerance of the weighted category shares of the adjacent label-coded year (round-trip verification property).
6. WHERE the disability measure is used in any pooled time series, THE Technical_Report SHALL mark the 2013-to-2014 break from 16.1% to 10.5% weighted prevalence and the 2019 spike to 14.3% as measurement artefacts.
7. IF a canonical variable is absent for a survey year, THEN THE Harmonisation_Layer SHALL return a missing column for that year and THE Data_Dictionary SHALL record the absence.

### Requirement 8: Visualisation Suite (Q2.2, 6 marks)

**User Story:** As an assessor, I want at least five purposeful figures with interpretations, so that the exploratory findings are legible and defensible.

#### Acceptance Criteria

1. THE Visualisation_Suite SHALL contain at least five figures.
2. THE Visualisation_Suite SHALL contain one figure showing the target class distribution.
3. THE Visualisation_Suite SHALL contain one figure showing missingness or data quality across variables and survey years.
4. THE Visualisation_Suite SHALL contain at least two figures showing a feature-target relationship.
5. THE Visualisation_Suite SHALL contain at least one figure showing a geographic, temporal or subgroup view.
6. FOR ALL figures in the Visualisation_Suite, each figure SHALL carry a title, axis units, readable axis labels and a written interpretation in the Technical_Report.
7. WHERE a figure presents population-level quantities, THE figure SHALL apply `person_wgt` and SHALL state the weighting in the caption.
8. WHERE a figure presents a time series spanning 2019 to 2021, THE figure SHALL annotate the reduced and telephonically collected 2020 and 2021 samples.

### Requirement 9: Split Protocol (Q2.3, 3 marks)

**User Story:** As an assessor, I want the split justified against the survey design, so that reported held-out performance is credible.

#### Acceptance Criteria

1. THE Split_Protocol SHALL define disjoint training, validation and test partitions, or a nested cross-validation design, before any transformation is fitted.
2. THE Technical_Report SHALL justify the Split_Protocol with reference to the repeated cross-sectional design of the GHS and to the repetition of PSU identifiers within a survey year.
3. WHERE a time-based split is used, THE Split_Protocol SHALL assign survey years 2012 to 2019 to training and validation and survey years 2021 and 2022 to test.
4. WHERE a time-based split is used, THE Technical_Report SHALL state that the SRD grant period falls only in the test partition and SHALL state the effect of that imbalance on generalisation claims.
5. WHERE a random or cross-validated split is used, THE Split_Protocol SHALL group records by PSU so that records sharing a PSU fall in the same partition.
6. THE Project_Team SHALL evaluate the test partition exactly once, at the step defined in Requirement 18.
7. IF any experiment requires a second test-partition evaluation, THEN THE Project_Team SHALL record the reason in the decision log and SHALL report the repeated evaluation in the Technical_Report.

### Requirement 10: Reproducible Preprocessing Pipeline (Q2.4, 5 marks)

**User Story:** As an assessor, I want preprocessing inside a single fitted pipeline, so that leakage is structurally prevented and the saved artefact matches what was evaluated.

#### Acceptance Criteria

1. THE Preprocessing_Pipeline SHALL perform cleaning, categorical encoding, numeric scaling, missing-value imputation, feature engineering and imbalance treatment as pipeline steps.
2. THE Preprocessing_Pipeline SHALL learn every statistic used for imputation, scaling and encoding from the training partition only.
3. THE Preprocessing_Pipeline SHALL exclude every leakage column listed in the Data_Audit from the input feature set.
4. THE Preprocessing_Pipeline SHALL serialise to a single file that the User_Interface loads without re-fitting.
5. WHERE household-context features are used, THE Preprocessing_Pipeline SHALL derive the features by aggregating person records within the combination of survey year and `uqnr`.
6. THE Technical_Report SHALL report the measured +0.039 AUC contribution of household-context features in the children stratum, from 0.824 to 0.863, as the justification for including the features.
7. WHEN an unseen categorical level is presented at prediction time, THE Preprocessing_Pipeline SHALL encode the unseen level without raising an error.
8. FOR ALL records, applying the Preprocessing_Pipeline twice to the same input SHALL produce identical transformed output (idempotence property).

### Requirement 11: Naive Baseline (Q3.1, 2 marks)

**User Story:** As an assessor, I want a transparent naive baseline, so that substantive model performance is interpretable against a floor.

#### Acceptance Criteria

1. THE Baseline_Model SHALL be a naive classifier using a majority-class or stratified strategy with a fixed random seed.
2. THE Baseline_Model SHALL be evaluated on the same held-out partition and with the same metric set as every model in the Model_Suite.
3. THE Technical_Report SHALL state what the Baseline_Model establishes and SHALL state the baseline value of the primary technical metric.
4. THE Technical_Report SHALL exclude the Baseline_Model from the count of four substantive models.

### Requirement 12: Substantive Model Suite (Q3.2 and Q3.3, 13 marks)

**User Story:** As an assessor, I want at least four substantive models from the required families trained on identical splits, so that the comparison is fair and the brief's model-set rule is met.

#### Acceptance Criteria

1. THE Model_Suite SHALL contain at least four substantive models in addition to the Baseline_Model.
2. THE Model_Suite SHALL contain at least two traditional models selected from logistic regression, decision tree, random forest, support vector machine, naive Bayes and gradient boosting.
3. THE Model_Suite SHALL contain at least two neural models, comprising a multi-layer perceptron and one further task-appropriate architecture for tabular data.
4. FOR ALL models in the Model_Suite, each model SHALL be trained on the identical training partition and evaluated on the identical held-out partition defined by the Split_Protocol.
5. WHEN a neural model is trained, THE Project_Team SHALL use the framework and version recorded under Requirement 2.
6. IF the recorded neural framework fails to install in the verified environment, THEN THE Project_Team SHALL record the failure in the decision log and SHALL select an alternative framework before training continues.

### Requirement 13: Training Documentation and Model Rationale (Q3.4 and Q3.5, 7 marks)

**User Story:** As an assessor, I want full training detail per model, so that every result can be reproduced from the report alone.

#### Acceptance Criteria

1. FOR ALL models in the Model_Suite, THE Technical_Report SHALL document architecture or hyperparameters, loss function, optimiser, epoch count, batch size, regularisation, early-stopping rule and random seed.
2. THE Technical_Report SHALL document the compute environment, including Python 3.14.3, pandas 2.3.3, scikit-learn 1.8.0, matplotlib 3.10.7 and the selected neural framework version.
3. THE Technical_Report SHALL state a rationale of at least two sentences linking each model family to the tabular, mixed-type, class-imbalanced character of the data.
4. WHEN a model is trained, THE training script SHALL set every random seed it controls to a value recorded in the Technical_Report.
5. FOR ALL models in the Model_Suite, retraining with the recorded seed and configuration SHALL reproduce the reported primary metric within a documented tolerance (reproducibility property).

### Requirement 14: Held-out Metric Set (Q4.1, 4 marks)

**User Story:** As an assessor, I want the same metric set for every model, so that the comparison is valid.

#### Acceptance Criteria

1. FOR ALL models in the Model_Suite and for the Baseline_Model, THE Evaluation_Report SHALL report accuracy and macro-F1 on the same held-out partition.
2. THE Evaluation_Report SHALL report precision, recall, weighted-F1 and ROC-AUC for every model.
3. WHERE the target class prior is imbalanced as defined in Requirement 4, THE Evaluation_Report SHALL report PR-AUC for every model.
4. WHERE PR-AUC is omitted, THE Technical_Report SHALL state the justification for the omission.
5. THE Evaluation_Report SHALL state, for each reported metric, whether `person_wgt` was applied, consistent with the decision recorded under Requirement 2.

### Requirement 15: Per-model Diagnostic Evidence (Q4.2, 5 marks)

**User Story:** As an assessor, I want per-model reports, matrices and a shared ROC figure, so that class-level behaviour is visible rather than hidden behind accuracy.

#### Acceptance Criteria

1. FOR ALL models in the Model_Suite, THE Evaluation_Report SHALL contain a classification report showing per-class precision, recall, F1 and support with named class labels.
2. FOR ALL models in the Model_Suite, THE Evaluation_Report SHALL contain a confusion matrix with labelled axes and absolute counts.
3. THE Evaluation_Report SHALL contain one figure plotting the ROC curve of every model on shared axes.
4. WHERE the target has more than two classes, THE ROC figure SHALL present one-versus-rest macro and micro curves.
5. THE Technical_Report SHALL interpret each confusion matrix in terms of the false-positive and false-negative costs stated in Requirement 4.

### Requirement 16: Model Comparison and Champion Selection (Q4.3, 3 marks)

**User Story:** As an assessor, I want the champion chosen on validation evidence, so that the final test result remains unbiased.

#### Acceptance Criteria

1. THE Evaluation_Report SHALL contain a comparison table listing every model against the full metric set.
2. THE Evaluation_Report SHALL contain a bar chart comparing the primary technical metric across every model and the Baseline_Model.
3. WHERE cross-validation is feasible, THE Evaluation_Report SHALL report the cross-validated mean and a variability measure or confidence interval per model.
4. THE Project_Team SHALL select the provisional champion using validation performance.
5. THE Technical_Report SHALL name the provisional champion, the strongest comparator and the validation values that determined the selection.

### Requirement 17: Overfitting Diagnosis (Q4.4, 4 marks)

**User Story:** As an assessor, I want overfitting quantified and controlled, so that reported generalisation is believable.

#### Acceptance Criteria

1. FOR ALL models in the Model_Suite, THE Evaluation_Report SHALL report the training and validation values of the primary technical metric and the numeric gap between the two values.
2. THE Evaluation_Report SHALL contain a learning curve for every model in the Model_Suite.
3. FOR ALL neural models, THE Evaluation_Report SHALL contain training and validation loss curves and training and validation metric curves per epoch.
4. THE Technical_Report SHALL diagnose each train-validation gap and SHALL name the control applied, selected from early stopping, dropout, weight decay, pruning, depth limits or data augmentation.
5. IF a train-validation gap exceeds a threshold recorded in the Technical_Report, THEN THE Project_Team SHALL apply at least one additional control and SHALL report the post-control gap.

### Requirement 18: Champion Optimisation (Q5, 10 marks)

**User Story:** As an assessor, I want exactly one model optimised under a documented protocol and a single locked test evaluation, so that the final claim is honest.

#### Acceptance Criteria

1. THE Optimisation_Study SHALL apply hyperparameter search to the single strongest validation model only.
2. THE Optimisation_Study SHALL document the search strategy, the search space with per-parameter ranges, the scoring rule and the number of trials.
3. THE Optimisation_Study SHALL score candidate configurations using cross-validation or the validation partition.
4. THE Optimisation_Study SHALL report pre-optimisation and post-optimisation validation results side by side and SHALL report the selected hyperparameter configuration.
5. WHEN the optimised configuration is selected, THE Project_Team SHALL lock the pipeline and SHALL evaluate the locked pipeline on the test partition exactly once.
6. THE Technical_Report SHALL state whether the measured improvement is practically meaningful for the named decision-maker and SHALL justify the statement with the service-delivery success indicator.
7. THE Optimisation_Study SHALL use the validation partition rather than the test partition for every tuning decision.

### Requirement 19: Ablation Study (Q6.1, 5 marks)

**User Story:** As an assessor, I want controlled ablations, so that the source of the system's performance is identified rather than asserted.

#### Acceptance Criteria

1. THE Ablation_Study SHALL contain at least three experiments, each altering exactly one component relative to the locked champion pipeline.
2. FOR ALL experiments in the Ablation_Study, the split, the random seed and the evaluation protocol SHALL remain identical to the locked champion configuration.
3. THE Ablation_Study SHALL include one experiment removing the household-context feature group.
4. THE Ablation_Study SHALL include one experiment removing or altering the imbalance treatment or a regularisation component.
5. THE Ablation_Study SHALL report the signed change in the primary technical metric for every experiment.
6. THE Technical_Report SHALL interpret each reported change in terms of what the system depends on, and SHALL compare the measured changes against the established leave-one-out values of `medaid` -0.069, `prov` -0.014, `race` -0.012 and `geotype` -0.006.

### Requirement 20: McNemar Validation (Q6.2, 5 marks)

**User Story:** As an assessor, I want a paired statistical test on identical held-out cases, so that the champion's advantage is shown to be more than noise.

#### Acceptance Criteria

1. THE McNemar_Test SHALL compare the optimised champion against the strongest comparator using predictions on identical held-out cases.
2. THE Evaluation_Report SHALL present the 2x2 correctness table with the counts of both-correct, champion-only-correct, comparator-only-correct and both-wrong.
3. THE Technical_Report SHALL state the null hypothesis, the alternative hypothesis and the significance level alpha = 0.05.
4. THE Technical_Report SHALL report the test statistic and the exact p-value.
5. WHERE the sum of the two discordant counts is below 25, THE McNemar_Test SHALL use the exact binomial form and THE Technical_Report SHALL state the use of the exact form.
6. THE Technical_Report SHALL distinguish the statistical conclusion from the practical significance of the measured difference.
7. THE Repository SHALL contain the paired prediction records used by the McNemar_Test.

### Requirement 21: User Interface Deployment (Q7, 10 marks)

**User Story:** As a public-service stakeholder, I want a working interface over the saved pipeline, so that I can obtain and interpret a prediction without running code myself.

#### Acceptance Criteria

1. THE User_Interface SHALL be built with Gradio, Streamlit, Flask or FastAPI with a web front end.
2. THE User_Interface SHALL load the saved end-to-end Preprocessing_Pipeline and champion model from disk and SHALL apply the loaded artefacts to every prediction.
3. THE User_Interface SHALL accept both manually entered input and uploaded file input.
4. WHEN a user submits valid input, THE User_Interface SHALL display the predicted class, the predicted probability or confidence, and a plain-language explanation of the prediction.
5. IF submitted input is missing a required field, contains an out-of-domain categorical value, or contains a numeric value outside the documented valid range, THEN THE User_Interface SHALL display a message naming the offending field and the accepted values, and SHALL retain the remaining user input.
6. THE User_Interface SHALL display per-feature guidance covering the accepted value domain of every input field.
7. THE User_Interface SHALL display a responsible-use notice stating that output supports population-level prioritisation rather than individual benefit adjudication.
8. THE User_Interface SHALL read configuration from environment variables or a committed non-secret configuration file, and SHALL present no personal records from the source data.
9. THE Repository SHALL contain deployment instructions and either a stable demonstration link or a single documented launch command.
10. THE Technical_Report SHALL demonstrate at least three valid input cases and at least one invalid input case with the resulting interface output.

### Requirement 22: Stakeholder Recommendation (Q8.1, 2 marks)

**User Story:** As a public-service stakeholder, I want a cautious, actionable recommendation, so that I know what to do and when not to trust the model.

#### Acceptance Criteria

1. THE Technical_Report SHALL state a recommendation addressed to the named decision-maker, expressed as a specific action on a specific population.
2. THE Technical_Report SHALL state the limitations and the uncertainty attached to the recommendation, including the structural breaks and the reduced 2020 and 2021 samples.
3. THE Technical_Report SHALL state the fairness findings per population group measured under Requirement 5.
4. THE Technical_Report SHALL state the monitoring needs, including the metric to track, the review interval and the drift signal that triggers retraining.
5. THE Technical_Report SHALL state at least three explicit conditions under which the model is unsuitable for use.

### Requirement 23: Reproducible Repository (Q8.2, 3 marks)

**User Story:** As an assessor, I want the package to run from documented instructions, so that reproducibility is verifiable rather than claimed.

#### Acceptance Criteria

1. THE Repository SHALL contain a README stating the problem, the execution order of scripts, the launch command for the User_Interface and the source links for the data.
2. THE Repository SHALL contain an environment or requirements file pinning every dependency version.
3. THE Repository SHALL contain the Data_Dictionary, the saved Preprocessing_Pipeline, the saved champion model and the User_Interface source code.
4. FOR ALL scripts in the Repository, each script SHALL set its random seeds from a single recorded configuration value.
5. THE Repository SHALL resolve data paths from a configurable root rather than from a hard-coded absolute local path.
6. WHEN the documented execution order is followed in a clean environment, THE Repository SHALL produce the reported figures and metric tables without manual intervention.
7. WHERE the data licence forbids redistribution, THE Repository SHALL omit the source data files and SHALL contain acquisition instructions instead.

### Requirement 24: Presentation and Contribution Statement (Q8.3, 2 marks)

**User Story:** As an assessor, I want a defended presentation and evidence of shared contribution, so that individual marks can be moderated fairly.

#### Acceptance Criteria

1. THE Project_Team SHALL deliver a presentation of 10 to 12 minutes including a live or recorded demonstration of the User_Interface.
2. THE Project_Team SHALL ensure that every team member presents a portion of the presentation.
3. THE Project_Team SHALL submit a one-page contribution statement signed by every member, listing each member's role and specific outputs.
4. THE Technical_Report SHALL label every adapted external code fragment and every artefact produced with artificial-intelligence assistance in accordance with institutional policy.
5. WHEN the presentation is delivered, THE Project_Team SHALL defend the target definition, the Split_Protocol, the leakage exclusions and the champion selection.
