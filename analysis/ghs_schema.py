"""
Single source of truth for the model's input contract.
Used by the inference layer for validation and by the Streamlit app for guidance,
so the UI can never drift from what the model was trained on.
"""
PROVINCES = ["Western Cape", "Eastern Cape", "Northern Cape", "Free State",
             "KwaZulu-Natal", "North West", "Gauteng", "Mpumalanga", "Limpopo"]
RACES = ["African/Black", "Coloured", "Indian/Asian", "White"]
SEXES = ["Male", "Female"]
GEOTYPES = ["Urban", "Traditional", "Rural formal"]
EDUC_BANDS = ["No schooling", "Primary or less", "Some secondary",
              "Matric (Grade 12)", "Post-school"]
LAB_STATUS = ["Employed", "Unemployed", "Not economically active"]
YESNO = ["Yes", "No"]
DISAB = ["Not disabled", "Disabled"]

# focal-person fields the user supplies directly
PERSON_FIELDS = {
    "age":        {"type": "int", "min": 0, "max": 17,
                   "help": "Age in years. This model covers children 0-17 only "
                           "(Child Support Grant eligibility age range)."},
    "sex":        {"type": "cat", "levels": SEXES, "help": "Sex as recorded by the survey."},
    "race":       {"type": "cat", "levels": RACES,
                   "help": "Population group. Retained because it carries measurable "
                           "signal (-0.012 AUC when removed) but see the fairness notice."},
    "prov":       {"type": "cat", "levels": PROVINCES, "help": "Province of residence."},
    "geotype":    {"type": "cat", "levels": GEOTYPES,
                   "help": "Settlement type. 'Traditional' means a traditional authority area."},
    "educ_band":  {"type": "cat", "levels": EDUC_BANDS,
                   "help": "Highest education level completed by this child."},
    "attend":     {"type": "cat", "levels": YESNO,
                   "help": "Is the child currently attending an educational institution?"},
    "medaid":     {"type": "cat", "levels": YESNO,
                   "help": "Is this child covered by a medical aid scheme? Strongest single "
                           "predictor in the model (-0.069 AUC when removed)."},
    "disab":      {"type": "cat", "levels": DISAB,
                   "help": "Disability status derived from the six functioning domains."},
    "lab_status": {"type": "cat", "levels": LAB_STATUS,
                   "help": "Labour-market status. Normally 'Not economically active' for "
                           "children; defined for ages 15-64 only."},
}

# roster member fields (used to DERIVE the 14 household features)
MEMBER_FIELDS = ["age", "sex", "educ_band", "lab_status", "medaid", "disab", "salary"]

# Primary task is csg_reach_gap: the positive class is the REACH FAILURE we want to find.
TARGET_LABELS = {0: "Likely already reached", 1: "Likely missing a grant"}

ELIGIBILITY_RULE = (
    "This model applies only to households that are **near-certainly income-eligible** "
    "for the Child Support Grant: no employed member and nobody covered by a medical "
    "aid scheme. Restricting to this group is what makes a 'no grant' answer "
    "interpretable as a **reach failure** rather than as ineligibility. Within this "
    "group 85.7% of children do receive a grant, so the 14.3% who do not are the "
    "population of interest."
)

RESPONSIBLE_USE = (
    "This tool estimates the **likelihood that a Child Support Grant-eligible child is "
    "NOT receiving the grant**, from survey patterns in the Statistics South Africa "
    "General Household Survey 2012-2022. It exists to help **prioritise limited outreach "
    "capacity**: at 5% coverage, targeting by this score reaches roughly 3 times more "
    "missed children than random outreach.\n\n"
    "It must **not** be used to adjudicate, approve, reduce or deny any individual's "
    "grant, and it is **not** a SASSA eligibility determination. A high score means "
    "'worth visiting', never 'is entitled'. A low score is not evidence of fraud or of "
    "ineligibility. Final eligibility can only be established by SASSA through the "
    "statutory means test."
)

FAIRNESS_NOTICE = (
    "Population group and sex are inputs to this model. Sex contributes nothing measurable "
    "(0.000 AUC change). Population group contributes a small amount (-0.012 AUC) and is "
    "retained to let per-group performance be audited rather than hidden. Any deployment "
    "should monitor per-group error rates, because historical inequality in grant reach is "
    "present in the training data and a model fitted to it can reproduce that inequality."
)

LIMITATIONS = [
    "Trained on survey years 2012-2018, validated on 2019-2020, tested on 2021-2022.",
    "Applies only to children aged 0-17 in households with no employed member and no "
    "medical aid cover. Outside that group the model is not valid and will refuse to score.",
    "Eligibility is approximated, not measured. The Child Support Grant means test keys "
    "on caregiver income, which the survey does not record reliably (salary is about 20% "
    "populated and partly bracket-imputed). The proxy therefore leaks: within the "
    "restricted group, non-receipt reaches 54% among White children and 71% among those "
    "with matric, which is more likely genuine ineligibility from assets or unrecorded "
    "income than a reach failure.",
    "Discrimination is modest: ROC-AUC about 0.69 against a 0.50 floor. The value is in "
    "ranking for outreach (about 3x lift at 5% coverage), not in individual certainty.",
    "Some apparent non-receipt is survey misreporting rather than a true reach failure.",
    "The 2020 and 2021 samples were roughly halved and collected telephonically under "
    "COVID-19 lockdown, so those years are less reliable.",
    "The survey carries no district or municipality code, so the finest geography "
    "available is province crossed with settlement type.",
]

