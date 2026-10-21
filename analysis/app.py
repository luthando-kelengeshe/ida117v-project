"""
Stakeholder-facing interface: Child Support Grant reach-gap prioritiser.
Brief Q7.  Run with:  streamlit run app.py

Loads the SAVED preprocessor, champion model and isotonic calibrator. Nothing refits.
Positive class = an eligible child who is NOT receiving a grant.
"""
import pandas as pd
import streamlit as st

import ghs_inference as gi
import ghs_schema as S

st.set_page_config(page_title="CSG reach-gap prioritiser", page_icon="🇿🇦",
                   layout="wide")
TASK = "csg_reach_gap"
NEA = "Not economically active"


@st.cache_resource(show_spinner="Loading saved model pipeline ...")
def load_champion(which=None):
    return gi.Champion(TASK, which=which)


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Model")
LABELS = {gi.CHAMPION: f"(final optimised champion) {gi.CHAMPION}"}
CHOICES = [LABELS.get(m, m) for m in gi.MODEL_CHOICES]
pick = st.sidebar.selectbox(
    "Scoring model", CHOICES, index=0,
    help="The optimised champion is the Q5 random-search winner, retrained on all "
         "training years and locked. The pre-optimisation models are kept for the "
         "comparison the brief requires. McNemar found no significant difference "
         "between the champion and Random Forest (p=0.82 on the outreach rule).")
which = {v: k for k, v in LABELS.items()}.get(pick, pick)

try:
    champ = load_champion(which)
except FileNotFoundError as e:
    st.error(f"Model artefacts not found.\n\n{e}")
    st.stop()

m = champ.metrics()
base = champ.base_rate()
if m:
    st.sidebar.metric("Validation ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
    st.sidebar.metric("Validation PR-AUC", f"{m.get('pr_auc', float('nan')):.3f}")
if base:
    st.sidebar.metric("Population reach-gap rate", f"{base*100:.1f}%")
st.sidebar.caption(
    f"Loaded: **{champ.name}**  \n"
    f"Features: **{champ.n_features()}**  \n"
    f"Calibrated: **{'yes (isotonic)' if champ.calibrator is not None else 'no'}**  \n"
    "Trained 2012-2018 · validated 2019-2020 · tested once on 2021-2022")
if champ.name == gi.CHAMPION:
    st.sidebar.success(
        "Serving the **final optimised** pipeline: random-search winner "
        "(512-256-128, dropout 0.1, pos_weight 3.0), 19 features after dropping the "
        "five that are constant inside the eligible population.")
st.sidebar.divider()
st.sidebar.subheader("Who this model covers")
st.sidebar.info(S.ELIGIBILITY_RULE)
st.sidebar.subheader("Responsible use")
st.sidebar.warning(S.RESPONSIBLE_USE)
st.sidebar.subheader("Fairness")
st.sidebar.warning(S.FAIRNESS_NOTICE)

# ----------------------------------------------------------------- header
st.title("Child Support Grant reach-gap prioritiser")
st.caption("Ranks eligible children by how likely they are to be **missing** a grant, "
           "so limited outreach capacity goes where it finds the most unreached "
           "children. Statistics South Africa General Household Survey 2012-2022.")
if base:
    st.info(f"**How to read a score.** {base*100:.1f}% of eligible children in this "
            f"group are not receiving a grant. A score well above that marks a "
            f"household worth visiting. Targeting the top 5% by score reaches about "
            f"**3x more** missed children than random outreach.")

tab1, tab2, tab3 = st.tabs(["Single household", "Batch upload (CSV)",
                            "Input guide & limitations"])

# ================================================================= TAB 1
with tab1:
    st.subheader("1 · The child")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 0, 17, 7, 1,
                              help=S.PERSON_FIELDS["age"]["help"])
        sex = st.selectbox("Sex", S.SEXES, help=S.PERSON_FIELDS["sex"]["help"])
        race = st.selectbox("Population group", S.RACES,
                            help=S.PERSON_FIELDS["race"]["help"])
    with c2:
        prov = st.selectbox("Province", S.PROVINCES, index=4,
                            help=S.PERSON_FIELDS["prov"]["help"])
        geotype = st.selectbox("Settlement type", S.GEOTYPES,
                               help=S.PERSON_FIELDS["geotype"]["help"])
        educ = st.selectbox("Highest education of the child", S.EDUC_BANDS, index=1,
                            help=S.PERSON_FIELDS["educ_band"]["help"])
    with c3:
        st.text_input("Child covered by medical aid?", value="No", disabled=True,
                      help="Fixed to 'No'. Households with medical aid are outside "
                           "the model's eligibility scope.")
        medaid = "No"
        disab = st.selectbox("Disability status", S.DISAB,
                             help=S.PERSON_FIELDS["disab"]["help"])
        attend = (st.selectbox("Currently attending school?", S.YESNO,
                               help=S.PERSON_FIELDS["attend"]["help"])
                  if gi.applicable("attend", age) else None)
        if attend is None:
            st.caption("_Attendance is not asked below age 5, so it is left blank "
                       "to match the survey._")
        lab = (st.selectbox("Labour-market status", [NEA, "Unemployed"], index=0,
                            help="Employed is unavailable: the model only covers "
                                 "households with no employed member.")
               if gi.applicable("lab_status", age) else None)
        if lab is None:
            st.caption("_Labour status is only recorded from age 15, so it is left "
                       "blank to match the survey._")

    st.subheader("2 · Other household members")
    st.caption("The model uses 14 household-level features. List the other people in "
               "the household and they are derived exactly as during training. "
               "Employed members and medical aid put the household out of scope.")
    default_roster = pd.DataFrame([
        {"age": 29, "sex": "Female", "educ_band": "Some secondary",
         "lab_status": "Unemployed", "medaid": "No", "disab": "Not disabled"},
    ])
    roster = st.data_editor(
        default_roster, num_rows="dynamic", use_container_width=True, key="roster",
        column_config={
            "age": st.column_config.NumberColumn("Age", min_value=0, max_value=110, step=1),
            "sex": st.column_config.SelectboxColumn("Sex", options=S.SEXES),
            "educ_band": st.column_config.SelectboxColumn("Education", options=S.EDUC_BANDS),
            "lab_status": st.column_config.SelectboxColumn(
                "Labour status", options=[NEA, "Unemployed", "Employed"]),
            "medaid": st.column_config.SelectboxColumn("Medical aid", options=S.YESNO),
            "disab": st.column_config.SelectboxColumn("Disability", options=S.DISAB),
        })

    if st.button("Score outreach priority", type="primary"):
        person = {"age": age, "sex": sex, "race": race, "prov": prov,
                  "geotype": geotype, "educ_band": educ, "attend": attend,
                  "medaid": medaid, "disab": disab, "lab_status": lab}
        try:
            members = [dict(r, salary=None) for r in roster.to_dict("records")
                       if r.get("age") is not None and str(r.get("age")) != "nan"]
            df = gi.build_row(person, members)
            prob, deltas = gi.explain(champ, df)
        except gi.InputError as e:
            st.error(f"**Cannot score this household.**\n\n{e}")
        except Exception as e:
            st.error(f"Unexpected error: {type(e).__name__}: {e}")
        else:
            band, action, lift = gi.risk_band(prob, base)
            pct = champ.percentile(prob)
            a, b = st.columns([1, 2])
            with a:
                st.metric("Risk band", band, delta=f"{lift:.1f}x population rate")
                st.metric("Probability of missing a grant", f"{prob*100:.1f}%")
                if pct is not None:
                    st.metric("Outreach priority", f"top {100-pct:.1f}%")
                st.progress(float(min(max(prob, 0.0), 1.0)))
                raw = float(champ.predict_proba(df, calibrated=False)[0])
                st.caption(f"Uncalibrated output was {raw*100:.1f}%. Calibration "
                           "corrects the overstatement caused by class-rebalanced "
                           "training.")
            with b:
                st.markdown(gi.plain_language(prob, deltas, base_rate=base, pct=pct))
            with st.expander("Show the 24 features sent to the model"):
                st.dataframe(df.T.rename(columns={0: "value"}),
                             use_container_width=True)

# ================================================================= TAB 2
with tab2:
    st.subheader("Rank many children from a CSV")
    needed = list(S.PERSON_FIELDS)
    st.caption("Required columns: " + ", ".join(f"`{c}`" for c in needed))
    st.caption("Optional columns describing the caregiver let each row represent a "
               "real household: `caregiver_age`, `caregiver_sex`, "
               "`caregiver_educ_band`, `caregiver_lab_status`, `caregiver_disab`. "
               "Without `caregiver_age` a row is treated as a single-person "
               "household, which is rejected for children under 15.")
    template = pd.DataFrame([
        {"age": 1, "sex": "Female", "race": "African/Black", "prov": "Mpumalanga",
         "geotype": "Traditional", "educ_band": "No schooling", "attend": "",
         "medaid": "No", "disab": "Not disabled", "lab_status": "",
         "caregiver_age": 23, "caregiver_sex": "Female",
         "caregiver_educ_band": "Some secondary",
         "caregiver_lab_status": "Unemployed", "caregiver_disab": "Not disabled"},
        {"age": 16, "sex": "Male", "race": "African/Black", "prov": "Gauteng",
         "geotype": "Urban", "educ_band": "Some secondary", "attend": "Yes",
         "medaid": "No", "disab": "Not disabled", "lab_status": NEA,
         "caregiver_age": 45, "caregiver_sex": "Female",
         "caregiver_educ_band": "Matric (Grade 12)",
         "caregiver_lab_status": "Unemployed", "caregiver_disab": "Not disabled"}])
    st.download_button("Download a template CSV",
                       template.to_csv(index=False).encode("utf-8"),
                       file_name="csg_reachgap_template.csv", mime="text/csv")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            frame = pd.read_csv(up)
        except Exception as e:
            st.error(f"That file could not be read as CSV: {e}")
        else:
            st.write(f"Received {len(frame):,} rows.")
            try:
                res, errs = gi.predict_batch(champ, frame)
            except gi.InputError as e:
                st.error(f"**Cannot score this file.**\n\n{e}")
            else:
                if errs:
                    st.warning(f"{len(errs)} row(s) skipped (invalid or out of scope).")
                    st.dataframe(pd.DataFrame(errs), use_container_width=True)
                if len(res):
                    cols = [c for c in needed if c in res.columns] + [
                        "probability_%", "lift_vs_population", "risk_band",
                        "outreach_percentile"]
                    st.success(f"Ranked {len(res):,} row(s), highest priority first.")
                    st.dataframe(res[cols], use_container_width=True)
                    st.download_button(
                        "Download ranked list",
                        res[cols].to_csv(index=False).encode("utf-8"),
                        file_name="csg_reachgap_ranked.csv", mime="text/csv")

# ================================================================= TAB 3
with tab3:
    st.subheader("Who this model covers")
    st.info(S.ELIGIBILITY_RULE)
    st.subheader("What each input means")
    st.dataframe(pd.DataFrame([
        {"field": f,
         "accepted values": (f"{d['min']}-{d['max']} (whole number)"
                             if d["type"] == "int" else ", ".join(d["levels"])),
         "guidance": d["help"]}
        for f, d in S.PERSON_FIELDS.items()]), use_container_width=True,
        hide_index=True)
    st.subheader("Risk bands")
    st.dataframe(pd.DataFrame(
        [{"band": n, "trigger": f"at least {t:.2f}x the population rate",
          "suggested action": a} for t, n, a in gi.BANDS],
    ), use_container_width=True, hide_index=True)
    st.subheader("Known limitations")
    for l in S.LIMITATIONS:
        st.markdown(f"- {l}")
    st.subheader("Responsible use")
    st.info(S.RESPONSIBLE_USE)
    st.subheader("Data source")
    st.markdown("Statistics South Africa, General Household Survey person files "
                "2012-2022 (773,311 person records; 97,390 in the eligible-by-proxy "
                "child population). Microdata is distributed de-identified and no "
                "personal records are bundled with this interface.")

