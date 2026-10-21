"""
Step 16: Acceptance evidence for brief Q7.4 on the PRIMARY task (csg_reach_gap).
Four valid cases and five invalid/out-of-scope cases, run against the same saved
artefacts the interface loads. Writes out/ui_acceptance_tests.csv.
"""
import os, sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghs_inference as gi
import ghs_schema as S

OUT = gi.OUT
champ = gi.Champion("csg_reach_gap")   # defaults to the final optimised champion
print(f"Loaded saved artefacts: {champ.name} | calibrator: {champ.calibrator is not None}")
print(f"Positive class = {S.TARGET_LABELS[1]}\n")

NEA = "Not economically active"
VALID = [
 ("V1 Infant, traditional area, no earner (expected HIGH risk of being missed)",
  dict(age=1, sex="Female", race="African/Black", prov="Mpumalanga",
       geotype="Traditional", educ_band="No schooling", attend=None, medaid="No",
       disab="Not disabled", lab_status=None),
  [dict(age=23, sex="Female", educ_band="Some secondary", lab_status="Unemployed",
        medaid="No", disab="Not disabled", salary=None)]),
 ("V2 Primary-school child, traditional area, no earner (expected LOW risk)",
  dict(age=8, sex="Male", race="African/Black", prov="Eastern Cape",
       geotype="Traditional", educ_band="Primary or less", attend="Yes", medaid="No",
       disab="Not disabled", lab_status=None),
  [dict(age=36, sex="Female", educ_band="Primary or less", lab_status=NEA,
        medaid="No", disab="Not disabled", salary=None),
   dict(age=61, sex="Female", educ_band="No schooling", lab_status=NEA,
        medaid="No", disab="Disabled", salary=None)]),
 ("V3 Late teen, urban, no earner (expected ELEVATED risk)",
  dict(age=17, sex="Female", race="Coloured", prov="Gauteng", geotype="Urban",
       educ_band="Some secondary", attend="Yes", medaid="No", disab="Not disabled",
       lab_status=NEA),
  [dict(age=45, sex="Female", educ_band="Matric (Grade 12)", lab_status="Unemployed",
        medaid="No", disab="Not disabled", salary=None)]),
 ("V4 Single-person edge case: lone child, empty roster",
  dict(age=5, sex="Male", race="African/Black", prov="KwaZulu-Natal",
       geotype="Rural formal", educ_band="Primary or less", attend="Yes", medaid="No",
       disab="Not disabled", lab_status=None),
  []),
]

INVALID = [
 ("I1 Age outside the 0-17 scope",
  dict(age=35, sex="Female", race="White", prov="Gauteng", geotype="Urban",
       educ_band="Post-school", attend="No", medaid="No", disab="Not disabled",
       lab_status="Unemployed"), []),
 ("I2 Province not a recognised level",
  dict(age=9, sex="Male", race="Coloured", prov="Cape Town", geotype="Urban",
       educ_band="Primary or less", attend="Yes", medaid="No", disab="Not disabled",
       lab_status=None), []),
 ("I3 Required field missing (medical aid)",
  dict(age=9, sex="Male", race="Coloured", prov="Western Cape", geotype="Urban",
       educ_band="Primary or less", attend="Yes", medaid=None, disab="Not disabled",
       lab_status=None), []),
 ("I4 OUT OF SCOPE: household has an employed member",
  dict(age=9, sex="Male", race="African/Black", prov="Limpopo", geotype="Traditional",
       educ_band="Primary or less", attend="Yes", medaid="No", disab="Not disabled",
       lab_status=None),
  [dict(age=40, sex="Male", educ_band="Matric (Grade 12)", lab_status="Employed",
        medaid="No", disab="Not disabled", salary=9000.0)]),
 ("I5 OUT OF SCOPE: child covered by medical aid",
  dict(age=9, sex="Male", race="White", prov="Gauteng", geotype="Urban",
       educ_band="Primary or less", attend="Yes", medaid="Yes", disab="Not disabled",
       lab_status=None), []),
]

rows = []
print("=" * 108); print("VALID CASES (in scope -> must score)"); print("=" * 108)
for label, person, roster in VALID:
    df = gi.build_row(person, roster)
    raw = float(champ.predict_proba(df, calibrated=False)[0])
    prob = float(champ.predict_proba(df)[0])
    br = champ.base_rate()
    band, action, lift = gi.risk_band(prob, br)
    pct = champ.percentile(prob)
    print(f"\n{label}")
    print(f"   raw={raw*100:5.1f}%  calibrated={prob*100:5.1f}%  "
          f"lift={lift:4.1f}x  band={band:<16} top {100-pct:4.1f}%")
    cls = band
    _, deltas = gi.explain(champ, df, top=3)
    for f, dd in deltas:
        print(f"      {gi.FRIENDLY.get(f,f):<42} {dd*100:+6.1f} pp")
    rows.append({"case": label, "outcome": "scored", "raw_%": round(raw*100,1),
                 "calibrated_%": round(prob*100,1), "prediction": cls})

print("\n" + "=" * 108)
print("INVALID / OUT-OF-SCOPE CASES (must be refused with a useful message)")
print("=" * 108)
fails = 0
for label, person, roster in INVALID:
    try:
        gi.build_row(person, roster)
        print(f"\n{label}\n   NOT REJECTED  <-- FAIL")
        fails += 1
        rows.append({"case": label, "outcome": "NOT REJECTED (FAIL)", "raw_%": None,
                     "calibrated_%": None, "prediction": None})
    except gi.InputError as e:
        print(f"\n{label}\n   refused: {e}")
        rows.append({"case": label, "outcome": f"refused: {e}", "raw_%": None,
                     "calibrated_%": None, "prediction": None})

t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "ui_acceptance_tests.csv"), index=False)
print("\n" + "=" * 108)
print(f"valid scored: {(t['outcome']=='scored').sum()}   "
      f"invalid refused: {t['outcome'].astype(str).str.startswith('refused').sum()}   "
      f"failures: {fails}")
print(f"wrote {os.path.join(OUT,'ui_acceptance_tests.csv')}")
sys.exit(1 if fails else 0)


