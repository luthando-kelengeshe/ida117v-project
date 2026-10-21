"""
Step 12: Full EDA for the social-grant prediction task.
Covers brief Q2.1 (data dictionary + audit) and Q2.2 (>=5 purposeful visualisations).
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ghs_harmonise as gh
import ghs_dataset as gd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 80)
OUT = gd.OUT_DIR
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)

print("Building analysis panel ...")
panel = gd.build_panel()
print(f"panel: {len(panel):,} x {panel.shape[1]}\n")

NUM = gd.PERSON_NUM + gd.HH_NUM
CAT = gd.PERSON_CAT
FEATS = NUM + CAT

# =============================================================== Q2.1 DICTIONARY
print("=" * 112); print("Q2.1a  DATA DICTIONARY"); print("=" * 112)
rows = []
for c in FEATS + ["grant_any", "weight", "uqnr", "psu", "stratum"]:
    s = panel[c]
    kind = "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
    if c in gd.HH_NUM: role, src = "engineered (household)", "derived: aggregate over (year, uqnr)"
    elif c == "lab_status": role, src = "engineered (person)", "derived: wage/bus/vol/ret/look/sto"
    elif c == "educ_band": role, src = "engineered (person)", "derived: education / Q15HIEDU / Q16hiedu"
    elif c == "grant_any": role, src = "TARGET", "soc_grant / Q31asg / Q132asg"
    elif c == "weight": role, src = "survey design", "person_wgt"
    elif c in ("uqnr", "psu", "stratum"): role, src = "survey design", c
    else: role, src = "raw (harmonised)", ", ".join(gh.ALIASES.get(c, [c]))
    dom = (f"[{np.nanmin(s):.2f}, {np.nanmax(s):.2f}]" if kind == "numeric"
           else " | ".join(map(str, sorted(s.dropna().unique())[:6])))
    rows.append({"variable": c, "role": role, "type": kind,
                 "source_columns": src, "domain_or_levels": dom[:88],
                 "non_null_pct": round(s.notna().mean() * 100, 1),
                 "n_unique": int(s.nunique())})
dd = pd.DataFrame(rows)
print(dd.to_string(index=False))
dd.to_csv(os.path.join(OUT, "data_dictionary.csv"), index=False)

# =============================================================== Q2.1 AUDIT
print("\n" + "=" * 112); print("Q2.1b  AUDIT: class balance"); print("=" * 112)
bal = []
for name, d in [("whole population", panel.dropna(subset=["grant_any"])),
                ("children 0-17", panel[panel["age"].between(0, 17)].dropna(subset=["grant_any"])),
                ("working age 18-59", panel[panel["age"].between(18, 59)].dropna(subset=["grant_any"])),
                ("older 60+", panel[panel["age"] >= 60].dropna(subset=["grant_any"]))]:
    bal.append({"population": name, "n": len(d),
                "pct_grant_Yes": round(d["grant_any"].mean() * 100, 2),
                "pct_grant_No": round((1 - d["grant_any"].mean()) * 100, 2),
                "imbalance_ratio": round(max(d["grant_any"].mean(), 1 - d["grant_any"].mean())
                                         / min(d["grant_any"].mean(), 1 - d["grant_any"].mean()), 2)})
balance = pd.DataFrame(bal); print(balance.to_string(index=False))
balance.to_csv(os.path.join(OUT, "audit_class_balance.csv"), index=False)

print("\n--- duplicates ---")
key = panel["year"].astype(str) + "|" + panel["uqnr"].astype(str) + "|" + panel["personnr"].astype(str)
print(f"exact duplicate person-keys (year|uqnr|personnr): {int(key.duplicated().sum())}")
print(f"fully duplicated feature rows: {int(panel[FEATS].duplicated().sum()):,} "
      f"({panel[FEATS].duplicated().mean()*100:.1f}%)  <- expected: features are coarse, not identifying")

print("\n--- target missingness (records with no usable grant answer) ---")
print(f"{int(panel['grant_any'].isna().sum()):,} of {len(panel):,} "
      f"({panel['grant_any'].isna().mean()*100:.2f}%) dropped from modelling")

print("\n--- missingness by feature and split ---")
miss = panel.groupby("split")[FEATS].apply(lambda d: (d.isna().mean() * 100).round(1))
print(miss.to_string())
miss.to_csv(os.path.join(OUT, "audit_missingness_by_split.csv"))

print("\n--- invalid values / outlier treatment ---")
print(f"age outside [0,110] set to NaN by harmoniser; remaining age range "
      f"[{panel['age'].min():.0f}, {panel['age'].max():.0f}]")
print(f"salary sentinel 888888888 and values outside (0, 5e6) dropped; "
      f"remaining non-null salary {panel['salary'].notna().sum():,} "
      f"({panel['salary'].notna().mean()*100:.1f}%)")
sal = panel["salary"].dropna()
print(f"salary percentiles  p50={sal.quantile(.5):,.0f}  p95={sal.quantile(.95):,.0f}  "
      f"p99={sal.quantile(.99):,.0f}  max={sal.max():,.0f}")
print(f"hh_salary_per_capita retained as engineered feature (p99="
      f"{panel['hh_salary_per_capita'].quantile(.99):,.0f})")

print("\n--- LEAKAGE: columns excluded by construction ---")
for c in gd.LEAKAGE_COLUMNS[:6]:
    print(f"  excluded: {c}")
print(f"  ... {len(gd.LEAKAGE_COLUMNS)} leakage columns total; none are loaded as features.")
print("  Evidence: grant-type follow-ups are ~100% populated among recipients and ~0% among")
print("  non-recipients; including them lifts held-out AUC from 0.906 to 0.993 (script 09).")

print("\n--- survey weights ---")
w = panel.groupby("year")["weight"].sum() / 1e6
print("summed person_wgt by year (millions):"); print(w.round(2).to_string())

# =============================================================== Q2.2 FIGURES
print("\n" + "=" * 112); print("Q2.2  VISUALISATIONS -> " + FIG); print("=" * 112)
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .3})
D = panel.dropna(subset=["grant_any"])

# F1 target distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
v = D["grant_any"].value_counts().sort_index()
ax[0].bar(["No grant (0)", "Grant (1)"], v.values, color=["#4878a8", "#c0392b"])
for i, n in enumerate(v.values):
    ax[0].text(i, n, f"{n:,}\n{n/len(D)*100:.1f}%", ha="center", va="bottom", fontsize=9)
ax[0].set_title("Target: social-grant receipt, whole population", fontweight="bold")
ax[0].set_ylabel("persons (count)"); ax[0].set_ylim(0, v.max() * 1.18)
st = balance.set_index("population").loc[["children 0-17", "working age 18-59", "older 60+"], "pct_grant_Yes"]
ax[1].barh(st.index, st.values, color="#c0392b")
for i, x in enumerate(st.values):
    ax[1].text(x + 1, i, f"{x:.1f}%", va="center", fontsize=9)
ax[1].set_title("Grant receipt by eligibility stratum", fontweight="bold")
ax[1].set_xlabel("% receiving a grant (unweighted)"); ax[1].set_xlim(0, 100)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F1_target_distribution.png"), bbox_inches="tight"); plt.close()

# F2 missingness
fig, ax = plt.subplots(figsize=(13, 5.2))
m = panel.groupby("year")[FEATS].apply(lambda d: d.isna().mean() * 100).T
im = ax.imshow(m.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns, rotation=0)
ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index, fontsize=8)
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        if m.values[i, j] > 1:
            ax.text(j, i, f"{m.values[i,j]:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if m.values[i, j] > 55 else "black")
ax.set_title("Missingness by feature and survey year (% of records)", fontweight="bold")
ax.grid(False); plt.colorbar(im, label="% missing")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F2_missingness.png"), bbox_inches="tight"); plt.close()

# F3 feature-target: age
fig, ax = plt.subplots(figsize=(12, 4.6))
Da = D.dropna(subset=["age"])
ag = Da.groupby(Da["age"].astype(int))["grant_any"].agg(["mean", "size"])
ag = ag[ag["size"] >= 80]
ax.plot(ag.index, ag["mean"] * 100, lw=2, color="#c0392b")
ax.axvline(18, ls="--", c="grey"); ax.axvline(60, ls="--", c="grey")
ax.annotate("CSG ends at 18", xy=(18, 70), xytext=(24, 82), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="grey"))
ax.annotate("Old-age grant starts at 60\n13.4% -> 51.3%", xy=(60, 51), xytext=(38, 20), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="grey"))
ax.set_title("Feature-target relationship: grant receipt by single year of age", fontweight="bold")
ax.set_xlabel("age (years)"); ax.set_ylabel("% receiving a grant"); ax.set_ylim(0, 100)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F3_feature_target_age.png"), bbox_inches="tight"); plt.close()

# F4 feature-target: medical aid + education
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
for a, col, ttl in [(ax[0], "medaid", "Medical aid cover"), (ax[1], "educ_band", "Highest education")]:
    d = D.dropna(subset=[col])
    order = gh.EDUC_ORDER if col == "educ_band" else ["Yes", "No"]
    r = d.groupby(col)["grant_any"].mean().reindex(order) * 100
    a.bar(range(len(r)), r.values, color="#4878a8")
    a.set_xticks(range(len(r))); a.set_xticklabels(r.index, rotation=20, ha="right", fontsize=8)
    for i, x in enumerate(r.values):
        a.text(i, x + 1, f"{x:.1f}%", ha="center", fontsize=8)
    a.set_title(f"Grant receipt by {ttl}", fontweight="bold"); a.set_ylabel("% receiving a grant")
    a.set_ylim(0, max(r.values) * 1.2)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F4_feature_target_medaid_educ.png"), bbox_inches="tight"); plt.close()

# F5 geographic
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
pv = (D[D["grant_any"].eq(1)].groupby("prov")["weight"].sum()
      / D.groupby("prov")["weight"].sum() * 100).sort_values()
ax[0].barh(pv.index, pv.values, color="#16a085")
for i, x in enumerate(pv.values): ax[0].text(x + .4, i, f"{x:.1f}%", va="center", fontsize=8)
ax[0].set_title("Grant receipt by province (survey-weighted)", fontweight="bold")
ax[0].set_xlabel("% of persons receiving a grant"); ax[0].set_xlim(0, pv.max() * 1.18)
gt = (D[D["grant_any"].eq(1)].groupby("geotype")["weight"].sum()
      / D.groupby("geotype")["weight"].sum() * 100).sort_values()
ax[1].barh(gt.index, gt.values, color="#8e44ad")
for i, x in enumerate(gt.values): ax[1].text(x + .4, i, f"{x:.1f}%", va="center", fontsize=8)
ax[1].set_title("Grant receipt by settlement type (survey-weighted)", fontweight="bold")
ax[1].set_xlabel("% of persons receiving a grant"); ax[1].set_xlim(0, gt.max() * 1.2)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F5_geographic.png"), bbox_inches="tight"); plt.close()

# F6 temporal + split
fig, ax = plt.subplots(figsize=(12, 4.6))
yr = (D[D["grant_any"].eq(1)].groupby("year")["weight"].sum()
      / D.groupby("year")["weight"].sum() * 100)
ax.plot(yr.index, yr.values, "o-", lw=2.5, color="#c0392b")
ax.axvspan(2011.6, 2018.4, color="#4878a8", alpha=.12)
ax.axvspan(2018.6, 2020.4, color="#f39c12", alpha=.16)
ax.axvspan(2020.6, 2022.4, color="#16a085", alpha=.14)
ax.text(2015, yr.min() - 1.2, "TRAIN 2012-2018", ha="center", fontsize=9, color="#2c3e50")
ax.text(2019.5, yr.min() - 1.2, "VAL 2019-20", ha="center", fontsize=9, color="#9a6700")
ax.text(2021.5, yr.min() - 1.2, "TEST 2021-22", ha="center", fontsize=9, color="#0e6655")
for x in (2020, 2021):
    ax.annotate("halved sample\n(phone survey)", xy=(x, yr.loc[x]), xytext=(x - .5, yr.loc[x] + 2.6),
                fontsize=7.5, ha="center", arrowprops=dict(arrowstyle="->", color="grey", lw=.8))
ax.set_title("Temporal view: grant receipt by survey year, with the time-based split",
             fontweight="bold")
ax.set_ylabel("% of persons receiving a grant (weighted)"); ax.set_xlabel("survey year")
ax.set_ylim(yr.min() - 2.4, yr.max() + 4.4)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F6_temporal_split.png"), bbox_inches="tight"); plt.close()

# F7 subgroup
fig, ax = plt.subplots(figsize=(11.5, 4.6))
sub = (D[D["grant_any"].eq(1)].groupby(["race", "geotype"])["weight"].sum()
       / D.groupby(["race", "geotype"])["weight"].sum() * 100).unstack()
sub.plot(kind="bar", ax=ax, width=.78, color=["#8c6d31", "#cc4c02", "#1f6f3f"])
ax.set_title("Subgroup view: grant receipt by population group and settlement type",
             fontweight="bold")
ax.set_ylabel("% receiving a grant (weighted)"); ax.set_xlabel("")
ax.tick_params(axis="x", rotation=12); ax.legend(title="settlement", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F7_subgroup.png"), bbox_inches="tight"); plt.close()

print("wrote F1..F7")
for f in sorted(os.listdir(FIG)): print("   ", f)

panel.to_parquet(os.path.join(OUT, "analysis_panel.parquet"), index=False)
print(f"\npanel cached -> {os.path.join(OUT, 'analysis_panel.parquet')}")

