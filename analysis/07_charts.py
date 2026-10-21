"""Step 7: Visual summary of the derived indicators."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

ind = pd.read_csv(os.path.join(OUT, "indicators_national.csv"), index_col="year")
att = pd.read_csv(os.path.join(OUT, "attainment_20plus.csv"), index_col="year")
med = pd.read_csv(os.path.join(OUT, "medaid_by_race.csv"), index_col="year")
unp = pd.read_csv(os.path.join(OUT, "unemployment_by_province.csv"), index_col="year")
geo = pd.read_csv(os.path.join(OUT, "matric_by_geotype.csv"), index_col="year")

fig, ax = plt.subplots(2, 3, figsize=(19, 10))
fig.suptitle("South Africa General Household Survey 2012-2022 — harmonised person-level findings\n"
             "773,311 records, survey-weighted", fontsize=15, fontweight="bold")

# 1. attainment composition
a = ax[0, 0]
att.plot(kind="area", stacked=True, ax=a, alpha=.85, linewidth=0,
         color=["#8c2d04", "#cc4c02", "#fe9929", "#41ab5d", "#1f6f3f"])
a.set_title("Educational attainment, adults 20+", fontweight="bold")
a.set_ylabel("% of adults 20+"); a.set_ylim(0, 100); a.set_xlabel("")
a.legend(fontsize=7, loc="lower left", ncol=2)

# 2. grants vs medical aid vs unemployment
a = ax[0, 1]
a.plot(ind.index, ind["social_grant"], "o-", lw=2.5, color="#c0392b", label="Receives a social grant")
a.plot(ind.index, ind["unemp_rate_strict"], "s-", lw=2.5, color="#2c3e50", label="Unemployment rate (15-64)")
a.plot(ind.index, ind["medical_aid"], "^-", lw=2.5, color="#2980b9", label="Covered by medical aid")
a.axvspan(2019.5, 2021.5, color="grey", alpha=.18)
a.annotate("COVID\n(phone survey)", xy=(2020.5, 0.06), xycoords=("data", "axes fraction"),
           ha="center", fontsize=8, color="#555")
a.set_title("Social protection, work and health cover", fontweight="bold")
a.set_ylabel("%"); a.legend(fontsize=8); a.grid(alpha=.3)

# 3. medical aid by population group
a = ax[0, 2]
for c, col in zip(med.columns, ["#b8860b", "#c0392b", "#8e44ad", "#16a085"]):
    a.plot(med.index, med[c], "o-", lw=2.2, label=c, color=col)
a.set_title("Medical aid coverage by population group", fontweight="bold")
a.set_ylabel("% covered"); a.set_ylim(0, 85); a.legend(fontsize=8); a.grid(alpha=.3)

# 4. unemployment by province
a = ax[1, 0]
for c in unp.columns:
    style = dict(lw=2.6, zorder=5) if c in ("Western Cape", "Mpumalanga") else dict(lw=1.2, alpha=.65)
    a.plot(unp.index, unp[c], "-", label=c, **style)
a.set_title("Unemployment rate by province", fontweight="bold")
a.set_ylabel("% of labour force"); a.legend(fontsize=6.5, ncol=2); a.grid(alpha=.3)

# 5. matric+ by geotype
a = ax[1, 1]
for c, col in zip(["Urban", "Traditional", "Rural formal"], ["#1f6f3f", "#cc4c02", "#8c6d31"]):
    if c in geo.columns:
        a.plot(geo.index, geo[c], "o-", lw=2.4, label=c, color=col)
a.set_title("Matric or higher, adults 20+, by settlement type", fontweight="bold")
a.set_ylabel("% with matric+"); a.set_ylim(0, 70); a.legend(fontsize=8); a.grid(alpha=.3)

# 6. sample size vs weighted population
a = ax[1, 2]
a.bar(ind.index, ind["n"] / 1000, color="#95a5a6", label="Sample (thousands of persons)")
a.set_ylabel("sample size (000s)"); a.set_xlabel("")
a2 = a.twinx()
a2.plot(ind.index, ind["pop_M"], "o-", color="#c0392b", lw=2.5, label="Weighted population (M)")
a2.set_ylabel("population (millions)", color="#c0392b")
a.set_title("Sample size vs weighted population\n(2020-21 samples roughly halved)", fontweight="bold")
a.legend(fontsize=8, loc="upper right"); a2.legend(fontsize=8, loc="lower right")

plt.tight_layout(rect=[0, 0, 1, 0.94])
p = os.path.join(OUT, "ghs_summary.png")
plt.savefig(p, dpi=135, bbox_inches="tight")
print("wrote", p)
