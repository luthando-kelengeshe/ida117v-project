"""
Step 18: Can we remove the eligibility/take-up conflation BY DESIGN?

Current problem: P(receives grant) mixes three things - eligibility, take-up, and
survey reporting. "High P but not receiving" is therefore a weak proxy for under-reach.

Idea: RESTRICT to children whose households are near-certainly income-eligible for
the Child Support Grant. Within that stratum eligibility is approximately constant,
so non-receipt is a REACH failure rather than ineligibility. Then flip the target so
the positive class IS the thing the stakeholder wants to find.
"""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

p = pd.read_parquet("out/analysis_panel.parquet")
k = p[p["age"].between(0, 17)].dropna(subset=["grant_any"]).copy()
print(f"all children 0-17: {len(k):,}   grant rate {k['grant_any'].mean():.3f}")

# candidate eligibility proxies, strictest last
k["no_earner"] = k["hh_no_earner"].eq(1)
k["no_medaid"] = k["hh_any_medaid"].eq(0)
k["low_pc"] = k["hh_salary_per_capita"].isna() | (k["hh_salary_per_capita"] < 1500)

defs = {
    "D0 all children (current design)": pd.Series(True, index=k.index),
    "D1 no medical aid in household": k["no_medaid"],
    "D2 no employed member": k["no_earner"],
    "D3 no earner AND no medical aid": k["no_earner"] & k["no_medaid"],
    "D4 no earner AND no medaid AND low income per capita":
        k["no_earner"] & k["no_medaid"] & k["low_pc"],
}
print()
print("ELIGIBILITY-PROXY RESTRICTIONS")
print("-" * 104)
print(f"{'definition':<56}{'n':>9}{'% of kids':>11}{'grant rate':>12}{'gap rate':>10}")
rows = []
for name, m in defs.items():
    d = k[m]
    gr = d["grant_any"].mean()
    wgap = d.loc[d["grant_any"].eq(0), "weight"].sum()
    print(f"{name:<56}{len(d):>9,}{m.mean()*100:>10.1f}%{gr:>12.3f}{1-gr:>10.3f}")
    rows.append({"definition": name, "n": len(d), "grant_rate": gr,
                 "gap_rate": 1 - gr, "weighted_gap_children": wgap})
res = pd.DataFrame(rows)
print()
print("Weighted national size of the REACH GAP (eligible-by-proxy but no grant):")
for _, r in res.iterrows():
    print(f"   {r['definition']:<56} {r['weighted_gap_children']/1e6:>6.2f} million children")

# does the gap still vary predictably inside the strictest stratum?
print()
print("=" * 104)
print("Within D3 (no earner AND no medical aid): does non-receipt still vary by group?")
print("=" * 104)
d3 = k[defs["D3 no earner AND no medical aid"]].copy()
d3["gap"] = 1 - d3["grant_any"]
for by in ["prov", "geotype", "race", "educ_band"]:
    g = (d3[d3["gap"].eq(1)].groupby(by)["weight"].sum()
         / d3.groupby(by)["weight"].sum() * 100).round(1).sort_values(ascending=False)
    print(f"\n-- non-receipt rate by {by} (weighted %) --")
    print(g.to_string())

print()
print("=" * 104)
print("Age profile of the gap inside D3 (is it concentrated anywhere?)")
print("=" * 104)
ag = d3.dropna(subset=["age"]).copy()
ag["band"] = pd.cut(ag["age"], [-1, 2, 5, 9, 13, 17],
                    labels=["0-2", "3-5", "6-9", "10-13", "14-17"])
g = (ag[ag["gap"].eq(1)].groupby("band", observed=True)["weight"].sum()
     / ag.groupby("band", observed=True)["weight"].sum() * 100).round(1)
print(g.to_string())

print()
print("Split sizes for D3 under the existing time-based split:")
print(d3.groupby("split").agg(n=("gap", "size"), gap_rate=("gap", "mean")).round(3).to_string())
res.to_csv("out/eligibility_restrictions.csv", index=False)
print("\nwrote out/eligibility_restrictions.csv")
