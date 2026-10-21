"""
Step 24: Definitive McNemar for Q6.2 - the OPTIMISED champion against the strongest
comparator (Random Forest) on the identical locked test cases from Q5.3.
This supersedes the pre-optimisation McNemar in script 21.
"""
import os, sys
import numpy as np, pandas as pd, joblib
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
OUT = "out"; TASK = "csg_reach_gap"

opt = np.load(os.path.join(OUT, f"optimised_probs_{TASK}.npz"))
old = np.load(os.path.join(OUT, f"val_test_probs_{TASK}.npz"))
cals = joblib.load(os.path.join(OUT, "models", f"calibrators_{TASK}.joblib"))["calibrators"]

y = opt["y_test"]; w = opt["w_test"]
assert np.array_equal(y, old["y_test"]), "test rows must be identical"
p_ch = opt["test"]                                   # optimised champion
p_co = np.clip(cals["Random Forest"].predict(old["Random Forest|test"]), 0, 1)
print(f"n = {len(y):,} identical locked test cases (2021-2022), positive rate {y.mean():.4f}\n")

def mcnemar(ca, cb, na, nb, ctx):
    both = int((ca & cb).sum()); a = int((ca & ~cb).sum())
    b = int((~ca & cb).sum()); nei = int((~ca & ~cb).sum()); disc = a + b
    print("=" * 96); print(ctx); print("=" * 96)
    print(f"  {'':<26}{nb+' correct':>22}{nb+' wrong':>20}")
    print(f"  {na+' correct':<26}{both:>22,}{a:>20,}")
    print(f"  {na+' wrong':<26}{b:>22,}{nei:>20,}")
    print(f"\n  both correct {both:,} | {na} only {a:,} | {nb} only {b:,} | both wrong {nei:,}")
    print(f"  discordant pairs = {disc:,}")
    print("  H0: equal error rates (discordant pairs split 50/50)")
    print("  H1: error rates differ        alpha = 0.05")
    pe = stats.binomtest(a, disc, 0.5).pvalue if disc else 1.0
    if disc < 25:
        print(f"  discordant < 25 -> EXACT binomial.  statistic={a}  p={pe:.6g}")
        stat, p = a, pe
    else:
        stat = (abs(a - b) - 1) ** 2 / disc
        pc = float(stats.chi2.sf(stat, 1))
        print(f"  chi-square with continuity correction: chi2={stat:.4f} (df=1) p={pc:.6g}")
        print(f"  exact binomial (confirmatory): p={pe:.6g}")
        p = pe
    aa, ab = float(ca.mean()), float(cb.mean())
    print(f"\n  accuracy {na} = {aa:.4f} | {nb} = {ab:.4f} | difference = {aa-ab:+.4f} "
          f"({(aa-ab)*100:+.2f} pp)")
    v = "REJECT H0" if p < .05 else "FAIL TO REJECT H0"
    print(f"  conclusion: {v} at alpha=0.05\n")
    return {"context": ctx, "both_correct": both, "champion_only": a,
            "comparator_only": b, "both_wrong": nei, "discordant": disc,
            "statistic": stat, "p_value": p, "acc_champion": aa,
            "acc_comparator": ab, "acc_diff": aa - ab, "conclusion": v}

rows = []
rows.append(mcnemar((p_ch >= .5) == (y == 1), (p_co >= .5) == (y == 1),
                    "Optimised MLP", "Random Forest",
                    "Q6.2a  LOCKED TEST, 0.5 decision threshold"))
k = int(len(y) * .10)
fa = np.zeros(len(y), bool); fa[np.argsort(-p_ch)[:k]] = True
fb = np.zeros(len(y), bool); fb[np.argsort(-p_co)[:k]] = True
rows.append(mcnemar(fa == (y == 1), fb == (y == 1), "Optimised MLP", "Random Forest",
                    "Q6.2b  LOCKED TEST, operational rule: flag top 10% for outreach"))

print("=" * 96); print("PRACTICAL SIGNIFICANCE: what the difference means operationally")
print("=" * 96)
base = float(np.average(y, weights=w))
for nm, p in [("Optimised MLP", p_ch), ("Random Forest", p_co)]:
    o = pd.DataFrame({"p": p, "y": y, "w": w}).sort_values("p", ascending=False)
    t10 = o.iloc[:k]
    hit = float(np.average(t10["y"], weights=t10["w"]))
    found = t10.loc[t10.y.eq(1), "w"].sum()
    print(f"  {nm:<16} top-10% hit rate {hit:.4f} (lift {hit/base:.2f}x), "
          f"weighted children found {found/1e3:,.0f}k")
pd.DataFrame(rows).to_csv(os.path.join(OUT, f"mcnemar_optimised_{TASK}.csv"), index=False)
print(f"\nwrote {OUT}/mcnemar_optimised_{TASK}.csv")
