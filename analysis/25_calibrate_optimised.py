"""
Step 25: Calibrate the OPTIMISED champion, then redo McNemar fairly.

Why: script 24 compared a calibrated Random Forest against an UNCALIBRATED optimised
MLP at a 0.5 threshold. That produced a 6.9 pp gap at p~1e-107, which measured the
missing calibration stage, not model quality. The rank-based operational test was
unaffected (p=0.676) because monotone calibration cannot change a ranking.

Fix: fit isotonic calibration for the optimised champion using the SAME protocol as
every other model (fit on validation year 2019, check on 2020), then rerun both tests.
"""
import os, sys
import numpy as np, pandas as pd, joblib
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
sys.stdout.reconfigure(encoding="utf-8")
import ghs_dataset as gd

OUT = "out"; MOD = os.path.join(OUT, "models"); TASK = "csg_reach_gap"
panel = pd.read_parquet(os.path.join(OUT, "analysis_panel.parquet"))
d, _, _ = gd.get_task(panel, TASK)
val_years = d[d["split"] == "val"]["year"].values

opt = np.load(os.path.join(OUT, f"optimised_probs_{TASK}.npz"))
old = np.load(os.path.join(OUT, f"val_test_probs_{TASK}.npz"))
cals = joblib.load(os.path.join(MOD, f"calibrators_{TASK}.joblib"))["calibrators"]
yv, yt, wt = opt["y_val"], opt["y_test"], opt["w_test"]
fit_m, chk_m = val_years == 2019, val_years == 2020

iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit(opt["val"][fit_m], yv[fit_m])
raw_c, cal_c = opt["val"][chk_m], iso.predict(opt["val"][chk_m]); yc = yv[chk_m]
print("CALIBRATION OF THE OPTIMISED CHAMPION (fit 2019, checked on 2020)")
print("-" * 86)
print(f"  mean predicted raw        {raw_c.mean():.4f}")
print(f"  mean predicted calibrated {cal_c.mean():.4f}")
print(f"  observed rate             {yc.mean():.4f}")
print(f"  bias   raw {raw_c.mean()-yc.mean():+.4f}  ->  calibrated {cal_c.mean()-yc.mean():+.4f}")
print(f"  Brier  raw {brier_score_loss(yc, raw_c):.4f}  ->  calibrated {brier_score_loss(yc, cal_c):.4f}")
print(f"  ROC-AUC unchanged by monotone calibration: "
      f"{roc_auc_score(yc, raw_c):.4f} -> {roc_auc_score(yc, cal_c):.4f}")

cals["Optimised MLP"] = iso
joblib.dump({"calibrators": cals, "fit_year": 2019, "check_year": 2020,
             "champion": "Optimised MLP"},
            os.path.join(MOD, f"calibrators_{TASK}.joblib"))
print(f"\n  saved calibrator for 'Optimised MLP' into calibrators_{TASK}.joblib")

p_ch = np.clip(iso.predict(opt["test"]), 0, 1)
p_co = np.clip(cals["Random Forest"].predict(old["Random Forest|test"]), 0, 1)

def mcnemar(ca, cb, na, nb, ctx):
    both = int((ca & cb).sum()); a = int((ca & ~cb).sum())
    b = int((~ca & cb).sum()); nei = int((~ca & ~cb).sum()); disc = a + b
    print("\n" + "=" * 96); print(ctx); print("=" * 96)
    print(f"  {'':<26}{nb+' correct':>22}{nb+' wrong':>20}")
    print(f"  {na+' correct':<26}{both:>22,}{a:>20,}")
    print(f"  {na+' wrong':<26}{b:>22,}{nei:>20,}")
    print(f"\n  discordant pairs = {disc:,}   ({na} only {a:,} | {nb} only {b:,})")
    print("  H0: equal error rates | H1: error rates differ | alpha = 0.05")
    pe = stats.binomtest(a, disc, 0.5).pvalue if disc else 1.0
    if disc < 25:
        print(f"  discordant < 25 -> EXACT binomial. statistic={a} p={pe:.6g}")
        stat, p = a, pe
    else:
        stat = (abs(a - b) - 1) ** 2 / disc
        print(f"  chi2 = {stat:.4f} (df=1)  p = {stats.chi2.sf(stat,1):.6g}")
        print(f"  exact binomial (confirmatory) p = {pe:.6g}")
        p = pe
    aa, ab = float(ca.mean()), float(cb.mean())
    print(f"\n  accuracy {na} = {aa:.4f} | {nb} = {ab:.4f} | diff = {aa-ab:+.4f} "
          f"({(aa-ab)*100:+.2f} pp)")
    v = "REJECT H0" if p < .05 else "FAIL TO REJECT H0"
    print(f"  conclusion: {v} at alpha=0.05")
    return {"context": ctx, "both_correct": both, "champion_only": a,
            "comparator_only": b, "both_wrong": nei, "discordant": disc,
            "statistic": stat, "p_value": p, "acc_champion": aa,
            "acc_comparator": ab, "acc_diff": aa - ab, "conclusion": v}

rows = [mcnemar((p_ch >= .5) == (yt == 1), (p_co >= .5) == (yt == 1),
                "Optimised MLP", "Random Forest",
                "Q6.2a  LOCKED TEST, 0.5 threshold (BOTH models calibrated)")]
k = int(len(yt) * .10)
fa = np.zeros(len(yt), bool); fa[np.argsort(-p_ch)[:k]] = True
fb = np.zeros(len(yt), bool); fb[np.argsort(-p_co)[:k]] = True
rows.append(mcnemar(fa == (yt == 1), fb == (yt == 1), "Optimised MLP", "Random Forest",
                    "Q6.2b  LOCKED TEST, operational rule: flag top 10%"))

print("\n" + "=" * 96); print("PRACTICAL SIGNIFICANCE"); print("=" * 96)
base = float(np.average(yt, weights=wt))
for nm, p in [("Optimised MLP", p_ch), ("Random Forest", p_co)]:
    o = pd.DataFrame({"p": p, "y": yt, "w": wt}).sort_values("p", ascending=False)
    t10 = o.iloc[:k]
    hit = float(np.average(t10["y"], weights=t10["w"]))
    print(f"  {nm:<16} ROC-AUC {roc_auc_score(yt,p,sample_weight=wt):.4f}  "
          f"PR-AUC {average_precision_score(yt,p,sample_weight=wt):.4f}  "
          f"top-10% hit {hit:.4f} (lift {hit/base:.2f}x)  "
          f"found {t10.loc[t10.y.eq(1),'w'].sum()/1e3:,.0f}k children")
pd.DataFrame(rows).to_csv(os.path.join(OUT, f"mcnemar_optimised_{TASK}.csv"), index=False)
np.savez_compressed(os.path.join(OUT, f"optimised_calibrated_{TASK}.npz"),
                    val=np.clip(iso.predict(opt["val"]),0,1), test=p_ch,
                    y_val=yv, y_test=yt, w_test=wt, w_val=opt["w_val"])
print(f"\nwrote {OUT}/mcnemar_optimised_{TASK}.csv and optimised_calibrated_{TASK}.npz")
