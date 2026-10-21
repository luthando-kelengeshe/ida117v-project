"""
Step 15: Probability calibration for the deployed interface.

Problem found: every model was trained with class rebalancing
(class_weight='balanced' / BCE pos_weight=0.454), so its sigmoid output is
calibrated to a 50/50 prior rather than the true ~74% prior. Ranking metrics
(ROC-AUC, PR-AUC) are unaffected, but the raw probability shown to a user is
understated by 16-18 percentage points, which would be actively misleading in
a stakeholder-facing tool.

Fix: fit an isotonic calibrator on validation predictions. Rebalanced training
is KEPT because it lifts minority-class recall (finding under-reached children
is the point), and calibration is applied afterwards as a separate stage.

Protocol: the calibrator is fitted on survey year 2019 and checked on 2020.
Both are validation years, so the 2021-2022 test set stays untouched.
"""
import json, os, sys
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); MOD = os.path.join(OUT, "models")
FIG = os.path.join(OUT, "figures")
TASK = os.environ.get("GHS_TASK", "csg_reach_gap")

import ghs_dataset as gd
panel = pd.read_parquet(os.path.join(OUT, "analysis_panel.parquet"))
d, _, _ = gd.get_task(panel, TASK)
val_years = d[d["split"] == "val"]["year"].values
d_probs = np.load(os.path.join(OUT, f"val_test_probs_{TASK}.npz"))
yv, yt = d_probs["y_val"], d_probs["y_test"]
fit_mask = val_years == 2019
chk_mask = val_years == 2020
print(f"calibration fit on 2019: n={fit_mask.sum():,} | check on 2020: n={chk_mask.sum():,}")

def ece(y, p, bins=10):
    """Expected calibration error."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -1e-9, 1 + 1e-9
    tot = 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0: continue
        tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return tot

MODELS = ["Logistic Regression", "Random Forest", "MLP (PyTorch)", "TabTransformer (PyTorch)"]
rows, calibrators = [], {}
for m in MODELS:
    pv = d_probs[m + "|val"]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pv[fit_mask], yv[fit_mask])
    calibrators[m] = iso
    raw, cal = pv[chk_mask], iso.predict(pv[chk_mask])
    y = yv[chk_mask]
    rows.append({
        "model": m,
        "mean_pred_raw": raw.mean(), "mean_pred_cal": cal.mean(), "observed": y.mean(),
        "bias_raw": raw.mean() - y.mean(), "bias_cal": cal.mean() - y.mean(),
        "brier_raw": brier_score_loss(y, raw), "brier_cal": brier_score_loss(y, cal),
        "ece_raw": ece(y, raw), "ece_cal": ece(y, cal),
        "auc_raw": roc_auc_score(y, raw), "auc_cal": roc_auc_score(y, cal)})

t = pd.DataFrame(rows).set_index("model")
print("\nCalibration check on validation year 2020 (held out from the calibrator fit)")
print(t.round(4).to_string())
t.round(5).to_csv(os.path.join(OUT, f"calibration_{TASK}.csv"))

with open(os.path.join(OUT, f"hyperparameters_{TASK}.json"), encoding="utf-8") as f:
    hp = json.load(f)
champ = hp["champion_by_val_auc"]
joblib.dump({"calibrators": calibrators, "fit_year": 2019, "check_year": 2020,
             "champion": champ},
            os.path.join(MOD, f"calibrators_{TASK}.joblib"))
print(f"\nsaved calibrators -> models/calibrators_{TASK}.joblib  (champion: {champ})")

# reliability diagram
fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.0))
for a, key, ttl in [(ax[0], "raw", "Before calibration"), (ax[1], "cal", "After isotonic calibration")]:
    for m in MODELS:
        pv = d_probs[m + "|val"]
        p = pv[chk_mask] if key == "raw" else calibrators[m].predict(pv[chk_mask])
        y = yv[chk_mask]
        q = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).agg(
            mp=("p", "mean"), obs=("y", "mean"))
        a.plot(g["mp"], g["obs"], "o-", ms=4, lw=1.6, label=m)
    a.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    a.set_xlabel("mean predicted probability"); a.set_ylabel("observed grant rate")
    a.set_title(ttl, fontweight="bold"); a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.grid(alpha=.3); a.legend(fontsize=7.5, loc="upper left")
fig.suptitle("Reliability diagram - validation year 2020 (task: " + TASK + ")",
             fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, .93])
plt.savefig(os.path.join(FIG, f"F12_calibration_{TASK}.png"), bbox_inches="tight")
print("wrote figures/F12_calibration_" + TASK + ".png")

