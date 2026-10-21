"""
Step 20: Regenerate the operational targeting evidence with RANDOM TIE-BREAKING.

Bug found: DummyClassifier assigns an identical probability to every record, so
sorting by score left ties in dataset order. That order correlates with survey year
and sample structure, which made a constant predictor appear to have lift. Breaking
ties at random restores the correct result: a constant predictor sits exactly on the
diagonal, which is what "no discriminative power" must look like.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)

OUT = "out"; FIG = os.path.join(OUT, "figures"); TASK = "csg_reach_gap"
SEED = 42
z = np.load(os.path.join(OUT, f"val_test_probs_{TASK}.npz"))
yv, wv = z["y_val"], z["w_val"]
cals = joblib.load(os.path.join(OUT, "models", f"calibrators_{TASK}.joblib"))["calibrators"]
MODELS = ["Baseline (Dummy)", "Logistic Regression", "Random Forest",
          "MLP (PyTorch)", "TabTransformer (PyTorch)"]
COLORS = dict(zip(MODELS, ["#7f8c8d", "#2980b9", "#16a085", "#c0392b", "#8e44ad"]))

rng = np.random.default_rng(SEED)
jitter = rng.permutation(len(yv))          # one shared random order for fair ties
base = float(np.average(yv, weights=wv))
print(f"validation base rate (random outreach hit rate) = {base:.4f}\n")

def ranked(p):
    o = pd.DataFrame({"p": p, "y": yv, "w": wv, "tie": jitter})
    return o.sort_values(["p", "tie"], ascending=[False, True])

rows = []
for m in MODELS:
    o = ranked(z[f"{m}|val"])
    tot = o.loc[o.y.eq(1), "w"].sum()
    rec = {"model": m}
    for frac in (0.05, 0.10, 0.20, 0.30):
        s = o.iloc[:max(int(len(o) * frac), 1)]
        hit = float(np.average(s["y"], weights=s["w"]))
        rec[f"prec@{int(frac*100)}%"] = round(hit, 4)
        rec[f"lift@{int(frac*100)}%"] = round(hit / base, 2)
        rec[f"recall@{int(frac*100)}%"] = round(s.loc[s.y.eq(1), "w"].sum() / tot, 4)
    rows.append(rec)
t = pd.DataFrame(rows).set_index("model")
print("OPERATIONAL TARGETING (validation, weighted, random tie-breaking)")
print(t.to_string())
t.to_csv(os.path.join(OUT, f"results_targeting_{TASK}.csv"))

fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.0))
for m in MODELS:
    o = ranked(z[f"{m}|val"])
    cw = o["w"].cumsum() / o["w"].sum()
    cg = (o["y"] * o["w"]).cumsum() / (o["y"] * o["w"]).sum()
    ax[0].plot(cw * 100, cg * 100, lw=2, color=COLORS[m], label=m)
    with np.errstate(divide="ignore", invalid="ignore"):
        ax[1].plot(cw * 100, cg / cw, lw=2, color=COLORS[m], label=m)
ax[0].plot([0, 100], [0, 100], "k--", lw=1, label="random outreach")
ax[0].set_xlabel("% of eligible population contacted (ranked by risk score)")
ax[0].set_ylabel("% of all reach-gap cases found")
ax[0].set_title("Cumulative gain: missed children found", fontweight="bold")
ax[0].legend(fontsize=8); ax[0].set_xlim(0, 100); ax[0].set_ylim(0, 100); ax[0].grid(alpha=.3)
ax[1].axhline(1, c="k", ls="--", lw=1, label="random outreach")
ax[1].set_xlabel("% of eligible population contacted (ranked by risk score)")
ax[1].set_ylabel("lift over random outreach")
ax[1].set_title("Lift: efficiency gain from targeting", fontweight="bold")
ax[1].legend(fontsize=8); ax[1].set_xlim(1, 100); ax[1].set_ylim(0, 4.2); ax[1].grid(alpha=.3)
fig.suptitle("Operational targeting - CSG reach gap, children 0-17 in eligible-by-proxy households",
             fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, .93])
plt.savefig(os.path.join(FIG, f"F13_targeting_{TASK}.png"), bbox_inches="tight")
print(f"\nrewrote {FIG}/F13_targeting_{TASK}.png")
