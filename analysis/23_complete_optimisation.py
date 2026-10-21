"""Completion step: write the optimisation JSON and figure F15 from saved logs,
so the 10-minute random search does not have to be repeated."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
OUT = "out"; FIG = os.path.join(OUT, "figures"); TASK = "csg_reach_gap"

tr = pd.read_csv(os.path.join(OUT, f"optuna_trials_{TASK}.csv"))
cvd = pd.read_csv(os.path.join(OUT, f"cv_leave_one_year_{TASK}.csv"))
pp = pd.read_csv(os.path.join(OUT, f"optimisation_prepost_{TASK}.csv"), index_col=0)
best = tr.sort_values("val_pr_auc", ascending=False).iloc[0]
pre_pr = float(pp.loc["pre-optimisation (locked champion)", "pr_auc"])

SPACE = {
 "hidden": ["(256, 128, 64)","(128, 64)","(256, 128)","(64, 32)","(512, 256, 128)","(128,)"],
 "p_drop": [0.0,0.1,0.2,0.3,0.4,0.5], "lr": [3e-4,5e-4,1e-3,2e-3],
 "wd": [0.0,1e-5,1e-4,1e-3], "bs": [256,512,1024,2048],
 "pos_weight": [None,1.0,2.0,3.0,5.494,8.0], "min_freq": [20,50,200],
 "use_sensitive": [True,False], "drop_constant": [True,False]}

meta = {
 "search": "random search", "n_trials": int(len(tr)), "seed": 42,
 "scoring_rule": "survey-weighted average precision (PR-AUC) on validation years 2019-2020",
 "early_stopping": "inner split - fit on 2012-2017, stop on 2018; validation used only for trial scoring",
 "test_policy": "test years 2021-2022 not used anywhere in the search; read once for the locked evaluation",
 "space": {k: [str(x) for x in v] for k, v in SPACE.items()},
 "selected": {k: (best[k].item() if hasattr(best[k], "item") else best[k])
              for k in SPACE},
 "selected_best_epoch": int(best["best_epoch"]),
 "val_pr_auc_pre": pre_pr,
 "val_pr_auc_post": float(best["val_pr_auc"]),
 "cv_leave_one_year": {
     "pr_auc_mean": float(cvd["pr_auc"].mean()), "pr_auc_sd": float(cvd["pr_auc"].std()),
     "roc_auc_mean": float(cvd["roc_auc"].mean()), "roc_auc_sd": float(cvd["roc_auc"].std()),
     "folds": int(len(cvd))},
}
with open(os.path.join(OUT, f"optimisation_{TASK}.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("wrote", f"{OUT}/optimisation_{TASK}.json")

srt = tr.sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(range(len(srt)), srt["val_pr_auc"], s=28, c="#2980b9", alpha=.85,
              label="random-search trials")
ax[0].axhline(pre_pr, c="#c0392b", ls="--", lw=1.8,
              label=f"pre-optimisation champion ({pre_pr:.4f})")
ax[0].axhline(srt["val_pr_auc"].max(), c="#16a085", lw=1.8,
              label=f"search winner ({srt['val_pr_auc'].max():.4f})")
ax[0].set_xlabel("trial (ranked best to worst)")
ax[0].set_ylabel("validation PR-AUC (survey-weighted)")
ax[0].set_title("Q5.1 Random search, 40 trials", fontweight="bold")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

ax[1].plot(cvd["held_out_year"], cvd["roc_auc"], "o-", lw=2, color="#8e44ad",
           ms=6, label="ROC-AUC")
ax[1].axhline(cvd["roc_auc"].mean(), c="#8e44ad", ls=":", lw=1.2,
              label=f"ROC mean {cvd['roc_auc'].mean():.3f} (sd {cvd['roc_auc'].std():.3f})")
ax2 = ax[1].twinx()
ax2.plot(cvd["held_out_year"], cvd["pr_auc"], "s--", lw=2, color="#e67e22",
         ms=6, label="PR-AUC")
ax2.set_ylabel("PR-AUC", color="#e67e22")
ax[1].set_xlabel("held-out survey year"); ax[1].set_ylabel("ROC-AUC", color="#8e44ad")
ax[1].set_title("Q4.3 Leave-one-year-out CV variability", fontweight="bold")
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left"); ax[1].grid(alpha=.3)
fig.suptitle(f"Champion optimisation and cross-validation - {TASK}", fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, .93])
plt.savefig(os.path.join(FIG, f"F15_optimisation_{TASK}.png"), bbox_inches="tight")
print("wrote", f"{FIG}/F15_optimisation_{TASK}.png")
