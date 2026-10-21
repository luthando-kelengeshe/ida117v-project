"""
Step 22: Brief Q5 - champion optimisation, plus the cross-validation variability
that Q4.3 asked for.

Q5.1  Random search over a documented space, applied ONLY to the strongest
      validation model (MLP: val ROC-AUC 0.6850 vs Random Forest 0.6840).
      Scoring rule: survey-weighted PR-AUC on the validation years 2019-2020.
      PR-AUC is used because the positive class is ~11% and ranking quality at the
      top of the list is what the outreach decision depends on.

Q5.2  Early stopping uses an INNER split (train on 2012-2017, stop on 2018) so the
      validation years are used only for trial scoring, never for stopping. The test
      years 2021-2022 are not touched anywhere in the search.

Q5.3  The winning configuration is locked and evaluated ONCE on the test set.

Also: leave-one-year-out CV over the training years for the winning config, which
supplies the mean and variability Q4.3 requires.
"""
import json, os, random, sys, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             accuracy_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import ghs_dataset as gd
from ghs_models import MLP

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 60)
SEED = 42
TASK = "csg_reach_gap"
OUT = gd.OUT_DIR; FIG = os.path.join(OUT, "figures"); MOD = os.path.join(OUT, "models")
N_TRIALS = 40

panel = pd.read_parquet(os.path.join(OUT, "analysis_panel.parquet"))
d, _, (NUM_ALL, CAT_ALL) = gd.get_task(panel, TASK)
DEAD = ["hh_n_employed", "hh_workers_per_capita", "hh_no_earner", "hh_any_medaid", "medaid"]
SENSITIVE = ["race", "sex"]

tr_full = d[d["split"] == "train"]
inner_tr = tr_full[tr_full["year"] <= 2017]
inner_va = tr_full[tr_full["year"] == 2018]
va = d[d["split"] == "val"]
te = d[d["split"] == "test"]
print(f"inner-train 2012-2017: {len(inner_tr):,} | inner-val 2018: {len(inner_va):,}")
print(f"selection val 2019-2020: {len(va):,} | locked test 2021-2022: {len(te):,}")
print(f"positive rate - train {tr_full['y'].mean():.4f}  val {va['y'].mean():.4f}  "
      f"test {te['y'].mean():.4f}\n")


def features(use_sensitive, drop_constant):
    num = [c for c in NUM_ALL if not (drop_constant and c in DEAD)]
    cat = [c for c in CAT_ALL if not (drop_constant and c in DEAD)]
    if not use_sensitive:
        cat = [c for c in cat if c not in SENSITIVE]
    return num, cat


def make_pre(num, cat, min_freq):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=min_freq,
                                               sparse_output=False))]), cat)])


def fit_mlp(cfg, fit_df, stop_df, score_dfs):
    """Train with early stopping on stop_df; return predictions for each score_df."""
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    num, cat = features(cfg["use_sensitive"], cfg["drop_constant"])
    pre = make_pre(num, cat, cfg["min_freq"]); pre.fit(fit_df[num + cat])
    A_fit = torch.tensor(pre.transform(fit_df[num + cat]), dtype=torch.float32)
    A_stop = torch.tensor(pre.transform(stop_df[num + cat]), dtype=torch.float32)
    y_fit = torch.tensor(fit_df["y"].values, dtype=torch.float32)
    y_stop = stop_df["y"].values
    model = MLP(A_fit.shape[1], hidden=tuple(cfg["hidden"]), p_drop=cfg["p_drop"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    pw = torch.tensor([cfg["pos_weight"]]) if cfg["pos_weight"] else None
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(y_fit); best = (-1, None, 0)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); perm = torch.randperm(n, generator=g)
        for i in range(0, n, cfg["bs"]):
            idx = perm[i:i + cfg["bs"]]
            if len(idx) < 2: continue
            opt.zero_grad()
            lossf(model(A_fit[idx], None), y_fit[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            ps = torch.sigmoid(model(A_stop, None)).numpy()
        sc = average_precision_score(y_stop, ps, sample_weight=stop_df["weight"].values)
        if sc > best[0]:
            best = (sc, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if ep - best[2] >= cfg["patience"]: break
    model.load_state_dict(best[1]); model.eval()
    outs = []
    for sdf in score_dfs:
        A = torch.tensor(pre.transform(sdf[num + cat]), dtype=torch.float32)
        with torch.no_grad():
            outs.append(torch.sigmoid(model(A, None)).numpy())
    return outs, best[2], model, pre, (num, cat)


SPACE = {
    "hidden": [(256, 128, 64), (128, 64), (256, 128), (64, 32), (512, 256, 128), (128,)],
    "p_drop": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "lr": [3e-4, 5e-4, 1e-3, 2e-3],
    "wd": [0.0, 1e-5, 1e-4, 1e-3],
    "bs": [256, 512, 1024, 2048],
    "pos_weight": [None, 1.0, 2.0, 3.0, 5.494, 8.0],
    "min_freq": [20, 50, 200],
    "use_sensitive": [True, False],
    "drop_constant": [True, False],
}
BASELINE_CFG = {"hidden": (256, 128, 64), "p_drop": 0.3, "lr": 1e-3, "wd": 1e-4,
                "bs": 1024, "pos_weight": 5.494, "min_freq": 50,
                "use_sensitive": True, "drop_constant": False,
                "epochs": 40, "patience": 6}

print("=" * 120)
print(f"Q5.1  RANDOM SEARCH - {N_TRIALS} trials, scoring = survey-weighted PR-AUC on 2019-2020")
print("=" * 120)
for k, v in SPACE.items():
    print(f"  {k:<16} {v}")
print()

rng = random.Random(SEED)
trials = []
t0 = time.time()
for t in range(N_TRIALS):
    cfg = {k: rng.choice(v) for k, v in SPACE.items()}
    cfg.update(epochs=40, patience=6)
    (pv,), ep, *_ = fit_mlp(cfg, inner_tr, inner_va, [va])
    ap = average_precision_score(va["y"], pv, sample_weight=va["weight"])
    auc = roc_auc_score(va["y"], pv, sample_weight=va["weight"])
    trials.append({**{k: (str(cfg[k]) if k == "hidden" else cfg[k]) for k in SPACE},
                   "best_epoch": ep, "val_pr_auc": ap, "val_roc_auc": auc})
    if (t + 1) % 8 == 0 or t == 0:
        print(f"  trial {t+1:>3}/{N_TRIALS}  PR-AUC={ap:.4f}  ROC={auc:.4f}  "
              f"({time.time()-t0:.0f}s elapsed)")

tr_df = pd.DataFrame(trials).sort_values("val_pr_auc", ascending=False)
tr_df.to_csv(os.path.join(OUT, f"optuna_trials_{TASK}.csv"), index=False)
print(f"\nsearch finished in {time.time()-t0:.0f}s")
print("\nTop 8 configurations by validation PR-AUC")
print(tr_df.head(8).to_string(index=False))

best = tr_df.iloc[0]
BEST_CFG = {k: (eval(best[k]) if k == "hidden" else best[k]) for k in SPACE}
BEST_CFG.update(epochs=40, patience=6)
print("\nSELECTED CONFIGURATION")
for k, v in BEST_CFG.items():
    print(f"  {k:<16} {v}")

# ---------------------------------------------------- Q4.3 leave-one-year-out CV
print("\n" + "=" * 120)
print("Q4.3  LEAVE-ONE-YEAR-OUT CROSS-VALIDATION of the selected config (training years)")
print("=" * 120)
cv = []
for yr in sorted(tr_full["year"].unique()):
    fit = tr_full[tr_full["year"] != yr]
    hold = tr_full[tr_full["year"] == yr]
    stop = fit[fit["year"] == fit["year"].max()]
    core = fit[fit["year"] != fit["year"].max()]
    (ph,), ep, *_ = fit_mlp(BEST_CFG, core, stop, [hold])
    cv.append({"held_out_year": yr, "n": len(hold),
               "pr_auc": average_precision_score(hold["y"], ph, sample_weight=hold["weight"]),
               "roc_auc": roc_auc_score(hold["y"], ph, sample_weight=hold["weight"])})
cvd = pd.DataFrame(cv)
print(cvd.round(4).to_string(index=False))
print(f"\n  PR-AUC  mean={cvd['pr_auc'].mean():.4f}  sd={cvd['pr_auc'].std():.4f}  "
      f"95% CI approx [{cvd['pr_auc'].mean()-1.96*cvd['pr_auc'].sem():.4f}, "
      f"{cvd['pr_auc'].mean()+1.96*cvd['pr_auc'].sem():.4f}]")
print(f"  ROC-AUC mean={cvd['roc_auc'].mean():.4f}  sd={cvd['roc_auc'].std():.4f}  "
      f"95% CI approx [{cvd['roc_auc'].mean()-1.96*cvd['roc_auc'].sem():.4f}, "
      f"{cvd['roc_auc'].mean()+1.96*cvd['roc_auc'].sem():.4f}]")
cvd.round(5).to_csv(os.path.join(OUT, f"cv_leave_one_year_{TASK}.csv"), index=False)

# ---------------------------------------------------- Q5.2 pre vs post
print("\n" + "=" * 120)
print("Q5.2  PRE- vs POST-OPTIMISATION (validation 2019-2020, survey-weighted)")
print("=" * 120)

def full_metrics(y, p, w, label):
    o = pd.DataFrame({"p": p, "y": y, "w": w}).sort_values("p", ascending=False)
    base = float(np.average(y, weights=w))
    t10 = o.iloc[:int(len(o) * .10)]
    t5 = o.iloc[:int(len(o) * .05)]
    yh = (p >= .5).astype(int)
    return {"config": label,
            "roc_auc": roc_auc_score(y, p, sample_weight=w),
            "pr_auc": average_precision_score(y, p, sample_weight=w),
            "macro_f1": f1_score(y, yh, average="macro", sample_weight=w),
            "accuracy": accuracy_score(y, yh, sample_weight=w),
            "lift@5%": float(np.average(t5["y"], weights=t5["w"])) / base,
            "lift@10%": float(np.average(t10["y"], weights=t10["w"])) / base}

(pv_base,), ep_b, *_ = fit_mlp(BASELINE_CFG, inner_tr, inner_va, [va])
(pv_best,), ep_o, *_ = fit_mlp(BEST_CFG, inner_tr, inner_va, [va])
cmp = pd.DataFrame([
    full_metrics(va["y"].values, pv_base, va["weight"].values, "pre-optimisation (locked champion)"),
    full_metrics(va["y"].values, pv_best, va["weight"].values, "post-optimisation (search winner)"),
]).set_index("config")
cmp.loc["change"] = cmp.iloc[1] - cmp.iloc[0]
print(cmp.round(4).to_string())
cmp.round(5).to_csv(os.path.join(OUT, f"optimisation_prepost_{TASK}.csv"))

# ---------------------------------------------------- Q5.3 single locked test eval
print("\n" + "=" * 120)
print("Q5.3  FINAL LOCKED EVALUATION - retrained on all training years, test read ONCE")
print("=" * 120)
(pv_fin, pt_fin), ep_f, model_f, pre_f, (num_f, cat_f) = fit_mlp(
    BEST_CFG, tr_full[tr_full["year"] <= 2017], tr_full[tr_full["year"] == 2018], [va, te])
final = pd.DataFrame([
    full_metrics(va["y"].values, pv_fin, va["weight"].values, "validation 2019-2020"),
    full_metrics(te["y"].values, pt_fin, te["weight"].values, "TEST 2021-2022 (locked, once)"),
]).set_index("config")
print(final.round(4).to_string())
final.round(5).to_csv(os.path.join(OUT, f"final_locked_{TASK}.csv"))
print(f"\n  features used: {len(num_f)} numeric + {len(cat_f)} categorical")
print(f"  sensitive attributes included: {BEST_CFG['use_sensitive']}")
print(f"  constant features dropped: {BEST_CFG['drop_constant']}")
print(f"  early-stopped at epoch {ep_f}")

torch.save(model_f.state_dict(), os.path.join(MOD, f"mlp_optimised_{TASK}.pt"))
import joblib
joblib.dump({"pre": pre_f, "num": num_f, "cat": cat_f, "cfg": {
    k: (list(v) if isinstance(v, tuple) else v) for k, v in BEST_CFG.items()}},
    os.path.join(MOD, f"optimised_bundle_{TASK}.joblib"))
np.savez_compressed(os.path.join(OUT, f"optimised_probs_{TASK}.npz"),
                    val=pv_fin, test=pt_fin, y_val=va["y"].values, y_test=te["y"].values,
                    w_val=va["weight"].values, w_test=te["weight"].values)
with open(os.path.join(OUT, f"optimisation_{TASK}.json"), "w", encoding="utf-8") as f:
    json.dump({"search": "random search", "n_trials": N_TRIALS, "seed": SEED,
               "scoring_rule": "survey-weighted average precision (PR-AUC) on 2019-2020",
               "early_stopping": "inner split: fit 2012-2017, stop on 2018",
               "space": {k: [str(x) for x in v] for k, v in SPACE.items()},
               "selected": {k: (list(v) if isinstance(v, tuple)
                                        else v.item() if hasattr(v, "item") else v)
                            for k, v in BEST_CFG.items()},
               "best_epoch_final": int(ep_f),
               "cv_pr_auc_mean": float(cvd["pr_auc"].mean()),
               "cv_pr_auc_sd": float(cvd["pr_auc"].std())}, f, indent=2)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(range(len(tr_df)), tr_df["val_pr_auc"].values, s=26, c="#2980b9", alpha=.8)
ax[0].axhline(cmp.loc["pre-optimisation (locked champion)", "pr_auc"], c="#c0392b", ls="--",
              lw=1.6, label="pre-optimisation champion")
ax[0].axhline(tr_df["val_pr_auc"].max(), c="#16a085", ls="-", lw=1.6, label="search winner")
ax[0].set_xlabel("trial (ranked best to worst)"); ax[0].set_ylabel("validation PR-AUC")
ax[0].set_title("Q5.1 Random search trials", fontweight="bold")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].errorbar(cvd["held_out_year"], cvd["pr_auc"], fmt="o-", lw=2, color="#8e44ad", ms=6)
ax[1].axhline(cvd["pr_auc"].mean(), c="k", ls="--", lw=1,
              label=f"mean {cvd['pr_auc'].mean():.3f} (sd {cvd['pr_auc'].std():.3f})")
ax[1].set_xlabel("held-out survey year"); ax[1].set_ylabel("PR-AUC")
ax[1].set_title("Q4.3 Leave-one-year-out CV variability", fontweight="bold")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.suptitle(f"Champion optimisation - {TASK}", fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, .93])
plt.savefig(os.path.join(FIG, f"F15_optimisation_{TASK}.png"), bbox_inches="tight")
print(f"\nwrote figures/F15_optimisation_{TASK}.png")
print("saved optimised model, bundle, probabilities and search log")

