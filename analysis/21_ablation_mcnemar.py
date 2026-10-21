"""
Step 21: Brief Q6 - controlled ablations (6.1) and McNemar's test (6.2).

Q6.1  Seven one-change-at-a-time experiments on the champion (MLP). Split, seed,
      epochs, batch size, early-stopping rule and evaluation protocol are held
      IDENTICAL to the locked configuration; exactly one thing changes per run.
      Categories covered, as the brief requires: feature group, preprocessing step,
      imbalance method, regulariser, architecture component.

Q6.2  McNemar's test between the champion (MLP) and the strongest comparator
      (Random Forest) on identical held-out cases.
"""
import json, os, random, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
import ghs_dataset as gd
from ghs_models import MLP

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 60)
SEED = 42
TASK = "csg_reach_gap"
OUT = gd.OUT_DIR; FIG = os.path.join(OUT, "figures")

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

panel = pd.read_parquet(os.path.join(OUT, "analysis_panel.parquet"))
d, _, (NUM, CAT) = gd.get_task(panel, TASK)
S = gd.split_xy(d, NUM, CAT)
Xtr, ytr, wtr, _ = S["train"]; Xva, yva, wva, _ = S["val"]
print(f"task={TASK}  train={len(ytr):,}  val={len(yva):,}  pos_rate_val={yva.mean():.4f}")

HH = gd.HH_NUM
DEAD = ["hh_n_employed", "hh_workers_per_capita", "hh_no_earner", "hh_any_medaid", "medaid"]
SENSITIVE = ["race", "sex"]
GEO = ["prov", "geotype"]
POS_W = float((1 - ytr.mean()) / ytr.mean())
BASE_HP = dict(hidden=(256, 128, 64), p_drop=0.3, lr=1e-3, wd=1e-4,
               epochs=40, bs=1024, patience=6, pos_weight=POS_W, scale=True)

def build_pre(num, cat, scale=True):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    num_steps.append(("sc", StandardScaler() if scale
                      else FunctionTransformer(lambda x: x, feature_names_out="one-to-one")))
    blocks = []
    if num:
        blocks.append(("num", Pipeline(num_steps), num))
    if cat:
        blocks.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                 sparse_output=False))]), cat))
    return ColumnTransformer(blocks)

def run_mlp(num, cat, hp):
    """Train the champion architecture under one controlled change."""
    seed_all()
    pre = build_pre(num, cat, hp["scale"]); pre.fit(Xtr[num + cat])
    A_tr = torch.tensor(pre.transform(Xtr[num + cat]), dtype=torch.float32)
    A_va = torch.tensor(pre.transform(Xva[num + cat]), dtype=torch.float32)
    Y_tr = torch.tensor(ytr, dtype=torch.float32); Y_va = torch.tensor(yva, dtype=torch.float32)
    model = MLP(A_tr.shape[1], hidden=tuple(hp["hidden"]), p_drop=hp["p_drop"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    pw = None if hp["pos_weight"] is None else torch.tensor([hp["pos_weight"]])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(Y_tr); best = (-1, None, 0)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(1, hp["epochs"] + 1):
        model.train(); perm = torch.randperm(n, generator=g)
        for i in range(0, n, hp["bs"]):
            idx = perm[i:i + hp["bs"]]
            if len(idx) < 2: continue
            opt.zero_grad()
            loss = lossf(model(A_tr[idx], None), Y_tr[idx]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(A_va, None)).numpy()
        a = roc_auc_score(yva, pv)
        if a > best[0]:
            best = (a, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if ep - best[2] >= hp["patience"]: break
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(A_va, None)).numpy()
    return pv, best[2]

def score(pv, label, n_feats, best_ep):
    yh = (pv >= .5).astype(int)
    o = pd.DataFrame({"p": pv, "y": yva, "w": wva}).sort_values("p", ascending=False)
    base = float(np.average(yva, weights=wva))
    top10 = o.iloc[:int(len(o) * .10)]
    return {"experiment": label, "n_features": n_feats, "best_epoch": best_ep,
            "roc_auc": roc_auc_score(yva, pv, sample_weight=wva),
            "pr_auc": average_precision_score(yva, pv, sample_weight=wva),
            "macro_f1": f1_score(yva, yh, average="macro", sample_weight=wva),
            "lift@10%": float(np.average(top10["y"], weights=top10["w"])) / base}

print("\n" + "=" * 118)
print("Q6.1  ABLATION STUDY - one controlled change per run, split/seed/protocol fixed")
print("=" * 118)

EXPERIMENTS = [
    ("A0  LOCKED CHAMPION (reference)", NUM, CAT, {}),
    ("A1  - household-context feature group", [c for c in NUM if c not in HH], CAT, {}),
    ("A2  - the 5 constant features (diagnostic)",
     [c for c in NUM if c not in DEAD], [c for c in CAT if c not in DEAD], {}),
    ("A3  - sensitive attributes (race, sex)", NUM, [c for c in CAT if c not in SENSITIVE], {}),
    ("A4  - geography (province, settlement)", NUM, [c for c in CAT if c not in GEO], {}),
    ("A5  imbalance treatment removed (pos_weight=1)", NUM, CAT, {"pos_weight": None}),
    ("A6  regulariser removed (dropout 0.3 -> 0.0)", NUM, CAT, {"p_drop": 0.0}),
    ("A7  architecture shrunk (3 layers -> 1x32)", NUM, CAT, {"hidden": (32,)}),
    ("A8  preprocessing: feature scaling removed", NUM, CAT, {"scale": False}),
]

rows = []
for label, num, cat, override in EXPERIMENTS:
    hp = dict(BASE_HP); hp.update(override)
    pv, ep = run_mlp(num, cat, hp)
    r = score(pv, label, len(num) + len(cat), ep)
    rows.append(r)
    print(f"  {label:<48} feats={r['n_features']:>3}  ep={ep:>2}  "
          f"ROC={r['roc_auc']:.4f}  PR={r['pr_auc']:.4f}  lift@10%={r['lift@10%']:.2f}")

ab = pd.DataFrame(rows).set_index("experiment")
ref = ab.loc["A0  LOCKED CHAMPION (reference)"]
for m in ["roc_auc", "pr_auc", "macro_f1", "lift@10%"]:
    ab[f"d_{m}"] = (ab[m] - ref[m]).round(4)
print("\nDELTAS vs locked champion (negative = ablation hurt performance)")
print(ab[["n_features", "roc_auc", "d_roc_auc", "pr_auc", "d_pr_auc",
          "lift@10%", "d_lift@10%"]].round(4).to_string())
ab.round(5).to_csv(os.path.join(OUT, f"ablation_{TASK}.csv"))

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4))
pl = ab.drop(index="A0  LOCKED CHAMPION (reference)")
cols = ["#c0392b" if v < 0 else "#16a085" for v in pl["d_pr_auc"]]
ax[0].barh(range(len(pl)), pl["d_pr_auc"], color=cols)
ax[0].set_yticks(range(len(pl))); ax[0].set_yticklabels(pl.index, fontsize=8)
ax[0].axvline(0, c="k", lw=.8); ax[0].invert_yaxis()
ax[0].set_xlabel("change in PR-AUC vs locked champion")
ax[0].set_title("Q6.1 Ablation effect on PR-AUC", fontweight="bold"); ax[0].grid(alpha=.3)
cols2 = ["#c0392b" if v < 0 else "#16a085" for v in pl["d_lift@10%"]]
ax[1].barh(range(len(pl)), pl["d_lift@10%"], color=cols2)
ax[1].set_yticks(range(len(pl))); ax[1].set_yticklabels([""] * len(pl))
ax[1].axvline(0, c="k", lw=.8); ax[1].invert_yaxis()
ax[1].set_xlabel("change in lift@10% vs locked champion")
ax[1].set_title("Q6.1 Ablation effect on outreach lift", fontweight="bold"); ax[1].grid(alpha=.3)
fig.suptitle(f"Ablation study - {TASK} (one change per run, seed {SEED}, identical split)",
             fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, .93])
plt.savefig(os.path.join(FIG, f"F14_ablation_{TASK}.png"), bbox_inches="tight"); plt.close()
print(f"wrote figures/F14_ablation_{TASK}.png")

# ===================================================================== Q6.2
print("\n" + "=" * 118)
print("Q6.2  McNEMAR'S TEST - champion vs strongest comparator, identical held-out cases")
print("=" * 118)
import joblib
z = np.load(os.path.join(OUT, f"val_test_probs_{TASK}.npz"))
cals = joblib.load(os.path.join(OUT, "models", f"calibrators_{TASK}.joblib"))["calibrators"]
CH, CO = "MLP (PyTorch)", "Random Forest"

def mcnemar(y, correct_a, correct_b, name_a, name_b, context):
    both = int(np.sum(correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    neither = int(np.sum(~correct_a & ~correct_b))
    disc = a_only + b_only
    print(f"\n--- {context} ---")
    print(f"  n = {len(y):,} identical held-out cases")
    print(f"\n  2x2 correctness table")
    print(f"  {'':<28}{name_b+' correct':>22}{name_b+' wrong':>22}")
    print(f"  {name_a+' correct':<28}{both:>22,}{a_only:>22,}")
    print(f"  {name_a+' wrong':<28}{b_only:>22,}{neither:>22,}")
    print(f"\n  both correct      : {both:,}")
    print(f"  {name_a} only correct : {a_only:,}")
    print(f"  {name_b} only correct : {b_only:,}")
    print(f"  both wrong        : {neither:,}")
    print(f"  discordant pairs  : {disc:,}")
    print(f"\n  H0: the two models have equal error rates (discordant pairs split 50/50)")
    print(f"  H1: the two models have different error rates")
    print(f"  alpha = 0.05")
    if disc < 25:
        p = stats.binomtest(a_only, disc, 0.5).pvalue if disc else 1.0
        print(f"  discordant n = {disc} < 25 -> EXACT binomial test used")
        print(f"  statistic (a_only) = {a_only}   exact p = {p:.6g}")
        stat = a_only
    else:
        stat = (abs(a_only - b_only) - 1) ** 2 / disc
        p = float(stats.chi2.sf(stat, 1))
        pe = stats.binomtest(a_only, disc, 0.5).pvalue
        print(f"  discordant n = {disc} >= 25 -> chi-square with continuity correction")
        print(f"  chi2 = {stat:.4f}  (df=1)   p = {p:.6g}")
        print(f"  exact binomial p (confirmatory) = {pe:.6g}")
        p = pe
    acc_a = float(np.mean(correct_a)); acc_b = float(np.mean(correct_b))
    print(f"\n  accuracy {name_a} = {acc_a:.4f}   accuracy {name_b} = {acc_b:.4f}   "
          f"difference = {acc_a-acc_b:+.4f}")
    verdict = "REJECT H0" if p < 0.05 else "FAIL TO REJECT H0"
    print(f"  conclusion: {verdict} at alpha=0.05")
    return {"context": context, "both_correct": both, f"{name_a}_only": a_only,
            f"{name_b}_only": b_only, "both_wrong": neither, "discordant": disc,
            "statistic": stat, "p_value": p, "acc_a": acc_a, "acc_b": acc_b,
            "acc_diff": acc_a - acc_b, "conclusion": verdict}

results = []
for part in ["val", "test"]:
    y = z[f"y_{part}"]
    pa = np.clip(cals[CH].predict(z[f"{CH}|{part}"]), 0, 1)
    pb = np.clip(cals[CO].predict(z[f"{CO}|{part}"]), 0, 1)
    ca = (pa >= .5).astype(int) == y
    cb = (pb >= .5).astype(int) == y
    results.append(mcnemar(y, ca, cb, CH, CO,
                           f"{part.upper()} set, 0.5 decision threshold (calibrated)"))
    # operational rule: flag the top 10% by score
    k = int(len(y) * .10)
    fa = np.zeros(len(y), bool); fa[np.argsort(-pa)[:k]] = True
    fb = np.zeros(len(y), bool); fb[np.argsort(-pb)[:k]] = True
    results.append(mcnemar(y, fa == (y == 1), fb == (y == 1), CH, CO,
                           f"{part.upper()} set, operational rule: flag top 10% for outreach"))

pd.DataFrame(results).to_csv(os.path.join(OUT, f"mcnemar_{TASK}.csv"), index=False)
print(f"\nwrote {OUT}/mcnemar_{TASK}.csv")
