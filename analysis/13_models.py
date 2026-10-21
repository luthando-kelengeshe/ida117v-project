"""
Step 13: Baseline + 4 substantive models for social-grant prediction.
Covers brief Q3.1 (naive baseline), Q3.2/Q3.3 (>=2 traditional + >=2 neural),
Q3.4 (training documentation), Q4.1/Q4.2/Q4.3 (metrics, matrices, ROC, comparison),
Q4.4 (overfitting: train-vs-val gaps and neural loss curves).

Models
  BASELINE     DummyClassifier(strategy="prior")
  TRADITIONAL  LogisticRegression                     (linear, interpretable)
  TRADITIONAL  RandomForestClassifier                 (bagged trees, non-linear)
  NEURAL       MLP                                    (PyTorch, one-hot input)
  NEURAL       TabTransformer                         (PyTorch, categorical embeddings
                                                       + multi-head self-attention)

Split is time-based (train 2012-18 / val 2019-20 / test 2021-22). The test set is
touched ONCE, at the very end, and only to report final numbers.
"""
import json, os, random, time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import ghs_dataset as gd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.use_deterministic_algorithms(False)

OUT = gd.OUT_DIR
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
MOD = os.path.join(OUT, "models"); os.makedirs(MOD, exist_ok=True)
TASK = os.environ.get("GHS_TASK", "csg_takeup")
DEV = "cpu"

print("=" * 110)
print(f"TASK = {TASK}   seed={SEED}   torch={torch.__version__}   device={DEV}")
print("=" * 110)

panel = pd.read_parquet(os.path.join(OUT, "analysis_panel.parquet"))
d, tcol, (NUM, CAT) = gd.get_task(panel, TASK)
S = gd.split_xy(d, NUM, CAT)
Xtr, ytr, wtr, _ = S["train"]; Xva, yva, wva, _ = S["val"]; Xte, yte, wte, _ = S["test"]
print(f"train {len(ytr):,} (pos {ytr.mean():.3f}) | val {len(yva):,} (pos {yva.mean():.3f}) "
      f"| test {len(yte):,} (pos {yte.mean():.3f})")
print(f"features: {len(NUM)} numeric + {len(CAT)} categorical\n")

# ----------------------------------------------------------------- preprocessors
def make_onehot():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                               sparse_output=False))]), CAT)])

def make_ordinal():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
                          ("od", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                unknown_value=-1))]), CAT)])

# ----------------------------------------------------------------- neural nets
class MLP(nn.Module):
    """Plain feed-forward net on one-hot encoded input."""
    def __init__(self, d_in, hidden=(256, 128, 64), p_drop=0.3):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x_num, x_cat=None):
        return self.net(x_num).squeeze(-1)

class TabTransformer(nn.Module):
    """
    Categorical features -> learned embeddings -> multi-head self-attention across
    the feature dimension, concatenated with normalised numerics -> MLP head.
    Architecturally distinct from the MLP: it learns interactions between
    categorical fields via attention rather than through one-hot concatenation.
    """
    def __init__(self, cardinalities, n_num, d_model=32, n_heads=4, n_blocks=2, p_drop=0.2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c + 2, d_model) for c in cardinalities])
        self.col = nn.Parameter(torch.randn(len(cardinalities), d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                         dim_feedforward=d_model * 4, dropout=p_drop,
                                         batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, num_layers=n_blocks)
        self.num_norm = nn.LayerNorm(n_num)
        d_cat = d_model * len(cardinalities)
        self.head = nn.Sequential(
            nn.Linear(d_cat + n_num, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(64, 1))
    def forward(self, x_num, x_cat):
        toks = torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.embs)], 1) + self.col
        z = self.tr(toks).flatten(1)
        return self.head(torch.cat([z, self.num_norm(x_num)], 1)).squeeze(-1)

def train_torch(model, tr, va, name, epochs=40, bs=1024, lr=1e-3,
                weight_decay=1e-4, patience=6, pos_weight=None):
    """Mini-batch training with early stopping on validation AUC."""
    Xn_tr, Xc_tr, y_tr = tr; Xn_va, Xc_va, y_va = va
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=.5, patience=2)
    pw = None if pos_weight is None else torch.tensor([pos_weight], device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(y_tr); hist = []; best = (-1, None, 0)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, generator=g); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2: continue
            opt.zero_grad()
            out = model(Xn_tr[idx].to(DEV), Xc_tr[idx].to(DEV))
            loss = lossf(out, y_tr[idx].to(DEV))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item() * len(idx)
        tr_loss = tot / n
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xn_va.to(DEV), Xc_va.to(DEV))).cpu().numpy()
            va_loss = lossf(model(Xn_va.to(DEV), Xc_va.to(DEV)), y_va.to(DEV)).item()
            pt = torch.sigmoid(model(Xn_tr.to(DEV), Xc_tr.to(DEV))).cpu().numpy()
        va_auc = roc_auc_score(y_va.numpy(), pv); tr_auc = roc_auc_score(y_tr.numpy(), pt)
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                     "train_auc": tr_auc, "val_auc": va_auc})
        sched.step(va_auc)
        if va_auc > best[0]:
            best = (va_auc, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if ep - best[2] >= patience:
            print(f"    early stop at epoch {ep} (best epoch {best[2]}, val AUC {best[0]:.4f})")
            break
        if ep % 5 == 0 or ep == 1:
            print(f"    ep{ep:>3} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                  f"train_auc={tr_auc:.4f} val_auc={va_auc:.4f}")
    model.load_state_dict(best[1])
    return model, pd.DataFrame(hist), best[2]

# ----------------------------------------------------------------- metric helper
def metrics(y, p, w=None, thr=0.5):
    yh = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, yh, sample_weight=w),
            "macro_f1": f1_score(y, yh, average="macro", sample_weight=w),
            "weighted_f1": f1_score(y, yh, average="weighted", sample_weight=w),
            "precision": precision_score(y, yh, zero_division=0, sample_weight=w),
            "recall": recall_score(y, yh, zero_division=0, sample_weight=w),
            "roc_auc": roc_auc_score(y, p, sample_weight=w),
            "pr_auc": average_precision_score(y, p, sample_weight=w)}

RESULTS, PROBS, CURVES, TIMES = {}, {}, {}, {}

# ================================================================= BASELINE Q3.1
print("-" * 110); print("Q3.1  NAIVE BASELINE: DummyClassifier(strategy='prior')"); print("-" * 110)
t0 = time.time()
dummy = Pipeline([("pre", make_onehot()), ("clf", DummyClassifier(strategy="prior", random_state=SEED))])
dummy.fit(Xtr, ytr)
TIMES["Baseline (Dummy)"] = time.time() - t0
for part, (X, y, w) in [("train", (Xtr, ytr, wtr)), ("val", (Xva, yva, wva)), ("test", (Xte, yte, wte))]:
    PROBS[("Baseline (Dummy)", part)] = dummy.predict_proba(X)[:, 1]
RESULTS["Baseline (Dummy)"] = {
    p: metrics(y, PROBS[("Baseline (Dummy)", p)], w)
    for p, (y, w) in [("train", (ytr, wtr)), ("val", (yva, wva)), ("test", (yte, wte))]}
print(f"  val accuracy={RESULTS['Baseline (Dummy)']['val']['accuracy']:.4f}  "
      f"macro_f1={RESULTS['Baseline (Dummy)']['val']['macro_f1']:.4f}  "
      f"roc_auc={RESULTS['Baseline (Dummy)']['val']['roc_auc']:.4f}")
print("  Establishes the floor: predicts the training prior for every person, so it")
print("  achieves majority-class accuracy with no discriminative power (AUC = 0.5).\n")

# ================================================================= TRADITIONAL
print("-" * 110); print("Q3.2  TRADITIONAL MODEL 1/2: Logistic Regression"); print("-" * 110)
t0 = time.time()
logreg = Pipeline([("pre", make_onehot()),
                   ("clf", LogisticRegression(max_iter=3000, C=1.0, penalty="l2",
                                              solver="lbfgs", class_weight="balanced",
                                              random_state=SEED, n_jobs=-1))])
logreg.fit(Xtr, ytr); TIMES["Logistic Regression"] = time.time() - t0
for part, X in [("train", Xtr), ("val", Xva), ("test", Xte)]:
    PROBS[("Logistic Regression", part)] = logreg.predict_proba(X)[:, 1]
RESULTS["Logistic Regression"] = {
    p: metrics(y, PROBS[("Logistic Regression", p)], w)
    for p, (y, w) in [("train", (ytr, wtr)), ("val", (yva, wva)), ("test", (yte, wte))]}
print(f"  fitted in {TIMES['Logistic Regression']:.1f}s  "
      f"val roc_auc={RESULTS['Logistic Regression']['val']['roc_auc']:.4f}\n")

print("-" * 110); print("Q3.2  TRADITIONAL MODEL 2/2: Random Forest"); print("-" * 110)
t0 = time.time()
rf = Pipeline([("pre", make_onehot()),
               ("clf", RandomForestClassifier(n_estimators=400, max_depth=18,
                                              min_samples_leaf=10, max_features="sqrt",
                                              class_weight="balanced_subsample",
                                              n_jobs=-1, random_state=SEED))])
rf.fit(Xtr, ytr); TIMES["Random Forest"] = time.time() - t0
for part, X in [("train", Xtr), ("val", Xva), ("test", Xte)]:
    PROBS[("Random Forest", part)] = rf.predict_proba(X)[:, 1]
RESULTS["Random Forest"] = {
    p: metrics(y, PROBS[("Random Forest", p)], w)
    for p, (y, w) in [("train", (ytr, wtr)), ("val", (yva, wva)), ("test", (yte, wte))]}
print(f"  fitted in {TIMES['Random Forest']:.1f}s  "
      f"val roc_auc={RESULTS['Random Forest']['val']['roc_auc']:.4f}\n")

# ================================================================= NEURAL PREP
print("-" * 110); print("Preparing tensors for the neural models"); print("-" * 110)
oh = make_onehot(); oh.fit(Xtr)
A_tr = torch.tensor(oh.transform(Xtr), dtype=torch.float32)
A_va = torch.tensor(oh.transform(Xva), dtype=torch.float32)
A_te = torch.tensor(oh.transform(Xte), dtype=torch.float32)
print(f"  one-hot design matrix: {A_tr.shape[1]} columns")

od = make_ordinal(); od.fit(Xtr)
def split_ord(X):
    Z = od.transform(X)
    xn = torch.tensor(Z[:, :len(NUM)], dtype=torch.float32)
    xc = torch.tensor(Z[:, len(NUM):] + 1, dtype=torch.long).clamp_(min=0)
    return xn, xc
N_tr, C_tr = split_ord(Xtr); N_va, C_va = split_ord(Xva); N_te, C_te = split_ord(Xte)
cards = [int(max(C_tr[:, i].max().item(), C_va[:, i].max().item(),
                 C_te[:, i].max().item())) + 1 for i in range(C_tr.shape[1])]
print(f"  categorical cardinalities: {dict(zip(CAT, cards))}")
Y_tr = torch.tensor(ytr, dtype=torch.float32); Y_va = torch.tensor(yva, dtype=torch.float32)
zeros = lambda t: torch.zeros((len(t), 1), dtype=torch.long)
pos_w = float((1 - ytr.mean()) / ytr.mean())
print(f"  BCE pos_weight = {pos_w:.3f} (inverse class prior)\n")

# ================================================================= NEURAL 1 MLP
print("-" * 110); print("Q3.2  NEURAL MODEL 1/2: MLP (PyTorch)"); print("-" * 110)
t0 = time.time()
mlp = MLP(A_tr.shape[1], hidden=(256, 128, 64), p_drop=0.3)
print(f"  params: {sum(p.numel() for p in mlp.parameters()):,}")
mlp, mlp_hist, mlp_best = train_torch(
    mlp, (A_tr, zeros(A_tr), Y_tr), (A_va, zeros(A_va), Y_va), "MLP",
    epochs=40, bs=1024, lr=1e-3, weight_decay=1e-4, patience=6, pos_weight=pos_w)
TIMES["MLP (PyTorch)"] = time.time() - t0
mlp.eval()
with torch.no_grad():
    for part, A in [("train", A_tr), ("val", A_va), ("test", A_te)]:
        PROBS[("MLP (PyTorch)", part)] = torch.sigmoid(mlp(A.to(DEV), zeros(A).to(DEV))).cpu().numpy()
RESULTS["MLP (PyTorch)"] = {
    p: metrics(y, PROBS[("MLP (PyTorch)", p)], w)
    for p, (y, w) in [("train", (ytr, wtr)), ("val", (yva, wva)), ("test", (yte, wte))]}
CURVES["MLP (PyTorch)"] = mlp_hist
print(f"  trained in {TIMES['MLP (PyTorch)']:.1f}s  best epoch {mlp_best}  "
      f"val roc_auc={RESULTS['MLP (PyTorch)']['val']['roc_auc']:.4f}\n")

# ================================================================= NEURAL 2 TAB
print("-" * 110); print("Q3.2  NEURAL MODEL 2/2: TabTransformer (PyTorch)"); print("-" * 110)
t0 = time.time()
tab = TabTransformer(cards, len(NUM), d_model=32, n_heads=4, n_blocks=2, p_drop=0.2)
print(f"  params: {sum(p.numel() for p in tab.parameters()):,}")
tab, tab_hist, tab_best = train_torch(
    tab, (N_tr, C_tr, Y_tr), (N_va, C_va, Y_va), "TabTransformer",
    epochs=30, bs=1024, lr=1e-3, weight_decay=1e-4, patience=5, pos_weight=pos_w)
TIMES["TabTransformer (PyTorch)"] = time.time() - t0
tab.eval()
with torch.no_grad():
    for part, (xn, xc) in [("train", (N_tr, C_tr)), ("val", (N_va, C_va)), ("test", (N_te, C_te))]:
        PROBS[("TabTransformer (PyTorch)", part)] = torch.sigmoid(
            tab(xn.to(DEV), xc.to(DEV))).cpu().numpy()
RESULTS["TabTransformer (PyTorch)"] = {
    p: metrics(y, PROBS[("TabTransformer (PyTorch)", p)], w)
    for p, (y, w) in [("train", (ytr, wtr)), ("val", (yva, wva)), ("test", (yte, wte))]}
CURVES["TabTransformer (PyTorch)"] = tab_hist
print(f"  trained in {TIMES['TabTransformer (PyTorch)']:.1f}s  best epoch {tab_best}  "
      f"val roc_auc={RESULTS['TabTransformer (PyTorch)']['val']['roc_auc']:.4f}\n")

# ================================================================= Q4 RESULTS
MODELS = list(RESULTS)
print("=" * 110); print("Q4.1  HELD-OUT METRICS (validation, survey-weighted)"); print("=" * 110)
tabv = pd.DataFrame({m: RESULTS[m]["val"] for m in MODELS}).T.round(4)
tabv["fit_seconds"] = [round(TIMES[m], 1) for m in MODELS]
print(tabv.to_string())
tabv.to_csv(os.path.join(OUT, f"results_val_{TASK}.csv"))

print("\n" + "=" * 110); print("Q4.4  OVERFITTING: train vs validation"); print("=" * 110)
gap = pd.DataFrame({
    "train_roc_auc": [RESULTS[m]["train"]["roc_auc"] for m in MODELS],
    "val_roc_auc": [RESULTS[m]["val"]["roc_auc"] for m in MODELS],
    "train_macro_f1": [RESULTS[m]["train"]["macro_f1"] for m in MODELS],
    "val_macro_f1": [RESULTS[m]["val"]["macro_f1"] for m in MODELS]}, index=MODELS)
gap["auc_gap"] = (gap["train_roc_auc"] - gap["val_roc_auc"]).round(4)
gap["f1_gap"] = (gap["train_macro_f1"] - gap["val_macro_f1"]).round(4)
print(gap.round(4).to_string())
gap.round(4).to_csv(os.path.join(OUT, f"results_overfit_{TASK}.csv"))

champ = max([m for m in MODELS if m != "Baseline (Dummy)"],
            key=lambda m: RESULTS[m]["val"]["roc_auc"])
runner = max([m for m in MODELS if m not in ("Baseline (Dummy)", champ)],
             key=lambda m: RESULTS[m]["val"]["roc_auc"])
print(f"\nQ4.3  provisional champion (by VALIDATION roc_auc): {champ}")
print(f"      strongest comparator:                        {runner}")

print("\n" + "=" * 110); print("Q4.2  CLASSIFICATION REPORTS (validation)"); print("=" * 110)
for m in MODELS:
    print(f"\n--- {m} ---")
    print(classification_report(yva, (PROBS[(m, "val")] >= .5).astype(int),
                                target_names=["no grant", "grant"], zero_division=0, digits=3))

print("=" * 110); print("FINAL TEST EVALUATION (reported once, for reference)"); print("=" * 110)
tabt = pd.DataFrame({m: RESULTS[m]["test"] for m in MODELS}).T.round(4)
print(tabt.to_string())
tabt.to_csv(os.path.join(OUT, f"results_test_{TASK}.csv"))

# ============================================== OPERATIONAL TARGETING (capacity view)
# For the reach-gap task, accuracy is the wrong question. The decision is
# "we can reach N households, which ones?" so report precision@k and lift.
print("\n" + "=" * 110)
print("OPERATIONAL TARGETING on validation: precision@k and lift over random outreach")
print("=" * 110)
op_rows = []
base_rate = float(np.average(yva, weights=wva))
for m in MODELS:
    if m == "Baseline (Dummy)":
        continue
    o = pd.DataFrame({"p": PROBS[(m, "val")], "y": yva, "w": wva}).sort_values(
        "p", ascending=False)
    tot = o.loc[o["y"].eq(1), "w"].sum()
    rec = {"model": m}
    for frac in (0.05, 0.10, 0.20, 0.30):
        s = o.iloc[:max(int(len(o) * frac), 1)]
        hit = float(np.average(s["y"], weights=s["w"]))
        rec[f"prec@{int(frac*100)}%"] = round(hit, 4)
        rec[f"lift@{int(frac*100)}%"] = round(hit / base_rate, 2)
        rec[f"recall@{int(frac*100)}%"] = round(s.loc[s["y"].eq(1), "w"].sum() / tot, 4)
    op_rows.append(rec)
op = pd.DataFrame(op_rows).set_index("model")
print(f"validation base rate (random outreach hit rate) = {base_rate:.4f}")
print(op.to_string())
op.to_csv(os.path.join(OUT, f"results_targeting_{TASK}.csv"))

# ================================================================= FIGURES
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .3})
COLORS = dict(zip(MODELS, ["#7f8c8d", "#2980b9", "#16a085", "#c0392b", "#8e44ad"]))

# confusion matrices
fig, axes = plt.subplots(1, len(MODELS), figsize=(4.0 * len(MODELS), 3.9))
for a, m in zip(axes, MODELS):
    cm = confusion_matrix(yva, (PROBS[(m, "val")] >= .5).astype(int))
    a.imshow(cm, cmap="Blues"); a.grid(False)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=10,
                   color="white" if cm[i, j] > cm.max() * .55 else "black")
    a.set_xticks([0, 1]); a.set_xticklabels(["pred 0", "pred 1"])
    a.set_yticks([0, 1]); a.set_yticklabels(["true 0", "true 1"])
    a.set_title(f"{m}\nval macro-F1 {RESULTS[m]['val']['macro_f1']:.3f}", fontsize=9, fontweight="bold")
fig.suptitle(f"Q4.2 Confusion matrices on validation (2019-2020) - task: {TASK}",
             fontweight="bold", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, .92])
plt.savefig(os.path.join(FIG, f"F8_confusion_{TASK}.png"), bbox_inches="tight"); plt.close()

# ROC shared axes
fig, ax = plt.subplots(figsize=(7.2, 6.2))
for m in MODELS:
    fpr, tpr, _ = roc_curve(yva, PROBS[(m, "val")], sample_weight=wva)
    ax.plot(fpr, tpr, lw=2, color=COLORS[m],
            label=f"{m} (AUC {RESULTS[m]['val']['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title(f"Q4.2 ROC curves, validation set - task: {TASK}", fontweight="bold")
ax.legend(fontsize=8.5, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(FIG, f"F9_roc_{TASK}.png"), bbox_inches="tight"); plt.close()

# comparison bars
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
mets = ["accuracy", "macro_f1", "roc_auc", "pr_auc"]
x = np.arange(len(mets)); wd = .16
for i, m in enumerate(MODELS):
    ax[0].bar(x + i * wd, [RESULTS[m]["val"][k] for k in mets], wd, label=m, color=COLORS[m])
ax[0].set_xticks(x + wd * 2); ax[0].set_xticklabels(mets)
ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("score")
ax[0].set_title("Q4.3 Validation metric comparison", fontweight="bold"); ax[0].legend(fontsize=8)
ax[1].bar(range(len(MODELS)), gap["auc_gap"].values,
          color=[COLORS[m] for m in MODELS])
ax[1].set_xticks(range(len(MODELS)))
ax[1].set_xticklabels([m.replace(" (PyTorch)", "") for m in MODELS], rotation=18, ha="right", fontsize=8)
ax[1].axhline(0, c="k", lw=.8)
ax[1].set_ylabel("train AUC - val AUC")
ax[1].set_title("Q4.4 Overfitting gap (higher = more overfit)", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG, f"F10_comparison_{TASK}.png"), bbox_inches="tight"); plt.close()

# neural curves
fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))
for j, (m, h) in enumerate(CURVES.items()):
    axes[0, j].plot(h["epoch"], h["train_loss"], "o-", label="train", color="#2980b9", ms=3)
    axes[0, j].plot(h["epoch"], h["val_loss"], "s-", label="validation", color="#c0392b", ms=3)
    axes[0, j].set_title(f"{m}: BCE loss per epoch", fontweight="bold", fontsize=10)
    axes[0, j].set_xlabel("epoch"); axes[0, j].set_ylabel("loss"); axes[0, j].legend(fontsize=8)
    axes[1, j].plot(h["epoch"], h["train_auc"], "o-", label="train", color="#2980b9", ms=3)
    axes[1, j].plot(h["epoch"], h["val_auc"], "s-", label="validation", color="#c0392b", ms=3)
    axes[1, j].set_title(f"{m}: ROC-AUC per epoch", fontweight="bold", fontsize=10)
    axes[1, j].set_xlabel("epoch"); axes[1, j].set_ylabel("ROC-AUC"); axes[1, j].legend(fontsize=8)
fig.suptitle(f"Q4.4 Neural training curves - task: {TASK}", fontweight="bold", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, .94])
plt.savefig(os.path.join(FIG, f"F11_neural_curves_{TASK}.png"), bbox_inches="tight"); plt.close()

# cumulative gain / lift curve
fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.0))
for m in MODELS:
    o = pd.DataFrame({"p": PROBS[(m, "val")], "y": yva, "w": wva}).sort_values("p", ascending=False)
    cw = o["w"].cumsum() / o["w"].sum()
    cg = (o["y"] * o["w"]).cumsum() / (o["y"] * o["w"]).sum()
    ax[0].plot(cw * 100, cg * 100, lw=2, color=COLORS[m], label=m)
    with np.errstate(divide="ignore", invalid="ignore"):
        ax[1].plot(cw * 100, cg / cw, lw=2, color=COLORS[m], label=m)
ax[0].plot([0, 100], [0, 100], "k--", lw=1, label="random outreach")
ax[0].set_xlabel("% of population contacted (ranked by risk score)")
ax[0].set_ylabel("% of all reach-gap cases found")
ax[0].set_title("Cumulative gain: how many missed children are found", fontweight="bold")
ax[0].legend(fontsize=8); ax[0].set_xlim(0, 100); ax[0].set_ylim(0, 100)
ax[1].axhline(1, c="k", ls="--", lw=1, label="random outreach")
ax[1].set_xlabel("% of population contacted (ranked by risk score)")
ax[1].set_ylabel("lift over random")
ax[1].set_title("Lift: efficiency gain from model-guided targeting", fontweight="bold")
ax[1].legend(fontsize=8); ax[1].set_xlim(1, 100); ax[1].set_ylim(0, None)
fig.suptitle(f"Operational targeting view - task: {TASK}", fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, .93])
plt.savefig(os.path.join(FIG, f"F13_targeting_{TASK}.png"), bbox_inches="tight"); plt.close()

for m, h in CURVES.items():
    h.to_csv(os.path.join(OUT, f"curves_{m.split()[0]}_{TASK}.csv"), index=False)
np.savez_compressed(os.path.join(OUT, f"val_test_probs_{TASK}.npz"),
                    **{f"{m}|{p}": PROBS[(m, p)] for (m, p) in PROBS},
                    y_val=yva, y_test=yte, w_val=wva, w_test=wte)

HP = {
 "seed": SEED, "task": TASK, "device": DEV, "torch": torch.__version__,
 "split": {"train": gd.TRAIN_YEARS, "val": gd.VAL_YEARS, "test": gd.TEST_YEARS},
 "n": {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))},
 "Baseline (Dummy)": {"strategy": "prior"},
 "Logistic Regression": {"penalty": "l2", "C": 1.0, "solver": "lbfgs",
                         "max_iter": 3000, "class_weight": "balanced"},
 "Random Forest": {"n_estimators": 400, "max_depth": 18, "min_samples_leaf": 10,
                   "max_features": "sqrt", "class_weight": "balanced_subsample"},
 "MLP (PyTorch)": {"hidden": [256, 128, 64], "dropout": 0.3, "batchnorm": True,
                   "optimiser": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
                   "loss": "BCEWithLogitsLoss", "pos_weight": round(pos_w, 4),
                   "batch_size": 1024, "max_epochs": 40, "early_stopping_patience": 6,
                   "best_epoch": int(mlp_best), "grad_clip": 5.0,
                   "input_dim": int(A_tr.shape[1]),
                   "params": int(sum(p.numel() for p in mlp.parameters()))},
 "TabTransformer (PyTorch)": {"d_model": 32, "n_heads": 4, "n_blocks": 2, "dropout": 0.2,
                              "head": [128, 64], "optimiser": "AdamW", "lr": 1e-3,
                              "weight_decay": 1e-4, "loss": "BCEWithLogitsLoss",
                              "pos_weight": round(pos_w, 4), "batch_size": 1024,
                              "max_epochs": 30, "early_stopping_patience": 5,
                              "best_epoch": int(tab_best), "grad_clip": 5.0,
                              "cardinalities": dict(zip(CAT, [int(c) for c in cards])),
                              "params": int(sum(p.numel() for p in tab.parameters()))},
 "champion_by_val_auc": champ, "strongest_comparator": runner,
 "fit_seconds": {k: round(v, 1) for k, v in TIMES.items()},
}
with open(os.path.join(OUT, f"hyperparameters_{TASK}.json"), "w", encoding="utf-8") as f:
    json.dump(HP, f, indent=2)
torch.save(mlp.state_dict(), os.path.join(MOD, f"mlp_{TASK}.pt"))
torch.save(tab.state_dict(), os.path.join(MOD, f"tabtransformer_{TASK}.pt"))
import joblib
joblib.dump(logreg, os.path.join(MOD, f"logreg_{TASK}.joblib"))
joblib.dump(rf, os.path.join(MOD, f"randomforest_{TASK}.joblib"))
joblib.dump({"onehot": oh, "ordinal": od, "num": NUM, "cat": CAT, "cards": cards},
            os.path.join(MOD, f"preprocessors_{TASK}.joblib"))
print(f"\nSaved models, preprocessors, curves, probabilities and hyperparameters under {OUT}")

