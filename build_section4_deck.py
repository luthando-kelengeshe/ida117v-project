"""
Build a PowerPoint deck for IDA117V Section 4 (Required Evidence and Experimental
Rules) AND Section 5 (Marking Rubric). Each Section-4 requirement is answered against
the build; each of the 8 rubric criteria gets a dedicated slide stating the Excellent
descriptor and the project's evidence against it (aiming for full marks).

All numbers are taken from artefacts in analysis/out/. Run from repo root:
    python build_section4_deck.py
Output: IDA117V_Section4_Evidence.pptx
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "analysis" / "out" / "figures"

# ---- palette -----------------------------------------------------------------
NAVY   = RGBColor(0x0F, 0x2E, 0x4E)
BLUE   = RGBColor(0x1C, 0x6E, 0xA8)
TEAL   = RGBColor(0x11, 0x87, 0x7A)
AMBER  = RGBColor(0xC8, 0x7A, 0x12)
GREY   = RGBColor(0x55, 0x5A, 0x60)
LIGHT  = RGBColor(0xF1, 0xF4, 0xF7)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
DARK   = RGBColor(0x22, 0x27, 0x2B)

EMU_W, EMU_H = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = EMU_W
prs.slide_height = EMU_H
BLANK = prs.slide_layouts[6]


# ---- helpers -----------------------------------------------------------------
def slide():
    return prs.slides.add_slide(BLANK)


def rect(s, x, y, w, h, color):
    from pptx.enum.shapes import MSO_SHAPE
    sp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def textbox(s, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    return tf


def para(tf, text, size=18, color=DARK, bold=False, align=PP_ALIGN.LEFT,
         space_after=6, bullet=False, level=0, first=False, italic=False):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    p.alignment = align
    p.level = level
    p.space_after = Pt(space_after)
    p.space_before = Pt(0)
    r = p.add_run()
    r.text = ("•  " + text) if bullet else text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = "Calibri"
    return p


def header(s, kicker, title):
    rect(s, 0, 0, EMU_W, Inches(1.25), NAVY)
    rect(s, 0, Inches(1.25), EMU_W, Inches(0.06), TEAL)
    tf = textbox(s, Inches(0.55), Inches(0.14), Inches(12.2), Inches(1.0))
    para(tf, kicker, size=13, color=RGBColor(0x9F, 0xC4, 0xE0), bold=True,
         space_after=2, first=True)
    para(tf, title, size=28, color=WHITE, bold=True, space_after=0)


def footer(s, n):
    tf = textbox(s, Inches(0.55), Inches(7.05), Inches(10), Inches(0.35))
    para(tf, "IDA117V  ·  Child Support Grant Reach-Gap Prioritiser  ·  Section 4 evidence",
         size=9, color=GREY, first=True)
    tf2 = textbox(s, Inches(12.3), Inches(7.05), Inches(0.8), Inches(0.35))
    para(tf2, str(n), size=10, color=GREY, align=PP_ALIGN.RIGHT, first=True)


def rubric_chip(s, text):
    """Small amber chip showing the rubric criterion the slide answers."""
    from pptx.enum.shapes import MSO_SHAPE
    w = Inches(5.6)
    sp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(0.55), Inches(1.42), w, Inches(0.42))
    sp.fill.solid(); sp.fill.fore_color.rgb = RGBColor(0xFB, 0xF0, 0xDC)
    sp.line.color.rgb = AMBER; sp.line.width = Pt(1)
    sp.shadow.inherit = False
    tf = sp.text_frame; tf.word_wrap = True
    tf.margin_top = Pt(2); tf.margin_bottom = Pt(2)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = "RUBRIC  " + text
    r.font.size = Pt(11); r.font.bold = True; r.font.color.rgb = AMBER
    r.font.name = "Calibri"


def body_box(s, top=Inches(2.05), left=Inches(0.55), width=Inches(6.0),
             height=Inches(4.7)):
    return textbox(s, left, top, width, height)


def add_figure(s, name, left, top, width):
    path = FIG / name
    if path.exists():
        s.shapes.add_picture(str(path), left, top, width=width)


def caption(s, text, left, top, width):
    tf = textbox(s, left, top, width, Inches(0.4))
    para(tf, text, size=11, color=GREY, italic=True, align=PP_ALIGN.CENTER,
         first=True)


# ---- content model -----------------------------------------------------------
def content_slide(n, kicker, title, rubric, intro, bullets, fig=None,
                  fig_caption=None, note=None):
    s = slide()
    header(s, kicker, title)
    rubric_chip(s, rubric)
    have_fig = fig is not None
    bw = Inches(6.1) if have_fig else Inches(12.2)
    tf = body_box(s, top=Inches(2.05), width=bw)
    if intro:
        para(tf, intro, size=16, color=BLUE, bold=True, first=True, space_after=8)
        first = False
    else:
        first = True
    for b in bullets:
        if isinstance(b, tuple):
            txt, lvl = b
        else:
            txt, lvl = b, 0
        para(tf, txt, size=14 if lvl == 0 else 12.5,
             color=DARK if lvl == 0 else GREY, bullet=True, level=lvl,
             first=first, space_after=5)
        first = False
    if note:
        para(tf, note, size=12, color=TEAL, italic=True, space_after=0)
    if have_fig:
        add_figure(s, fig, Inches(6.9), Inches(2.15), Inches(6.0))
        if fig_caption:
            caption(s, fig_caption, Inches(6.9), Inches(6.55), Inches(6.0))
    footer(s, n)
    return s


def crit_slide(n, crit_no, crit_name, marks, excellent, evidence, target=None):
    """A rubric-criterion slide: Excellent descriptor + project evidence."""
    s = slide()
    header(s, f"RUBRIC CRITERION {crit_no}  ·  {marks} MARKS", crit_name)
    # 'Excellent' descriptor panel
    rect(s, Inches(0.55), Inches(1.5), Inches(12.25), Inches(0.95),
         RGBColor(0xED, 0xF4, 0xEF))
    tf = textbox(s, Inches(0.75), Inches(1.56), Inches(11.9), Inches(0.9))
    para(tf, "EXCELLENT (80–100%) descriptor", size=12, color=TEAL, bold=True,
         first=True, space_after=2)
    para(tf, excellent, size=13, color=DARK, italic=True, space_after=0)
    # evidence
    tf2 = textbox(s, Inches(0.55), Inches(2.65), Inches(12.25), Inches(3.9))
    para(tf2, "Evidence in this project", size=16, color=BLUE, bold=True,
         first=True, space_after=8)
    for b in evidence:
        if isinstance(b, tuple):
            txt, lvl = b
        else:
            txt, lvl = b, 0
        para(tf2, txt, size=13.5 if lvl == 0 else 12,
             color=DARK if lvl == 0 else GREY, bullet=True, level=lvl,
             space_after=5)
    if target:
        from pptx.enum.shapes import MSO_SHAPE
        sp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.55), Inches(6.5), Inches(12.25),
                                Inches(0.45))
        sp.fill.solid(); sp.fill.fore_color.rgb = NAVY
        sp.line.fill.background(); sp.shadow.inherit = False
        p = sp.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
        r = p.add_run(); r.text = "TARGET  " + target
        r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = WHITE
        r.font.name = "Calibri"
    footer(s, n)
    return s


# ============================= TITLE ==========================================
s = slide()
rect(s, 0, 0, EMU_W, EMU_H, NAVY)
rect(s, 0, Inches(4.55), EMU_W, Inches(0.08), TEAL)
tf = textbox(s, Inches(0.9), Inches(1.4), Inches(11.5), Inches(3.0))
para(tf, "IDA117V  ·  INTRODUCTION TO DATA SCIENCE  ·  GROUP PROJECT",
     size=15, color=RGBColor(0x9F, 0xC4, 0xE0), bold=True, first=True,
     space_after=10)
para(tf, "Required Evidence & Experimental Rules", size=40, color=WHITE,
     bold=True, space_after=6)
para(tf, "Section 4 of the project template, answered against the build — "
         "and mapped to the marking rubric", size=20,
     color=RGBColor(0xC9, 0xD7, 0xE4), space_after=0)
tf2 = textbox(s, Inches(0.9), Inches(4.8), Inches(11.5), Inches(2.2))
para(tf2, "Child Support Grant Reach-Gap Prioritiser", size=22, color=TEAL,
     bold=True, first=True, space_after=8)
para(tf2, "Who is eligible for the Child Support Grant but not receiving it?  "
          "StatsSA General Household Survey, person files 2012–2022.",
     size=15, color=RGBColor(0xC9, 0xD7, 0xE4), space_after=4)
para(tf2, "Task: csg_reach_gap  ·  prediction unit: one child 0–17  ·  "
          "primary metric: precision@k / lift  ·  seed = 42",
     size=13, color=RGBColor(0x9F, 0xC4, 0xE0), space_after=0)

# ============================= AGENDA / MAPPING ===============================
s = slide()
header(s, "HOW THIS DECK IS ORGANISED",
       "Section 4 requirements → rubric criteria")
rows = [
    ("Data split integrity", "Crit. 2 — Data audit, EDA & pipeline (18)"),
    ("Model set (baseline + 4: 2 traditional, 2 neural)", "Crit. 3 — Baseline & ≥4 models (22)"),
    ("Core visual evidence", "Crit. 2 & 4 — EDA / evaluation"),
    ("Classification evidence", "Crit. 4 — Evaluation & comparison (16)"),
    ("Optimisation", "Crit. 5 — Champion optimisation (10)"),
    ("Ablation", "Crit. 6 — Ablation & McNemar (10)"),
    ("McNemar validation", "Crit. 6 — Ablation & McNemar (10)"),
    ("Overfitting", "Crit. 4 — Evaluation & overfitting (16)"),
    ("User-interface deployment", "Crit. 7 — UI deployment (7)"),
    ("Reproducibility", "Crit. 8 — Reproducibility & defence (7)"),
]
# two-column table-ish layout
colx = [Inches(0.55), Inches(6.9)]
tops = Inches(2.05)
left_tf = textbox(s, Inches(0.55), Inches(2.0), Inches(6.1), Inches(4.9))
right_tf = textbox(s, Inches(6.9), Inches(2.0), Inches(6.0), Inches(4.9))
half = (len(rows) + 1) // 2
for i, (req, crit) in enumerate(rows):
    tf = left_tf if i < half else right_tf
    first = (i == 0) or (i == half)
    para(tf, req, size=15, color=NAVY, bold=True, bullet=True, first=first,
         space_after=1)
    para(tf, crit, size=12, color=AMBER, level=1, space_after=8)
footer(s, 2)

# ============================= 1. SPLIT INTEGRITY =============================
content_slide(
    3, "REQUIREMENT 1 — DATA SPLIT INTEGRITY",
    "Split before fit, on identical held-out cases",
    "Criterion 2 (18) — leakage-safe pipeline; Criterion 4 (16) — shared held-out cases",
    "Split by survey YEAR (group/time split) — no household leaks across sets.",
    [
        "Train 2012–2018 = 71,080  ·  Validation 2019–2020 = 12,849  ·  Test 2021–2022 = 13,461",
        ("Source: hyperparameters_csg_reach_gap.json (split, n)", 1),
        "Split-before-fit: imputer, encoder and scaler fitted on TRAIN years only, then applied forward",
        ("Fitted transforms persisted in preprocessors_* and optimised_bundle_*.joblib — re-used, never re-fitted", 1),
        "Shared held-out cases: every final comparison (table, ROC, confusion, McNemar) uses the same locked test set",
        "Test read exactly ONCE — final_locked row 'TEST 2021-2022 (locked, once)'",
    ],
    fig="F6_temporal_split.png",
    fig_caption="F6 — temporal train / validation / test split",
)

# ============================= 2. MODEL SET ===================================
content_slide(
    4, "REQUIREMENT 2 — MODEL SET",
    "Naïve baseline + 4 substantive models",
    "Criterion 3 (22) — compliant 2+2 model set, seeds & hyperparameters disclosed",
    "Two traditional + two neural, plus a transparent naïve baseline. Seed = 42 throughout.",
    [
        "Baseline — DummyClassifier (prior strategy)",
        "Traditional 1 — Logistic Regression (L2, C=1.0, class_weight=balanced)",
        "Traditional 2 — Random Forest (400 trees, depth 18, leaf 10, balanced_subsample)",
        "Neural 1 — MLP 256-128-64, dropout 0.3, BatchNorm, AdamW lr 1e-3, pos_weight 5.49, 55,169 params",
        "Neural 2 — TabTransformer (d_model 32, 4 heads, 2 blocks, 74,943 params)",
        "All hyperparameters + seeds disclosed in hyperparameters_csg_reach_gap.json",
        ("Saved artefacts under analysis/out/models/ (.joblib and .pt)", 1),
    ],
    note="Rationale: linear vs tree vs two neural families bracket the hypothesis space on tabular survey data.",
)

# ============================= 3. CORE VISUAL EVIDENCE ========================
s = slide()
header(s, "REQUIREMENT 3 — CORE VISUAL EVIDENCE",
       "Every required figure is produced")
rubric_chip(s, "Criterion 2 (18) & Criterion 4 (16) — visual evidence")
tf = textbox(s, Inches(0.55), Inches(2.0), Inches(12.3), Inches(0.5))
para(tf, "15 figures (F1–F15) cover EDA, comparison, confusion, ROC, learning "
         "curves, calibration, targeting, ablation and optimisation.",
     size=15, color=BLUE, bold=True, first=True)
grid = [
    ("F10_comparison_csg_reach_gap.png", "F10 — model comparison"),
    ("F9_roc_csg_reach_gap.png", "F9 — ROC (common figure)"),
    ("F8_confusion_csg_reach_gap.png", "F8 — confusion matrices"),
    ("F13_targeting_csg_reach_gap.png", "F13 — targeting / lift"),
]
positions = [(Inches(0.55), Inches(2.6)), (Inches(3.75), Inches(2.6)),
             (Inches(0.55), Inches(4.9)), (Inches(3.75), Inches(4.9))]
for (fn, cap), (x, y) in zip(grid, positions):
    add_figure(s, fn, x, y, Inches(3.0))
    caption(s, cap, x, y + Inches(2.0), Inches(3.0))
tf2 = textbox(s, Inches(7.1), Inches(2.6), Inches(5.8), Inches(4.3))
para(tf2, "Also produced", size=15, color=NAVY, bold=True, first=True, space_after=6)
for t in ["F1 target distribution", "F2 missingness / data quality",
          "F3 age vs target", "F4 medical aid & education vs target",
          "F5 geographic", "F6 temporal split", "F7 subgroup",
          "F11 neural loss/metric curves", "F12 calibration",
          "F14 ablation", "F15 optimisation"]:
    para(tf2, t, size=13, color=DARK, bullet=True, space_after=4)
footer(s, 5)

# ============================= 4. CLASSIFICATION EVIDENCE =====================
s = slide()
header(s, "REQUIREMENT 4 — CLASSIFICATION EVIDENCE",
       "Per-model metrics, class-sensitive reading")
rubric_chip(s, "Criterion 4 (16) — consistent multi-metric evaluation")
# table
from pptx.util import Cm
rows_tbl = [
    ["Model", "Acc", "Macro-F1", "Prec", "Rec", "ROC-AUC", "PR-AUC"],
    ["Dummy", "0.888", "0.470", "0.00", "0.00", "0.500", "0.112"],
    ["LogReg", "0.602", "0.500", "0.17", "0.67", "0.680", "0.231"],
    ["Rand. Forest", "0.769", "0.589", "0.24", "0.48", "0.705", "0.261"],
    ["MLP", "0.660", "0.537", "0.19", "0.65", "0.702", "0.257"],
    ["TabTransformer", "0.526", "0.456", "0.16", "0.75", "0.687", "0.244"],
]
nr, nc = len(rows_tbl), len(rows_tbl[0])
gt = s.shapes.add_table(nr, nc, Inches(0.55), Inches(2.1), Inches(7.2),
                        Inches(2.7)).table
widths = [Inches(1.9)] + [Inches(0.88)] * (nc - 1)
for c, w in enumerate(widths):
    gt.columns[c].width = w
for r in range(nr):
    for c in range(nc):
        cell = gt.cell(r, c)
        cell.text = rows_tbl[r][c]
        pr = cell.text_frame.paragraphs[0]
        pr.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        run = pr.runs[0]; run.font.size = Pt(11)
        run.font.name = "Calibri"
        if r == 0:
            run.font.bold = True; run.font.color.rgb = WHITE
            cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        else:
            run.font.color.rgb = DARK
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if r % 2 else WHITE
tf = textbox(s, Inches(0.55), Inches(5.0), Inches(12.3), Inches(2.0))
para(tf, "Held-out TEST metrics (results_test_csg_reach_gap.csv)", size=12,
     color=GREY, italic=True, first=True, space_after=8)
para(tf, "Class-sensitive reading: positives (eligible-but-not-reached) are ~11%, "
         "so accuracy misleads — Dummy scores 0.888 accuracy with 0 precision and 0 recall.",
     size=14, color=DARK, bullet=True, space_after=5)
para(tf, "Decision is capacity-constrained → report PR-AUC and lift@k. On the locked test "
         "the champion finds 3.5× (lift@5%) and 2.77× (lift@10%) more missed children than random.",
     size=14, color=TEAL, bullet=True, space_after=0)
footer(s, 6)

# ============================= 5. OPTIMISATION ================================
content_slide(
    7, "REQUIREMENT 5 — OPTIMISATION",
    "Validation-led search, test read once",
    "Criterion 5 (10) — validation-led search; honest pre/post & one-time test",
    "Only the strongest validation model (the MLP) is tuned. Random search, 40 trials, seed 42.",
    [
        "Scoring rule: survey-weighted PR-AUC on validation years 2019–2020",
        "Space: hidden sizes, dropout 0–0.5, lr, weight decay, batch, pos_weight, min_freq, drop_constant, use_sensitive",
        "Selected: hidden (512,256,128), dropout 0.1, lr 2e-3, wd 1e-5, batch 2048, pos_weight 3.0, drop_constant True",
        "Pre → post (validation): PR-AUC 0.235 → 0.244 ; macro-F1 0.523 → 0.587 ; lift@5% 3.07 → 3.34",
        "Locked test, read ONCE: ROC-AUC 0.691, PR-AUC 0.271, lift@5% 3.52, lift@10% 2.77",
        "Stability: leave-one-year-out CV PR-AUC 0.307 ± 0.035 over 7 folds",
    ],
    fig="F15_optimisation_csg_reach_gap.png",
    fig_caption="F15 — optimisation search & result",
    note="Test years 2021–2022 appear nowhere in the search — practically meaningful gain, not test tuning.",
)

# ============================= 6. ABLATION ===================================
content_slide(
    8, "REQUIREMENT 6 — ABLATION",
    "Nine one-change-at-a-time experiments",
    "Criterion 6 (10) — ≥3 controlled ablations with insightful deltas",
    "Split, seed and protocol held fixed against locked champion A0 (exceeds the minimum of 3).",
    [
        "A1 − household context:  ΔPR-AUC −0.017, Δlift@10% −0.21  (largest useful loss)",
        "A6 − dropout (0.3→0.0):  Δlift@10% −0.25  (regulariser matters most for lift)",
        "A4 − geography:          ΔPR-AUC −0.012, Δlift@10% −0.19",
        "A5 − imbalance weight:   ΔPR-AUC −0.010  (helps rare-positive recall)",
        "A8 − feature scaling:    ΔPR-AUC −0.014",
        "A2 − 5 constant features: ΔPR-AUC −0.003  (they ARE the eligibility filter → dropped)",
        "A3 − sensitive attrs:    ΔPR-AUC −0.005  (cheap to remove → fairness lever)",
    ],
    fig="F14_ablation_csg_reach_gap.png",
    fig_caption="F14 — ablation deltas (ablation_csg_reach_gap.csv)",
    note="Depends on: household context, geography, imbalance handling, dropout. Search picked drop_constant=True independently.",
)

# ============================= 7. McNEMAR ====================================
s = slide()
header(s, "REQUIREMENT 7 — McNEMAR VALIDATION",
       "Paired test on identical locked cases")
rubric_chip(s, "Criterion 6 (10) — correct paired test, table, p-value, conclusion")
tf = textbox(s, Inches(0.55), Inches(2.0), Inches(6.1), Inches(4.8))
para(tf, "Optimised champion vs strongest comparator (Random Forest), both calibrated.",
     size=15, color=BLUE, bold=True, first=True, space_after=8)
para(tf, "H₀: equal error rates (discordant counts symmetric)", size=14, color=DARK,
     bullet=True, space_after=3)
para(tf, "H₁: error rates differ     α = 0.05", size=14, color=DARK, bullet=True,
     space_after=8)
para(tf, "Q6.2a — 0.5 threshold:  stat 3.613,  p = 0.057  → fail to reject H₀",
     size=14, color=NAVY, bullet=True, space_after=3)
para(tf, "Q6.2b — operational rule (top 10%):  stat 0.050,  p = 0.82  → fail to reject H₀",
     size=14, color=NAVY, bullet=True, space_after=8)
para(tf, "Statistical vs practical: models are statistically indistinguishable, "
         "yet the champion ranks missed children better (PR-AUC 0.234→0.257; "
         "lift@10% 2.59→2.77) ≈ 24,000 more children found per round.",
     size=13, color=TEAL, bullet=True, space_after=0)
# 2x2 table
rows_tbl = [
    ["0.5 threshold", "Comp. correct", "Comp. wrong"],
    ["Champ correct", "11,935", "49"],
    ["Champ wrong", "31", "1,446"],
]
gt = s.shapes.add_table(3, 3, Inches(7.1), Inches(2.5), Inches(5.6),
                        Inches(2.1)).table
for r in range(3):
    for c in range(3):
        cell = gt.cell(r, c); cell.text = rows_tbl[r][c]
        p0 = cell.text_frame.paragraphs[0]
        p0.alignment = PP_ALIGN.CENTER
        run = p0.runs[0]; run.font.size = Pt(14); run.font.name = "Calibri"
        if r == 0 or c == 0:
            run.font.bold = True; run.font.color.rgb = WHITE
            cell.fill.solid(); cell.fill.fore_color.rgb = BLUE
        else:
            run.font.color.rgb = DARK; cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT
caption(s, "2×2 correctness table — mcnemar_optimised_csg_reach_gap.csv",
        Inches(7.1), Inches(4.65), Inches(5.6))
footer(s, 9)

# ============================= 8. OVERFITTING ================================
content_slide(
    10, "REQUIREMENT 8 — OVERFITTING",
    "Quantified train–validation gap + controls",
    "Criterion 4 (16) — quantified overfitting diagnosis & controls",
    "Train-vs-validation AUC gap, with neural learning curves (results_overfit_csg_reach_gap.csv).",
    [
        "Logistic Regression: train 0.657 / val 0.663  → gap −0.006",
        "Random Forest:       train 0.849 / val 0.684  → gap 0.165  (overfits)",
        "MLP (PyTorch):       train 0.701 / val 0.685  → gap 0.016  (generalises)",
        "TabTransformer:      train 0.664 / val 0.670  → gap −0.006",
        "Diagnosis: RF overfits; MLP chosen for optimisation partly for its small gap",
        "Controls: dropout, weight decay, BatchNorm, grad-clip, early stopping (MLP stopped epoch 4/40, patience 6)",
        ("Ablation A6 confirms dropout is load-bearing (−0.25 lift@10% without it)", 1),
    ],
    fig="F11_neural_curves_csg_reach_gap.png",
    fig_caption="F11 — neural loss / metric curves",
)

# ============================= 9. UI DEPLOYMENT ==============================
content_slide(
    11, "REQUIREMENT 9 — USER-INTERFACE DEPLOYMENT",
    "Streamlit app loads the saved pipeline",
    "Criterion 7 (7) — runs saved end-to-end pipeline; validates input; safeguards",
    "analysis/app.py  ·  launch: cd analysis && streamlit run app.py  ·  health /_stcore/health → ok",
    [
        "Loads saved artefacts only — nothing re-fitted at inference:",
        ("optimised_bundle (preprocessor + 19-feature contract), mlp_optimised.pt, calibrators.joblib", 1),
        "Returns calibrated probability + lift-based risk band (Very high/High/Elevated/Typical) + outreach percentile",
        "Plain-language explanation; model selector to compare against the other models",
        "Input validation with specific error messages; responsible-use notice in the guide tab",
        "Acceptance tests (ui_acceptance_tests.csv): 4 valid cases scored + 5 invalid/out-of-scope refused",
        ("Exceeds template minimum of 3 valid + 1 invalid case", 1),
    ],
    note="Refusals include: age outside 0–17; bad province; missing field; household has an earner; on medical aid.",
)

# ============================= 10. REPRODUCIBILITY ===========================
content_slide(
    12, "REQUIREMENT 10 — REPRODUCIBILITY",
    "Runs from documented instructions",
    "Criterion 8 (7) — complete code & deployment reproducibility",
    "No hidden steps, no secrets, no local-only paths.",
    [
        "Execution order documented in analysis/README.md §2 (15 numbered scripts)",
        "No absolute paths — every script resolves paths relative to itself",
        "Task chosen by env var: GHS_TASK=csg_reach_gap",
        "Fixed seeds: SEED = 42 everywhere (also recorded in the optimisation JSON)",
        "Environment: requirements.txt — Python 3.14.3, pandas 2.3.3, scikit-learn 1.8.0, torch 2.14.1+cpu, streamlit 1.52.1",
        "No credentials / no personal data; GHS microdata de-identified; all compute local",
        "Caveat: *.csv and *.joblib via Git LFS — run `git lfs install && git lfs checkout` first (README §6a)",
    ],
)

# ============================= RUBRIC SECTION DIVIDER ========================
s = slide()
rect(s, 0, 0, EMU_W, EMU_H, NAVY)
rect(s, 0, Inches(3.4), EMU_W, Inches(0.08), AMBER)
tf = textbox(s, Inches(0.9), Inches(2.4), Inches(11.5), Inches(2.6))
para(tf, "SECTION 5", size=16, color=AMBER, bold=True, first=True, space_after=6)
para(tf, "Marking Rubric — criterion by criterion", size=34, color=WHITE,
     bold=True, space_after=10)
para(tf, "Each of the 8 criteria: the Excellent descriptor, then the project's "
         "evidence against it. Target: full marks, 100/100.",
     size=16, color=RGBColor(0xC9, 0xD7, 0xE4), space_after=0)
footer(s, 13)

# ---- Criterion 1 -------------------------------------------------------------
crit_slide(
    14, 1, "Problem, stakeholder, data & ethics", 10,
    "Compelling service-delivery need; precise task and costs; authoritative, "
    "well-licensed dataset; excellent contextual, POPIA and fairness reasoning.",
    [
        "Need: ~750,000 children (2022) near-certainly eligible for the Child Support Grant but not receiving it; outreach capacity is finite",
        "Stakeholder & decision: SASSA / DSD deciding which households to visit first — 'right grant, right person, right time and place'",
        "Precise task: binary classification; unit = one child 0–17; target 1 = eligible but NOT receiving; population = households with no earner and no medical aid",
        "Costs: false positive = a wasted visit to an already-reached household; false negative = a genuinely missed child stays missed",
        "Data: StatsSA General Household Survey 2012–2022, person files — authoritative national survey, de-identified, lawful reuse",
        "POPIA / privacy: microdata de-identified; uqnr is a household survey id, not a person id; all compute local; no records uploaded",
        "Fairness: race and sex retained but ablation A3 shows removing them costs only 0.005 PR-AUC — a ready fairness lever; output is population-level, never an individual adjudication",
    ],
    target="Compelling need + precise task/costs + licensed data + POPIA & fairness reasoning → aiming 10/10",
)

# ---- Criterion 2 -------------------------------------------------------------
crit_slide(
    15, 2, "Data audit, EDA, visualisation & pipeline", 18,
    "Thorough audit; insightful, accessible visuals; rigorous split; leakage-safe, "
    "reproducible preparation with strong justification.",
    [
        "Data dictionary: 24 modelling features + target + survey-design columns documented (data_dictionary.csv) with type, source, levels, non-null %",
        "Audit: class balance (positives ~11% in-scope), duplicates, missingness BY SPLIT (audit_missingness_by_split.csv), invalid values, leakage review",
        "Missingness handled honestly: survey skip patterns respected (lab_status hidden <15, attendance <5) — not imputed as data",
        "7 EDA figures F1–F7: target distribution, missingness, two feature–target views, geographic, temporal and subgroup",
        "Rigorous split: by survey YEAR (group/time), fitted on train only — no household leaks (F6)",
        "Reproducible pipeline: imputation, encoding, scaling, 14 engineered household features, imbalance treatment — all inside persisted preprocessors",
    ],
    target="Thorough audit + insightful visuals + rigorous leakage-safe pipeline → aiming 18/18",
)

# ---- Criterion 3 -------------------------------------------------------------
crit_slide(
    16, 3, "Baseline & ≥4 models", 22,
    "Naïve baseline plus compliant, reproducible 2+2 model set; strong rationales "
    "and complete neural/training detail.",
    [
        "Naïve baseline: DummyClassifier (prior) — establishes the no-skill floor (0.888 accuracy, 0 precision/recall on the rare class)",
        "Traditional ×2: Logistic Regression + Random Forest",
        "Neural ×2: MLP (256-128-64) + TabTransformer — the task-appropriate tabular architecture",
        "Trained on identical year-based splits; seed = 42 for every model",
        "Complete neural detail: loss (BCEWithLogits, pos_weight), AdamW, lr, weight decay, batch, epochs, early stopping, grad-clip, param counts — hyperparameters_csg_reach_gap.json",
        "Rationale per family: linear vs non-linear tree vs dense neural vs attention-over-features brackets the hypothesis space on tabular survey data",
        "Reproducible: all five models saved under analysis/out/models/",
    ],
    target="Compliant 2+2 + baseline + full training detail + strong rationale → aiming 22/22",
)

# ---- Criterion 4 -------------------------------------------------------------
crit_slide(
    17, 4, "Evaluation, comparison & overfitting", 16,
    "Consistent multi-metric evaluation; complete reports/matrices/ROC; excellent "
    "visual comparison and quantified overfitting diagnosis/controls.",
    [
        "Same held-out metrics for every model: accuracy, macro-F1, weighted-F1, precision, recall, ROC-AUC, PR-AUC (results_test_*.csv)",
        "PR-AUC included because positives are ~11% — the imbalance-aware metric that matches a capacity-constrained decision",
        "Confusion matrix per model (F8); ROC curves on one common figure (F9); comparison table + bar chart (F10)",
        "Champion selected on VALIDATION, not test; stability via leave-one-year-out CV (PR-AUC 0.307 ± 0.035, 7 folds)",
        "Overfitting quantified: train–val AUC gap RF 0.165 (overfits) vs MLP 0.016 (generalises)",
        "Controls: dropout, weight decay, BatchNorm, grad-clip, early stopping (epoch 4/40); neural loss/metric curves (F11)",
    ],
    target="Consistent metrics + full reports/ROC + quantified overfitting & controls → aiming 16/16",
)

# ---- Criterion 5 -------------------------------------------------------------
crit_slide(
    18, 5, "Champion optimisation", 10,
    "Validation-led, well-designed search; reproducible configuration; honest "
    "pre/post and one-time test evaluation with practical interpretation.",
    [
        "Only the strongest validation model (MLP) tuned — random search, 40 trials, seed 42",
        "Scoring rule: survey-weighted PR-AUC on validation 2019–2020; documented 9-dimension search space",
        "Selected config disclosed: (512,256,128), dropout 0.1, lr 2e-3, wd 1e-5, batch 2048, pos_weight 3.0, drop_constant True",
        "Honest pre/post on validation: PR-AUC 0.235→0.244, macro-F1 0.523→0.587, lift@5% 3.07→3.34",
        "Test read ONCE for the locked result: ROC-AUC 0.691, PR-AUC 0.271, lift@10% 2.77",
        "Practical meaning: ≈ 24,000 additional children found per outreach round vs the pre-optimisation model",
    ],
    target="Validation-led search + reproducible config + honest one-time test + practical reading → aiming 10/10",
)

# ---- Criterion 6 -------------------------------------------------------------
crit_slide(
    19, 6, "Ablation & McNemar validation", 10,
    "≥3 controlled ablations with insightful deltas; correct paired test, "
    "assumptions, contingency table, p-value and nuanced conclusion.",
    [
        "9 one-change ablations (minimum is 3), split/seed/protocol fixed vs locked champion A0 (ablation_csg_reach_gap.csv, F14)",
        "Insightful deltas: dropout removed −0.25 lift@10%; household context −0.21; geography −0.19; constant features ~0 (they ARE the filter)",
        "McNemar on identical locked cases, BOTH models calibrated (fixes an earlier apples-to-oranges comparison)",
        "H₀ equal error rates; H₁ differ; α = 0.05; 2×2 table reported (both correct 11,935; champ-only 49; comp-only 31; both wrong 1,446)",
        "Result: 0.5 threshold stat 3.613 p=0.057; operational rule stat 0.050 p=0.82 → fail to reject H₀",
        "Nuanced conclusion: statistically indistinguishable, yet champion preferred on practical ranking (PR-AUC & lift)",
    ],
    target="9 ablations + correct calibrated McNemar + nuanced stat-vs-practical reading → aiming 10/10",
)

# ---- Criterion 7 -------------------------------------------------------------
crit_slide(
    20, 7, "User-interface deployment", 7,
    "Working, polished interface runs the saved end-to-end pipeline; validates "
    "input; presents useful predictions and confidence; includes safeguards and "
    "fully reproducible deployment.",
    [
        "Streamlit app (analysis/app.py) loads SAVED artefacts only — optimised bundle, optimised MLP weights, isotonic calibrators; nothing re-fitted",
        "Single-household and batch-upload tabs; roster-based household features derived exactly as in training",
        "Output: calibrated probability + lift-based risk band + outreach percentile + plain-language explanation",
        "Safeguards: input validation with specific messages, skip-pattern-aware fields, responsible-use notice, no secrets/no personal data",
        "Acceptance tests: 4 valid cases scored + 5 invalid/out-of-scope refused (exceeds 3 valid + 1 invalid minimum)",
        "Reproducible launch: cd analysis && streamlit run app.py; health check /_stcore/health → ok",
    ],
    target="Polished interface on saved pipeline + validation + safeguards + reproducible launch → aiming 7/7",
)

# ---- Criterion 8 -------------------------------------------------------------
crit_slide(
    21, 8, "Conclusions, reproducibility & team defence", 7,
    "Actionable, cautious recommendation; transparent limitations/fairness; "
    "complete code and deployment reproducibility; polished defence and credible "
    "shared contribution.",
    [
        "Actionable recommendation: use for population-level outreach prioritisation — visit the highest-percentile eligible households first",
        "Cautious conditions: must NOT adjudicate, approve, reduce or deny an individual's grant; carries no legal standing",
        "Limitations/uncertainty stated: ROC-AUC ~0.69 on a genuinely hard reframed target; survey-based, not an administrative register",
        "Fairness: sensitive-attribute ablation quantifies the cost of dropping race/sex; monitoring needs flagged",
        "Reproducibility: README execution order, requirements.txt, seed=42, no absolute paths, no secrets; Git LFS caveat documented",
        "Package complete: README, data dictionary, saved models/pipeline, UI code, figures, metric tables, McNemar predictions",
        "Team defence: suggested roles map to the pipeline; every member can defend the full flow",
    ],
    target="Actionable + cautious + fully reproducible package + credible defence → aiming 7/7",
)

# ============================= RUBRIC SUMMARY ================================
s = slide()
header(s, "RUBRIC COVERAGE", "All 8 criteria answered — target 100/100")
tf = textbox(s, Inches(0.55), Inches(2.0), Inches(12.3), Inches(0.5))
para(tf, "Every criterion has verifiable evidence — no criterion sits at zero.",
     size=14, color=BLUE, bold=True, first=True)
rows_tbl = [
    ["Rubric criterion", "Marks", "Headline evidence"],
    ["1  Problem, stakeholder, data & ethics", "10", "750k missed children; precise task/costs; StatsSA GHS; POPIA & fairness"],
    ["2  Data audit, EDA & pipeline", "18", "Data dictionary, audit by split, F1–F7, leakage-safe year split"],
    ["3  Baseline & ≥4 models", "22", "Dummy + 2 traditional + 2 neural; seeds & full training detail"],
    ["4  Evaluation & overfitting", "16", "Per-model metrics, ROC/confusion, train–val gap, curves"],
    ["5  Champion optimisation", "10", "Random search, pre/post, locked one-time test"],
    ["6  Ablation & McNemar", "10", "9 ablations; calibrated McNemar p=0.057 / 0.82"],
    ["7  UI deployment", "7", "Streamlit loads saved pipeline; 4 valid + 5 invalid cases"],
    ["8  Reproducibility & defence", "7", "README order, seeds, requirements, no secrets"],
    ["TOTAL", "100", "Target: full marks"],
]
nr, nc = len(rows_tbl), 3
gt = s.shapes.add_table(nr, nc, Inches(0.55), Inches(2.5), Inches(12.25),
                        Inches(4.1)).table
gt.columns[0].width = Inches(3.95)
gt.columns[1].width = Inches(0.95)
gt.columns[2].width = Inches(7.35)
for r in range(nr):
    is_total = (r == nr - 1)
    for c in range(nc):
        cell = gt.cell(r, c); cell.text = rows_tbl[r][c]
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p0 = cell.text_frame.paragraphs[0]
        p0.alignment = PP_ALIGN.CENTER if c == 1 else PP_ALIGN.LEFT
        run = p0.runs[0]; run.font.size = Pt(11.5); run.font.name = "Calibri"
        if r == 0:
            run.font.bold = True; run.font.color.rgb = WHITE
            cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        elif is_total:
            run.font.bold = True; run.font.color.rgb = WHITE
            cell.fill.solid(); cell.fill.fore_color.rgb = TEAL
        else:
            run.font.color.rgb = DARK; cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if r % 2 else WHITE
footer(s, 22)

# ============================= CLOSING ======================================
s = slide()
rect(s, 0, 0, EMU_W, EMU_H, NAVY)
rect(s, 0, Inches(3.4), EMU_W, Inches(0.08), TEAL)
tf = textbox(s, Inches(0.9), Inches(2.3), Inches(11.5), Inches(3.0))
para(tf, "Every requirement is backed by a saved artefact", size=30, color=WHITE,
     bold=True, first=True, space_after=10)
para(tf, "Split integrity · 4 models + baseline · full figure set · class-sensitive "
         "metrics · validation-led optimisation · 9 ablations · McNemar · "
         "overfitting controls · deployed UI · reproducible repo",
     size=16, color=RGBColor(0xC9, 0xD7, 0xE4), space_after=12)
para(tf, "Numbers read from analysis/out/  ·  seed = 42  ·  test read once",
     size=14, color=TEAL, bold=True, space_after=0)

out = ROOT / "IDA117V_Section4_Evidence.pptx"
try:
    prs.save(str(out))
except PermissionError:
    out = ROOT / "IDA117V_Section4_and_Rubric_Evidence.pptx"
    prs.save(str(out))
    print("NOTE: original file was locked (open in PowerPoint); saved under a new name.")
print("Saved:", out)
print("Slides:", len(prs.slides._sldIdLst))
