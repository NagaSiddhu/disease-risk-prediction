import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BLUE, LBLUE, ORANGE, LORANGE, GREY = "#2F6DB5", "#DCE8F7", "#E07B39", "#FBE6D8", "#555B66"


def box(ax, x, y, w, h, text, fc=LBLUE, ec=BLUE, size=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", color="#1b1f27")


def arrow(ax, p, q, color=GREY):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color))


# ---------------- Fig 1.1 categories of risk factors
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
box(ax, 3.7, 2.55, 2.6, 1.0, "Disease\nrisk factors", fc=LORANGE, ec=ORANGE, size=11, bold=True)
box(ax, 3.4, 4.7, 3.2, 1.3, "Non-modifiable\n• Age and sex\n• Family history\n• Genetics", size=9)
box(ax, 0.1, 2.3, 3.1, 1.5, "Lifestyle (modifiable)\n• Diet and obesity\n• Physical inactivity\n• Smoking, alcohol", size=9)
box(ax, 6.8, 2.3, 3.1, 1.5, "Clinical measurements\n• Blood pressure, glucose\n• Cholesterol, BMI\n• Heart rate, ECG", size=9)
box(ax, 3.1, 0.1, 3.8, 1.4, "Laboratory / imaging findings\n• Cell-nucleus measurements\n• Tumour size and texture\n• Biopsy and scan features", size=9)
arrow(ax, (5, 4.65), (5, 3.6)); arrow(ax, (3.25, 3.05), (3.65, 3.05))
arrow(ax, (6.75, 3.05), (6.35, 3.05)); arrow(ax, (5, 1.55), (5, 2.5))
plt.tight_layout(); plt.savefig("figures/fig_risk_factors.png", dpi=200); plt.close()

# ---------------- Fig 1.2 screening workflow
fig, ax = plt.subplots(figsize=(9, 2.9))
ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")
steps = ["Patient\nrecords /\nmeasurements", "Pre-processing\n(cleaning,\nscaling)", "ML risk model\n(trained on\nlabelled cases)",
         "Risk score\nLow / High", "Clinician review\nand early\nintervention"]
xs = [0.1, 2.5, 4.9, 7.3, 9.7]
for i, (x, s) in enumerate(zip(xs, steps)):
    box(ax, x, 0.9, 2.1, 1.5, s, fc=LORANGE if i == 2 else LBLUE, ec=ORANGE if i == 2 else BLUE,
        size=9, bold=(i == 2))
    if i < len(xs) - 1:
        arrow(ax, (x + 2.15, 1.65), (xs[i + 1] - 0.05, 1.65))
ax.text(6, 0.35, "Model output supports, but never replaces, the clinician's decision", ha="center",
        fontsize=9, style="italic", color=GREY)
plt.tight_layout(); plt.savefig("figures/fig_workflow.png", dpi=200); plt.close()

# ---------------- Fig 3.1 architecture
fig, ax = plt.subplots(figsize=(7.2, 9.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 13.4); ax.axis("off")


def group(y0, y1, label):
    ax.add_patch(Rectangle((0.9, y0), 8.6, y1 - y0, fill=False, ls="--", ec="#333", lw=1.2))
    ax.add_patch(Rectangle((0.05, y0), 0.7, y1 - y0, fc=BLUE, ec=BLUE))
    ax.text(0.4, (y0 + y1) / 2, label, rotation=90, ha="center", va="center", color="white",
            fontsize=9, fontweight="bold")


group(11.1, 13.2, "Data Collection")
box(ax, 1.3, 11.4, 3.5, 1.4, "Breast Cancer Wisconsin\n(Diagnostic) dataset\n569 patients, 30 features")
box(ax, 5.5, 11.4, 3.6, 1.4, "Label annotation\nHigh risk = 1 (malignant)\nLow risk = 0 (benign)")
arrow(ax, (4.85, 12.1), (5.45, 12.1))
arrow(ax, (5, 11.0), (5, 10.55))

group(7.9, 10.5, "Pre-processing")
box(ax, 1.3, 9.3, 3.5, 0.9, "Missing-value and\nduplicate check")
box(ax, 5.5, 9.3, 3.6, 0.9, "Stratified split\n80% train / 20% test")
box(ax, 1.3, 8.1, 3.5, 0.9, "Correlation analysis and\nfeature inspection")
box(ax, 5.5, 8.1, 3.6, 0.9, "Standardisation\n(fitted on training data only)")
arrow(ax, (4.85, 9.75), (5.45, 9.75)); arrow(ax, (7.3, 9.25), (7.3, 9.05))
arrow(ax, (5.45, 8.55), (4.85, 8.55))
arrow(ax, (5, 7.8), (5, 7.3))

group(3.7, 7.2, "Proposed Model")
box(ax, 1.3, 6.0, 1.9, 0.9, "Logistic\nRegression", size=8.5)
box(ax, 3.4, 6.0, 1.9, 0.9, "Support Vector\nMachine", size=8.5)
box(ax, 5.5, 6.0, 1.9, 0.9, "Random\nForest", size=8.5)
box(ax, 7.6, 6.0, 1.7, 0.9, "Gradient\nBoosting", size=8.5)
box(ax, 1.3, 4.7, 8.0, 0.8, "Hyper-parameter tuning: GridSearchCV, 5-fold, F1 scoring", size=9)
box(ax, 2.2, 3.85, 6.2, 0.6, "Soft-voting ensemble (averaged probabilities)  →  risk score",
    fc=LORANGE, ec=ORANGE, size=9, bold=True)
for x in (2.25, 4.35, 6.45, 8.45):
    arrow(ax, (x, 5.95), (x, 5.55))
arrow(ax, (5.3, 4.65), (5.3, 4.5))
arrow(ax, (5, 3.6), (5, 3.1))

group(0.3, 3.0, "Evaluation")
box(ax, 1.3, 1.7, 3.5, 0.9, "Accuracy · Precision\nRecall · F1-score")
box(ax, 5.5, 1.7, 3.6, 0.9, "ROC-AUC · Precision–Recall\ncurve")
box(ax, 1.3, 0.5, 3.5, 0.9, "Confusion matrix")
box(ax, 5.5, 0.5, 3.6, 0.9, "Cross-validation and\nfeature importance")
plt.tight_layout(); plt.savefig("figures/fig_architecture.png", dpi=200); plt.close()
print("done")
