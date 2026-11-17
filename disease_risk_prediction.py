# Disease Risk Prediction using Machine Learning
# Dataset: Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, GridSearchCV, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             precision_recall_curve, average_precision_score)

warnings.filterwarnings("ignore")
SEED = 42
FIG = "figures/"

# ---------------------------------------------------------------- 1. Load data
raw = load_breast_cancer(as_frame=True)
X = raw.data
y = 1 - raw.target            # 1 = malignant (high risk), 0 = benign (low risk)
print("Shape:", X.shape, "| Missing:", int(X.isna().sum().sum()),
      "| Duplicates:", int(X.duplicated().sum()))
print(y.value_counts().rename({1: "High risk", 0: "Low risk"}))

# ---------------------------------------------------------------- 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

# ---------------------------------------------------------------- 3. Models + tuning
def pipe(clf):
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])

search_space = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=5000)),
                            {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "KNN": (pipe(KNeighborsClassifier()),
            {"clf__n_neighbors": [3, 5, 7, 9, 11], "clf__weights": ["uniform", "distance"]}),
    "Naive Bayes": (pipe(GaussianNB()),
                    {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]}),
    "Decision Tree": (pipe(DecisionTreeClassifier(random_state=SEED)),
                      {"clf__max_depth": [3, 5, 7, None], "clf__min_samples_leaf": [1, 3, 5]}),
    "SVM": (pipe(SVC(probability=True, random_state=SEED)),
            {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]}),
    "Random Forest": (pipe(RandomForestClassifier(random_state=SEED)),
                      {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10],
                       "clf__min_samples_leaf": [1, 2]}),
    "Gradient Boosting": (pipe(GradientBoostingClassifier(random_state=SEED)),
                          {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                           "clf__max_depth": [2, 3]}),
    "MLP": (pipe(MLPClassifier(max_iter=2000, random_state=SEED)),
            {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-4, 1e-2]}),
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
tuned, best_params = {}, {}
for name, (model, grid) in search_space.items():
    gs = GridSearchCV(model, grid, cv=inner_cv, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned[name] = gs.best_estimator_
    best_params[name] = {k.replace("clf__", ""): (v if not isinstance(v, tuple) else list(v))
                         for k, v in gs.best_params_.items()}
    print(f"{name:20s} best CV F1 = {gs.best_score_:.4f}  {best_params[name]}")

# Proposed model: soft-voting ensemble of the tuned LR, SVM, RF and GB models
proposed_name = "Proposed Ensemble"
tuned[proposed_name] = VotingClassifier(
    estimators=[("lr", clone(tuned["Logistic Regression"])),
                ("svm", clone(tuned["SVM"])),
                ("rf", clone(tuned["Random Forest"])),
                ("gb", clone(tuned["Gradient Boosting"]))],
    voting="soft")

# ---------------------------------------------------------------- 4. Evaluate
results, cv_results, probs = {}, {}, {}
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
for name, model in tuned.items():
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    probs[name] = p
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, p),
    }
    cv = cross_validate(clone(model), X_train, y_train, cv=outer_cv,
                        scoring=["accuracy", "f1", "recall"], n_jobs=-1)
    cv_results[name] = {k: (float(cv[f"test_{k}"].mean()), float(cv[f"test_{k}"].std()))
                        for k in ["accuracy", "f1", "recall"]}
    print(name, {k: round(v, 4) for k, v in results[name].items()})

best_pred = (probs[proposed_name] >= 0.5).astype(int)
cm = confusion_matrix(y_test, best_pred)
tn, fp, fn, tp = cm.ravel()

# ---------------------------------------------------------------- 5. Figures
sns.set_theme(style="whitegrid", font_scale=1.0)
BLUE, ORANGE, GREY = "#2F6DB5", "#E07B39", "#8A8F98"

# 5.1 class distribution
plt.figure(figsize=(5, 3.6))
counts = y.value_counts().sort_index()
ax = sns.barplot(x=["Low risk\n(benign)", "High risk\n(malignant)"], y=counts.values,
                 palette=[BLUE, ORANGE])
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, str(v), ha="center")
plt.ylabel("Number of patients"); plt.title("Class distribution")
plt.tight_layout(); plt.savefig(FIG + "fig_class_dist.png", dpi=200); plt.close()

# 5.2 correlation heat-map of the ten "mean" features
mean_cols = [c for c in X.columns if c.startswith("mean ")]
plt.figure(figsize=(7, 5.6))
sns.heatmap(X[mean_cols].corr(), cmap="Blues", annot=True, fmt=".1f", annot_kws={"size": 7},
            cbar_kws={"label": "Pearson correlation"})
plt.title("Correlation between mean features")
plt.tight_layout(); plt.savefig(FIG + "fig_corr.png", dpi=200); plt.close()

# 5.3 classification metrics for all approaches (dot plot)
order = sorted(results, key=lambda n: results[n]["f1"])
fig, ax = plt.subplots(figsize=(7.5, 4.6))
markers = {"accuracy": "o", "precision": "s", "recall": "^", "f1": "D"}
colors = {"accuracy": BLUE, "precision": ORANGE, "recall": "#3C9D6B", "f1": "#7A4FB5"}
for m in markers:
    ax.scatter([results[n][m] for n in order], range(len(order)), marker=markers[m],
               color=colors[m], label=m.capitalize() if m != "f1" else "F1-score", s=45, zorder=3)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
ax.set_xlabel("Score on hold-out test set"); ax.set_xlim(0.87, 1.01)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
plt.title("Classification metrics for various approaches")
plt.tight_layout(); plt.savefig(FIG + "fig_metrics.png", dpi=200); plt.close()

# 5.4 cross-validated accuracy
order_cv = sorted(cv_results, key=lambda n: cv_results[n]["accuracy"][0])
plt.figure(figsize=(7.5, 4.4))
plt.errorbar([cv_results[n]["accuracy"][0] for n in order_cv], range(len(order_cv)),
             xerr=[cv_results[n]["accuracy"][1] for n in order_cv], fmt="o", color=BLUE,
             ecolor=GREY, capsize=3)
plt.yticks(range(len(order_cv)), order_cv)
plt.xlabel("5-fold cross-validated accuracy (mean ± SD)")
plt.title("Cross-validation performance")
plt.tight_layout(); plt.savefig(FIG + "fig_cv.png", dpi=200); plt.close()

# 5.5 confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low risk", "High risk"], yticklabels=["Low risk", "High risk"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix (proposed model)")
plt.tight_layout(); plt.savefig(FIG + "fig_cm.png", dpi=200); plt.close()

# 5.6 ROC curves
plt.figure(figsize=(6, 5))
for name in tuned:
    if name == proposed_name:
        continue
    f, t, _ = roc_curve(y_test, probs[name])
    plt.plot(f, t, color=GREY, alpha=0.5, lw=1)
f, t, _ = roc_curve(y_test, probs[proposed_name])
plt.plot(f, t, color=ORANGE, lw=2.5, label=f"Proposed (AUC = {results[proposed_name]['auc']:.3f})")
plt.plot([], [], color=GREY, label="Other models")
plt.plot([0, 1], [0, 1], "--", color="black", lw=1)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(FIG + "fig_roc.png", dpi=200); plt.close()

# 5.7 precision-recall curve
pr, rc, _ = precision_recall_curve(y_test, probs[proposed_name])
ap = average_precision_score(y_test, probs[proposed_name])
plt.figure(figsize=(6, 5))
plt.plot(rc, pr, color=BLUE, lw=2.5, label=f"Average precision = {ap:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall curve (proposed model)")
plt.ylim(0.5, 1.02); plt.legend(loc="lower left"); plt.tight_layout()
plt.savefig(FIG + "fig_pr.png", dpi=200); plt.close()

# 5.8 permutation importance
imp = permutation_importance(tuned[proposed_name], X_test, y_test, n_repeats=20,
                             random_state=SEED, scoring="f1", n_jobs=-1)
imp_df = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean,
                        "sd": imp.importances_std})
          .sort_values("importance", ascending=False).head(10).iloc[::-1])
plt.figure(figsize=(7, 4.4))
plt.barh(imp_df["feature"], imp_df["importance"], xerr=imp_df["sd"], color=BLUE, ecolor=GREY)
plt.xlabel("Drop in F1-score when the feature is shuffled")
plt.title("Top 10 features (permutation importance)")
plt.tight_layout(); plt.savefig(FIG + "fig_importance.png", dpi=200); plt.close()

# 5.9 learning curve
sizes, tr, va = learning_curve(clone(tuned[proposed_name]), X_train, y_train,
                               train_sizes=np.linspace(0.2, 1.0, 6), cv=outer_cv,
                               scoring="accuracy", n_jobs=-1)
plt.figure(figsize=(6.5, 4.4))
for arr, lab, c in [(tr, "Training accuracy", BLUE), (va, "Validation accuracy", ORANGE)]:
    plt.plot(sizes, arr.mean(1), "o-", color=c, label=lab)
    plt.fill_between(sizes, arr.mean(1) - arr.std(1), arr.mean(1) + arr.std(1), color=c, alpha=0.15)
plt.xlabel("Number of training samples"); plt.ylabel("Accuracy")
plt.title("Learning curve (proposed model)"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG + "fig_learning.png", dpi=200); plt.close()

# ---------------------------------------------------------------- 6. Save numbers
summary = {
    "n_samples": int(len(X)), "n_features": int(X.shape[1]),
    "n_high_risk": int(y.sum()), "n_low_risk": int((1 - y).sum()),
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "missing": int(X.isna().sum().sum()), "duplicates": int(X.duplicated().sum()),
    "results": results, "cv": cv_results, "best_params": best_params,
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "avg_precision": float(ap),
    "top_features": imp_df.iloc[::-1]["feature"].tolist(),
    "sklearn_version": __import__("sklearn").__version__,
}
json.dump(summary, open("results.json", "w"), indent=2)
print("Confusion matrix:\n", cm)
