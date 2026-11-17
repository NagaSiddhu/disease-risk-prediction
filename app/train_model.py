"""Train the proposed soft-voting ensemble and save it for the app.

Run once:  python train_model.py
Creates:   model.joblib  (model + feature info + test metrics + example patients)
Uses the same split (80/20, seed 42) and the same tuned hyper-parameters as the report.
"""
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SEED = 42


def pipe(clf):
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def build_model():
    # Hyper-parameters selected by GridSearchCV in the report (Table 3.1)
    return VotingClassifier(
        estimators=[
            ("Logistic Regression", pipe(LogisticRegression(C=1, max_iter=5000))),
            ("SVM", pipe(SVC(C=10, gamma="scale", probability=True, random_state=SEED))),
            ("Random Forest", pipe(RandomForestClassifier(n_estimators=100, max_depth=5,
                                                          min_samples_leaf=1, random_state=SEED))),
            ("Gradient Boosting", pipe(GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                                  max_depth=2, random_state=SEED))),
        ],
        voting="soft",
    )


def main():
    raw = load_breast_cancer(as_frame=True)
    X, y = raw.data, 1 - raw.target          # 1 = high risk (malignant), 0 = low risk (benign)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)

    model = build_model().fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred), "F1-score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, p),
    }

    # one clearly low-risk and one clearly high-risk example from the unseen test set
    lo = X_test[y_test == 0].iloc[[int(np.argmin(p[(y_test == 0).values]))]]
    hi = X_test[y_test == 1].iloc[[int(np.argmax(p[(y_test == 1).values]))]]
    bundle = {
        "model": model, "features": list(X.columns),
        "defaults": X_train.median().to_dict(),
        "min": X_train.min().to_dict(), "max": X_train.max().to_dict(),
        "metrics": metrics, "n_train": len(X_train), "n_test": len(X_test),
        "examples": {"Example: low-risk patient": lo.iloc[0].to_dict(),
                     "Example: borderline patient": X_test.iloc[[int(np.argmin(np.abs(p - 0.5)))]].iloc[0].to_dict(),
                     "Example: high-risk patient": hi.iloc[0].to_dict()},
    }
    joblib.dump(bundle, "model.joblib")
    print("Saved model.joblib")
    for k, v in metrics.items():
        print(f"  {k:10s} {v:.4f}")


if __name__ == "__main__":
    main()
