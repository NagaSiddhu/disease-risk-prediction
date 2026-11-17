# Disease Risk Prediction using Machine Learning

Mini project: estimate whether a patient is **high risk** or **low risk** from 30 numerical measurements, using a
tuned soft-voting ensemble (Logistic Regression + SVM + Random Forest + Gradient Boosting), and a Streamlit web
app to try it.

Author: [your name] &nbsp;|&nbsp; Guide: [guide name] &nbsp;|&nbsp; [college / university]

## Results (hold-out test set, 114 patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Proposed soft-voting ensemble | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9957 |

Confusion matrix: 72 true negatives, 0 false positives, 3 false negatives, 39 true positives.
The ensemble **matches** the best single models (SVM, Random Forest, MLP) rather than clearly beating them; the
full comparison of 8 baselines, 5-fold cross-validation, feature importance and limitations is in
[`docs/Disease_Risk_Prediction_Mini_Project_Report.docx`](docs/Disease_Risk_Prediction_Mini_Project_Report.docx).

## Dataset
Breast Cancer Wisconsin (Diagnostic), 569 patients, 30 features. It ships with scikit-learn, so nothing needs to be
downloaded. Label: 1 = high risk (malignant), 0 = low risk (benign).

## Project structure
```
disease_risk_prediction.py   full experiment: preprocessing, tuning, 9 models, evaluation, figures
make_diagrams.py             draws the three diagram figures used in the report
results.json                 numbers produced by the experiment (used in the report)
figures/                     all figures
docs/                        the project report (Word)
app/                         Streamlit web app (see app/README.md)
```

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python disease_risk_prediction.py     # reproduce every number and figure (about 1 minute)

cd app
python train_model.py                 # creates model.joblib
streamlit run app.py                  # opens http://localhost:8501
```

## Limitations
Small, clean, single-source dataset; no external validation; diagnostic measurements rather than long-term risk
factors. **Educational project, not a medical device.**
