# Disease Risk Predictor (Streamlit app)

Web app for the mini project "Disease Risk Prediction using Machine Learning".

## Run it

```bash
pip install -r requirements.txt
python train_model.py        # one time: trains the model and creates model.joblib
streamlit run app.py         # opens http://localhost:8501
```

## What it does
- **Single patient:** enter the 30 measurements (or start from a typical / example patient) and get the
  probability of high risk, a LOW/HIGH label, and the probability from each of the four models in the ensemble.
- **Batch (CSV):** upload many patients, get predictions, download the results.
- **Model performance:** test-set metrics and figures from the report.
- **Sidebar threshold:** lower it to catch more high-risk patients at the cost of more false alarms.

## Files
| File | Purpose |
|---|---|
| `train_model.py` | trains the proposed soft-voting ensemble (same settings as the report), saves `model.joblib` |
| `core.py` | prediction helpers (no UI code) |
| `app.py` | the Streamlit interface |
| `figures/` | figures shown in the performance tab |

Educational demo only. Not a medical device.
