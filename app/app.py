import pandas as pd
import streamlit as st
from pathlib import Path

from core import load_bundle, predict_risk, member_probabilities, check_columns, label_for

st.set_page_config(page_title="Disease Risk Predictor", page_icon="🩺", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle()


try:
    B = get_bundle()
except FileNotFoundError:
    st.error("model.joblib not found. Run `python train_model.py` once in this folder, then reload the page.")
    st.stop()

FEATURES = B["features"]
GROUPS = {
    "Mean values": [f for f in FEATURES if f.startswith("mean ")],
    "Error (standard error) values": [f for f in FEATURES if f.endswith(" error")],
    "Worst values": [f for f in FEATURES if f.startswith("worst ")],
}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("High-risk threshold", 0.05, 0.95, 0.50, 0.05,
                          help="A patient is flagged HIGH RISK when the predicted probability is at or above this value. "
                               "Lowering it catches more high-risk patients (higher recall) but raises more false alarms.")
    st.caption("Default 0.50 is the setting used for all results in the project report.")
    st.divider()
    st.markdown("**About**  \nSoft-voting ensemble of Logistic Regression, SVM, Random Forest and Gradient Boosting, "
                "trained on the Breast Cancer Wisconsin (Diagnostic) dataset.")
    st.warning("Educational demo only. It is **not** a medical device and must not be used to diagnose "
               "or treat anyone.", icon="⚠️")

st.title("🩺 Disease Risk Predictor")
st.caption("Enter a patient's 30 cell-nucleus measurements and the model estimates the probability of high risk.")

tab_single, tab_batch, tab_perf = st.tabs(["Single patient", "Batch (CSV)", "Model performance"])

# ------------------------------------------------------------------ single patient
with tab_single:
    def init_values(source):
        for f in FEATURES:
            st.session_state[f"f_{f}"] = float(source[f])

    if "f_initialised" not in st.session_state:
        init_values(B["defaults"])
        st.session_state["f_initialised"] = True

    presets = {"Typical patient (median values)": B["defaults"], **B["examples"]}

    def on_preset():
        init_values(presets[st.session_state["preset"]])

    st.radio("Start from", list(presets), key="preset", horizontal=True, on_change=on_preset)

    for group, feats in GROUPS.items():
        with st.expander(group, expanded=(group == "Mean values")):
            cols = st.columns(3)
            for i, f in enumerate(feats):
                lo, hi = B["min"][f], B["max"][f]
                cols[i % 3].number_input(
                    f, min_value=float(lo * 0.5), max_value=float(hi * 1.5),
                    step=float((hi - lo) / 200 or 0.001), format="%.5f", key=f"f_{f}")

    row = {f: st.session_state[f"f_{f}"] for f in FEATURES}
    prob = float(predict_risk(B, pd.DataFrame([row]))[0])
    label = label_for(prob, threshold)

    st.subheader("Result")
    c1, c2 = st.columns([1, 2])
    c1.metric("Predicted probability of high risk", f"{prob * 100:.1f}%")
    with c2:
        st.progress(min(max(prob, 0.0), 1.0))
        if label == "HIGH RISK":
            st.error(f"**{label}**: probability {prob:.2f} is at or above the threshold {threshold:.2f}. "
                     "A clinician should review this patient.")
        else:
            st.success(f"**{label}**: probability {prob:.2f} is below the threshold {threshold:.2f}.")

    st.markdown("**Do the individual models agree?**")
    members = member_probabilities(B, row)
    st.bar_chart(pd.Series(members, name="Probability of high risk"), horizontal=True, height=220)
    spread = max(members.values()) - min(members.values())
    if spread > 0.3:
        st.info("The four models disagree noticeably for this patient, so treat this prediction with extra caution.")

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write("Upload a CSV with one patient per row and the 30 feature columns (same names as in the single-patient tab).")
    sample = pd.DataFrame(list(B["examples"].values()))[FEATURES]
    st.download_button("Download a sample CSV (2 patients)", sample.to_csv(index=False).encode(),
                       "sample_patients.csv", "text/csv")
    up = st.file_uploader("Patient CSV", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:  # unreadable file
            st.error(f"Could not read the file: {e}")
            st.stop()
        missing = check_columns(B, df)
        if missing:
            st.error(f"{len(missing)} required column(s) are missing, for example: {', '.join(missing[:5])}")
        else:
            p = predict_risk(B, df)
            out = df.copy()
            out["risk_probability"] = p.round(4)
            out["prediction"] = [label_for(x, threshold) for x in p]
            st.success(f"Scored {len(out)} patients: {(out['prediction'] == 'HIGH RISK').sum()} flagged as high risk "
                       f"at threshold {threshold:.2f}.")
            st.dataframe(out[["risk_probability", "prediction"] + FEATURES[:4]], use_container_width=True)
            st.download_button("Download results", out.to_csv(index=False).encode(),
                               "risk_predictions.csv", "text/csv")

# ------------------------------------------------------------------ performance
with tab_perf:
    st.write(f"Results on the unseen hold-out test set ({B['n_test']} patients; model trained on {B['n_train']}).")
    cols = st.columns(len(B["metrics"]))
    for c, (k, v) in zip(cols, B["metrics"].items()):
        c.metric(k, f"{v:.4f}")
    st.caption("Only 114 test patients: one patient changes accuracy by about 0.9 percentage points. "
               "See the project report for cross-validation results.")
    fig_dir = Path(__file__).parent / "figures"
    imgs = [("Confusion matrix", "fig_cm.png"), ("ROC curves", "fig_roc.png"),
            ("Precision–Recall curve", "fig_pr.png"), ("Top features", "fig_importance.png")]
    cols = st.columns(2)
    for i, (title, fn) in enumerate(imgs):
        if (fig_dir / fn).exists():
            cols[i % 2].image(str(fig_dir / fn), caption=title)
