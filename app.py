
import gzip
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib

st.set_page_config(page_title="NeuroSense | EEG Workload Monitor", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.stApp { background: #0b1220; color: #e7eef8; }
[data-testid="stSidebar"] { background: #111b2e; }
h1,h2,h3 { color: #eaf4ff; }
[data-testid="stMetric"] { background: #142238; border: 1px solid #263b58; padding: 14px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent
def load_npy(name):
    p = BASE / name
    return np.load(p, allow_pickle=False) if p.exists() else None

@st.cache_resource
def load_model():
    p = BASE / "eeg_workload_model.pkl.gz"
    if p.exists():
        try:
            with gzip.open(p, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Could not load model: {e}")
    return None



@st.cache_data
def load_csv(name):
    p = BASE / name
    return pd.read_csv(p) if p.exists() else None

features = load_npy("STEW_features_demo.npy")
labels = load_npy("STEW_labels_demo.npy")
subjects = load_npy("STEW_subjects_demo.npy")
predictions = load_csv("test_predictions.csv")
comparison = load_csv("model_comparison.csv")
model = load_model()

st.sidebar.title("🧠 NeuroSense")
st.sidebar.caption("EEG-Based Workload Monitoring")
page = st.sidebar.radio("Navigate", ["Overview", "EEG Explorer", "Workload Analysis", "Model Evaluation"])
st.sidebar.markdown("---")
st.sidebar.caption("Research prototype • Stored STEW dataset")
st.sidebar.info("This demonstration uses saved EEG samples. It is not connected to a live wireless EEG device.")

if page == "Overview":
    st.title("NeuroSense")
    st.subheader("EEG-Based Cognitive Workload Monitoring")
    st.write("Explore stored EEG samples, frequency-band features and machine-learning evaluation results.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("EEG channels", "14")
    c2.metric("Sampling rate", "128 Hz")
    c3.metric("Window duration", "2 seconds")
    c4.metric("Extracted features", "70")
    st.markdown("### Processing pipeline")
    st.markdown("**STEW dataset → 1–40 Hz bandpass filtering → Welch PSD features → ML classification → subject-wise evaluation**")
    st.markdown("### Dashboard modules")
    st.write("- **EEG Explorer:** inspect example feature values.")
    st.write("- **Workload Analysis:** inspect saved model predictions.")
    st.write("- **Model Evaluation:** compare reported models and saved predictions.")
    st.warning("Numeric labels are shown without low/high names until the original STEW label mapping is verified.")
    st.caption("Software-only prototype. Live wireless EEG acquisition is future work.")

elif page == "EEG Explorer":
    st.title("EEG Explorer")
    if features is None or labels is None:
        st.error("Demo feature/label files are missing from the deployment folder.")
        st.stop()
    idx = st.slider("Select demonstration sample", 0, len(features)-1, 0)
    st.metric("Recorded numeric class", str(int(labels[idx])))
    row = features[idx]
    bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    if row.size == 70:
        arr = row.reshape(14,5)
        channel = st.selectbox("EEG channel", list(range(1,15)), format_func=lambda x:f"Channel {x}")
        df = pd.DataFrame({"Frequency band":bands, "Feature value":arr[channel-1]})
        fig = px.bar(df, x="Frequency band", y="Feature value", title=f"Band features — Channel {channel}")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1220", plot_bgcolor="#0b1220")
        st.plotly_chart(fig, use_container_width=True)
        all_df = pd.DataFrame(arr, columns=bands)
        all_df.insert(0, "Channel", [f"Ch {i}" for i in range(1,15)])
        st.markdown("### Feature values across all channels")
        st.dataframe(all_df, use_container_width=True)
    else:
        st.info(f"Feature vector has {row.size} values; expected 70 (14 channels × 5 bands).")

elif page == "Workload Analysis":
    st.title("Workload Analysis")
    if features is None:
        st.error("Demo feature file is missing.")
        st.stop()
    idx = st.slider("Sample index", 0, len(features)-1, 0, key="pred_idx")
    st.metric("Ground-truth numeric label", str(int(labels[idx])) if labels is not None else "Unavailable")
    if model is not None:
        try:
            pred = int(np.asarray(model.predict(features[idx].reshape(1,-1))).ravel()[0])
            st.metric("Model-predicted numeric label", str(pred))
        except Exception as e:
            st.error(f"Saved model prediction failed: {e}")
    else:
        st.warning("Model file not included or could not be loaded. Saved results may still be viewed below.")
    st.caption("Numeric classes are displayed without low/high labels until the dataset mapping is confirmed.")
    if predictions is not None:
        st.markdown("### Saved test predictions")
        st.dataframe(predictions.head(20), use_container_width=True)

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    if comparison is not None and not comparison.empty:
        st.markdown("### Model comparison")
        st.dataframe(comparison, use_container_width=True)
        num_cols = comparison.select_dtypes(include=np.number).columns.tolist()
        name_col = next((c for c in comparison.columns if c not in num_cols), None)
        metric_col = next((c for c in num_cols if "accuracy" in c.lower()), num_cols[0] if num_cols else None)
        if name_col and metric_col:
            fig = px.bar(comparison, x=name_col, y=metric_col, title=f"Model comparison — {metric_col}")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1220", plot_bgcolor="#0b1220")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model comparison CSV missing or empty.")
    if predictions is not None:
        st.markdown("### Saved prediction results")
        st.dataframe(predictions.head(30), use_container_width=True)
    st.markdown("### Evaluation notes")
    st.write("For generalisation claims, use subject-wise train/test separation and report accuracy with precision, recall and F1-score.")
    st.caption("Only metrics available in the included CSV files are displayed.")
