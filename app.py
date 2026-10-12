from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
FS = 128
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

st.set_page_config(
    page_title="NeuroSense | EEG Workload Monitor",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.stApp {background:#0b1220;color:#e7eef8;}
[data-testid="stSidebar"] {background:#111b2e;}
h1,h2,h3 {color:#eaf4ff;}
[data-testid="stMetric"] {
    background:#142238;border:1px solid #263b58;
    padding:14px;border-radius:12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_npy(name):
    path = BASE / name
    return np.load(path, allow_pickle=False) if path.exists() else None


@st.cache_data
def load_csv(name):
    path = BASE / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def load_model():
    path = BASE / "eeg_workload_model.pkl"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        return None


X = load_npy("STEW_X_demo.npy")
features = load_npy("STEW_features_demo.npy")
labels = load_npy("STEW_labels_demo.npy")
subjects = load_npy("STEW_subjects_demo.npy")
comparison = load_csv("model_comparison.csv")
predictions = load_csv("test_predictions.csv")
model = load_model()

st.sidebar.title("🧠 NeuroSense")
st.sidebar.caption("EEG-Based Workload Monitoring")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EEG Explorer", "Workload Analysis", "Model Evaluation"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Research prototype • STEW EEG dataset")
st.sidebar.info(
    "This demonstration uses stored EEG samples. "
    "It is not connected to a live wireless EEG device."
)

if X is None or features is None or labels is None or subjects is None:
    st.error("One or more required demo data files are missing.")
    st.stop()

# Keep all arrays aligned to the same demo windows.
n = min(len(X), len(features), len(labels), len(subjects))
X = X[:n]
features = features[:n]
labels = labels[:n]
subjects = subjects[:n]


def make_band_table(row):
    arr = row.reshape(14, 5)
    df = pd.DataFrame(arr, columns=BANDS)
    df.insert(0, "Channel", [f"Channel {i}" for i in range(1, 15)])
    return df


def predict_one(idx):
    if model is None:
        return None, None
    try:
        sample = features[idx].reshape(1, -1)
        pred = int(np.asarray(model.predict(sample)).ravel()[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sample)[0]
        return pred, proba
    except Exception as exc:
        st.warning(f"Prediction could not be generated: {exc}")
        return None, None


if page == "Overview":
    st.title("NeuroSense")
    st.subheader("EEG-Based Cognitive Workload Monitoring")
    st.write(
        "Explore stored EEG samples, frequency-band features, "
        "participant data and machine-learning results."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EEG channels", "14")
    c2.metric("Sampling rate", "128 Hz")
    c3.metric("Window duration", "2 seconds")
    c4.metric("Extracted features", "70")

    st.markdown("### Processing pipeline")
    st.markdown(
        "**STEW dataset → 1–40 Hz bandpass filtering → "
        "Welch PSD features → ML classification → subject-wise evaluation**"
    )

    st.markdown("### Dashboard modules")
    st.write("- **EEG Explorer:** choose a participant and inspect EEG signals.")
    st.write("- **Workload Analysis:** view actual and predicted numeric labels.")
    st.write("- **Model Evaluation:** compare saved model metrics and results.")

    st.info(
        f"Interactive demo: {n} stored windows from "
        f"{len(np.unique(subjects))} participants."
    )
    st.warning(
        "Numeric classes are displayed as 0 and 1; "
        "confirm the original dataset label mapping before naming them low/high."
    )
    st.caption("Software-only prototype. Live wireless EEG acquisition is future work.")


elif page == "EEG Explorer":
    st.title("EEG Explorer")

    participant_ids = sorted(np.unique(subjects).tolist())
    participant = st.selectbox("Select participant", participant_ids)

    participant_indices = np.where(subjects == participant)[0]
    selected_idx = st.selectbox(
        "Select sample for this participant",
        participant_indices.tolist(),
        format_func=lambda i: (
            f"Sample {int(np.where(participant_indices == i)[0][0]) + 1}"
        ),
    )

    left, right = st.columns(2)
    left.metric("Selected sample", int(selected_idx) + 1)
    right.metric("Recorded numeric class", int(labels[selected_idx]))

    st.markdown("### EEG waveform")
    channel = st.selectbox(
        "Choose EEG channel",
        list(range(1, 15)),
        format_func=lambda c: f"Channel {c}",
        key="wave_channel",
    )

    signal = X[selected_idx, channel - 1, :]
    time = np.arange(len(signal)) / FS

    wave_fig = go.Figure()
    wave_fig.add_trace(
        go.Scatter(
            x=time,
            y=signal,
            mode="lines",
            name=f"Channel {channel}",
            line=dict(width=1.5),
        )
    )
    wave_fig.update_layout(
        title=f"EEG waveform — Participant {participant}, Channel {channel}",
        xaxis_title="Time (seconds)",
        yaxis_title="Signal amplitude (dataset units)",
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
    )
    st.plotly_chart(wave_fig, use_container_width=True)

    if features.shape[1] == 70:
        arr = features[selected_idx].reshape(14, 5)
        band_df = pd.DataFrame({
            "Frequency band": BANDS,
            "Feature value": arr[channel - 1],
        })

        st.markdown("### Frequency-band features")
        band_fig = px.bar(
            band_df,
            x="Frequency band",
            y="Feature value",
            title=f"Frequency-band features — Channel {channel}",
        )
        band_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
        )
        st.plotly_chart(band_fig, use_container_width=True)

        st.markdown("### Feature values across all 14 channels")
        st.dataframe(make_band_table(features[selected_idx]), use_container_width=True)
    else:
        st.error("Expected 70 features per sample (14 channels × 5 bands).")


elif page == "Workload Analysis":
    st.title("Workload Analysis")

    participant_ids = sorted(np.unique(subjects).tolist())
    participant = st.selectbox(
        "Select participant", participant_ids, key="work_participant"
    )
    participant_indices = np.where(subjects == participant)[0]
    idx = st.selectbox(
        "Select sample",
        participant_indices.tolist(),
        format_func=lambda i: f"Sample {int(np.where(participant_indices == i)[0][0]) + 1}",
        key="work_sample",
    )

    pred, proba = predict_one(idx)

    c1, c2 = st.columns(2)
    c1.metric("Ground-truth numeric label", int(labels[idx]))
    c2.metric(
        "Model-predicted numeric label",
        str(pred) if pred is not None else "Unavailable",
    )

    if pred is not None:
        if pred == int(labels[idx]):
            st.success("The model prediction matches this sample's recorded label.")
        else:
            st.warning("The model prediction differs from this sample's recorded label.")

    if proba is not None:
        st.markdown("### Model prediction probabilities")
        probability_df = pd.DataFrame({
            "Numeric class": [str(c) for c in model.classes_],
            "Probability": proba,
        })
        fig = px.bar(
            probability_df,
            x="Numeric class",
            y="Probability",
            title="Predicted probability by class",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These predictions use stored demo features and the trained model; "
        "they are not live EEG measurements."
    )

    if predictions is not None:
        st.markdown("### Saved test predictions")
        st.dataframe(predictions.head(20), use_container_width=True)


elif page == "Model Evaluation":
    st.title("Model Evaluation")

    if comparison is not None and not comparison.empty:
        st.markdown("### Model comparison")
        st.dataframe(comparison, use_container_width=True)

        numeric_cols = comparison.select_dtypes(include=np.number).columns.tolist()
        name_col = next(
            (c for c in comparison.columns if c not in numeric_cols), None
        )
        accuracy_col = next(
            (c for c in numeric_cols if "accuracy" in c.lower()), None
        )
        f1_col = next(
            (c for c in numeric_cols if "f1" in c.lower()), None
        )

        if name_col and accuracy_col:
            fig = px.bar(
                comparison,
                x=name_col,
                y=accuracy_col,
                title="Model comparison — Accuracy",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1220",
                plot_bgcolor="#0b1220",
            )
            st.plotly_chart(fig, use_container_width=True)

        if name_col and f1_col:
            fig = px.bar(
                comparison,
                x=name_col,
                y=f1_col,
                title="Model comparison — F1 score",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1220",
                plot_bgcolor="#0b1220",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model comparison CSV is missing or empty.")

    if predictions is not None:
        st.markdown("### Saved test prediction results")
        st.dataframe(predictions.head(30), use_container_width=True)

    st.markdown("### Evaluation notes")
    st.write(
        "Use subject-wise train/test separation and report accuracy, "
        "precision, recall and F1 score when evaluating generalisation."
    )
    st.caption(
        "The participant selector and waveform display use the 100-window demo subset."
    )
