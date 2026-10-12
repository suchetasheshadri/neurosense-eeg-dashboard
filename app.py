
import io
import gzip
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="NeuroSense | EEG Workload Monitor",
    page_icon="🧠",
    layout="wide"
)

BASE = Path(__file__).parent

st.markdown("""
<style>
.stApp { background: #0b1220; color: #e7eef8; }
[data-testid="stSidebar"] { background: #111b2e; }
h1, h2, h3 { color: #eaf4ff; }
[data-testid="stMetric"] {
    background: #142238;
    border: 1px solid #263b58;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_npy(name):
    path = BASE / name
    if path.exists():
        return np.load(path, allow_pickle=False)
    return None


@st.cache_data
def load_csv(name):
    path = BASE / name
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_model():
    path = BASE / "eeg_workload_model.pkl.gz"

    if not path.exists():
        return None

    try:
        with gzip.open(path, "rb") as f:
            model_bytes = f.read()
        return joblib.load(io.BytesIO(model_bytes))
    except Exception as exc:
        st.error(f"Could not load the trained model: {exc}")
        return None


features = load_npy("STEW_features_demo.npy")
labels = load_npy("STEW_labels_demo.npy")
subjects = load_npy("STEW_subjects_demo.npy")
raw_eeg = load_npy("STEW_X_demo.npy")
comparison = load_csv("model_comparison.csv")
predictions = load_csv("test_predictions.csv")
model = load_model()

bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

st.sidebar.title("🧠 NeuroSense")
st.sidebar.caption("EEG-Based Workload Monitoring")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EEG Explorer", "Workload Analysis", "Model Evaluation"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Research prototype • STEW EEG dataset")
st.sidebar.info(
    "This demonstration uses stored EEG samples. "
    "It is not connected to a live wireless EEG device."
)


def get_subjects():
    if subjects is None or len(subjects) != len(features):
        return None
    return np.unique(subjects)


def select_sample(section_key):
    """Return the selected participant ID and demo-array index."""
    unique_subjects = get_subjects()

    if unique_subjects is None:
        st.warning(
            "Participant IDs are unavailable or do not match the demo samples. "
            "Please check STEW_subjects_demo.npy."
        )
        idx = st.selectbox(
            "Select demonstration sample",
            range(len(features)),
            key=f"{section_key}_sample"
        )
        return None, int(idx)

    subject = st.selectbox(
        "Select participant",
        unique_subjects.tolist(),
        key=f"{section_key}_subject"
    )

    choices = np.flatnonzero(subjects == subject)

    idx = st.selectbox(
        "Select sample for this participant",
        choices.tolist(),
        format_func=lambda i: f"Sample {i}",
        key=f"{section_key}_sample"
    )

    return subject, int(idx)


def feature_dataframe(idx):
    row = features[idx]

    if row.size != 70:
        return None

    values = row.reshape(14, 5)
    df = pd.DataFrame(values, columns=bands)
    df.insert(0, "Channel", [f"Channel {i}" for i in range(1, 15)])
    return df


# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.title("NeuroSense")
    st.subheader("EEG-Based Cognitive Workload Monitoring")
    st.write(
        "Explore EEG feature patterns, participant samples, workload "
        "predictions and machine-learning evaluation results."
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
    st.write("🧠 **EEG Explorer:** participant selection, channel features and graphs.")
    st.write("📊 **Workload Analysis:** actual labels and model predictions.")
    st.write("📈 **Model Evaluation:** compare classification performance.")

    if features is not None:
        st.success(f"{len(features)} demonstration samples are available.")

    st.warning(
        "Labels remain numeric until the original dataset's class mapping "
        "has been verified."
    )
    st.caption(
        "Software-only prototype. Live wireless EEG acquisition is future work."
    )


# ---------------- EEG EXPLORER ----------------
elif page == "EEG Explorer":
    st.title("EEG Explorer")

    if features is None or len(features) == 0:
        st.error("STEW_features_demo.npy is missing or empty.")
        st.stop()

    participant, idx = select_sample("explorer")

    c1, c2 = st.columns(2)
    c1.metric("Selected sample", idx)
    if labels is not None and idx < len(labels):
        c2.metric("Recorded numeric class", int(labels[idx]))

    row = features[idx]

    if row.size != 70:
        st.error("Expected 70 features: 14 channels × 5 frequency bands.")
        st.stop()

    arr = row.reshape(14, 5)

    st.markdown("### Frequency-band features")

    channel = st.selectbox(
        "Choose EEG channel",
        list(range(1, 15)),
        format_func=lambda ch: f"Channel {ch}",
        key="explorer_channel"
    )

    band_df = pd.DataFrame({
        "Frequency band": bands,
        "Feature value": arr[channel - 1]
    })

    fig = px.bar(
        band_df,
        x="Frequency band",
        y="Feature value",
        title=f"Frequency-band features — Channel {channel}"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature values across all 14 channels")
    all_df = pd.DataFrame(arr, columns=bands)
    all_df.insert(0, "Channel", [f"Channel {i}" for i in range(1, 15)])
    st.dataframe(all_df, use_container_width=True, hide_index=True)

    # Actual waveform is shown only when raw demo EEG is supplied.
    st.markdown("### EEG waveform")

    if raw_eeg is not None and idx < len(raw_eeg):
        raw = raw_eeg[idx]

        if raw.ndim == 2 and raw.shape[0] == 14:
            raw_channel = st.selectbox(
                "Choose waveform channel",
                list(range(1, 15)),
                format_func=lambda ch: f"Channel {ch}",
                key="wave_channel"
            )

            sampling_rate = 128
            times = np.arange(raw.shape[1]) / sampling_rate

            wave_fig = go.Figure()
            wave_fig.add_trace(go.Scatter(
                x=times,
                y=raw[raw_channel - 1],
                mode="lines",
                name=f"Channel {raw_channel}"
            ))
            wave_fig.update_layout(
                title="Stored EEG signal",
                xaxis_title="Time (seconds)",
                yaxis_title="Signal amplitude",
                template="plotly_dark",
                paper_bgcolor="#0b1220",
                plot_bgcolor="#0b1220"
            )
            st.plotly_chart(wave_fig, use_container_width=True)
        else:
            st.info("Raw EEG file exists, but its array shape needs checking.")
    else:
        st.info(
            "The deployed demo currently contains extracted features, not raw "
            "EEG windows. Add STEW_X_demo.npy to display actual waveforms."
        )


# ---------------- WORKLOAD ANALYSIS ----------------
elif page == "Workload Analysis":
    st.title("Workload Analysis")

    if features is None or labels is None:
        st.error("Demo feature or label file is missing.")
        st.stop()

    participant, idx = select_sample("workload")

    c1, c2 = st.columns(2)
    c1.metric("Selected sample", idx)

    if participant is not None:
        c2.metric("Participant ID", str(participant))

    c1, c2 = st.columns(2)
    c1.metric("Ground-truth numeric label", int(labels[idx]))

    if model is not None:
        try:
            sample = features[idx].reshape(1, -1)
            pred = int(np.asarray(model.predict(sample)).ravel()[0])
            c2.metric("Model-predicted numeric label", pred)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(sample)[0]
                prob_df = pd.DataFrame({
                    "Class": [str(c) for c in model.classes_],
                    "Probability": probabilities
                })
                fig = px.bar(
                    prob_df,
                    x="Class",
                    y="Probability",
                    title="Model prediction probabilities"
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b1220",
                    plot_bgcolor="#0b1220",
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    else:
        st.warning("The trained model could not be loaded.")

    st.caption(
        "Numeric classes are displayed without low/high names until the "
        "dataset label mapping is confirmed."
    )

    if predictions is not None:
        st.markdown("### Saved test predictions")
        if participant is not None and "subject_id" in predictions.columns:
            participant_rows = predictions[
                predictions["subject_id"].astype(str) == str(participant)
            ]
            if not participant_rows.empty:
                st.dataframe(participant_rows, use_container_width=True)
            else:
                st.info(
                    "No saved prediction rows for this participant. "
                    "The selected sample's model prediction above is calculated "
                    "from its feature vector."
                )
        else:
            st.dataframe(predictions.head(30), use_container_width=True)


# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":
    st.title("Model Evaluation")

    if comparison is not None and not comparison.empty:
        st.markdown("### Model comparison")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

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
                title="Model comparison — Accuracy"
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1220",
                plot_bgcolor="#0b1220"
            )
            st.plotly_chart(fig, use_container_width=True)

        if name_col and f1_col:
            fig = px.bar(
                comparison,
                x=name_col,
                y=f1_col,
                title="Model comparison — F1 score"
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1220",
                plot_bgcolor="#0b1220"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model comparison CSV is missing or empty.")

    if predictions is not None:
        st.markdown("### Saved prediction results")
        st.dataframe(predictions.head(30), use_container_width=True)

    st.markdown("### Evaluation notes")
    st.write(
        "Subject-wise train/test separation is important when evaluating "
        "generalisation to participants not seen during training. Report "
        "accuracy, precision, recall and F1-score where available."
    )
    st.caption("Metrics are taken from the saved model comparison file.")
