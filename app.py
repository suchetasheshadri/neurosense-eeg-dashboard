from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def workload_name(label):
    return {
        0: "Low Workload",
        1: "High Workload"
    }.get(int(label), "Unknown")


BASE = Path(__file__).parent
FS = 128
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
PAGES = ["Overview", "Workload Insights", "EEG Explorer", "Workload Analysis", "Model Evaluation"]

st.set_page_config(page_title="NeuroSense | EEG Workload Monitoring", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp {background:#0b1220;color:#e7eef8;}
[data-testid="stSidebar"] {background:#111b2e;}
h1,h2,h3 {color:#eaf4ff;}
[data-testid="stMetric"] {background:#142238;border:1px solid #263b58;padding:14px;border-radius:12px;}
.ns-panel {background:#142238;border:1px solid #263b58;border-radius:13px;padding:16px 18px;margin:8px 0 12px;}
.ns-muted {color:#a9bad0;font-size:.91rem;}
.ns-kicker {color:#79c7ff;font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
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



import gzip
import io
import joblib

@st.cache_resource
def load_model():
    model_path = BASE / "eeg_workload_model.pkl.gz"

    if not model_path.exists():
        raise FileNotFoundError("Model file is missing.")

    compressed_data = model_path.read_bytes()
    model_data = gzip.decompress(compressed_data)

    return joblib.load(io.BytesIO(model_data))

try:
    model = load_model()
    st.success("Random Forest model loaded successfully!")
except Exception as e:
    model = None
    st.error(f"Model loading failed: {type(e).__name__}: {e}")


X = load_npy("STEW_X_demo.npy")
features = load_npy("STEW_features_demo.npy")
labels = load_npy("STEW_labels_demo.npy")
subjects = load_npy("STEW_subjects_demo.npy")
comparison = load_csv("model_comparison.csv")
predictions = load_csv("test_predictions.csv")

st.sidebar.title("🧠 NeuroSense")
st.sidebar.caption("EEG-Based Workload Monitoring")
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Research prototype • STEW EEG dataset")
st.sidebar.info("This demonstration uses stored EEG samples. It is not connected to a live wireless EEG device.")

required = {
    "STEW_X_demo.npy": X,
    "STEW_features_demo.npy": features,
    "STEW_labels_demo.npy": labels,
    "STEW_subjects_demo.npy": subjects,
}
missing = [name for name, value in required.items() if value is None]
if missing:
    st.error("Required demo file(s) missing: " + ", ".join(missing))
    st.stop()

# Align the arrays in case a demo file contains fewer rows than another.
n = min(len(X), len(features), len(labels), len(subjects))
X, features, labels, subjects = X[:n], features[:n], labels[:n], subjects[:n]


def themed(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#e7eef8"),
        margin=dict(l=20, r=20, t=55, b=25),
    )
    if height:
        fig.update_layout(height=height)
    return fig


def band_table(row):
    df = pd.DataFrame(row.reshape(14, 5), columns=BANDS)
    df.insert(0, "Channel", [f"Channel {i}" for i in range(1, 15)])
    return df


def predict_one(idx):
    if model is None:
        return None, None
    try:
        sample = features[idx].reshape(1, -1)
        pred = int(np.asarray(model.predict(sample)).ravel()[0])
        proba = model.predict_proba(sample)[0] if hasattr(model, "predict_proba") else None
        return pred, proba
    except Exception as exc:
        st.warning(f"Prediction could not be generated: {exc}")
        return None, None


def sample_controls(prefix):
    participant_ids = sorted(np.unique(subjects).tolist())
    participant = st.selectbox("Select participant", participant_ids, key=f"{prefix}_participant")
    indices = np.where(subjects == participant)[0]
    idx = st.selectbox(
        "Select EEG sample",
        indices.tolist(),
        format_func=lambda i: f"Sample {int(np.where(indices == i)[0][0]) + 1}",
        key=f"{prefix}_sample",
    )
    return participant, indices, idx


# ---------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------
if page == "Overview":
    st.title("NeuroSense")
    st.subheader("EEG-Based Cognitive Workload Monitoring")
    st.markdown(
        '<p class="ns-muted">A software research prototype for exploring EEG signals, '
        'extracting frequency-band features and classifying stored dataset labels.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EEG channels", "14")
    c2.metric("Sampling rate", "128 Hz")
    c3.metric("Window duration", "2 seconds")
    c4.metric("Extracted features", "70")

    st.markdown("### Project at a glance")
    st.write(
        "NeuroSense analyses previously recorded EEG windows from the STEW dataset. "
        "Each window is represented by frequency-band features, which are supplied "
        "to a trained Random Forest classifier."
    )
    st.markdown("### Processing pipeline")
    st.markdown(
        "**STEW EEG data → 1–40 Hz bandpass filtering → Welch PSD features "
        "→ Random Forest classification → subject-wise evaluation**"
    )

    st.markdown("### Explore NeuroSense")
    a, b, c = st.columns(3)
    a.markdown('<div class="ns-panel"><b>Workload Insights</b><br><span class="ns-muted">Understand the model output and processing steps.</span></div>', unsafe_allow_html=True)
    b.markdown('<div class="ns-panel"><b>EEG Explorer</b><br><span class="ns-muted">Inspect stored waveforms and frequency-band features.</span></div>', unsafe_allow_html=True)
    c.markdown('<div class="ns-panel"><b>Model Evaluation</b><br><span class="ns-muted">Compare models and review saved test predictions.</span></div>', unsafe_allow_html=True)

    st.info(f"Interactive demonstration: {n:,} stored EEG windows from {len(np.unique(subjects))} participants.")
    st.warning(
        "The original mapping of numerical classes 0 and 1 has not yet been verified. "
        "They are therefore displayed as numeric classes, not named low/high workload."
    )
    st.caption("Software-only prototype. Live wireless EEG acquisition is future work. This is not a medical diagnostic tool.")


# ---------------------------------------------------------------------
# WORKLOAD INSIGHTS: NEW PLAIN-LANGUAGE EXPLANATION SECTION
# ---------------------------------------------------------------------
elif page == "Workload Insights":
    st.title("Workload Insights")
    st.markdown(
        '<p class="ns-muted">A plain-language guide to the selected prediction, '
        'the processing pipeline and the limits of what the result means.</p>',
        unsafe_allow_html=True,
    )
    participant, indices, idx = sample_controls("insights")
    pred, proba = predict_one(idx)
    actual = int(labels[idx])

    st.markdown("### Selected EEG sample")
    a, b, c = st.columns(3)
    a.metric("Participant", str(participant))
    b.metric("Sample number", int(np.where(indices == idx)[0][0]) + 1)
    c.metric("Recorded dataset class", str(actual))

    st.markdown("### Model result")
    if pred is None:
        st.error("The trained model is unavailable, so a prediction cannot be displayed.")
    else:
        left, right = st.columns(2)
        left.metric("Predicted class", f"Class {pred}")
        right.metric("Recorded class", f"Class {actual}", delta="Match" if pred == actual else "Different")
        if pred == actual:
            st.success("For this sample, the model prediction matches the recorded dataset label.")
        else:
            st.warning("For this sample, the model prediction differs from the recorded dataset label.")

        if proba is not None:
            prob_df = pd.DataFrame({
                "Numeric class": [f"Class {c}" for c in model.classes_],
                "Model probability": proba,
            })
            fig = px.bar(
                prob_df, x="Numeric class", y="Model probability",
                title="Model output probability by class",
                text=prob_df["Model probability"].map(lambda v: f"{v:.1%}"),
            )
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(themed(fig, 330), use_container_width=True)
            st.caption("These are model output probabilities for this stored sample, not clinical certainty scores.")

    st.markdown("### How the system works")
    steps = [
        ("1 · EEG sample", "A short, previously recorded EEG window is selected. The dataset contains 14 channels sampled at 128 Hz."),
        ("2 · Signal filtering", "The preprocessing pipeline applies a 1–40 Hz bandpass filter to the EEG signal."),
        ("3 · Feature extraction", "Welch power spectral density is used to calculate five band features for each of 14 channels: 70 features per window."),
        ("4 · Classification", "The trained Random Forest receives the 70 features and predicts one of the two numerical classes."),
    ]
    for title, detail in steps:
        st.markdown(f'<div class="ns-panel"><b>{title}</b><br><span class="ns-muted">{detail}</span></div>', unsafe_allow_html=True)

    st.markdown("### EEG frequency bands")
    band_info = [
        ("Delta", "1–4 Hz", "Often associated with deep sleep activity."),
        ("Theta", "4–8 Hz", "Can be associated with drowsiness, memory and internal processing."),
        ("Alpha", "8–13 Hz", "Often observed during relaxed wakefulness."),
        ("Beta", "13–30 Hz", "Often associated with alertness and active mental processing."),
        ("Gamma", "30–40 Hz", "Studied in relation to higher-level information processing."),
    ]
    cols = st.columns(5)
    for col, (name, hz, note) in zip(cols, band_info):
        with col:
            st.markdown(f'<div class="ns-panel"><b>{name}</b><br><span class="ns-kicker">{hz}</span><br><span class="ns-muted">{note}</span></div>', unsafe_allow_html=True)
    st.caption("These are broad scientific associations, not direct or unique indicators of a person's mental state.")

    st.markdown("### How to interpret the result")
    st.markdown("""
    - **A class is a model output:** it is the category selected by the trained algorithm.
    - **A match is not proof of perfect performance:** it means only that this sample's prediction agrees with its stored label.
    - **A mismatch is possible:** EEG patterns vary, and machine-learning models make classification errors.
    - **This is not live monitoring:** the dashboard uses stored samples, not a connected wireless EEG headset.
    - **This is not a medical tool:** do not use these outputs to make health or safety decisions.
    """)
    st.info("Keep Class 0 and Class 1 as numerical labels until the original dataset label mapping has been verified.")
    st.markdown("### General wellbeing note")
    st.write(
        "If you personally feel mentally tired while studying or working, a short break, "
        "water and a quieter workspace may help. These are general suggestions and are "
        "not generated from the EEG class prediction."
    )


# ---------------------------------------------------------------------
# EEG EXPLORER
# ---------------------------------------------------------------------
elif page == "EEG Explorer":
    st.title("EEG Explorer")
    st.markdown("<p class='ns-muted'>Explore a participant's recorded brain signals and their frequency patterns.</p>", unsafe_allow_html=True)
    participant, indices, idx = sample_controls("explorer")

    left, right = st.columns(2)
    left.metric("Sample information", int(np.where(indices == idx)[0][0]) + 1)
    right.metric("Recorded workload level", int(labels[idx]))

    st.markdown("### Brain signal waveform")
    st.caption("How the brain's electrical signal changes over 2 seconds.")
    channel = st.selectbox("Choose an electrode channel", list(range(1, 15)), format_func=lambda c: f"Channel {c}", key="wave_channel")
    signal = X[idx, channel - 1, :]
    time = np.arange(len(signal)) / FS

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signal, mode="lines", name=f"Channel {channel}", line=dict(width=1.6, color="#75bfff")))
    fig.update_layout(
        title=f"Participant {participant} · Channel {channel}",
        xaxis_title="Time (seconds)",
        yaxis_title="Signal amplitude (dataset units)",
    )
    st.plotly_chart(themed(fig, 420), use_container_width=True)

    if features.shape[1] == 70:
        values = features[idx].reshape(14, 5)
        band_df = pd.DataFrame({"Frequency band": BANDS, "Feature value": values[channel - 1]})
        st.markdown("### Frequency-band features")
        st.caption("Extracted numerical features for the selected channel; not direct workload measurements.")
        fig = px.bar(band_df, x="Frequency band", y="Feature value", title=f"Frequency-band features · Channel {channel}", text_auto=".3f")
        st.plotly_chart(themed(fig, 360), use_container_width=True)

        st.markdown("### Feature values across all 14 channels")
        st.dataframe(band_table(features[idx]).round(4), use_container_width=True, hide_index=True)
    else:
        st.error("Expected 70 features per sample (14 channels × 5 bands).")


# ---------------------------------------------------------------------
# WORKLOAD ANALYSIS
# ---------------------------------------------------------------------
elif page == "Workload Analysis":
    st.title("Workload Analysis")
    st.markdown('<p class="ns-muted">Compare the recorded dataset label with the model prediction for a selected window.</p>', unsafe_allow_html=True)
    participant, indices, idx = sample_controls("work")

if idx is None:
    st.warning("Please select an EEG sample.")
    st.stop()

pred, proba = predict_one(idx)

    c1, c2 = st.columns(2)
    label_names = {
    0: "Low Workload",
    1: "High Workload"
}

labels_for_display = load_npy("STEW_labels_demo.npy")

if labels_for_display is None:
    st.error("EEG labels file could not be loaded.")
    st.stop()

actual_label = int(labels_for_display[idx])
c1.metric(
    "Actual Workload",
    label_names.get(actual_label, "Unknown")
)

c2.metric(
    "Predicted Workload",
    label_names.get(int(pred), "Unknown") if pred is not None else "Unavailable"
)

if pred is not None:
    if int(pred) == 1:
        st.warning("High Workload detected. Consider taking a short break.")
    else:
        st.success("Low Workload detected.")

    if pred is not None:
        if pred == int(labels[idx]):
            st.success("The model prediction matches this sample's recorded label.")
        else:
            st.warning("The model prediction differs from this sample's recorded label.")

    if proba is not None:
        st.markdown("### Model prediction probabilities")
        prob_df = pd.DataFrame({"Numeric class": [str(c) for c in model.classes_], "Probability": proba})
        fig = px.bar(prob_df, x="Numeric class", y="Probability", title="Predicted probability by class", text=prob_df["Probability"].map(lambda v: f"{v:.1%}"))
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(themed(fig, 350), use_container_width=True)

    st.caption("Predictions use stored demo features and the trained model; they are not live EEG measurements.")
    if predictions is not None:
        st.markdown("### Saved test predictions")
        st.dataframe(predictions.head(20), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------------------
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.markdown('<p class="ns-muted">Compare the tested classifiers and review saved test-set predictions.</p>', unsafe_allow_html=True)

    if comparison is not None and not comparison.empty:
        st.markdown("### Model comparison")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        numeric_cols = comparison.select_dtypes(include=np.number).columns.tolist()
        name_col = next((c for c in comparison.columns if c not in numeric_cols), None)
        accuracy_col = next((c for c in numeric_cols if "accuracy" in c.lower()), None)
        f1_col = next((c for c in numeric_cols if "f1" in c.lower()), None)

        if name_col and accuracy_col:
            fig = px.bar(comparison, x=name_col, y=accuracy_col, title="Model comparison · Accuracy", text_auto=".2f")
            st.plotly_chart(themed(fig, 380), use_container_width=True)
            st.caption("Accuracy is the proportion of test samples classified correctly.")

        if name_col and f1_col:
            fig = px.bar(comparison, x=name_col, y=f1_col, title="Model comparison · F1 score", text_auto=".3f")
            st.plotly_chart(themed(fig, 380), use_container_width=True)
            st.caption("F1 combines precision and recall into a single measure.")
    else:
        st.warning("Model comparison CSV is missing or empty.")

    if predictions is not None:
        st.markdown("### Saved test prediction results")
        st.dataframe(predictions.head(30), use_container_width=True, hide_index=True)

    st.markdown("### Evaluation notes")
    st.write(
        "The project uses subject-wise train/test separation so that test participants "
        "are not used during model training. Accuracy and F1 score should be considered "
        "together when interpreting generalisation performance."
    )
    st.caption("Participant selection and waveform display use the stored demonstration subset.")
