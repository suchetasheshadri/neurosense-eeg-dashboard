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
PAGES = [
    "Overview", "Workload Insights", "EEG Explorer", "Workload Analysis",
    "Recording Comparison", "Signal Quality Check", "Personalised Feedback",
    "Model Evaluation"
]

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


def assess_signal_quality(window):
    """Basic screening heuristics; not a clinical or validated EEG quality score."""
    arr = np.asarray(window, dtype=float)
    finite = np.isfinite(arr).all()
    if not finite:
        return {
            "status": "Check required",
            "notes": ["The selected recording contains non-finite values."],
            "max_abs": np.nan,
            "flat_channels": 0,
        }

    channel_std = np.std(arr, axis=1)
    channel_ptp = np.ptp(arr, axis=1)
    flat_channels = int(np.sum((channel_std < 1e-8) | (channel_ptp < 1e-8)))
    max_abs = float(np.max(np.abs(arr)))
    notes = []
    if flat_channels:
        notes.append(f"{flat_channels} channel(s) are nearly flat in this window.")
    if max_abs > 1e6:
        notes.append("The signal contains an unusually large amplitude value; check the recording units or possible artefacts.")
    if not notes:
        notes.append("No flat channels or extreme numerical values were detected by these basic checks.")

    return {
        "status": "Review recommended" if notes and (flat_channels or max_abs > 1e6) else "No basic issue detected",
        "notes": notes,
        "max_abs": max_abs,
        "flat_channels": flat_channels,
    }


def sample_controls(prefix):
    participant_ids = sorted(np.unique(subjects).tolist())
    participant = st.selectbox("Choose a participant", participant_ids, key=f"{prefix}_participant")
    indices = np.where(subjects == participant)[0]
    idx = st.selectbox(
        "Choose an EEG recording",
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
    st.markdown(
        '<p class="ns-muted">Compare the recorded dataset label with the model prediction for a selected window.</p>',
        unsafe_allow_html=True
    )

    participant, indices, idx = sample_controls("work")

    if idx is None:
        st.warning("Please select an EEG sample.")
        st.stop()

    pred, proba = predict_one(idx)

    actual_label = int(labels[idx])

    c1, c2 = st.columns(2)
    c1.metric("Recorded dataset class", f"Class {actual_label}")
    c2.metric("Predicted dataset class", f"Class {int(pred)}" if pred is not None else "Unavailable")

    if pred is not None:
        if int(pred) == actual_label:
            st.success("The model prediction matches this sample's recorded dataset label.")
        else:
            st.warning("The model prediction differs from this sample's recorded dataset label.")

    if proba is not None and model is not None:
        st.markdown("### Model prediction probabilities")
        prob_df = pd.DataFrame({
            "Numeric class": [f"Class {int(c)}" for c in model.classes_],
            "Probability": proba,
        })
        fig = px.bar(
            prob_df,
            x="Numeric class",
            y="Probability",
            title="Prediction probability by dataset class",
            text=prob_df["Probability"].map(lambda v: f"{v:.1%}"),
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(themed(fig, 350), use_container_width=True)

    st.caption(
        "Predictions use stored EEG demo features and the trained model. "
        "They are not live EEG measurements."
    )

    if predictions is not None:
        st.markdown("### Saved test predictions")
        st.dataframe(
            predictions.head(20),
            use_container_width=True,
            hide_index=True
        )


# ---------------------------------------------------------------------
# RECORDING COMPARISON
# ---------------------------------------------------------------------
elif page == "Recording Comparison":
    st.title("EEG Recording Comparison")
    st.markdown(
        '<p class="ns-muted">Compare two stored EEG recordings using their waveforms, frequency-band features and model outputs.</p>',
        unsafe_allow_html=True,
    )

    participant_ids = sorted(np.unique(subjects).tolist())
    left, right = st.columns(2)
    with left:
        st.markdown("### Recording A")
        participant_a = st.selectbox("Participant A", participant_ids, key="compare_participant_a")
        indices_a = np.where(subjects == participant_a)[0]
        idx_a = st.selectbox(
            "EEG window A", indices_a.tolist(),
            format_func=lambda i: f"Window {int(np.where(indices_a == i)[0][0]) + 1}",
            key="compare_window_a",
        )
    with right:
        st.markdown("### Recording B")
        participant_b = st.selectbox("Participant B", participant_ids, index=min(1, len(participant_ids)-1), key="compare_participant_b")
        indices_b = np.where(subjects == participant_b)[0]
        idx_b = st.selectbox(
            "EEG window B", indices_b.tolist(),
            format_func=lambda i: f"Window {int(np.where(indices_b == i)[0][0]) + 1}",
            key="compare_window_b",
        )

    pred_a, proba_a = predict_one(idx_a)
    pred_b, proba_b = predict_one(idx_b)
    st.markdown("### Prediction summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recording A · Class", f"Class {pred_a}" if pred_a is not None else "Unavailable")
    m2.metric("Recording A · Actual", f"Class {int(labels[idx_a])}")
    m3.metric("Recording B · Class", f"Class {pred_b}" if pred_b is not None else "Unavailable")
    m4.metric("Recording B · Actual", f"Class {int(labels[idx_b])}")
    st.caption("Class numbers are shown because the original Class 0 / Class 1 workload mapping has not been verified.")

    channel = st.selectbox("Channel to compare", list(range(1, 15)), format_func=lambda c: f"Channel {c}", key="compare_channel")
    time = np.arange(X.shape[2]) / FS
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=X[idx_a, channel-1, :], mode="lines", name=f"Participant {participant_a}"))
    fig.add_trace(go.Scatter(x=time, y=X[idx_b, channel-1, :], mode="lines", name=f"Participant {participant_b}"))
    fig.update_layout(title=f"EEG waveform comparison · Channel {channel}", xaxis_title="Time (seconds)", yaxis_title="Signal amplitude (dataset units)")
    st.plotly_chart(themed(fig, 420), use_container_width=True)

    if features.shape[1] == 70:
        band_a = features[idx_a].reshape(14, 5).mean(axis=0)
        band_b = features[idx_b].reshape(14, 5).mean(axis=0)
        band_compare = pd.DataFrame({
            "Frequency band": BANDS * 2,
            "Mean feature value": np.concatenate([band_a, band_b]),
            "Recording": [f"Participant {participant_a}"] * 5 + [f"Participant {participant_b}"] * 5,
        })
        fig = px.bar(band_compare, x="Frequency band", y="Mean feature value", color="Recording", barmode="group", title="Mean frequency-band features across 14 channels")
        st.plotly_chart(themed(fig, 380), use_container_width=True)
        st.caption("These are averages of the extracted PSD features across channels, not direct measurements of workload.")


# ---------------------------------------------------------------------
# SIGNAL QUALITY CHECK
# ---------------------------------------------------------------------
elif page == "Signal Quality Check":
    st.title("EEG Signal Quality Check")
    st.markdown(
        '<p class="ns-muted">A basic screening of the selected stored EEG window for missing values, nearly flat channels and unusually large amplitudes.</p>',
        unsafe_allow_html=True,
    )
    participant, indices, idx = sample_controls("quality")
    result = assess_signal_quality(X[idx])
    q1, q2, q3 = st.columns(3)
    q1.metric("Participant", str(participant))
    q2.metric("Nearly flat channels", result["flat_channels"])
    q3.metric("Maximum absolute amplitude", f'{result["max_abs"]:.4g}' if np.isfinite(result["max_abs"]) else "Invalid values")

    if result["status"] == "No basic issue detected":
        st.success("No basic signal issue detected by the screening checks.")
    else:
        st.warning("Review this recording before interpreting its model output.")
    for note in result["notes"]:
        st.write("• " + note)

    st.markdown("### Channel-level signal overview")
    quality_df = pd.DataFrame({
        "Channel": [f"Channel {i+1}" for i in range(14)],
        "Standard deviation": np.std(X[idx], axis=1),
        "Peak-to-peak range": np.ptp(X[idx], axis=1),
    })
    st.dataframe(quality_df.round(5), use_container_width=True, hide_index=True)
    st.markdown("### Selected EEG waveform")
    channel = st.selectbox("Choose a channel", list(range(1, 15)), format_func=lambda c: f"Channel {c}", key="quality_channel")
    fig = go.Figure(go.Scatter(x=np.arange(X.shape[2]) / FS, y=X[idx, channel-1, :], mode="lines", name=f"Channel {channel}"))
    fig.update_layout(title=f"Participant {participant} · Channel {channel}", xaxis_title="Time (seconds)", yaxis_title="Signal amplitude (dataset units)")
    st.plotly_chart(themed(fig, 360), use_container_width=True)
    st.info("This is a simple numerical screening aid, not a validated EEG artefact detector or medical-grade signal-quality assessment. Thresholds and findings should be interpreted in the context of the dataset and signal units.")


# ---------------------------------------------------------------------
# PERSONALISED FEEDBACK
# ---------------------------------------------------------------------
elif page == "Personalised Feedback":
    st.title("Recording Summary & Feedback")
    st.markdown(
        '<p class="ns-muted">A plain-language summary of one stored EEG recording, its model output and basic signal checks.</p>',
        unsafe_allow_html=True,
    )
    participant, indices, idx = sample_controls("feedback")
    pred, proba = predict_one(idx)
    quality = assess_signal_quality(X[idx])

    st.markdown("### Your selected recording")
    a, b, c = st.columns(3)
    a.metric("Participant", str(participant))
    b.metric("Recording number", int(np.where(indices == idx)[0][0]) + 1)
    c.metric("Signal screening", quality["status"])

    st.markdown("### What the model found")
    if pred is None:
        st.error("A prediction is unavailable because the trained model could not process this recording.")
    else:
        st.subheader(f"Predicted dataset category: Class {pred}")
        st.write(f"The model assigned this stored EEG recording to numerical class {pred}.")
        st.caption("The numerical class is a machine-learning output. It should not be interpreted as low or high workload until the dataset label mapping is verified.")
        if proba is not None:
            best_prob = float(np.max(proba))
            st.metric("Model probability for selected class", f"{best_prob:.1%}")
            st.caption("This is the model's probability output, not clinical certainty.")
        actual = int(labels[idx])
        if pred == actual:
            st.success("The predicted class matches the stored dataset label for this window.")
        else:
            st.warning("The predicted class differs from the stored dataset label for this window.")

    st.markdown("### Signal check")
    if quality["status"] == "No basic issue detected":
        st.success("No flat channels or extreme numerical values were detected by the basic screening checks.")
    else:
        st.warning("The signal screening raised a point for review:")
        for note in quality["notes"]:
            st.write("• " + note)

    st.markdown("### Suggested next step")
    if quality["status"] != "No basic issue detected":
        st.info("Review the waveform and signal statistics, or select another recording before interpreting the prediction.")
    elif pred is None:
        st.info("Check that the trained model and feature files are available, then try again.")
    else:
        st.info("Compare this recording with another participant or window to explore how the EEG features and model output differ.")
    st.caption("NeuroSense is a research prototype using previously recorded STEW data. It does not provide a diagnosis, measure a person's current mental state, or replace professional assessment.")


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
