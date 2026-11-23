from pathlib import Path
import gzip
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import welch, spectrogram, butter, sosfiltfilt

# ============================================================
# NEUROSENSE | EEG-BASED COGNITIVE STATE ANALYSIS
# Software research prototype using stored STEW EEG recordings
# ============================================================

BASE = Path(__file__).parent
FS = 128
CHANNELS = 14
WINDOW_SAMPLES = 256
WINDOW_SECONDS = WINDOW_SAMPLES / FS
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 40),
}
BAND_NAMES = list(BANDS.keys())

st.set_page_config(
    page_title="NeuroSense | EEG Cognitive Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------- STYLE ---------------------------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #080e1b; color: #e8eef9; }
[data-testid="stSidebar"] { background: #101a2b; border-right: 1px solid #24344d; }
[data-testid="stHeader"] { background: rgba(8,14,27,.92); }
h1,h2,h3 { color: #f1f6ff; letter-spacing: -.02em; }
p, li, label { color: #d5deed; }
.ns-hero {
  background: linear-gradient(115deg,#122b49 0%,#14213b 55%,#1b2444 100%);
  border: 1px solid #2d4d70; border-radius: 20px;
  padding: 30px 32px; margin: 8px 0 22px 0;
}
.ns-eyebrow { color:#70d6cf; font-size: .78rem; font-weight:800; letter-spacing:.16em; }
.ns-hero h1 { font-size: 2.5rem; margin:.35rem 0 .5rem 0; }
.ns-hero p { color:#c5d5e9; font-size:1.05rem; max-width:850px; }
.ns-card {
  background:#111e31; border:1px solid #263d5b; border-radius:15px;
  padding:18px 20px; min-height:120px;
}
.ns-card-title { color:#8edbd4; font-size:.78rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
.ns-card-value { color:#f3f7ff; font-size:1.65rem; font-weight:750; margin-top:8px; }
.ns-muted { color:#9fb1c9; font-size:.9rem; }
.ns-note {
  background:#14253a; border-left:4px solid #62c9c0; border-radius:8px;
  padding:12px 15px; color:#d8e4f2; margin:10px 0;
}
.ns-warning {
  background:#33291b; border-left:4px solid #e7ad55; border-radius:8px;
  padding:12px 15px; color:#f6e7c9; margin:10px 0;
}
div[data-testid="stMetric"] {
  background:#111e31; border:1px solid #263d5b; padding:14px 16px;
  border-radius:13px;
}
.stButton>button { border-radius:10px; font-weight:650; }
section[data-testid="stSidebar"] .stRadio label { padding:5px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------- DATA LOADING ------------------------
@st.cache_data
def load_npy(filename):
    path = BASE / filename
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)

@st.cache_data
def load_csv(filename):
    path = BASE / filename
    return pd.read_csv(path) if path.exists() else None

@st.cache_resource
def load_model():
    path = BASE / "eeg_workload_model.pkl.gz"
    if not path.exists():
        return None
    raw = gzip.decompress(path.read_bytes())
    return joblib.load(io.BytesIO(raw))

X = load_npy("STEW_X_demo.npy")
FEATURES = load_npy("STEW_features_demo.npy")
LABELS = load_npy("STEW_labels_demo.npy")
SUBJECTS = load_npy("STEW_subjects_demo.npy")
COMPARISON = load_csv("model_comparison.csv")
PREDICTIONS = load_csv("test_predictions.csv")
MODEL = load_model()

# ----------------------- UTILITIES --------------------------
def themed(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080e1b",
        plot_bgcolor="#080e1b",
        font=dict(color="#e8eef9"),
        margin=dict(l=28, r=24, t=58, b=35),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#203149", zerolinecolor="#203149")
    fig.update_yaxes(gridcolor="#203149", zerolinecolor="#203149")
    return fig

def safe_filter(signal):
    """Apply the project's 1–40 Hz Butterworth bandpass for display/quality checks."""
    try:
        sos = butter(4, [1, 40], btype="bandpass", fs=FS, output="sos")
        return sosfiltfilt(sos, signal)
    except Exception:
        return np.asarray(signal, dtype=float)

def band_features_for_window(index):
    if FEATURES is not None and FEATURES.ndim == 2 and FEATURES.shape[1] == 70:
        return FEATURES[index].reshape(CHANNELS, 5)
    return None

def get_prediction(index):
    if MODEL is None or FEATURES is None:
        return None, None
    try:
        row = FEATURES[index].reshape(1, -1)
        prediction = int(np.asarray(MODEL.predict(row)).ravel()[0])
        probabilities = MODEL.predict_proba(row)[0] if hasattr(MODEL, "predict_proba") else None
        return prediction, probabilities
    except Exception:
        return None, None

def signal_quality(signal):
    """Simple screening indicators; not a clinical or validated EEG quality score."""
    x = np.asarray(signal, dtype=float)
    finite = np.isfinite(x).all()
    if not finite:
        return {"status": "Check required", "reason": "The signal contains missing or invalid values.", "peak": np.nan, "std": np.nan, "flat": True}
    peak = float(np.max(np.abs(x)))
    std = float(np.std(x))
    flat = std < 1e-6
    # Generic outlier flag relative to this recording; not an EEG diagnostic threshold.
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median))) + 1e-12
    robust_outliers = int(np.sum(np.abs(x - median) > 8 * 1.4826 * mad))
    if flat:
        status, reason = "Flat signal", "This channel has almost no variation. Check the selected recording."
    elif robust_outliers > 3:
        status, reason = "Unusual fluctuations", "Several unusually large deviations were found. Inspect the waveform before interpreting it."
    else:
        status, reason = "No obvious issue detected", "The basic screening checks did not identify a flat signal or repeated extreme deviations."
    return {"status": status, "reason": reason, "peak": peak, "std": std, "flat": flat, "outliers": robust_outliers}

def channel_label(c):
    return f"Channel {int(c)+1}"

# ------------------------ SIDEBAR ----------------------------
st.sidebar.markdown("## 🧠 NeuroSense")
st.sidebar.caption("EEG-Based Cognitive State Analysis")
st.sidebar.markdown("---")
PAGES = [
    "Overview",
    "EEG Explorer",
    "Brain Activity",
    "Workload Analysis",
    "Dataset",
    "Model Evaluation",
]
page = st.sidebar.radio("NAVIGATION", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**Research prototype**")
st.sidebar.caption("Stored STEW EEG recordings · 14 channels · 128 Hz")
st.sidebar.caption("No live wireless EEG device is connected.")

if any(v is None for v in [X, FEATURES, LABELS, SUBJECTS]):
    st.error("One or more required demonstration files are missing.")
    st.write("Required files: `STEW_X_demo.npy`, `STEW_features_demo.npy`, `STEW_labels_demo.npy`, `STEW_subjects_demo.npy`.")
    st.stop()

n = min(len(X), len(FEATURES), len(LABELS), len(SUBJECTS))
X, FEATURES, LABELS, SUBJECTS = X[:n], FEATURES[:n], LABELS[:n], SUBJECTS[:n]
if X.ndim != 3 or X.shape[1:] != (14, 256):
    st.error(f"Unexpected EEG array shape: {X.shape}. Expected (windows, 14, 256).")
    st.stop()
if FEATURES.ndim != 2 or FEATURES.shape[1] != 70:
    st.error(f"Unexpected feature array shape: {FEATURES.shape}. Expected (windows, 70).")
    st.stop()

PARTICIPANTS = sorted(np.unique(SUBJECTS).tolist())

def participant_selector(key="participant"):
    p = st.selectbox("Choose participant", PARTICIPANTS, key=key)
    idxs = np.where(SUBJECTS == p)[0]
    selected = st.selectbox(
        "Choose EEG window",
        idxs.tolist(),
        format_func=lambda idx: f"Recording {int(np.where(idxs == idx)[0][0])+1}  ·  Dataset row {idx}",
        key=f"{key}_window",
    )
    return p, idxs, int(selected)

def render_14_channel(index):
    fig = go.Figure()
    t = np.arange(X.shape[2]) / FS
    offsets = []
    for ch in range(14):
        y = X[index, ch, :].astype(float)
        scale = np.std(y) or 1.0
        offsets.append(scale)
        fig.add_trace(go.Scatter(
            x=t, y=y + ch * (np.nanmax(np.abs(X[index])) * 2.5),
            mode="lines", name=f"CH {ch+1}", line=dict(width=1.15)
        ))
    fig.update_layout(title="All 14 EEG channels · selected 2-second window",
                      xaxis_title="Time (seconds)", yaxis_title="Stacked signal amplitude",
                      showlegend=True)
    st.plotly_chart(themed(fig, 560), use_container_width=True)

def render_single_channel(index, channel):
    raw = X[index, channel, :].astype(float)
    filtered = safe_filter(raw)
    t = np.arange(len(raw)) / FS
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=raw, name="Original", line=dict(color="#65a9ff", width=1.4)))
    fig.add_trace(go.Scatter(x=t, y=filtered, name="1–40 Hz filtered", line=dict(color="#55d6bd", width=1.4)))
    fig.update_layout(title=f"{channel_label(channel)} · original and filtered signal",
                      xaxis_title="Time (seconds)", yaxis_title="Amplitude (dataset units)")
    st.plotly_chart(themed(fig, 390), use_container_width=True)
    return raw, filtered

def render_psd(signal, title="Power Spectral Density"):
    f, p = welch(signal, fs=FS, nperseg=min(128, len(signal)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=p, mode="lines", fill="tozeroy",
                             line=dict(color="#7b9cff", width=2), name="PSD"))
    for name, (lo, hi) in BANDS.items():
        fig.add_vrect(x0=lo, x1=hi, opacity=.10, line_width=0,
                      annotation_text=name, annotation_position="top")
    fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="Power spectral density",
                      xaxis_range=[0, 45])
    st.plotly_chart(themed(fig, 390), use_container_width=True)

def feature_band_means(index):
    matrix = band_features_for_window(index)
    if matrix is None:
        return None
    return pd.Series(np.mean(matrix, axis=0), index=BAND_NAMES)

def show_band_chart(index):
    means = feature_band_means(index)
    if means is None:
        st.info("Band feature data is unavailable for this recording.")
        return
    df = means.rename_axis("Frequency band").reset_index(name="Mean PSD feature")
    fig = px.bar(df, x="Frequency band", y="Mean PSD feature", color="Frequency band",
                 title="Average frequency-band features across 14 channels")
    st.plotly_chart(themed(fig, 360), use_container_width=True)
    st.caption("Values are averages of the model's extracted mean-PSD features. They are not direct measurements of workload.")

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown("""
    <div class="ns-hero">
      <div class="ns-eyebrow">BIOMEDICAL SIGNAL PROCESSING · MACHINE LEARNING</div>
      <h1>NEUROSENSE</h1>
      <p>Explore EEG recordings, understand brain-wave frequency patterns and examine
      machine-learning workload classifications through one interactive research platform.</p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    a.metric("EEG participants", f"{len(PARTICIPANTS)}", help="Participants represented in the demonstration subset.")
    b.metric("EEG channels", "14")
    c.metric("Sampling frequency", "128 Hz")
    d.metric("Recording duration", "2 seconds / window")

    st.markdown("### What can you explore?")
    cards = st.columns(3)
    with cards[0]:
        st.markdown('<div class="ns-card"><div class="ns-card-title">01 · SIGNAL EXPLORATION</div><h3>EEG Explorer</h3><p class="ns-muted">Inspect all 14 channels, an individual electrode, filtered waveforms and the frequency spectrum.</p></div>', unsafe_allow_html=True)
    with cards[1]:
        st.markdown('<div class="ns-card"><div class="ns-card-title">02 · BRAIN ACTIVITY</div><h3>Frequency Analysis</h3><p class="ns-muted">Explore Delta, Theta, Alpha, Beta and Gamma features, plus derived ratios for research exploration.</p></div>', unsafe_allow_html=True)
    with cards[2]:
        st.markdown('<div class="ns-card"><div class="ns-card-title">03 · MODEL INSIGHTS</div><h3>Workload Analysis</h3><p class="ns-muted">View model classifications, compare recordings and read plain-language explanations.</p></div>', unsafe_allow_html=True)

    st.markdown("### How NeuroSense works")
    flow = st.columns(5)
    stages = [
        ("01", "STEW dataset", "Stored EEG recordings"),
        ("02", "Preprocessing", "1–40 Hz bandpass"),
        ("03", "Feature extraction", "Welch PSD · 70 features"),
        ("04", "Classification", "Random Forest model"),
        ("05", "Visualisation", "Interactive analysis"),
    ]
    for col, (num, title, desc) in zip(flow, stages):
        with col:
            st.markdown(f'<div class="ns-card"><div class="ns-card-title">{num}</div><b>{title}</b><p class="ns-muted">{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("### Project scope")
    st.markdown("""
    <div class="ns-note">
    <b>What this prototype does:</b> analyses previously recorded EEG data, visualises signal
    characteristics and presents the output of a trained workload classifier.
    </div>
    <div class="ns-warning">
    <b>What it does not do:</b> it is not connected to a wireless headset and does not diagnose
    drowsiness, emotion, attention disorders or medical conditions. Those require separately
    labelled data and validated models.
    </div>
    """, unsafe_allow_html=True)
    st.caption("Dataset: STEW (Simultaneous Task EEG Workload). The dashboard uses the demonstration files stored with this application.")

# ============================================================
# PAGE 2: EEG EXPLORER
# ============================================================
elif page == "EEG Explorer":
    st.title("EEG Signal Explorer")
    st.write("Select a participant and recording to inspect the electrical signals captured across the 14 EEG channels.")
    p, idxs, idx = participant_selector("explorer")
    m1, m2, m3 = st.columns(3)
    m1.metric("Participant", str(p))
    m2.metric("Recording within participant", f"{int(np.where(idxs == idx)[0][0])+1} / {len(idxs)}")
    m3.metric("Signal duration", f"{WINDOW_SECONDS:.1f} seconds")

    st.markdown("### 1. Complete 14-channel recording")
    render_14_channel(idx)

    st.markdown("### 2. Inspect an individual channel")
    channel = st.selectbox("Choose an electrode", list(range(14)), format_func=channel_label, key="explorer_channel")
    raw, filtered = render_single_channel(idx, channel)

    st.markdown("### 3. Frequency spectrum (PSD)")
    render_psd(filtered, f"Filtered signal PSD · {channel_label(channel)}")

    with st.expander("View time-frequency spectrogram"):
        f, t, sxx = spectrogram(filtered, fs=FS, nperseg=64, noverlap=48)
        fig = go.Figure(go.Heatmap(x=t, y=f, z=10*np.log10(sxx + 1e-12),
                                   colorscale="Viridis", colorbar=dict(title="Power (dB)")))
        fig.update_layout(title="Spectrogram · frequency content over time",
                          xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)", yaxis_range=[0, 45])
        st.plotly_chart(themed(fig, 420), use_container_width=True)

# ============================================================
# PAGE 3: BRAIN ACTIVITY
# ============================================================
elif page == "Brain Activity":
    st.title("Brain Activity & Frequency Bands")
    st.write("Explore the five frequency-band features extracted from each EEG recording.")
    p, idxs, idx = participant_selector("brain")
    st.markdown("### The five EEG frequency bands")
    band_descriptions = [
        ("Delta", "1–4 Hz", "Low-frequency activity; commonly studied in sleep-related EEG research."),
        ("Theta", "4–8 Hz", "Studied in relation to memory, drowsiness and cognitive processing."),
        ("Alpha", "8–13 Hz", "Often associated with relaxed wakefulness and changes in attention."),
        ("Beta", "13–30 Hz", "Often studied in relation to alertness and active mental processing."),
        ("Gamma", "30–40 Hz", "Studied in relation to higher-level information processing."),
    ]
    cols = st.columns(5)
    for col, (name, hz, desc) in zip(cols, band_descriptions):
        with col:
            st.markdown(f'<div class="ns-card"><div class="ns-card-title">{hz}</div><h3>{name}</h3><p class="ns-muted">{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("### Feature profile for the selected recording")
    show_band_chart(idx)
    matrix = band_features_for_window(idx)
    if matrix is not None:
        df = pd.DataFrame(matrix, columns=BAND_NAMES)
        df.insert(0, "EEG channel", [f"Channel {i}" for i in range(1, 15)])
        with st.expander("View all 70 extracted features"):
            st.dataframe(df.round(6), use_container_width=True, hide_index=True)

        st.markdown("### Exploratory frequency ratios")
        band_avg = np.mean(matrix, axis=0)
        band = dict(zip(BAND_NAMES, band_avg))
        alpha_beta = float(band["Alpha"] / (band["Beta"] + 1e-12))
        theta_beta = float(band["Theta"] / (band["Beta"] + 1e-12))
        r1, r2 = st.columns(2)
        r1.metric("Alpha / Beta feature ratio", f"{alpha_beta:.3f}")
        r2.metric("Theta / Beta feature ratio", f"{theta_beta:.3f}")
        st.info("These are exploratory ratios calculated from the extracted mean-PSD features. They are not validated measures of focus, drowsiness or emotion in this project.")
    st.markdown("### What about focus, drowsiness and emotion?")
    st.markdown("""
    EEG frequency bands are studied in research on attention, sleepiness and emotional processing.
    However, this project was trained only for the dataset's two workload classes. It has no
    validated focus, drowsiness or emotion labels, so NeuroSense does not claim to detect those
    states. The frequency features can be explored as research indicators, not as user diagnoses.
    """)

# ============================================================
# PAGE 4: WORKLOAD ANALYSIS
# ============================================================
elif page == "Workload Analysis":
    st.title("Cognitive Workload Analysis")
    st.write("Examine the trained model's output for a selected EEG recording and understand what the result means.")
    p, idxs, idx = participant_selector("workload")
    actual = int(LABELS[idx])
    prediction, probabilities = get_prediction(idx)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Model classification")
        if prediction is None:
            st.error("The trained model could not produce a prediction. Check that the model file is available.")
        else:
            st.markdown(f'<div class="ns-card"><div class="ns-card-title">PREDICTED DATASET CLASS</div><div class="ns-card-value">Class {prediction}</div><p class="ns-muted">This is the category assigned by the trained Random Forest model.</p></div>', unsafe_allow_html=True)
            if prediction == actual:
                st.success("The model output matches the stored dataset label for this recording.")
            else:
                st.warning("The model output differs from the stored dataset label for this recording.")
    with right:
        st.markdown("### Recording information")
        st.metric("Participant", str(p))
        st.metric("Recording number", f"{int(np.where(idxs == idx)[0][0])+1}")
        st.metric("Stored dataset label", f"Class {actual}")

    if probabilities is not None:
        prob_df = pd.DataFrame({"Class": [f"Class {int(c)}" for c in MODEL.classes_], "Probability": probabilities})
        fig = px.bar(prob_df, x="Class", y="Probability", color="Class",
                     title="Model output probabilities", text=prob_df["Probability"].map(lambda v: f"{v:.1%}"))
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(themed(fig, 330), use_container_width=True)
        st.caption("Model probabilities are not clinical certainty scores.")

    st.markdown("### What should the user understand?")
    st.markdown("""
    - The displayed class is the model's classification of this stored EEG window.
    - A matching result means the model output agrees with the dataset label for this window.
    - A mismatch is a normal possibility in machine-learning classification.
    - The project currently predicts only the two dataset classes.
    """)
    st.warning("The original STEW numeric class mapping must be verified before Class 0 and Class 1 can safely be renamed Low Workload and High Workload.")
    st.markdown("### General wellbeing guidance")
    st.markdown("""
    If you personally feel tired or overloaded while studying, consider a short break,
    hydration and a comfortable study environment. This is general wellbeing guidance;
    it is not generated as a diagnosis from the EEG prediction.
    """)

    st.markdown("### Signal screening for this recording")
    channel_checks = [signal_quality(X[idx, ch, :]) for ch in range(14)]
    bad = sum(q["status"] != "No obvious issue detected" for q in channel_checks)
    q1, q2, q3 = st.columns(3)
    q1.metric("Channels screened", "14")
    q2.metric("Channels flagged", str(bad))
    q3.metric("Screening method", "Basic checks")
    qdf = pd.DataFrame({
        "Channel": [f"Channel {i+1}" for i in range(14)],
        "Screening result": [q["status"] for q in channel_checks],
        "Peak absolute amplitude": [q["peak"] for q in channel_checks],
        "Standard deviation": [q["std"] for q in channel_checks],
    })
    st.dataframe(qdf.round(5), use_container_width=True, hide_index=True)
    st.caption("Basic screening only. Thresholds are generic and not clinically validated.")

# ============================================================
# PAGE 5: DATASET
# ============================================================
elif page == "Dataset":
    st.title("STEW Dataset")
    st.write("Explore the source data used to develop and demonstrate NeuroSense.")
    unique_subjects = np.unique(SUBJECTS)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Windows in this app", f"{n:,}")
    d2.metric("Participants represented", str(len(unique_subjects)))
    d3.metric("EEG channels per window", "14")
    d4.metric("Samples per channel", "256")
    st.markdown("### Dataset structure")
    structure = pd.DataFrame({
        "Parameter": ["Dataset", "EEG channels", "Sampling frequency", "Samples per window", "Window duration", "Extracted features", "Feature construction"],
        "Description": ["STEW – Simultaneous Task EEG Workload", "14", "128 Hz", "256", "2 seconds", "70 per window", "5 mean-PSD band features × 14 channels"],
    })
    st.dataframe(structure, use_container_width=True, hide_index=True)
    st.markdown("### Participant selection")
    subject_counts = pd.Series(SUBJECTS).value_counts().sort_index()
    fig = px.bar(x=subject_counts.index.astype(str), y=subject_counts.values,
                 labels={"x": "Participant ID", "y": "Number of stored windows"},
                 title="Number of demonstration windows available per participant")
    st.plotly_chart(themed(fig, 370), use_container_width=True)
    st.markdown("### Explore a dataset recording")
    p, idxs, idx = participant_selector("dataset")
    st.write(f"Selected participant **{p}** · recording **{int(np.where(idxs == idx)[0][0])+1}**")
    st.write(f"Raw EEG array shape for this window: `{X[idx].shape}` (14 channels × 256 samples).")
    st.write(f"Feature vector shape: `{FEATURES[idx].shape}` (70 features).")
    with st.expander("View raw numerical EEG samples"):
        st.dataframe(pd.DataFrame(X[idx].T, columns=[f"Channel {i}" for i in range(1, 15)]).round(4), use_container_width=True)
    st.warning(f"This deployed application contains {n:,} demonstration windows. The full original STEW dataset contains 28,512 windows from 48 participants; do not confuse the demo subset with the complete dataset.")

# ============================================================
# PAGE 6: MODEL EVALUATION
# ============================================================
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write("Technical performance results are placed here for project review and transparency.")
    if COMPARISON is not None and not COMPARISON.empty:
        st.markdown("### Classifier comparison")
        st.dataframe(COMPARISON, use_container_width=True, hide_index=True)
        numeric = COMPARISON.select_dtypes(include=np.number).columns.tolist()
        name_col = next((c for c in COMPARISON.columns if c not in numeric), None)
        acc_col = next((c for c in numeric if "accuracy" in c.lower()), None)
        f1_col = next((c for c in numeric if "f1" in c.lower()), None)
        if name_col and acc_col:
            fig = px.bar(COMPARISON, x=name_col, y=acc_col, color=name_col,
                         title="Accuracy comparison", text_auto=".2f")
            st.plotly_chart(themed(fig, 360), use_container_width=True)
        if name_col and f1_col:
            fig = px.bar(COMPARISON, x=name_col, y=f1_col, color=name_col,
                         title="F1-score comparison", text_auto=".3f")
            st.plotly_chart(themed(fig, 360), use_container_width=True)
    else:
        st.info("Model comparison file is not available.")

    if PREDICTIONS is not None:
        st.markdown("### Saved test predictions")
        st.dataframe(PREDICTIONS.head(50), use_container_width=True, hide_index=True)

    if MODEL is not None and hasattr(MODEL, "feature_importances_"):
        st.markdown("### Random Forest feature importance")
        importances = np.asarray(MODEL.feature_importances_)
        if len(importances) == 70:
            names = [f"Ch {ch+1} · {band}" for ch in range(14) for band in BAND_NAMES]
            imp = pd.DataFrame({"Feature": names, "Importance": importances})
            imp = imp.nlargest(15, "Importance").sort_values("Importance")
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                         title="15 most influential model features")
            st.plotly_chart(themed(fig, 500), use_container_width=True)
            st.caption("Feature importance describes how the fitted Random Forest used input features. It does not establish causation.")
    st.caption("The model was evaluated on held-out participants. Accuracy and F1-score are dataset evaluation metrics, not measures of clinical performance.")

st.sidebar.markdown("---")
st.sidebar.caption("NeuroSense · Academic research prototype")
