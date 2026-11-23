from pathlib import Path
import gzip, io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import butter, sosfiltfilt, welch

# ============================================================
# NEUROSENSE | A guided EEG analysis application
# Built for the stored STEW EEG research dataset
# ============================================================

ROOT = Path(__file__).parent
FS = 128
BAND_RANGES = {
    "Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 13),
    "Beta": (13, 30), "Gamma": (30, 40)
}
BAND_NAMES = list(BAND_RANGES)

# IMPORTANT: verify this encoding against the exact y-label creation used during training.
# This display assumes 0 = Low Workload and 1 = High Workload.
WORKLOAD_LABELS = {0: "Low Workload", 1: "High Workload"}

def workload_name(value):
    try:
        return WORKLOAD_LABELS[int(value)]
    except (KeyError, TypeError, ValueError):
        return f"Unmapped label ({value})"

st.set_page_config(
    page_title="NeuroSense | Brain Signal Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- DESIGN -----------------------------
st.markdown("""
<style>
.stApp {background:#080e1b;color:#eaf0fb}
[data-testid="stSidebar"] {background:#101a2b;border-right:1px solid #243650}
[data-testid="stHeader"] {background:rgba(8,14,27,.9)}
h1,h2,h3 {color:#f3f7ff;letter-spacing:-.025em}
p,li,label {color:#d5dfef}
.ns-hero {
 background:linear-gradient(115deg,#102c4b,#172544 60%,#20264a);
 border:1px solid #315477;border-radius:22px;padding:32px;margin:6px 0 24px
}
.ns-hero .eyebrow {color:#74d9d0;font-weight:800;letter-spacing:.16em;font-size:.76rem}
.ns-hero h1 {font-size:2.65rem;margin:.45rem 0}
.ns-hero p {font-size:1.06rem;color:#c7d7eb;max-width:900px}
.ns-card {background:#111f33;border:1px solid #2a405d;border-radius:16px;padding:19px;min-height:135px}
.ns-card small {color:#79d6cf;font-weight:800;letter-spacing:.1em}
.ns-card p {color:#aebed3;font-size:.92rem}
.ns-step {background:#14263d;border-left:4px solid #65cfc4;padding:13px 16px;border-radius:9px;margin:8px 0}
.ns-note {background:#14263d;border:1px solid #2d4b69;border-radius:12px;padding:14px 17px}
.ns-warn {background:#33291b;border:1px solid #72552c;border-radius:12px;padding:14px 17px}
div[data-testid="stMetric"] {background:#111f33;border:1px solid #2a405d;padding:13px 15px;border-radius:13px}
.stButton>button {border-radius:10px;font-weight:700;min-height:2.7rem}
</style>
""", unsafe_allow_html=True)

# ----------------------------- LOAD DATA -----------------------------
@st.cache_data
def load_array(name):
    p = ROOT / name
    return np.load(p, allow_pickle=False) if p.exists() else None

@st.cache_data
def load_table(name):
    p = ROOT / name
    return pd.read_csv(p) if p.exists() else None

@st.cache_resource
def load_classifier():
    p = ROOT / "eeg_workload_model.pkl.gz"
    if not p.exists():
        return None
    return joblib.load(io.BytesIO(gzip.decompress(p.read_bytes())))

raw = load_array("STEW_X_demo.npy")
features = load_array("STEW_features_demo.npy")
labels = load_array("STEW_labels_demo.npy")
people = load_array("STEW_subjects_demo.npy")
model = load_classifier()
model_table = load_table("model_comparison.csv")
saved_predictions = load_table("test_predictions.csv")

# ----------------------------- HELPERS -----------------------------
def style_plot(fig, height=370):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080e1b", plot_bgcolor="#080e1b",
        font_color="#eaf0fb", height=height, margin=dict(l=25,r=25,t=55,b=35)
    )
    fig.update_xaxes(gridcolor="#203149")
    fig.update_yaxes(gridcolor="#203149")
    return fig

def get_prediction(i):
    if model is None:
        return None, None
    try:
        x = features[i].reshape(1, -1)
        pred = int(np.asarray(model.predict(x)).ravel()[0])
        prob = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None
        return pred, prob
    except Exception:
        return None, None

def filtered(signal):
    try:
        sos = butter(4, [1,40], btype="bandpass", fs=FS, output="sos")
        return sosfiltfilt(sos, np.asarray(signal, dtype=float))
    except Exception:
        return np.asarray(signal, dtype=float)

def get_person_rows(person):
    return np.where(people == person)[0]

def participant_name(person):
    return f"Participant {int(person):02d}"

def choose_recording(person, key):
    rows = get_person_rows(person)
    if len(rows) == 0:
        return None, rows
    chosen = st.selectbox(
        "Choose a recording",
        list(range(len(rows))),
        format_func=lambda k: f"Recording {k+1:03d}",
        key=key
    )
    return int(rows[chosen]), rows

def feature_matrix(i):
    return features[i].reshape(14, 5)

def band_means(i):
    return np.mean(feature_matrix(i), axis=0)


def coherence_matrix(i, band=(8, 30)):
    """Mean magnitude-squared coherence for every pair of EEG sensors."""
    from scipy.signal import coherence
    signals = np.asarray(raw[i], dtype=float)
    matrix = np.eye(signals.shape[0], dtype=float)
    for a in range(signals.shape[0]):
        for b in range(a + 1, signals.shape[0]):
            f, cxy = coherence(signals[a], signals[b], fs=FS, nperseg=128)
            keep = (f >= band[0]) & (f <= band[1])
            value = float(np.mean(cxy[keep])) if np.any(keep) else 0.0
            matrix[a, b] = matrix[b, a] = value
    return matrix


def local_feature_sensitivity(i):
    """Model-agnostic one-feature-at-a-time sensitivity around this recording."""
    if model is None or not hasattr(model, "predict_proba"):
        return None
    x = np.asarray(features[i], dtype=float).reshape(-1)
    try:
        classes = list(model.classes_)
        pred = int(np.asarray(model.predict(x.reshape(1, -1))).ravel()[0])
        class_pos = classes.index(pred)
        base_prob = float(model.predict_proba(x.reshape(1, -1))[0, class_pos])
        reference = np.nanmedian(features, axis=0)
        effects = []
        for j in range(len(x)):
            altered = x.copy()
            altered[j] = reference[j]
            changed_prob = float(model.predict_proba(altered.reshape(1, -1))[0, class_pos])
            effects.append(base_prob - changed_prob)
        names = [f"Sensor {ch+1} · {band}" for ch in range(14) for band in BAND_NAMES]
        return pd.DataFrame({"Feature": names, "Sensitivity": effects,
                             "Absolute effect": np.abs(effects)}).sort_values("Absolute effect", ascending=False)
    except Exception:
        return None


def prediction_confidence(i):
    pred, probs = get_prediction(i)
    if pred is None or probs is None:
        return pred, None, "Unavailable"
    confidence = float(np.max(probs))
    # This is a display band, not a calibrated probability guarantee.
    level = "Higher model confidence" if confidence >= .75 else ("Moderate model confidence" if confidence >= .60 else "Low model confidence")
    return pred, confidence, level

def show_band_chart(i, title="Frequency pattern in this recording"):
    values = band_means(i)
    df = pd.DataFrame({"Brain-wave band": BAND_NAMES, "Feature value": values})
    fig = px.bar(df, x="Brain-wave band", y="Feature value", color="Brain-wave band", title=title)
    st.plotly_chart(style_plot(fig, 350), use_container_width=True)

def show_all_sensors(i):
    t = np.arange(raw.shape[2])/FS
    fig = go.Figure()
    spacing = max(float(np.max(np.abs(raw[i]))) * 2.6, 1.0)
    for ch in range(14):
        fig.add_trace(go.Scatter(
            x=t, y=raw[i,ch,:] + ch*spacing, mode="lines",
            name=f"Sensor {ch+1}", line=dict(width=1.2)
        ))
    fig.update_layout(title="Brain signals recorded by all 14 sensors",
                      xaxis_title="Time (seconds)",
                      yaxis_title="Signals shown in separate rows")
    st.plotly_chart(style_plot(fig, 540), use_container_width=True)

def show_sensor(i, sensor):
    sig = raw[i, sensor, :]
    filt = filtered(sig)
    t = np.arange(len(sig))/FS
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t,y=sig,name="Original signal",line=dict(color="#73a9ff")))
    fig.add_trace(go.Scatter(x=t,y=filt,name="Filtered signal",line=dict(color="#5dd5bd")))
    fig.update_layout(title=f"Sensor {sensor+1}: original and cleaned signal",
                      xaxis_title="Time (seconds)",yaxis_title="Signal strength (dataset units)")
    st.plotly_chart(style_plot(fig,360),use_container_width=True)
    f, psd = welch(filt, fs=FS, nperseg=128)
    fig2 = go.Figure(go.Scatter(x=f,y=psd,mode="lines",fill="tozeroy",line=dict(color="#a58cff")))
    for name,(lo,hi) in BAND_RANGES.items():
        fig2.add_vrect(x0=lo,x1=hi,opacity=.09,line_width=0,
                       annotation_text=name,annotation_position="top")
    fig2.update_layout(title="Where the signal's energy appears across frequencies",
                       xaxis_title="Frequency (Hz)",yaxis_title="Power spectral density",
                       xaxis_range=[0,45])
    st.plotly_chart(style_plot(fig2,360),use_container_width=True)
    return sig

def show_prediction(i):
    pred, probs = get_prediction(i)
    if pred is None:
        st.error("The saved machine-learning model is not available. Check the model file in the project repository.")
        return
    st.markdown(f"### Model result: **{workload_name(pred)}**")
    if probs is not None:
        df = pd.DataFrame({"Predicted workload":[workload_name(c) for c in model.classes_],
                           "Model output":[float(v) for v in probs]})
        fig = px.bar(df,x="Predicted workload",y="Model output",color="Predicted workload",
                     title="Model output for each workload level",
                     text=df["Model output"].map(lambda v:f"{v:.1%}"))
        fig.update_yaxes(range=[0,1],tickformat=".0%")
        st.plotly_chart(style_plot(fig,300),use_container_width=True)
    st.caption("These are model output scores, not medical certainty. The displayed workload names use the WORKLOAD_LABELS mapping defined near the top of app.py; confirm that mapping matches the labels used when the model was trained.")

# ----------------------------- NAVIGATION -----------------------------
st.sidebar.markdown("## 🧠 NeuroSense")
st.sidebar.caption("Understand brain signals. Explore workload patterns.")
st.sidebar.markdown("---")
pages = [
    "Welcome",
    "Explore an EEG recording",
    "Understand brain-wave activity",
    "Workload report",
    "Compare recordings",
    "About the dataset",
    "Brain connectivity",
    "Why did the AI predict this?",
    "Workload pattern over recordings",
    "Research results"
]
page = st.sidebar.radio("ANALYSIS SECTIONS", pages, label_visibility="visible", key="nav_page")
st.sidebar.markdown("---")
st.sidebar.caption("Research prototype · Stored EEG data")
st.sidebar.caption("No live headset is connected.")

if raw is None or features is None or labels is None or people is None:
    st.error("The application cannot find all four EEG demonstration files.")
    st.write("Please keep these files beside app.py: `STEW_X_demo.npy`, `STEW_features_demo.npy`, `STEW_labels_demo.npy`, `STEW_subjects_demo.npy`.")
    st.stop()

n = min(len(raw),len(features),len(labels),len(people))
raw,features,labels,people = raw[:n],features[:n],labels[:n],people[:n]
if raw.ndim != 3 or raw.shape[1:] != (14,256) or features.ndim != 2 or features.shape[1] != 70:
    st.error("The data files do not have the expected dimensions. Expected 14 sensors × 256 samples and 70 features per recording.")
    st.stop()
participant_ids = sorted(np.unique(people).tolist())

# ============================================================
# 1. WELCOME
# ============================================================
if page == "Welcome":
    st.markdown("""
    <div class="ns-hero">
      <div class="eyebrow">EEG · COGNITIVE WORKLOAD RESEARCH</div>
      <h1>NEUROSENSE</h1>
      <p><b>From brain signals to interpretable workload evidence.</b><br>
      NeuroSense analyses multichannel EEG recordings, extracts frequency patterns,
      examines relationships between sensors and uses machine learning to study
      workload-related brain activity.</p>
    </div>
    """, unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("EEG sensors", "14")
    k2.metric("Sampling rate", "128 Hz")
    k3.metric("Window duration", "2 seconds")
    k4.metric("Features per window", "70")
    st.markdown("## What makes this more than a classifier?")
    n1,n2,n3 = st.columns(3)
    with n1:
        st.markdown('<div class="ns-card"><small>01 · SIGNAL RELATIONSHIPS</small><h3>Brain connectivity</h3><p>Measure frequency-domain synchrony between sensor pairs, not only the activity at each sensor.</p></div>', unsafe_allow_html=True)
    with n2:
        st.markdown('<div class="ns-card"><small>02 · MODEL TRANSPARENCY</small><h3>Prediction explanation</h3><p>Show which sensor and frequency-band features most influence an individual model result.</p></div>', unsafe_allow_html=True)
    with n3:
        st.markdown('<div class="ns-card"><small>03 · RECORDING PATTERNS</small><h3>Across-window analysis</h3><p>Explore how model outputs and EEG features vary across the stored windows for a participant.</p></div>', unsafe_allow_html=True)
    st.markdown("## Analysis workflow")
    steps = st.columns(5)
    for col, (num, title, desc) in zip(steps, [
        ("1", "EEG recordings", "14 sensors · STEW dataset"),
        ("2", "Preprocessing", "1–40 Hz bandpass filter"),
        ("3", "Feature analysis", "PSD bands + connectivity"),
        ("4", "Machine learning", "Random Forest classification"),
        ("5", "Interpretation", "Feature contribution + trends")
    ]):
        with col:
            st.markdown(f'<div class="ns-card"><small>STEP {num}</small><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ns-note"><b>Project scope:</b> this is an offline research prototype using stored STEW EEG data.
    It does not acquire signals wirelessly, diagnose a person or directly identify emotion, attention or fatigue.
    The model predicts the dataset's two workload-related classes.</div>
    """, unsafe_allow_html=True)
    st.markdown("## Open an analysis")
    st.write("Choose a section from the navigation panel. Start with EEG Signal Analysis, then explore Brain Connectivity and AI Explanation.")

# ============================================================
# 2. EXPLORE A RECORDING
# ============================================================
elif page == "Explore an EEG recording":
    st.title("Explore an EEG recording")
    st.write("Choose a person and a recording. NeuroSense will show the brain signals captured during that short recording.")
    person=st.selectbox("1. Choose a participant",participant_ids,format_func=participant_name,key="explore_person")
    idx,rows=choose_recording(person,"explore_record")
    st.markdown('<div class="ns-step"><b>What you are looking at</b><br>Each recording lasts 2 seconds. Fourteen sensors record tiny electrical changes from different positions on the scalp.</div>',unsafe_allow_html=True)
    st.markdown("## The 14 recorded EEG traces")
    st.info("These are the 14 channels stored in the STEW data file, plotted as separate vertically offset traces so they can be viewed together. They are not a picture of the brain or a scalp electrode-position map. The vertical spacing is added only for readability; it is not a physical sensor distance.")
    show_all_sensors(idx)
    st.markdown("## Zoom in on one sensor")
    sensor=st.selectbox("Choose a sensor position",list(range(14)),format_func=lambda x:f"Sensor {x+1}",key="explore_sensor")
    sig=show_sensor(idx,sensor)
    st.markdown("### About this waveform")
    st.write("The horizontal axis shows time in seconds. The vertical axis shows the recorded signal amplitude in the dataset’s numerical units. The filtered trace suppresses frequencies outside the selected 1–40 Hz passband; it is not a separate recording.")

# ============================================================
# 3. UNDERSTAND BRAIN-WAVE ACTIVITY
# ============================================================
elif page == "Understand brain-wave activity":
    st.title("Understand brain-wave activity")
    st.write("An EEG recording contains a mixture of electrical activity at different frequencies. NeuroSense groups the frequency content into five conventional bands.")
    bands = [
        ("Delta","1–4 Hz","Slow-frequency activity; commonly studied in sleep and other brain-state research."),
        ("Theta","4–8 Hz","Studied in memory, learning and cognitive processing."),
        ("Alpha","8–13 Hz","Often observed during relaxed wakefulness; its pattern can change with task conditions."),
        ("Beta","13–30 Hz","Studied in active mental processing and sensorimotor activity."),
        ("Gamma","30–40 Hz","Studied in higher-frequency neural processing; interpretation requires care because muscle noise can affect this range.")
    ]
    cols=st.columns(5)
    for col,(name,hz,desc) in zip(cols,bands):
        with col:
            st.markdown(f'<div class="ns-card"><small>{hz}</small><h3>{name}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
    st.markdown("## Explore frequency activity in one recording")
    person=st.selectbox("Choose a participant",participant_ids,format_func=participant_name,key="bands_person")
    idx,rows=choose_recording(person,"bands_record")
    values=band_means(idx)
    df=pd.DataFrame({"Frequency band":BAND_NAMES,"Mean PSD feature":values})
    fig=px.bar(df,x="Frequency band",y="Mean PSD feature",color="Frequency band",
               title="Average power spectral density feature by frequency band")
    st.plotly_chart(style_plot(fig,380),use_container_width=True)

    st.markdown("## Mean PSD features across the 14 sensors")
    table=pd.DataFrame(features[idx].reshape(14,5),columns=BAND_NAMES)
    table.insert(0,"Sensor", [f"Sensor {i}" for i in range(1,15)])
    st.dataframe(table.round(6),use_container_width=True,hide_index=True)
    st.markdown("### How to read this table")
    st.markdown("""
    Each **row** represents one of the 14 EEG sensors. Each **column** represents one frequency band.
    A cell is the average Power Spectral Density (PSD) feature extracted from that sensor's signal within that band's frequency range.

    - **PSD** describes how the signal's power is distributed over frequency.
    - **Hz** (hertz) describes frequency: 1 Hz means one cycle per second.
    - **Mean PSD feature** is calculated by averaging the PSD values across frequency bins inside the selected band. It is a numerical signal feature, not a workload score.
    - A larger value means more measured signal power density in that frequency range for that sensor, compared with a smaller value from the same processing pipeline.
    - Compare values within the same band and processing setup. Do not assume that a larger Alpha or Beta value automatically means more focus, stress or workload.

    **Units:** PSD units are the square of the original EEG signal unit per hertz (for example, µV²/Hz if the source signal is stored in microvolts). This app preserves the numerical scale of the supplied feature file; the source file's physical scaling must be checked before displaying a definite µV²/Hz unit. Therefore, the table is shown as *mean PSD feature values* rather than assigning an unverified physical unit.
    """)
    st.markdown("### Why are there 70 features?")
    st.write("There are 14 sensors and 5 frequency bands per sensor. The feature vector therefore contains 14 × 5 = 70 numbers for each two-second EEG window. The Random Forest uses this 70-value description as its input.")
    st.markdown("### What can these values tell us?")
    st.write("They describe the frequency composition of the recorded EEG. They help compare recordings and provide inputs to the workload model. They do not independently diagnose attention, emotion, tiredness or a medical condition.")
    st.markdown("### Research note: focus and emotions")
    st.write("EEG frequency bands are studied in attention, sleepiness and emotion research. However, this project's trained model has workload labels only. NeuroSense therefore does not claim to recognise a person's focus or emotion.")

# ============================================================
# 4. WORKLOAD REPORT
# ============================================================
elif page == "Workload report":
    st.title("Workload report")
    st.write("Select a stored EEG recording to view the model's workload prediction and the dataset's reference label.")
    person=st.selectbox("Choose a participant",participant_ids,format_func=participant_name,key="report_person")
    idx,rows=choose_recording(person,"report_record")
    pred,probs=get_prediction(idx)
    actual=int(labels[idx])
    if pred is None:
        st.error("The trained model could not create a result. Check that the model file is present.")
    else:
        x,y=st.columns(2)
        with x:
            st.markdown(f'<div class="ns-card"><small>AI MODEL PREDICTION</small><div style="font-size:2rem;font-weight:800;margin-top:10px">{workload_name(pred)}</div><p>Predicted workload class for this recording.</p></div>',unsafe_allow_html=True)
        with y:
            st.markdown(f'<div class="ns-card"><small>DATASET REFERENCE</small><div style="font-size:2rem;font-weight:800;margin-top:10px">{workload_name(actual)}</div><p>Workload label stored with this recording.</p></div>',unsafe_allow_html=True)
        if pred==actual:
            st.success("The model prediction matches the dataset reference label for this recording.")
        else:
            st.warning("The model prediction does not match the dataset reference label for this recording.")
        if probs is not None:
            df=pd.DataFrame({"Workload level":[workload_name(c) for c in model.classes_],"Model output":probs})
            fig=px.bar(df,x="Workload level",y="Model output",color="Workload level",
                       text=df["Model output"].map(lambda v:f"{v:.1%}"),
                       title="Model output for low- and high-workload classes")
            fig.update_yaxes(range=[0,1],tickformat=".0%")
            st.plotly_chart(style_plot(fig,310),use_container_width=True)
            st.caption("The percentages are the model's output scores for the two classes. They are not medical certainty and may not be calibrated probabilities.")
    st.markdown("## Understanding your result")
    st.write("The model analyses 70 frequency-domain features extracted from the 14 EEG sensors and selects the workload class whose learned patterns best match this recording.")
    st.markdown("## General wellbeing information")
    st.write("This page provides general information only. If a person feels mentally overloaded during study or work, it may help to pause briefly, take a screen break, hydrate, and return to the task after resting. Maintain regular sleep, meals and movement. These suggestions are not generated from the EEG prediction and are not medical advice.")
    st.caption("NeuroSense is a research prototype. It does not diagnose a health condition or measure a person's overall wellbeing.")

# ============================================================
# 5. COMPARE RECORDINGS
# ============================================================
elif page == "Compare recordings":
    st.title("Compare two EEG recordings")
    st.write("Choose two recordings to compare their brain-wave patterns and the AI model's outputs.")
    left,right=st.columns(2)
    with left:
        st.markdown("### Recording A")
        p1=st.selectbox("Choose first participant",participant_ids,format_func=participant_name,key="compare_p1")
        i1,r1=choose_recording(p1,"compare_r1")
    with right:
        st.markdown("### Recording B")
        p2=st.selectbox("Choose second participant",participant_ids,format_func=participant_name,key="compare_p2")
        i2,r2=choose_recording(p2,"compare_r2")
    if i1 is not None and i2 is not None:
        pred1,_=get_prediction(i1)
        pred2,_=get_prediction(i2)
        a,b=st.columns(2)
        a.metric("Recording A · predicted workload",workload_name(pred1) if pred1 is not None else "Unavailable")
        b.metric("Recording B · predicted workload",workload_name(pred2) if pred2 is not None else "Unavailable")
        df=pd.DataFrame({"Brain-wave band":BAND_NAMES,
                         "Recording A":band_means(i1),
                         "Recording B":band_means(i2)})
        fig=go.Figure()
        fig.add_trace(go.Bar(x=df["Brain-wave band"],y=df["Recording A"],name=f"{participant_name(p1)}"))
        fig.add_trace(go.Bar(x=df["Brain-wave band"],y=df["Recording B"],name=f"{participant_name(p2)}"))
        fig.update_layout(barmode="group",title="Compare the five frequency-band features",
                          xaxis_title="Frequency band",yaxis_title="Extracted feature value")
        st.plotly_chart(style_plot(fig,390),use_container_width=True)
        with st.expander("Compare the two recordings' waveforms"):
            s1=st.selectbox("Choose a sensor to compare",list(range(14)),format_func=lambda x:f"Sensor {x+1}",key="compare_sensor")
            t=np.arange(256)/FS
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=t,y=raw[i1,s1,:],name=f"{participant_name(p1)}"))
            fig.add_trace(go.Scatter(x=t,y=raw[i2,s1,:],name=f"{participant_name(p2)}"))
            fig.update_layout(title="Same sensor position, two recordings",xaxis_title="Time (seconds)",yaxis_title="Signal strength")
            st.plotly_chart(style_plot(fig,360),use_container_width=True)
        st.caption("Differences are descriptive comparisons of stored recordings. They do not establish that one participant is more focused, sleepy or emotionally different.")

# ============================================================
# 6. DATASET
# ============================================================
elif page == "About the dataset":
    st.title("About the STEW dataset")
    st.write("NeuroSense uses stored multichannel EEG recordings from the STEW (Simultaneous Task EEG Workload) research dataset.")
    a,b,c,d=st.columns(4)
    a.metric("Demo windows loaded",f"{n:,}")
    b.metric("Participants in demo",len(participant_ids))
    c.metric("Sensors per window",14)
    d.metric("Samples per sensor",256)
    st.markdown("## What does one EEG window contain?")
    st.write("Each window is a short segment of EEG sampled from 14 sensors. At 128 samples per second, 256 samples represent 2 seconds of signal from each sensor.")
    st.markdown("## How 70 features are formed")
    st.markdown("""
    <div class="ns-step"><b>Step 1 · 14 sensor signals</b><br>Each sensor provides a 256-sample waveform for the selected two-second window.</div>
    <div class="ns-step"><b>Step 2 · Frequency analysis</b><br>Welch's method estimates the Power Spectral Density (PSD) of each sensor signal.</div>
    <div class="ns-step"><b>Step 3 · Five frequency ranges</b><br>The PSD is summarised over Delta, Theta, Alpha, Beta and Gamma bands.</div>
    <div class="ns-step"><b>Step 4 · Feature vector</b><br>Five values from each of 14 sensors produce 70 numerical features. These are supplied to the trained Random Forest model.</div>
    """,unsafe_allow_html=True)
    st.markdown("## Dataset size versus this website's demonstration data")
    st.write("The full project dataset contains 28,512 EEG windows from 48 participants. This deployed website uses a smaller demonstration subset, so the number of participants and windows shown above may be lower. The model evaluation results shown later were produced using the project’s subject-wise train/test split, not by treating this demonstration subset as the full dataset.")
    st.markdown("## Why we do not display raw numerical tables here")
    st.write("Raw sample-by-sample values and all 70 feature numbers are useful for programming and model training, but they are difficult to interpret without signal-processing context. The EEG Signal Analysis, Brain-Wave Analysis and Brain Connectivity sections present these data as waveforms, frequency summaries and sensor-relationship visualisations instead.")
    person=st.selectbox("Choose a participant to explore",participant_ids,format_func=participant_name,key="dataset_person")
    idx,rows=choose_recording(person,"dataset_record")
    st.markdown("### Selected recording summary")
    p,q,r=st.columns(3)
    p.metric("Recording duration","2 seconds")
    q.metric("EEG sensors", "14")
    r.metric("Extracted features","70")
    st.write("Open **Explore an EEG recording** to view the actual signal traces, or **Understand brain-wave activity** to inspect the frequency-band features.")

# ============================================================
# 7. BRAIN CONNECTIVITY
# ============================================================
elif page == "Brain connectivity":
    st.title("Brain connectivity analysis")
    st.write("This analysis compares pairs of EEG sensor signals. It estimates whether activity in a selected frequency range has a consistent relationship between two sensors.")
    person = st.selectbox("Choose a participant", participant_ids, format_func=participant_name, key="connect_person")
    idx, rows = choose_recording(person, "connect_record")
    band_choice = st.selectbox("Frequency range", ["Alpha (8–13 Hz)", "Beta (13–30 Hz)", "Alpha + Beta (8–30 Hz)"], key="connect_band")
    ranges = {"Alpha (8–13 Hz)": (8,13), "Beta (13–30 Hz)": (13,30), "Alpha + Beta (8–30 Hz)": (8,30)}
    matrix = coherence_matrix(idx, ranges[band_choice])
    fig = px.imshow(matrix, x=[f"Sensor {i+1}" for i in range(14)], y=[f"Sensor {i+1}" for i in range(14)],
                    zmin=0, zmax=1, color_continuous_scale="Viridis",
                    labels={"x":"Sensor","y":"Sensor","color":"Coherence"},
                    title=f"How strongly pairs of sensors are related · {band_choice}")
    st.plotly_chart(style_plot(fig, 540), use_container_width=True)
    st.markdown("### How to read this heatmap")
    st.markdown("""
    - The **horizontal axis** names the first sensor in a pair; the **vertical axis** names the second sensor.
    - Find where a row and column meet to see the coherence value for that sensor pair.
    - The **colour scale runs from 0 to 1**. Values closer to 1 indicate a more consistent frequency relationship in the selected band; values closer to 0 indicate a weaker or less consistent relationship.
    - The diagonal is 1 because each sensor is being compared with itself. It is not a meaningful connection between two different sensors.
    - The heatmap is symmetric: comparing Sensor 2 with Sensor 7 gives the same pairwise value as Sensor 7 with Sensor 2.
    """)
    pairs = [(a,b,matrix[a,b]) for a in range(14) for b in range(a+1,14)]
    pairs.sort(key=lambda x:x[2], reverse=True)
    st.markdown("### Examples of the strongest relationships in this recording")
    for a,b,v in pairs[:5]:
        st.write(f"**Sensor {a+1} and Sensor {b+1}:** coherence = {v:.3f}")
    st.markdown("""
    **What is coherence?** Magnitude-squared coherence is a signal-processing measure that describes how consistently two signals share activity at the same frequencies. It is calculated from their power and cross-power spectra.

    **What does it not mean?** A high value does not prove that one brain region is communicating directly with another, that one sensor causes another to change, or that the person has high or low workload. It is an estimate from this short two-second recording and should be interpreted as an exploratory signal relationship.
    """)

# ============================================================
# 8. EXPLAINABLE AI
# ============================================================
elif page == "Why did the AI predict this?":
    st.title("Why did the AI predict this?")
    st.write("A machine-learning prediction can be difficult to interpret. This page shows which EEG features the Random Forest appears to rely on for one selected recording.")
    person = st.selectbox("Choose a participant", participant_ids, format_func=participant_name, key="explain_person")
    idx, rows = choose_recording(person, "explain_record")
    pred, confidence, level = prediction_confidence(idx)
    if pred is None:
        st.error("The trained model or prediction probabilities are unavailable.")
    else:
        a,b,c = st.columns(3)
        a.metric("Predicted workload", workload_name(pred))
        b.metric("Model output for this class", f"{confidence:.1%}" if confidence is not None else "Unavailable")
        c.metric("Output level", level)
        st.caption("The percentage is the model's output score, not a guaranteed or clinically calibrated probability.")
        explanation = local_feature_sensitivity(idx)
        if explanation is None:
            st.warning("Individual feature sensitivity could not be calculated with the available model.")
        else:
            top = explanation.head(12).copy()
            top = top.sort_values("Sensitivity")
            fig = px.bar(top, x="Sensitivity", y="Feature", orientation="h",
                         title="How individual features affect this recording's model output",
                         color="Sensitivity", color_continuous_scale="RdBu")
            st.plotly_chart(style_plot(fig, 500), use_container_width=True)
            st.markdown("### How to read this graph")
            st.markdown("""
            The analysis changes **one feature at a time**: it replaces that feature with the median value for that feature in the demonstration data, then checks how the model's output changes.

            - A **positive sensitivity** means replacing the actual feature with its median reduced the predicted-class score. In this local test, the actual feature value supported the current prediction.
            - A **negative sensitivity** means replacing it with the median increased the predicted-class score. In this local test, the actual feature value moved the score away from the predicted class.
            - A value near **zero** means this one-at-a-time replacement had little effect on the model output.

            The horizontal axis is the change in the model score (not EEG voltage or PSD). The feature name identifies the sensor and frequency band.
            """)
            st.write("This is a local sensitivity check, not SHAP. It helps explore model behaviour but does not prove that a feature causes workload or that the model is using a biologically meaningful mechanism.")
        if hasattr(model, "feature_importances_") and len(model.feature_importances_) == 70:
            st.markdown("## Which features matter most to the trained model overall?")
            names = [f"Sensor {ch+1} · {band}" for ch in range(14) for band in BAND_NAMES]
            imp = pd.DataFrame({"Feature":names, "Importance":model.feature_importances_}).nlargest(10,"Importance").sort_values("Importance")
            fig2 = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Ten features with the highest Random Forest importance")
            st.plotly_chart(style_plot(fig2, 420), use_container_width=True)
            st.markdown("""
            **How to read this graph:** each bar represents one of the 70 inputs used by the Random Forest. A longer bar means that feature had greater importance according to the model's tree-based importance calculation across the training process. Importance values are relative model scores; they are not percentages of brain activity and do not indicate that a sensor is more important biologically.
            """)
    st.markdown("""
    <div class="ns-warn"><b>Scientific limitation:</b> both feature importance and local sensitivity describe how this particular trained model uses its inputs. They are not causal explanations, medical interpretations or proof that a specific frequency band determines workload.</div>
    """, unsafe_allow_html=True)


# ============================================================
# 9. ACROSS-WINDOW PATTERNS
# ============================================================
elif page == "Workload pattern over recordings":
    st.title("Workload pattern across stored recordings")
    st.write("Explore how the model's class output and EEG features vary across the available windows for one participant.")
    person = st.selectbox("Choose a participant", participant_ids, format_func=participant_name, key="trend_person")
    rows = get_person_rows(person)
    records = []
    for order, idx in enumerate(rows):
        pred, probs = get_prediction(int(idx))
        band_vals = band_means(int(idx))
        records.append({"Window order":order+1, "Predicted workload":workload_name(pred) if pred is not None else "Unavailable",
                        "Model confidence":float(np.max(probs)) if probs is not None else np.nan,
                        "Alpha":band_vals[2], "Beta":band_vals[3], "Theta":band_vals[1]})
    trend = pd.DataFrame(records)
    if trend.empty:
        st.info("No recordings are available for this participant.")
    else:
        fig = px.scatter(trend, x="Window order", y="Model confidence", color="Predicted workload",
                         title="Model output confidence across stored windows",
                         hover_data=["Alpha","Beta","Theta"])
        fig.update_yaxes(range=[0,1], tickformat=".0%")
        st.plotly_chart(style_plot(fig, 370), use_container_width=True)
        fig2 = go.Figure()
        for band in ["Theta","Alpha","Beta"]:
            fig2.add_trace(go.Scatter(x=trend["Window order"], y=trend[band], mode="lines",
                                      name=f"{band} feature"))
        fig2.update_layout(title="Selected frequency-band features across windows",
                           xaxis_title="Window order in the available dataset subset",
                           yaxis_title="Extracted feature value")
        st.plotly_chart(style_plot(fig2, 390), use_container_width=True)
        class_counts = trend["Predicted workload"].value_counts(dropna=True)
        st.markdown("### Distribution of predicted workload")
        st.dataframe(pd.DataFrame({"Predicted workload":class_counts.index,
                                   "Number of windows":class_counts.values}), use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="ns-warn"><b>Important:</b> this visualises the order of windows in the stored demonstration data. It is not a validated continuous-time workload trace, live monitoring or proof of a person's workload changing over time. Temporal interpretation requires verified timestamps or task sequence metadata.</div>
    """, unsafe_allow_html=True)


# ============================================================
# 10. RESEARCH RESULTS
# ============================================================
elif page == "Research results":
    st.title("Research results")
    st.write("This section is intended for technical review: it shows how the tested machine-learning models performed.")
    if model_table is not None and not model_table.empty:
        st.markdown("## Compare the tested models")
        st.dataframe(model_table,use_container_width=True,hide_index=True)
        numeric=model_table.select_dtypes(include=np.number).columns.tolist()
        namecol=next((c for c in model_table.columns if c not in numeric),None)
        acccol=next((c for c in numeric if "accuracy" in c.lower()),None)
        f1col=next((c for c in numeric if "f1" in c.lower()),None)
        if namecol and acccol:
            fig=px.bar(model_table,x=namecol,y=acccol,color=namecol,title="Model accuracy comparison",text_auto=".2f")
            st.plotly_chart(style_plot(fig,350),use_container_width=True)
        if namecol and f1col:
            fig=px.bar(model_table,x=namecol,y=f1col,color=namecol,title="Model F1-score comparison",text_auto=".3f")
            st.plotly_chart(style_plot(fig,350),use_container_width=True)
    else:
        st.info("The model comparison file was not found in the repository.")
    if model is not None and hasattr(model,"feature_importances_") and len(model.feature_importances_)==70:
        st.markdown("## Which EEG features influenced the Random Forest?")
        names=[f"Sensor {ch+1} · {band}" for ch in range(14) for band in BAND_NAMES]
        imp=pd.DataFrame({"Feature":names,"Importance":model.feature_importances_}).nlargest(15,"Importance").sort_values("Importance")
        fig=px.bar(imp,x="Importance",y="Feature",orientation="h",title="15 most influential model features")
        st.plotly_chart(style_plot(fig,490),use_container_width=True)
        st.caption("Feature importance describes how the trained model used its inputs. It does not prove that a feature causes workload.")
    st.markdown("""
    <div class="ns-note"><b>Evaluation note:</b> the model was assessed using participants held out from training.
    Accuracy and F1-score describe classification performance on the test set; they are not clinical performance measures.</div>
    """,unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("NeuroSense · ECE mini project")
