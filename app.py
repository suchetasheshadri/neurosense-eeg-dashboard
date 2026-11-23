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
        format_func=lambda k: f"Recording {k+1:03d} (demo window)",
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
    "About the dataset",
    "Explore an EEG recording",
    "Understand brain-wave activity",
    "General well-being",
    "Workload report",
    "Compare recordings",
    "Brain connectivity",
    "Why did the AI predict this?",
    "Workload pattern over recordings",
    "Research results"
]
page = st.sidebar.radio("ANALYSIS SECTIONS", pages, label_visibility="visible", key="nav_page")
st.sidebar.markdown("---")
st.sidebar.caption("Research prototype · Stored EEG data")
st.sidebar.caption("Demo subset of held-out test participants")
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
      <p><b>An EEG-based cognitive workload analysis and interpretation platform.</b><br>
      NeuroSense studies short, multichannel EEG recordings to understand their frequency
      composition, examine relationships between sensors, and estimate workload-related
      classes using a trained machine-learning model.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("## What problem does the project address?")
    st.write("Mental workload describes the amount of mental effort a person experiences while carrying out a task. EEG records small electrical signals from the scalp. These signals contain patterns that researchers can analyse when studying workload. Looking at a waveform alone is difficult, so NeuroSense turns the recordings into measurable features and interpretable visual summaries.")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("EEG sensors", "14")
    k2.metric("Sampling rate", "128 Hz")
    k3.metric("Window duration", "2 seconds")
    k4.metric("Model inputs", "70 features")
    st.markdown("## What does NeuroSense do?")
    steps = st.columns(4)
    for col, (num, title, desc) in zip(steps, [
        ("01", "Reads EEG", "Loads short recordings from the STEW research dataset."),
        ("02", "Analyses signals", "Filters signals and studies five frequency bands."),
        ("03", "Estimates workload", "Uses a trained Random Forest to predict a workload class."),
        ("04", "Explains patterns", "Explores sensor connectivity, feature influence and recording trends.")
    ]):
        with col:
            st.markdown(f'<div class="ns-card"><small>{num}</small><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.markdown("## What is the project's contribution?")
    n1,n2,n3 = st.columns(3)
    with n1:
        st.markdown('<div class="ns-card"><small>01 · SIGNAL ANALYSIS</small><h3>Frequency patterns</h3><p>Summarises EEG activity into Delta, Theta, Alpha, Beta and Gamma features for each sensor.</p></div>', unsafe_allow_html=True)
    with n2:
        st.markdown('<div class="ns-card"><small>02 · SENSOR RELATIONSHIPS</small><h3>Connectivity exploration</h3><p>Uses coherence to explore frequency relationships between pairs of EEG sensors.</p></div>', unsafe_allow_html=True)
    with n3:
        st.markdown('<div class="ns-card"><small>03 · INTERPRETABILITY</small><h3>Explainable model analysis</h3><p>Shows which extracted features influence a model result and how outputs vary across stored windows.</p></div>', unsafe_allow_html=True)
    st.markdown("## Who can use this website?")
    st.write("The interface is designed for students, researchers and visitors who want to understand how EEG signals are processed and how a machine-learning model analyses workload-related patterns. Each technical graph is accompanied by a plain-language explanation.")
    st.markdown("""
    <div class="ns-note"><b>Scope and limitations:</b> NeuroSense is an offline research prototype using stored STEW data. It does not connect to a wireless EEG headset or provide live monitoring. Its trained model predicts the two workload classes represented in the dataset. It does not diagnose medical conditions or independently detect emotions, focus, stress or fatigue.</div>
    """, unsafe_allow_html=True)
    st.markdown("## Suggested way to explore")
    st.write("Start with **About the dataset** to understand the participants and recordings. Then explore a signal, learn what the frequency bands mean, and finally inspect the workload prediction and model evaluation.")

# ============================================================
# 2. EXPLORE A RECORDING
# ============================================================
elif page == "Explore an EEG recording":
    st.title("Explore an EEG recording")
    st.write("Choose a person and a recording. NeuroSense will show the brain signals captured during that short recording.")
    person=st.selectbox("1. Choose a participant",participant_ids,format_func=participant_name,key="explore_person")
    idx,rows=choose_recording(person,"explore_record")
    st.markdown(f'<div class="ns-step"><b>Why these participant and recording numbers?</b><br>The participant number is the original ID from the STEW dataset. This website displays a compact subset of held-out test participants, so IDs are not renumbered consecutively. Recording 001, 002 and so on are simply the order of the available demo windows for that participant; they are not task names or session numbers.</div>',unsafe_allow_html=True)
    st.markdown('<div class="ns-step"><b>What you are looking at</b><br>Each displayed window lasts 2 seconds. Fourteen EEG channels record small voltage changes from scalp electrodes. The traces are separated vertically for visibility; their vertical positions do not represent actual electrode locations.</div>',unsafe_allow_html=True)
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
    st.write("An EEG signal is made up of electrical activity that changes over time. Frequency analysis separates this complex signal into slower and faster patterns. Researchers group frequencies into bands so they can describe and compare the signal in a consistent way.")
    bands = [
        ("Delta","1–4 Hz","The slowest band shown here. Delta activity is commonly studied in sleep and other brain-state research."),
        ("Theta","4–8 Hz","Studied in memory, learning and cognitive processing. Its meaning depends on the task, sensor location and recording conditions."),
        ("Alpha","8–13 Hz","Often prominent during relaxed wakefulness, particularly with reduced visual input. Alpha can change during tasks, but it is not a direct focus meter."),
        ("Beta","13–30 Hz","Often studied in active cognitive and sensorimotor processing. It can also be affected by muscle activity."),
        ("Gamma","30–40 Hz","A higher-frequency range studied in neural processing. It is especially important to interpret carefully because muscle artefacts can contaminate higher frequencies.")
    ]
    cols=st.columns(5)
    for col,(name,hz,desc) in zip(cols,bands):
        with col:
            st.markdown(f'<div class="ns-card"><small>{hz}</small><h3>{name}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
    st.markdown("## Why analyse these five bands?")
    st.write("The raw EEG waveform contains many frequencies mixed together. Band analysis summarises the signal into interpretable frequency ranges. The project uses these five ranges to create a compact description of each sensor's activity. The values are inputs to the machine-learning model; no single band independently proves that a person is focused, stressed, tired or experiencing a particular emotion.")
    st.markdown("## What is Power Spectral Density (PSD)?")
    st.write("Power Spectral Density, or PSD, describes how the power of a signal is distributed over frequency. Instead of looking only at how the voltage changes over time, PSD helps answer: which frequency ranges contain more or less signal power?")
    st.markdown("""
    - **Time-domain graph:** shows signal amplitude as time passes.
    - **Frequency-domain graph:** shows how the signal is distributed across frequencies.
    - **PSD graph:** shows the estimated power per unit frequency. Peaks indicate frequencies where more signal power is concentrated.
    - **Welch's method:** estimates PSD by dividing the signal into overlapping segments, calculating spectra and averaging them. Averaging helps produce a steadier estimate than using one raw periodogram.
    """)
    st.markdown("## Explore frequency activity in one recording")
    person=st.selectbox("Choose a participant",participant_ids,format_func=participant_name,key="bands_person")
    idx,rows=choose_recording(person,"bands_record")
    values=band_means(idx)
    df=pd.DataFrame({"Frequency band":BAND_NAMES,"Mean PSD feature":values})
    fig=px.bar(df,x="Frequency band",y="Mean PSD feature",color="Frequency band",
               title="Average extracted PSD feature in each frequency band")
    st.plotly_chart(style_plot(fig,380),use_container_width=True)
    st.markdown("### How to read this bar chart")
    st.write("Each bar represents one frequency band for the selected recording. A taller bar means the extracted mean PSD feature is numerically larger for that band. Compare the bars as features produced by the same processing pipeline; do not interpret a taller bar as automatically meaning higher workload or better concentration.")
    st.markdown("## Mean PSD features across the 14 sensors")
    table=pd.DataFrame(features[idx].reshape(14,5),columns=BAND_NAMES)
    table.insert(0,"Sensor", [f"Sensor {i}" for i in range(1,15)])
    st.dataframe(table.round(6),use_container_width=True,hide_index=True)
    st.markdown("### How to read this table")
    st.markdown("""
    - **Each row** is one sensor (one EEG channel).
    - **Each frequency column** is one band: Delta, Theta, Alpha, Beta or Gamma.
    - **Each cell** is the mean PSD feature extracted for that sensor and band in the selected two-second recording.
    - **The numbers are not workload scores.** They are signal features supplied to the machine-learning model.
    - A larger number means a larger estimated power-density feature in that band under this processing pipeline. It does not by itself mean more focus, stress or workload.

    **What are the units?** PSD units depend on the physical unit and scaling of the original EEG signal. If the source signal is in microvolts, PSD is normally expressed in µV²/Hz. The app's demo feature file preserves its numerical scale, but the exact source scaling has not been verified here. For that reason, the table is labelled *mean PSD feature value* and no physical unit is asserted.

    **How should I compare values?** Compare the same frequency band across sensors, or compare bands within this same recording while remembering that band widths differ. Avoid comparing absolute values across different datasets or preprocessing pipelines unless their units and scaling are consistent.
    """)
    st.markdown("## Why are there 70 features?")
    st.write("There are 14 sensors and 5 frequency bands per sensor. Therefore, 14 × 5 = 70 numerical features describe one two-second EEG window. The Random Forest receives these 70 values as its input.")
    st.markdown("## What can and cannot be inferred?")
    st.write("These features describe the frequency composition of recorded EEG. Together with the trained model, they support workload-class estimation. They do not independently diagnose attention, emotion, tiredness or a medical condition. The project's current labels are workload labels only.")

# ============================================================
# 4. GENERAL WELL-BEING
# ============================================================
elif page == "General well-being":
    st.title("General well-being: practical information")
    st.write("Mental workload is one part of a person's experience during a task. A demanding task can require sustained attention, memory and decision-making. EEG workload research studies signal patterns associated with these demands, but an EEG prediction is not a complete measure of someone's wellbeing.")
    st.markdown("## Everyday practices that may support wellbeing")
    tips = [
        ("Take short breaks","After a period of demanding work, step away briefly, look away from the screen and allow time to reset."),
        ("Protect sleep","A consistent sleep routine supports everyday cognitive functioning and recovery."),
        ("Stay hydrated and eat regularly","Basic hydration and regular meals can help support general wellbeing during study or work."),
        ("Move between study sessions","Light movement or a short walk can break up long periods of sitting."),
        ("Break large tasks into smaller steps","Clear priorities and manageable work intervals can make demanding tasks easier to organise."),
        ("Notice persistent difficulties","If stress, sleep problems or concentration difficulties persist or interfere with daily life, consider speaking with a qualified health professional.")
    ]
    for title,desc in tips:
        st.markdown(f'<div class="ns-step"><b>{title}</b><br>{desc}</div>',unsafe_allow_html=True)
    st.markdown("## How this relates to NeuroSense")
    st.write("The suggestions above are general educational information. They are not personalised from EEG measurements and are not generated by the Random Forest model. NeuroSense does not determine whether a person is healthy, stressed, emotionally well or unwell.")
    st.warning("Do not use a workload prediction to make medical decisions. This project is an educational research prototype, not a clinical device or a substitute for professional assessment.")

# ============================================================
# 4. WORKLOAD REPORT
# ============================================================
elif page == "Workload report":
    st.title("Workload report")
    st.write("Choose a stored EEG window to see the Random Forest prediction and compare it with the reference label supplied with the dataset.")
    person=st.selectbox("Choose a participant",participant_ids,format_func=participant_name,key="report_person")
    idx,rows=choose_recording(person,"report_record")
    pred,probs=get_prediction(idx)
    actual=int(labels[idx])
    if pred is None:
        st.error("The trained model could not create a result. Check that the model file is present.")
    else:
        x,y=st.columns(2)
        with x:
            st.markdown(f'<div class="ns-card"><small>MODEL PREDICTION</small><div style="font-size:2rem;font-weight:800;margin-top:10px">{workload_name(pred)}</div><p>What the Random Forest predicts from this EEG window.</p></div>',unsafe_allow_html=True)
        with y:
            st.markdown(f'<div class="ns-card"><small>DATASET REFERENCE LABEL</small><div style="font-size:2rem;font-weight:800;margin-top:10px">{workload_name(actual)}</div><p>The reference label stored with this EEG window.</p></div>',unsafe_allow_html=True)
        if pred==actual:
            st.success("For this selected window, the model prediction agrees with the dataset reference label.")
        else:
            st.warning("For this selected window, the model prediction differs from the dataset reference label. Individual mistakes are expected; overall performance is evaluated in Research Results.")
        if probs is not None:
            df=pd.DataFrame({"Workload level":[workload_name(c) for c in model.classes_],"Model output":probs})
            fig=px.bar(df,x="Workload level",y="Model output",color="Workload level",
                       text=df["Model output"].map(lambda v:f"{v:.1%}"),
                       title="How strongly the model favours each workload class")
            fig.update_yaxes(range=[0,1],tickformat=".0%")
            st.plotly_chart(style_plot(fig,310),use_container_width=True)
            st.markdown("### How to read the model output")
            st.write("Each bar is the model's output score for one workload class. If one bar is higher, the model favours that class more strongly for this recording. These scores are not guaranteed to be calibrated probabilities and should not be treated as medical certainty.")
    st.markdown("## How does the system produce this report?")
    st.write("The selected EEG window is represented by 70 PSD features (five frequency bands from each of 14 sensors). The trained Random Forest processes these features and returns one of the two workload classes. The dataset reference label is shown separately so the prediction can be checked for this example.")
    st.info("The prediction describes the model's interpretation of a stored two-second EEG window. It does not measure a person's complete mental state, diagnose a condition or establish their ability to work or study.")

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
        st.markdown("### How to interpret the comparison")
        st.write("The grouped bars place the same frequency band from Recording A and Recording B next to each other. For each band, compare the bar heights: the taller bar has the larger extracted feature value in that recording. This is a descriptive comparison, not proof that one participant has higher workload.")
        delta_df = pd.DataFrame({
            "Frequency band": BAND_NAMES,
            "Recording A": band_means(i1),
            "Recording B": band_means(i2),
        })
        delta_df["Difference (B − A)"] = delta_df["Recording B"] - delta_df["Recording A"]
        delta_df["Larger feature value"] = np.where(delta_df["Difference (B − A)"] > 0,
                                                       "Recording B",
                                                       np.where(delta_df["Difference (B − A)"] < 0, "Recording A", "Equal"))
        st.markdown("### Which recording has the larger feature value in each band?")
        st.dataframe(delta_df.round(6), use_container_width=True, hide_index=True)
        st.write("A positive difference means Recording B has a larger feature value in that band; a negative difference means Recording A is larger. The size of the difference is in the feature file's numerical scale, not a percentage of workload.")
        with st.expander("Compare the two recordings' waveforms"):
            s1=st.selectbox("Choose a sensor to compare",list(range(14)),format_func=lambda x:f"Sensor {x+1}",key="compare_sensor")
            t=np.arange(256)/FS
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=t,y=raw[i1,s1,:],name=f"{participant_name(p1)}"))
            fig.add_trace(go.Scatter(x=t,y=raw[i2,s1,:],name=f"{participant_name(p2)}"))
            fig.update_layout(title="Same sensor position, two recordings",xaxis_title="Time (seconds)",yaxis_title="Signal strength")
            st.plotly_chart(style_plot(fig,360),use_container_width=True)
            st.write("This graph overlays the same sensor position from both recordings. The horizontal axis is time in seconds and the vertical axis is signal amplitude in the dataset's numerical units. Differences in shape or amplitude are visible, but the plot alone does not identify their cause.")
        st.caption("Differences are descriptive comparisons of stored recordings. They do not establish that one participant is more focused, sleepy or emotionally different.")

# ============================================================
# 6. DATASET
# ============================================================
elif page == "About the dataset":
    st.title("About the STEW dataset")
    st.write("This project uses the STEW dataset — Simultaneous Task EEG Workload — a research dataset created to study EEG patterns associated with mental workload during multitasking.")
    st.markdown("## The original dataset")
    a,b,c,d = st.columns(4)
    a.metric("Participants", "48")
    b.metric("EEG windows", "28,512")
    c.metric("EEG sensors", "14")
    d.metric("Sampling rate", "128 Hz")
    st.markdown("""
    The full dataset used in the project contains recordings from **48 participants**. Each recording window contains data from **14 EEG channels**. Every channel has **256 samples**, captured at **128 samples per second**.

    This means that one window represents:

    \[
    \text{Duration} = \frac{256\text{ samples}}{128\text{ samples/second}} = 2\text{ seconds}
    \]

    In simple terms, for every two-second window, the dataset stores 256 measurements from each of the 14 sensors.
    """)
    st.markdown("## What is an EEG channel or sensor?")
    st.write("An EEG channel is a signal recorded from one electrode position on the scalp. The 14 channels are 14 separate signal streams. They are not 14 pictures of the brain and the channel numbers shown in this website should not be treated as physical electrode names or anatomical locations unless the original electrode-position mapping is supplied.")
    st.markdown("## What are the recordings and why are participant numbers not consecutive?")
    st.write("The full dataset includes 48 participants, but this website is designed to demonstrate the model on the held-out test participants. The project split was participant-wise: 38 participants were used for model training and 10 different participants were reserved for testing. The test participant IDs are therefore not necessarily 1, 2, 3, and so on.")
    st.write("The IDs shown in the selection menu identify the original participant IDs in the dataset. They may look random because the test participants were selected as a held-out group rather than renumbered from 1 to 10. This is intentional: the model is being evaluated on people it did not see during training.")
    st.markdown("## Why does each participant have only a limited number of recordings here?")
    st.write("The full research dataset contains many windows per participant. To keep the online application lightweight and quick to load, the website uses a smaller demonstration subset extracted from the held-out test data. The recording selector therefore shows only the windows included in this demonstration subset, not every recording from the original dataset.")
    st.write(f"**Currently loaded in this website:** {n:,} demonstration windows from {len(participant_ids)} participant IDs.")
    counts = pd.Series(people).value_counts().sort_index()
    counts_df = pd.DataFrame({"Original participant ID":[int(x) for x in counts.index],
                              "Demonstration recordings available":[int(x) for x in counts.values]})
    st.dataframe(counts_df, use_container_width=True, hide_index=True)
    st.caption("This table describes the reduced demonstration files bundled with the website, not the full STEW dataset.")
    st.markdown("## How the project split the data")
    split_a, split_b = st.columns(2)
    with split_a:
        st.markdown('<div class="ns-card"><small>TRAINING SET</small><h2>38 participants</h2><p>22,572 EEG windows were used to train and compare the machine-learning models.</p></div>', unsafe_allow_html=True)
    with split_b:
        st.markdown('<div class="ns-card"><small>TEST SET</small><h2>10 participants</h2><p>5,940 EEG windows from participants not used in training were reserved for evaluation.</p></div>', unsafe_allow_html=True)
    st.write("This is called a subject-independent or participant-wise split. It is more demanding than randomly splitting windows from the same people, because the test participants are unseen during training. It helps assess whether a model can generalise to different participants.")
    st.markdown("## How one EEG window becomes 70 model features")
    st.markdown("""
    <div class="ns-step"><b>1. Raw EEG window</b><br>14 sensor signals × 256 samples for a two-second segment.</div>
    <div class="ns-step"><b>2. Preprocessing</b><br>A Butterworth bandpass filter keeps the 1–40 Hz range used by this project.</div>
    <div class="ns-step"><b>3. Power Spectral Density (PSD)</b><br>Welch's method estimates how the signal's power is distributed across frequencies.</div>
    <div class="ns-step"><b>4. Frequency-band summaries</b><br>For each sensor, PSD values are summarised in Delta, Theta, Alpha, Beta and Gamma bands.</div>
    <div class="ns-step"><b>5. Machine-learning input</b><br>14 sensors × 5 band values = 70 features for one EEG window.</div>
    """, unsafe_allow_html=True)
    st.markdown("## Files used by the website")
    st.write("The application uses four demonstration data files: `STEW_X_demo.npy` contains the EEG waveforms; `STEW_features_demo.npy` contains the extracted 70-feature vectors; `STEW_labels_demo.npy` contains the reference workload labels; and `STEW_subjects_demo.npy` identifies the participant associated with each window. The trained Random Forest is loaded separately from `eeg_workload_model.pkl.gz`.")
    st.markdown("## Important distinction: full dataset vs website demo")
    st.warning("The full project dataset contains 28,512 windows from 48 participants. The website may show fewer windows and fewer participants because it loads a compact demonstration subset. The model's reported evaluation is based on the subject-wise test set, not merely the small subset displayed in this website.")
    st.markdown("## What does the dataset allow us to study?")
    st.write("The STEW dataset supports workload-related EEG analysis because it contains task recordings and workload reference labels. In this project, the trained model predicts the two workload classes. It does not provide emotion labels or validated focus labels, so this dataset and model cannot be used to claim emotion or focus detection.")

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
    **What is coherence?** Magnitude-squared coherence is a signal-processing measure that describes how consistently two signals share activity at the same frequencies. It is calculated from their power and cross-power spectra. The value is dimensionless and ranges from 0 to 1.

    **Why is the heatmap symmetric?** Coherence is a pairwise relationship, so the value for Sensor A compared with Sensor B is the same as Sensor B compared with Sensor A. The diagonal compares each sensor with itself and is therefore 1 by definition.

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
        st.markdown("### How to read the confidence graph")
        st.write("Each dot represents one stored EEG window in the order it appears in this participant's demonstration subset. The vertical position is the highest model output score among the two workload classes. A value closer to 1 means the model favoured one class more strongly; a value closer to 0.5 means the two class scores are closer together. This is a model score, not a measure of how certain a human expert should be.")
        st.write("The dot colour shows which workload class the model selected for that window. A change in colour means the predicted class changed between windows. It does not by itself prove that the participant's real workload changed.")
        fig2 = go.Figure()
        for band in ["Theta","Alpha","Beta"]:
            fig2.add_trace(go.Scatter(x=trend["Window order"], y=trend[band], mode="lines",
                                      name=f"{band} feature"))
        fig2.update_layout(title="Selected frequency-band features across windows",
                           xaxis_title="Window order in the available dataset subset",
                           yaxis_title="Extracted feature value")
        st.plotly_chart(style_plot(fig2, 390), use_container_width=True)
        st.markdown("### How to read the frequency-feature trend")
        st.write("The lines show the extracted Theta, Alpha and Beta feature values for successive stored windows. The horizontal axis is the order of windows in the demo subset, not verified clock time. The vertical axis is the numerical feature value. A rise or fall shows that the extracted band feature changed between windows; it is not automatically a rise or fall in workload.")
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
    st.write("This section explains how the machine-learning models were compared and how the selected Random Forest performed on participants held out from training.")
    st.markdown("## Why compare multiple models?")
    st.write("Different algorithms learn patterns in different ways. The project compared Support Vector Machine (SVM), Random Forest, K-Nearest Neighbours (KNN) and a Multi-Layer Perceptron (MLP). Comparing them on the same test set helps identify which model performed best under the project's evaluation setup.")
    known = pd.DataFrame({
        "Model":["SVM","Random Forest","KNN","MLP"],
        "Accuracy":[0.5461,0.7165,0.5682,0.6258],
        "F1-score":[0.6617,0.7393,0.6002,0.6743]
    })
    st.markdown("### Recorded model comparison")
    st.dataframe(known.style.format({"Accuracy":"{:.2%}","F1-score":"{:.4f}"}),use_container_width=True,hide_index=True)
    fig=px.bar(known,x="Model",y="Accuracy",text=known["Accuracy"].map(lambda v:f"{v:.2%}"),
               title="Accuracy: proportion of test windows classified correctly")
    fig.update_yaxes(range=[0,1],tickformat=".0%")
    st.plotly_chart(style_plot(fig,350),use_container_width=True)
    fig2=px.bar(known,x="Model",y="F1-score",text=known["F1-score"].map(lambda v:f"{v:.3f}"),
                title="F1-score: balance between precision and recall")
    fig2.update_yaxes(range=[0,1])
    st.plotly_chart(style_plot(fig2,350),use_container_width=True)
    st.markdown("### What do these evaluation terms mean?")
    st.markdown("""
    - **Accuracy:** the fraction of all test windows for which the predicted class matched the reference label. For example, 71.65% means roughly 72 out of every 100 test windows were classified correctly.
    - **Precision:** when the model predicts a particular class, how often that prediction is correct.
    - **Recall:** of all test windows that truly belong to a class, how many the model successfully identifies.
    - **F1-score:** a combined measure of precision and recall. It is useful when we want to consider both false alarms and missed examples.
    """)
    st.markdown("## Why was Random Forest selected?")
    st.write("Among the four compared models, Random Forest achieved the highest recorded accuracy (71.65%) and F1-score (0.7393) in this project evaluation. It was therefore selected for the dashboard's workload predictions. This does not mean it will be correct for every recording or every person.")
    st.markdown("## Evaluation design: unseen participants")
    st.write("The split was performed by participant: 38 participants (22,572 windows) were used for training, and 10 different participants (5,940 windows) were held out for testing. This is important because windows from the same participant can share individual signal characteristics. Keeping participants separate gives a more meaningful test of generalisation to people not used during training.")
    st.markdown("## Random Forest confusion matrix")
    cm = np.array([[1868,1102],[582,2388]])
    cm_df = pd.DataFrame(cm, index=["Actual Low Workload","Actual High Workload"],
                         columns=["Predicted Low Workload","Predicted High Workload"])
    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                       labels={"x":"Model prediction","y":"Dataset reference","color":"Number of windows"},
                       title="Random Forest test-set confusion matrix")
    st.plotly_chart(style_plot(fig_cm,390),use_container_width=True)
    st.markdown("""
    **How to read the confusion matrix:** rows are the actual reference labels from the dataset; columns are the model's predictions. The diagonal cells are correct predictions. The off-diagonal cells are errors: a low-workload example predicted as high, or a high-workload example predicted as low.

    - 1,868 actual low-workload windows were predicted as low.
    - 1,102 actual low-workload windows were predicted as high.
    - 582 actual high-workload windows were predicted as low.
    - 2,388 actual high-workload windows were predicted as high.

    The total is 5,940 test windows, with 4,256 correct predictions, giving 71.65% accuracy.
    """)
    st.markdown("## What does this result tell us — and what does it not tell us?")
    st.write("The results show that Random Forest learned workload-related patterns from the training participants and classified a majority of windows from unseen test participants correctly. The remaining errors show that EEG workload classification is not perfect. These metrics describe performance on this dataset and split only; they are not clinical validation, and they do not establish emotion, focus or medical-state detection.")
    if model is not None and hasattr(model,"feature_importances_") and len(model.feature_importances_)==70:
        st.markdown("## Which input features does the trained model use?")
        names=[f"Sensor {ch+1} · {band}" for ch in range(14) for band in BAND_NAMES]
        imp=pd.DataFrame({"Feature":names,"Importance":model.feature_importances_}).nlargest(15,"Importance").sort_values("Importance")
        fig=px.bar(imp,x="Importance",y="Feature",orientation="h",title="15 features with the highest Random Forest importance")
        st.plotly_chart(style_plot(fig,490),use_container_width=True)
        st.write("Each bar is one of the 70 input features. Longer bars indicate greater importance according to the Random Forest's built-in tree-based calculation. These values are relative model-importance scores, not percentages of EEG power and not proof that a sensor causes workload.")
    st.markdown("""
    <div class="ns-note"><b>Evaluation summary:</b> Random Forest was selected because it achieved the strongest recorded performance among the compared models. The participant-wise split provides a test on unseen participants. NeuroSense remains a research prototype and should not be used for clinical or individual high-stakes decisions.</div>
    """,unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("NeuroSense · ECE mini project")
