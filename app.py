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

def check_signal(signal):
    x = np.asarray(signal, dtype=float)
    if not np.isfinite(x).all():
        return "Needs review", "The signal contains missing or invalid values."
    sd = float(np.std(x))
    med = float(np.median(x))
    mad = float(np.median(np.abs(x-med))) + 1e-12
    outliers = int(np.sum(np.abs(x-med) > 8*1.4826*mad))
    if sd < 1e-6:
        return "Nearly flat", "There is almost no change in this sensor's signal."
    if outliers > 3:
        return "Unusual spikes", "The signal contains several unusually large changes. Inspect it before interpreting the result."
    return "No obvious issue", "The basic screening did not find a flat signal or repeated extreme spikes."

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
    st.markdown(f"### Model result: **Category {pred}**")
    if probs is not None:
        df = pd.DataFrame({"Model category":[f"Category {int(c)}" for c in model.classes_],
                           "Model output":[float(v) for v in probs]})
        fig = px.bar(df,x="Model category",y="Model output",color="Model category",
                     title="How strongly the model favoured each category",
                     text=df["Model output"].map(lambda v:f"{v:.1%}"))
        fig.update_yaxes(range=[0,1],tickformat=".0%")
        st.plotly_chart(style_plot(fig,300),use_container_width=True)
    st.caption("These are model outputs, not medical certainty. The original numeric category meanings must be confirmed before naming them low or high workload.")

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
    "Research results"
]
page = st.sidebar.radio("GO TO", pages, label_visibility="collapsed")
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
      <div class="eyebrow">BIOMEDICAL SIGNAL ANALYSIS PLATFORM</div>
      <h1>NEUROSENSE</h1>
      <p><b>See what is inside an EEG recording.</b><br>
      Explore brain signals, understand five brain-wave frequency ranges and examine
      how a machine-learning model classifies workload-related EEG patterns.</p>
    </div>
    """,unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    a.metric("Participants in this app",len(participant_ids))
    b.metric("Brain sensors",14)
    c.metric("Sampling speed","128 / sec")
    d.metric("Recording length","2 seconds")
    st.markdown("## Start exploring")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="ns-card"><small>01 · LOOK AT THE SIGNAL</small><h3>Explore a recording</h3><p>See the brain signals from all sensors, view one sensor more closely and inspect its frequency spectrum.</p></div>',unsafe_allow_html=True)
        if st.button("Open EEG explorer",use_container_width=True):
            st.session_state["nav_target"]="Explore an EEG recording"
    with c2:
        st.markdown('<div class="ns-card"><small>02 · UNDERSTAND THE PATTERN</small><h3>Brain-wave activity</h3><p>Learn what Delta, Theta, Alpha, Beta and Gamma mean and see the values extracted from a recording.</p></div>',unsafe_allow_html=True)
        if st.button("Explore brain-wave bands",use_container_width=True):
            st.session_state["nav_target"]="Understand brain-wave activity"
    with c3:
        st.markdown("<div class='ns-card'><small>03 · READ THE RESULT</small><h3>Workload report</h3><p>See the model's category, its output probabilities and a plain-language explanation.</p></div>",unsafe_allow_html=True)
        if st.button("View workload report",use_container_width=True):
            st.session_state["nav_target"]="Workload report"
    if "nav_target" in st.session_state:
        st.info(f"Use the left navigation to open **{st.session_state.pop('nav_target')}**.")
    st.markdown("## How NeuroSense works")
    steps=st.columns(5)
    for col,(num,title,desc) in zip(steps,[
        ("1","EEG dataset","Previously recorded brain signals"),
        ("2","Signal cleaning","Keep the 1–40 Hz range"),
        ("3","Brain-wave features","Five frequency ranges per sensor"),
        ("4","AI model","Classifies workload-related patterns"),
        ("5","Visual report","Charts and understandable results")]):
        with col:
            st.markdown(f'<div class="ns-card"><small>STEP {num}</small><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
    st.markdown("""
    <div class="ns-note"><b>What this project currently does:</b> analyses stored EEG recordings from the STEW dataset.
    It is a software research prototype, not a live headset or medical diagnostic system.</div>
    """,unsafe_allow_html=True)

# ============================================================
# 2. EXPLORE A RECORDING
# ============================================================
elif page == "Explore an EEG recording":
    st.title("Explore an EEG recording")
    st.write("Choose a person and a recording. NeuroSense will show the brain signals captured during that short recording.")
    person=st.selectbox("1. Choose a participant",participant_ids,format_func=participant_name,key="explore_person")
    idx,rows=choose_recording(person,"explore_record")
    st.markdown('<div class="ns-step"><b>What you are looking at</b><br>Each recording lasts 2 seconds. Fourteen sensors record tiny electrical changes from different positions on the scalp.</div>',unsafe_allow_html=True)
    st.markdown("## The full brain-signal picture")
    show_all_sensors(idx)
    st.markdown("## Zoom in on one sensor")
    sensor=st.selectbox("Choose a sensor position",list(range(14)),format_func=lambda x:f"Sensor {x+1}",key="explore_sensor")
    sig=show_sensor(idx,sensor)
    status,reason=check_signal(sig)
    st.markdown("### Quick signal check")
    if status=="No obvious issue":
        st.success(f"{status}: {reason}")
    else:
        st.warning(f"{status}: {reason}")
    st.caption("This is a basic automated screening check, not a clinical EEG quality certification.")

# ============================================================
# 3. UNDERSTAND BRAIN-WAVE ACTIVITY
# ============================================================
elif page == "Understand brain-wave activity":
    st.title("Understand brain-wave activity")
    st.write("EEG signals contain activity at different frequencies. Researchers group these frequencies into five broad bands.")
    bands = [
        ("Delta","1–4 Hz","Slow brain-wave activity; studied especially in sleep research."),
        ("Theta","4–8 Hz","Studied in memory, drowsiness and cognitive processing research."),
        ("Alpha","8–13 Hz","Often observed during relaxed wakefulness; changes with attention."),
        ("Beta","13–30 Hz","Studied in alertness and active mental processing."),
        ("Gamma","30–40 Hz","Studied in complex information-processing research.")
    ]
    cols=st.columns(5)
    for col,(name,hz,desc) in zip(cols,bands):
        with col:
            st.markdown(f'<div class="ns-card"><small>{hz}</small><h3>{name}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
    st.markdown("## Explore the frequency pattern in a recording")
    person=st.selectbox("Choose a participant",participant_ids,format_func=participant_name,key="bands_person")
    idx,rows=choose_recording(person,"bands_record")
    values=band_means(idx)
    df=pd.DataFrame({"Brain-wave band":BAND_NAMES,"Extracted feature value":values})
    fig=px.bar(df,x="Brain-wave band",y="Extracted feature value",color="Brain-wave band",
               title="Five frequency-band features in the selected recording")
    st.plotly_chart(style_plot(fig,380),use_container_width=True)
    st.markdown("### Values across the 14 sensors")
    table=pd.DataFrame(features[idx].reshape(14,5),columns=BAND_NAMES)
    table.insert(0,"Sensor", [f"Sensor {i}" for i in range(1,15)])
    st.dataframe(table.round(5),use_container_width=True,hide_index=True)
    st.markdown("""
    <div class="ns-note"><b>How to read this:</b> taller bars mean a larger extracted feature value for that frequency range in this recording.
    They do not directly mean more or less focus, sleepiness or emotion.</div>
    """,unsafe_allow_html=True)
    st.markdown("### Research note: focus, sleepiness and emotion")
    st.write("EEG frequency bands are studied in research on attention, sleepiness and emotional processing. However, this project’s trained model predicts only the two workload dataset categories. It has not been trained or validated to recognise emotion, focus or drowsiness, so NeuroSense does not label a person with those states.")

# ============================================================
# 4. WORKLOAD REPORT
# ============================================================
elif page == "Workload report":
    st.title("Workload report")
    st.write("This page turns the model output for one selected recording into a simple report.")
    person=st.selectbox("Choose a participant",participant_ids,format_func=participant_name,key="report_person")
    idx,rows=choose_recording(person,"report_record")
    st.markdown("## Result")
    pred,probs=get_prediction(idx)
    actual=int(labels[idx])
    if pred is None:
        st.error("The trained model could not create a result. Check that the model file is present.")
    else:
        x,y=st.columns([1,1])
        with x:
            st.markdown(f'<div class="ns-card"><small>MODEL RESULT</small><div style="font-size:2.2rem;font-weight:800;margin-top:10px">Category {pred}</div><p>This is the category selected by the trained AI model for this recording.</p></div>',unsafe_allow_html=True)
        with y:
            st.markdown(f'<div class="ns-card"><small>DATASET REFERENCE</small><div style="font-size:2.2rem;font-weight:800;margin-top:10px">Category {actual}</div><p>This is the category stored with the original dataset recording.</p></div>',unsafe_allow_html=True)
        if pred==actual:
            st.success("For this recording, the model result agrees with the category stored in the dataset.")
        else:
            st.warning("For this recording, the model result differs from the category stored in the dataset.")
        if probs is not None:
            df=pd.DataFrame({"Category":[f"Category {int(c)}" for c in model.classes_],"Model output":probs})
            fig=px.bar(df,x="Category",y="Model output",color="Category",text=df["Model output"].map(lambda v:f"{v:.1%}"),title="Model's output for each category")
            fig.update_yaxes(range=[0,1],tickformat=".0%")
            st.plotly_chart(style_plot(fig,310),use_container_width=True)
    st.markdown("## What does this result mean?")
    st.write("The AI model has compared the recording's 70 extracted EEG features with patterns it learned during training and selected one of the two dataset categories.")
    st.markdown("""
    <div class="ns-warn"><b>Important:</b> We have not yet confirmed from the original STEW label documentation which numerical category represents low workload and which represents high workload. Until verified, the report uses Category 0 and Category 1 rather than making an unsupported low/high claim.</div>
    """,unsafe_allow_html=True)
    st.markdown("## Signal check")
    checks=[check_signal(raw[idx,ch,:]) for ch in range(14)]
    flagged=sum(s!="No obvious issue" for s,_ in checks)
    st.metric("Sensors with a basic screening flag",f"{flagged} of 14")
    checkdf=pd.DataFrame({"Sensor":[f"Sensor {i+1}" for i in range(14)],
                          "Screening result":[s for s,_ in checks],
                          "Explanation":[r for _,r in checks]})
    st.dataframe(checkdf,use_container_width=True,hide_index=True)
    st.markdown("## General wellbeing information")
    st.write("If someone personally feels tired or mentally overloaded, taking a short break, drinking water and using a comfortable study environment may help. These are general wellbeing suggestions, not recommendations inferred from a medical assessment.")

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
        a.metric("Recording A · model category",f"Category {pred1}" if pred1 is not None else "Unavailable")
        b.metric("Recording B · model category",f"Category {pred2}" if pred2 is not None else "Unavailable")
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
    st.title("About the EEG dataset")
    st.write("NeuroSense uses the STEW (Simultaneous Task EEG Workload) dataset, a research dataset of recorded EEG signals.")
    a,b,c,d=st.columns(4)
    a.metric("Windows in this app",f"{n:,}")
    b.metric("Participants shown",len(participant_ids))
    c.metric("Sensors per recording",14)
    d.metric("Samples per sensor",256)
    st.markdown("## What is inside one recording?")
    st.markdown("""
    - **14 sensors:** small electrodes placed at different positions on the scalp.
    - **128 samples per second:** how often each sensor's signal is recorded.
    - **2 seconds:** the duration of one short recording window.
    - **70 extracted features:** five frequency-band values from each of the 14 sensors.
    """)
    counts=pd.Series(people).value_counts().sort_index()
    fig=px.bar(x=[participant_name(x) for x in counts.index],y=counts.values,
               labels={"x":"Participant","y":"Recordings available"},
               title="Recordings available for each participant in this app")
    st.plotly_chart(style_plot(fig,390),use_container_width=True)
    person=st.selectbox("Choose a participant to preview their data",participant_ids,format_func=participant_name,key="dataset_person")
    idx,rows=choose_recording(person,"dataset_record")
    with st.expander("See the numerical signal values for this recording"):
        rawdf=pd.DataFrame(raw[idx].T,columns=[f"Sensor {i+1}" for i in range(14)])
        st.dataframe(rawdf.round(4),use_container_width=True)
    with st.expander("See the 70 features extracted from this recording"):
        featdf=pd.DataFrame(features[idx].reshape(14,5),columns=BAND_NAMES)
        featdf.insert(0,"Sensor",[f"Sensor {i+1}" for i in range(1,15)])
        st.dataframe(featdf.round(5),use_container_width=True)
    st.markdown(f'<div class="ns-warn"><b>Dataset transparency:</b> this deployed app contains {n:,} demonstration windows. The full STEW dataset used in the project has 28,512 windows from 48 participants. The deployed subset should not be described as the complete dataset.</div>',unsafe_allow_html=True)

# ============================================================
# 7. RESEARCH RESULTS
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
    if saved_predictions is not None:
        with st.expander("View saved test predictions"):
            st.dataframe(saved_predictions.head(50),use_container_width=True,hide_index=True)
    st.markdown("""
    <div class="ns-note"><b>Evaluation note:</b> the model was assessed using participants held out from training.
    Accuracy and F1-score describe classification performance on the test set; they are not clinical performance measures.</div>
    """,unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("NeuroSense · ECE mini project")
