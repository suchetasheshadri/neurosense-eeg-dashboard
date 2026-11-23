from pathlib import Path
import gzip, io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------- PAGE SETUP ----------------------------
st.set_page_config(page_title="NeuroSense", page_icon="🧠", layout="wide")
BASE = Path(__file__).parent
FS = 128
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

st.markdown("""
<style>
.stApp {background:#f5f8fc; color:#17243a;}
[data-testid="stSidebar"] {background:#eaf0f8;}
.block-container {padding-top:2rem; max-width:1200px;}
h1,h2,h3 {color:#17243a;}
.ns-card {background:white; border:1px solid #dce5f0; border-radius:16px; padding:18px;}
.ns-small {color:#53657d; font-size:0.94rem;}
div.stButton > button {border-radius:10px; min-height:2.8rem; font-weight:600;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_npy(name):
    p = BASE / name
    return np.load(p, allow_pickle=False) if p.exists() else None

@st.cache_data
def load_csv(name):
    p = BASE / name
    return pd.read_csv(p) if p.exists() else None

@st.cache_resource
def load_model():
    p = BASE / "eeg_workload_model.pkl.gz"
    if not p.exists():
        return None
    return joblib.load(io.BytesIO(gzip.decompress(p.read_bytes())))

X = load_npy("STEW_X_demo.npy")
features = load_npy("STEW_features_demo.npy")
labels = load_npy("STEW_labels_demo.npy")
subjects = load_npy("STEW_subjects_demo.npy")
model = load_model()
comparison = load_csv("model_comparison.csv")
predictions = load_csv("test_predictions.csv")

if any(v is None for v in [X, features, labels, subjects]):
    st.error("Some required project data files are missing. Please check that all four STEW demo .npy files are in the same GitHub folder as app.py.")
    st.stop()

n = min(len(X), len(features), len(labels), len(subjects))
X, features, labels, subjects = X[:n], features[:n], labels[:n], subjects[:n]

if features.ndim != 2 or features.shape[1] != 70:
    st.error(f"Expected 70 extracted features per EEG window, but found shape {features.shape}.")
    st.stop()


def predict(idx):
    if model is None:
        return None, None
    row = features[idx].reshape(1, -1)
    pred = int(np.asarray(model.predict(row)).ravel()[0])
    probs = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None
    return pred, probs


def signal_check(window):
    a = np.asarray(window, dtype=float)
    if not np.isfinite(a).all():
        return "Needs review", ["The recording contains missing or invalid numerical values."], "error"
    std = np.std(a, axis=1)
    ptp = np.ptp(a, axis=1)
    flat = int(np.sum((std < 1e-8) | (ptp < 1e-8)))
    max_abs = float(np.max(np.abs(a)))
    notes = []
    if flat:
        notes.append(f"{flat} of the 14 channels are almost flat in this window.")
    if max_abs > 1e6:
        notes.append("The recording contains an unusually large numerical amplitude. Check the source data and units.")
    if notes:
        return "Please review", notes, "warning"
    return "No obvious issue found", ["No flat channels or extreme numerical values were detected by these simple checks."], "success"


def sample_options(participant):
    return np.where(subjects == participant)[0].tolist()


def sample_label(indices, idx):
    return f"Recording {indices.index(idx) + 1}"


def show_recording_result(idx, participant, heading=True):
    if heading:
        st.subheader("Your recording result")
    pred, probs = predict(idx)
    q_status, q_notes, q_kind = signal_check(X[idx])
    left, right = st.columns(2)
    with left:
        st.markdown("#### 1. Model result")
        if pred is None:
            st.error("The trained model could not be loaded. Check eeg_workload_model.pkl.gz.")
        else:
            st.metric("Predicted dataset category", f"Class {pred}")
            st.caption("This is the model's numerical category. The original Class 0 / Class 1 meaning has not been verified, so we do not label it low or high workload.")
            if probs is not None:
                prob_df = pd.DataFrame({"Class": [f"Class {int(c)}" for c in model.classes_], "Model output": probs})
                fig = px.bar(prob_df, x="Class", y="Model output", range_y=[0,1], text=prob_df["Model output"].map(lambda x:f"{x:.1%}"), title="How strongly the model favoured each class")
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=55,b=10), yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("These are model probabilities, not medical certainty scores.")
    with right:
        st.markdown("#### 2. Recording check")
        st.metric("Basic signal check", q_status)
        for note in q_notes:
            st.write("• " + note)
        st.caption("This is a simple numerical screening check, not a clinically validated EEG quality assessment.")

    st.markdown("#### 3. What should you take away?")
    if pred is not None:
        st.info(f"For Participant {participant}, the selected 2-second recording was assigned to **Class {pred}** by the trained Random Forest model.")
        st.write("You can explore the waveform and frequency-band values below, or compare this recording with another participant's recording. The class should not be interpreted as a diagnosis or a direct measurement of someone's mental state.")

    with st.expander("See the EEG signal (technical details)"):
        ch = st.selectbox("Choose a channel", list(range(1,15)), format_func=lambda x:f"Channel {x}", key=f"channel_{participant}_{idx}")
        t = np.arange(X.shape[2]) / FS
        fig = go.Figure(go.Scatter(x=t, y=X[idx, ch-1, :], mode="lines", name=f"Channel {ch}"))
        fig.update_layout(title=f"Participant {participant} · Channel {ch}", xaxis_title="Time (seconds)", yaxis_title="Signal amplitude", height=330, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
        band_values = features[idx].reshape(14,5)[ch-1]
        band_df = pd.DataFrame({"Frequency band":BANDS, "Extracted feature value":band_values})
        fig2 = px.bar(band_df, x="Frequency band", y="Extracted feature value", title=f"Frequency features for Channel {ch}")
        fig2.update_layout(height=300, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------- NAVIGATION ----------------------------
st.sidebar.title("🧠 NeuroSense")
st.sidebar.caption("EEG recording analysis")
page = st.sidebar.radio("Where would you like to go?", ["Home", "Analyse a recording", "Compare two recordings", "Project information"])
st.sidebar.divider()
st.sidebar.caption("Uses stored STEW EEG recordings. No live wireless headset is connected.")

# ---------------------------- HOME ----------------------------
if page == "Home":
    st.title("Welcome to NeuroSense")
    st.markdown("### Explore an EEG recording in a few simple steps.")
    st.write("NeuroSense is a student-built research prototype. It analyses previously recorded EEG data, checks for a few obvious signal issues, and uses a machine-learning model to assign a numerical category to each recording.")
    a,b,c = st.columns(3)
    a.markdown('<div class="ns-card"><h3>① Choose</h3><p>Select a participant and one of their EEG recordings.</p></div>', unsafe_allow_html=True)
    b.markdown('<div class="ns-card"><h3>② Analyse</h3><p>View the model category and a basic signal check.</p></div>', unsafe_allow_html=True)
    c.markdown('<div class="ns-card"><h3>③ Understand</h3><p>Explore the signal or compare it with another recording.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.info(f"This demonstration contains {n:,} EEG windows from {len(np.unique(subjects))} participants. Each window is 2 seconds long and contains 14 channels.")
    st.warning("Important: Class 0 and Class 1 are shown as numerical categories because the original label mapping has not been verified. This prototype is not a medical or diagnostic tool.")
    st.markdown("### Start here")
    st.write("Choose **Analyse a recording** from the left menu. You do not need to understand the technical graphs to use the main result.")

# ---------------------------- ANALYSE ONE ----------------------------
elif page == "Analyse a recording":
    st.title("Analyse a recording")
    st.write("Choose a participant, choose one recording, then press the button. NeuroSense will show the result in plain language.")
    participant_ids = sorted(np.unique(subjects).tolist())
    participant = st.selectbox("Step 1 — Choose a participant", participant_ids)
    inds = sample_options(participant)
    idx = st.selectbox("Step 2 — Choose a recording", inds, format_func=lambda i: sample_label(inds, i))
    if st.button("Analyse my recording", type="primary", use_container_width=True):
        st.session_state["analysed_sample"] = (int(participant), int(idx))
    if "analysed_sample" in st.session_state:
        saved_participant, saved_idx = st.session_state["analysed_sample"]
        if saved_idx < n and subjects[saved_idx] == saved_participant:
            show_recording_result(saved_idx, saved_participant)
        else:
            st.info("Choose a recording and press Analyse my recording to see its result.")
    else:
        st.markdown('<div class="ns-card"><h4>Ready when you are</h4><p>After you press <b>Analyse my recording</b>, your result will appear here.</p></div>', unsafe_allow_html=True)

# ---------------------------- COMPARE ----------------------------
elif page == "Compare two recordings":
    st.title("Compare two recordings")
    st.write("Choose two stored EEG recordings and compare what the model sees. This is useful for exploring differences between recordings; it is not a comparison of two people's health.")
    ids = sorted(np.unique(subjects).tolist())
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Recording A")
        pa = st.selectbox("Participant A", ids, key="pa")
        ia_list = sample_options(pa)
        ia = st.selectbox("Recording A", ia_list, format_func=lambda i: sample_label(ia_list,i), key="ia")
    with col2:
        st.markdown("#### Recording B")
        pb = st.selectbox("Participant B", ids, index=min(1,len(ids)-1), key="pb")
        ib_list = sample_options(pb)
        ib = st.selectbox("Recording B", ib_list, format_func=lambda i: sample_label(ib_list,i), key="ib")
    if st.button("Compare recordings", type="primary", use_container_width=True):
        st.session_state["comparison_samples"] = (int(ia), int(ib), int(pa), int(pb))
    if "comparison_samples" in st.session_state:
        ia2, ib2, pa2, pb2 = st.session_state["comparison_samples"]
        if ia2 < n and ib2 < n:
            pred_a, prob_a = predict(ia2)
            pred_b, prob_b = predict(ib2)
            st.markdown("### Comparison summary")
            c1,c2 = st.columns(2)
            c1.metric("Recording A · model category", f"Class {pred_a}" if pred_a is not None else "Unavailable")
            c2.metric("Recording B · model category", f"Class {pred_b}" if pred_b is not None else "Unavailable")
            st.caption("The class numbers are model outputs. Their low/high workload meaning has not been verified.")
            band_a = features[ia2].reshape(14,5).mean(axis=0)
            band_b = features[ib2].reshape(14,5).mean(axis=0)
            df = pd.DataFrame({"Frequency band":BANDS, "Recording A":band_a, "Recording B":band_b})
            long = df.melt(id_vars="Frequency band", var_name="Recording", value_name="Average extracted feature")
            fig = px.bar(long, x="Frequency band", y="Average extracted feature", color="Recording", barmode="group", title="Compare frequency-band features across all 14 channels")
            fig.update_layout(height=380, margin=dict(l=10,r=10,t=55,b=10))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("View both EEG waveforms"):
                channel = st.selectbox("Choose a channel to compare", list(range(1,15)), format_func=lambda x:f"Channel {x}", key="compare_channel")
                t = np.arange(X.shape[2])/FS
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=t,y=X[ia2,channel-1,:],mode="lines",name=f"Participant {pa2}"))
                fig2.add_trace(go.Scatter(x=t,y=X[ib2,channel-1,:],mode="lines",name=f"Participant {pb2}"))
                fig2.update_layout(title=f"Channel {channel} waveform comparison",xaxis_title="Time (seconds)",yaxis_title="Signal amplitude",height=380)
                st.plotly_chart(fig2,use_container_width=True)
            qa = signal_check(X[ia2])[0]
            qb = signal_check(X[ib2])[0]
            st.write(f"**Basic signal check:** Recording A — {qa}; Recording B — {qb}.")

# ---------------------------- ABOUT ----------------------------
elif page == "Project information":
    st.title("About NeuroSense")
    st.write("NeuroSense is a software-based EEG analysis prototype developed as an ECE student project.")
    st.markdown("### What happens behind the scenes?")
    st.markdown("1. EEG recordings come from the STEW dataset.\n2. Each selected recording contains 14 channels and 256 samples per channel.\n3. The project uses 70 pre-extracted frequency-band features per recording.\n4. A trained Random Forest model predicts a numerical class.\n5. The dashboard presents the result and visualises the selected signal.")
    st.markdown("### Project facts")
    a,b,c,d = st.columns(4)
    a.metric("EEG channels", "14")
    b.metric("Sampling rate", "128 Hz")
    c.metric("Window duration", "2 s")
    d.metric("Features", "70")
    st.markdown("### Important limitations")
    st.write("This application uses stored dataset recordings, not live wireless EEG. The numerical class labels have not been mapped to low/high workload here. Signal checks are simple screening rules, not clinical validation. Model probabilities are not medical certainty scores.")
    if comparison is not None:
        with st.expander("Model comparison results"):
            st.dataframe(comparison, use_container_width=True, hide_index=True)
    if predictions is not None:
        with st.expander("Saved test predictions"):
            st.dataframe(predictions.head(30), use_container_width=True, hide_index=True)
