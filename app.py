        "water and a quieter workspace may help. These are general suggestions and are "
        "not generated from the EEG class prediction."
    )


# ---------------------------------------------------------------------
# EEG EXPLORER
# ---------------------------------------------------------------------
elif page == "EEG Explorer":
    st.title("EEG Explorer")
    st.markdown('<p class="ns-muted">Inspect a stored EEG waveform and its extracted frequency-band features.</p>', unsafe_allow_html=True)
    participant, indices, idx = sample_controls("explorer")

    left, right = st.columns(2)
    left.metric("Selected sample", int(np.where(indices == idx)[0][0]) + 1)
    right.metric("Recorded numeric class", int(labels[idx]))

    st.markdown("### EEG waveform")
    st.caption("Signal amplitude over the two-second selected sample.")
    channel = st.selectbox("Choose EEG channel", list(range(1, 15)), format_func=lambda c: f"Channel {c}", key="wave_channel")
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
    pred, proba = predict_one(idx)

    c1, c2 = st.columns(2)
    c1.metric("Ground-truth numeric label", int(labels[idx]))
    c2.metric("Model-predicted numeric label", str(pred) if pred is not None else "Unavailable")

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
