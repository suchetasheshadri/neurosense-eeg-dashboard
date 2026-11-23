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
