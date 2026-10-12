# NeuroSense — EEG Workload Monitoring Dashboard

Streamlit prototype for exploring stored EEG frequency-band features, numeric labels, saved predictions and model comparison.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional data files beside app.py
- `STEW_features_demo.npy`: small demonstration subset of the 70-feature matrix
- `STEW_labels_demo.npy`: matching numeric labels
- `STEW_subjects_demo.npy`: matching subject IDs
- `eeg_workload_model.pkl`: trained scikit-learn model
- `model_comparison.csv` and `test_predictions.csv`: saved results

The full raw STEW dataset is intentionally not included. This is a software-only demo, not a live wireless EEG acquisition system. Verify the original numeric label mapping before assigning low/high names.
