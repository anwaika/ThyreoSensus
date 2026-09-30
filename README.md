# ThyreoSensus-AI — Full Application

## What changed from v1
- Real ECG waveform upload now supported: upload a raw signal (CSV/NPY),
  the app runs NeuroKit2 signal processing to extract HR/PR/QRS/QT (and
  an approximate axis if 12-lead columns I and aVF are present), then
  feeds those real extracted values into your trained model.
- Still supports manual entry and single-row CSV upload as alternatives.
- Batch mode now accepts either a feature CSV or multiple raw waveform
  files at once.
- Coffee/athlete questions are included but explicitly marked as NOT
  used by the model — see "Important honesty note" below.
- Removed the separate Model Performance Dashboard and About pages.
  Accuracy caveat and per-prediction SHAP explanation are now shown
  directly alongside every prediction instead.
- Added an ECG segment visualization: real waveform with detected R-peaks
  if you uploaded one, or a schematic PQRST illustration sized to the
  entered intervals if you used manual entry — both color-coded red/green
  against normal clinical reference ranges.

## Important honesty note: what the model actually knows
Your trained RF1 model was trained ONLY on:
- 7 ECG-derived measurements: heart rate, PR interval, QRS duration,
  QT interval, P/QRS/T axis
- 11 demographic features: age, sex, ethnicity, race

It was NOT trained on caffeine intake or physical activity level.
Those two questions are in the UI because you asked for them, but they
are stored for your own notes only and do not affect the prediction.
To make them real, you would need to:
1. Pull NHANES III's Household Adult File dietary/activity variables
2. Merge them into your training data by SEQN
3. Retrain RF1 (or any model) with these as new features
4. Re-run the full evaluation pipeline
This is a real modeling task, not a UI change — ask if you want help
scoping it.

## Setup

1. Export your trained model from Colab:
   ```python
   import joblib
   joblib.dump(rf1_final, "/content/drive/MyDrive/ThyreoSensus/rf1_final_deployed.pkl")
   ```
   Download the `.pkl` to this folder.

2. Install dependencies:
   ```
   pip install -r requirements.txt --break-system-packages
   ```

3. Run:
   ```
   streamlit run app.py
   ```

## Waveform upload format

- CSV: columns named by lead (`I`, `II`, `III`, `aVR`, `aVL`, `aVF`,
  `V1`-`V6`), one row per sample. A single column (any name, or just
  `II`) also works for HR/PR/QRS/QT extraction — axis will default.
- NPY: shape `(n_samples, n_leads)`, matching your MIMIC preprocessing
  format — e.g. `(5000, 12)` for a 10-second 500 Hz 12-lead recording.
  Columns are assumed in standard lead order (I, II, III, aVR, aVL,
  aVF, V1-V6).

## Testing it's using your real model, not fake numbers

Take one of your actual preprocessed MIMIC `.npy` files (e.g.
`preprocessed/normalized/43805018.npy` from your earlier pipeline) and
upload it in "Upload ECG waveform" mode — you should see the real
signal plotted with R-peaks marked, and extracted HR/PR/QRS/QT that
look physiologically plausible (HR roughly 60-100 for a normal-looking
trace). This confirms the pipeline is running real signal processing,
not placeholder values.

## Known limitations to state alongside any demo
- Hyperthyroid recall is ~19% on held-out test (4/21 correctly detected)
  — always show the accuracy caveat, don't present predictions as
  certain, especially for Hyperthyroid.
- Axis extraction from an uploaded waveform is a simplified
  approximation (lead I / aVF amplitude ratio), not the exact algorithm
  NHANES used for its labels — noted in the app's axis message.
- P-axis and T-axis are not reliably auto-extracted from raw waveform
  in this pipeline version and default to a fixed value when using
  waveform upload mode.
