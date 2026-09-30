<<<<<<< HEAD
# ThyreoSensus-AI 

Explainable AI screening tool for thyroid disorders (Hypothyroidism / Hyperthyroidism) from ECG-derived measurements and patient demographics.

> **Research prototype.** Not a diagnostic replacement for serum TSH / Free T4 testing.

---

## Overview

Thyroid dysfunction is currently diagnosed via blood panels that take 24–48 hours to return. Thyroid hormone directly affects cardiac electrophysiology (heart rate, QT/PR intervals, QRS voltage), so ThyreoSensus-AI screens for likely thyroid state directly from ECG-derived features as a faster triage signal.

**3-class output:** Euthyroid / Hypothyroid / Hyperthyroid

## Datasets

| Dataset | Role | Why |
|---|---|---|
| [NHANES III](https://wwwn.cdc.gov/nchs/nhanes/nhanes3/) (CDC, public) | Primary — ECG-derived features + real thyroid labels | Only public dataset with both linked to the same patients |
| [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) (PhysioNet, public) | Separate waveform-pretraining track | Raw 12-lead waveform, but no thyroid labels |

**Note:** NHANES III provides ECG-*derived* measurements (heart rate, PR/QRS/QT intervals, axes), not raw waveform. MIMIC-IV-ECG cannot be fused with NHANES at the patient level — no shared IDs.

Final labeled cohort: **7,455 patients** — Euthyroid 90.15%, Hypothyroid 7.97%, Hyperthyroid 1.88%. Labels derived via TSH/T4 thresholds from Hollowell et al. (2002).

## Features (18 predictors)

- **7 ECG-derived:** heart rate, PR interval, QRS duration, QT interval, P/QRS/T axis
- **11 demographic:** age, BMI proxy, sex, ethnicity, race (one-hot)

## Methodology

Patient-level 70/15/15 stratified split (5,218 / 1,118 / 1,119), zero leakage verified. Full ablation study across:

- Single-modality baselines: ECG-only, Tabular-only
- Fusion: early (concatenation), late (probability blending), ensembling
- Imbalance handling: SMOTE, SMOTENC, synthetic augmentation, focal loss, cost-sensitive weighting, threshold tuning

All evaluated under 5-fold stratified cross-validation with per-class recall reported (not just accuracy, which is misleading given 90% majority class).

## Final Model

**Random Forest** (`n_estimators=300, max_depth=5, min_samples_leaf=5, class_weight="balanced"`)

| Metric | Score |
|---|---|
| Accuracy | 63.72% |
| Balanced Accuracy | 46.16% |
| Macro-F1 | 35.84% |
| ROC-AUC (OvR) | 68.65% |

**Per-class recall:** Euthyroid 65.5%, Hypothyroid 53.9%, **Hyperthyroid 19.0%** (4/21 test cases detected)

Explicit limitation: severe class imbalance (140 total Hyperthyroid patients) is the primary constraint on minority-class performance — confirmed by testing 8+ imbalance-mitigation techniques, all rejected for degrading overall performance without meaningfully fixing this.

## Explainability

SHAP (TreeExplainer) applied to the final model. Key findings:
- ECG features contribute more to disorder classes (Hypo 39.9%, Hyper 41.9%) than to Euthyroid (28.6%) — physiologically consistent
- Top ECG feature (T-axis) confirmed by both SHAP and independent ANOVA analysis — convergent, not causal, evidence

## Application

Interactive Streamlit dashboard (`app/`):
- **Predict:** single-patient screening via raw ECG waveform upload (NeuroKit2 feature extraction), manual entry, or CSV
- **Batch:** score multiple patients from a CSV or multiple waveform files
- Live SHAP explanation and normal-range comparison for every prediction

### Setup
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
Place your trained model as `app/rf1_final_deployed.pkl` (joblib-dumped `RandomForestClassifier`).

## Project Structure
```
├── data_pipeline/       # NHANES + MIMIC parsing, cleaning, label construction
├── models/               # Training scripts for all tested approaches
├── notebooks/            # Colab notebooks (MIMIC pretraining, fusion experiments)
├── app/                  # Streamlit dashboard
├── results/               # Final figures, confusion matrices, SHAP plots
└── README.md
```

## Limitations

- Trained on ECG-*derived* features, not raw waveform (NHANES III limitation)
- Hyperthyroid detection recall is low (~19%) due to extreme class rarity (1.88% prevalence)
- Not trained on caffeine/physical-activity data — not a confounder-adjusted model despite UI fields for these (advisory only)
- Published literature with stronger results (AUC 0.88–0.93) used private hospital-scale datasets not publicly accessible

## References

Key literature — full list in `docs/references.bib`:
- Choi et al. (2022), *Eur Heart J Digit Health* — ECG biomarker for hyperthyroidism
- Lin et al. (2024), *Commun Med* — AI-ECG for hyperthyroidism detection
- Kłosowicz et al. (2025), *J Clin Med* — thyroid-ECG meta-analysis
- Ribeiro et al. (2020), *Nat Commun* — 12-lead ECG deep learning
- Taleban et al. (2025), *Biomed Signal Process Control* — XAI in ECG systematic review

## License

Research/academic project. Not for clinical use.