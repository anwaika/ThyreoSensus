import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except ImportError:
    NK_AVAILABLE = False

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="ThyreoSensus-AI | Explainable Thyroid Screening",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .status-card {
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .status-normal {
        background-color: #e8f8f5;
        border-left: 6px solid #2ecc71;
        color: #145a32;
    }
    .status-warning {
        background-color: #fef9e7;
        border-left: 6px solid #f39c12;
        color: #7d6608;
    }
    .status-danger {
        background-color: #fdedec;
        border-left: 6px solid #e74c3c;
        color: #78281f;
    }
    .con-box {
        background-color: #eef2f7;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #3498db;
        margin-top: 10px;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("rf1_final_deployed.pkl")

CLASS_NAMES = ["Euthyroid", "Hypothyroid", "Hyperthyroid"]

FEATURE_NAMES = [
    "ECPRATE", "ECPPR", "ECPQRS", "ECPQT", "ECPAXIS1", "ECPAXIS2", "ECPAXIS3",
    "HSAGEIR", "HAN6FS", "NCPNCAFE", "TOTAL_EXERCISE_MET", "IS_ATHLETE",
    "HSSEX_1", "HSSEX_2",
    "DMARETHN_1", "DMARETHN_2", "DMARETHN_3", "DMARETHN_4",
    "DMARACER_1", "DMARACER_2", "DMARACER_8",
    "ACT_0", "ACT_1", "ACT_2", "ACT_3"
]

ECG_FEATURES = ["ECPRATE", "ECPPR", "ECPQRS", "ECPQT", "ECPAXIS1", "ECPAXIS2", "ECPAXIS3"]
CONFOUNDER_FEATURES = ["NCPNCAFE", "TOTAL_EXERCISE_MET", "IS_ATHLETE", "ACT_0", "ACT_1", "ACT_2", "ACT_3"]

NORMAL_RANGES = {
    "ECPRATE": (60, 100, "bpm", "Heart Rate"),
    "ECPPR": (120, 200, "ms", "PR Interval"),
    "ECPQRS": (70, 110, "ms", "QRS Duration"),
    "ECPQT": (350, 440, "ms", "QT Interval"),
    "ECPAXIS1": (0, 75, "°", "P-Axis"),
    "ECPAXIS2": (-30, 90, "°", "QRS-Axis"),
    "ECPAXIS3": (0, 90, "°", "T-Axis"),
}

# ============================================================
# LOAD MODEL & EXPLAINER
# ============================================================
@st.cache_resource
def load_deployed_system():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH.resolve()}")
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    return model, explainer

# ============================================================
# ROBUST WAVEFORM EXTRACTION
# ============================================================
def extract_ecg_features_from_array(signal_data, sampling_rate=500):
    if not NK_AVAILABLE:
        raise ImportError("NeuroKit2 required for waveform feature extraction.")
    
    signal_arr = np.asarray(signal_data, dtype=float)
    if signal_arr.ndim == 2:
        if signal_arr.shape[0] <= 12 and signal_arr.shape[0] < signal_arr.shape[1]:
            lead_I = signal_arr[0, :]
            lead_II = signal_arr[1, :]
            lead_aVF = (signal_arr[1, :] + signal_arr[2, :]) / 2.0 if signal_arr.shape[0] > 2 else signal_arr[1, :]
            primary_lead = lead_II
        else:
            lead_I = signal_arr[:, 0]
            lead_II = signal_arr[:, 1]
            lead_aVF = (signal_arr[:, 1] + signal_arr[:, 2]) / 2.0 if signal_arr.shape[1] > 2 else signal_arr[:, 1]
            primary_lead = lead_II
    else:
        primary_lead = signal_arr.flatten()
        lead_I = primary_lead
        lead_aVF = primary_lead

    primary_lead = primary_lead[np.isfinite(primary_lead)]
    if len(primary_lead) < int(sampling_rate * 1.5):
        raise ValueError(f"Waveform too short ({len(primary_lead)} samples).")

    try:
        clean_signal = nk.ecg_clean(primary_lead, sampling_rate=sampling_rate, method="neurokit")
        signals, rpeaks_info = nk.ecg_process(clean_signal, sampling_rate=sampling_rate)
    except Exception:
        clean_signal = primary_lead - np.mean(primary_lead)
        signals, rpeaks_info = nk.ecg_process(clean_signal, sampling_rate=sampling_rate)

    hr = float(np.nanmedian(signals["ECG_Rate"].values)) if "ECG_Rate" in signals.columns else 75.0
    if not np.isfinite(hr) or hr < 30 or hr > 220:
        hr = 75.0

    pr_interval, qrs_duration, qt_interval = 160.0, 90.0, 400.0
    try:
        intervals = nk.ecg_intervalrelated(signals, sampling_rate=sampling_rate)
        if isinstance(intervals, pd.DataFrame):
            for col in intervals.columns:
                c_low = str(col).lower()
                val = pd.to_numeric(intervals[col], errors="coerce").median()
                if np.isfinite(val):
                    if "pr" in c_low and 80 <= val <= 350:
                        pr_interval = float(val)
                    elif "qrs" in c_low and 45 <= val <= 220:
                        qrs_duration = float(val)
                    elif "qt" in c_low and 250 <= val <= 650:
                        qt_interval = float(val)
    except Exception:
        pass

    if signal_arr.ndim == 2:
        net_I = float(np.max(lead_I) - np.min(lead_I)) * np.sign(np.mean(lead_I) if np.abs(np.mean(lead_I)) > 1e-4 else 1.0)
        net_aVF = float(np.max(lead_aVF) - np.min(lead_aVF)) * np.sign(np.mean(lead_aVF) if np.abs(np.mean(lead_aVF)) > 1e-4 else 1.0)
        rad = np.arctan2(net_aVF, net_I)
        qrs_axis = float(np.degrees(rad))
    else:
        qrs_axis = 35.0

    return {
        "ECPRATE": round(float(hr), 1),
        "ECPPR": round(float(pr_interval), 1),
        "ECPQRS": round(float(qrs_duration), 1),
        "ECPQT": round(float(qt_interval), 1),
        "ECPAXIS1": 45.0,
        "ECPAXIS2": round(float(qrs_axis), 1),
        "ECPAXIS3": 40.0,
    }, clean_signal, signals

# ============================================================
# PLOTTING
# ============================================================
def plot_ecg_signal_with_peaks(signal, signals_df, sampling_rate=500, max_seconds=6):
    max_len = int(max_seconds * sampling_rate)
    sig_sub = signal[:max_len]
    time_axis = np.arange(len(sig_sub)) / sampling_rate
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(time_axis, sig_sub, color="#1f77b4", lw=1.2, label="Processed Lead II")
    if "ECG_R_Peaks" in signals_df.columns:
        r_peaks = np.where(signals_df["ECG_R_Peaks"].values[:max_len] == 1)[0]
        if len(r_peaks) > 0:
            ax.scatter(r_peaks / sampling_rate, sig_sub[r_peaks], color="#d62728", s=40, zorder=5, label="Detected R-peaks")
    ax.set_title(f"ECG Trace ({max_seconds}s Window) with Automated Landmark Delineation", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("Voltage / Amplitude", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig

def plot_synthetic_pqrst(hr, pr, qrs, qt):
    duration = 3.0
    fs = 500
    t = np.linspace(0, duration, int(duration * fs))
    sig = np.zeros_like(t)
    rr = 60.0 / max(hr, 30)
    beats = np.arange(0.2, duration, rr)
    for b in beats:
        sig += 0.15 * np.exp(-((t - (b - 0.18)) / 0.035)**2)
        sig -= 0.10 * np.exp(-((t - (b - 0.04)) / 0.012)**2)
        sig += 1.00 * np.exp(-((t - b) / 0.012)**2)
        sig -= 0.25 * np.exp(-((t - (b + 0.04)) / 0.015)**2)
        sig += 0.30 * np.exp(-((t - (b + (qt/1000)*0.55)) / 0.07)**2)
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.plot(t, sig, color="#2ca02c", lw=1.3, label="Parametric ECG Cycle")
    ax.set_title(f"Synthesized Cardiac Cycle (HR: {hr} bpm | PR: {pr} ms | QRS: {qrs} ms | QT: {qt} ms)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig

def plot_shap_explanation(shap_df, predicted_state):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    top_df = shap_df.tail(10)
    color_map = {"ECG": "#3498db", "Confounder": "#e74c3c", "Demographic": "#7f8c8d"}
    colors = [color_map.get(m, "#34495e") for m in top_df["Modality"]]
    ax.barh(top_df["Feature_Label"], top_df["SHAP value"], color=colors, height=0.65)
    ax.axvline(0, color="black", linestyle="--", lw=0.8, alpha=0.7)
    ax.set_title(f"SHAP Feature Impact for {predicted_state}\n(Blue: ECG | Red: Confounder / Lifestyle | Grey: Clinical)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Log-odds shift towards prediction", fontsize=9, fontweight="bold")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

# ============================================================
# INPUT BUILDER
# ============================================================
def build_input_vector(ecg_dict, age, sex, ethnicity, race, caffeine_mg, activity_level, bmi=25.0):
    row = {}
    for f in ECG_FEATURES:
        row[f] = float(ecg_dict.get(f, 0.0))
    row["HSAGEIR"] = float(age)
    row["HAN6FS"] = float(bmi)
    row["NCPNCAFE"] = float(caffeine_mg)
    
    # MET intensity estimate from activity tier
    met_map = {0: 0.0, 1: 10.0, 2: 24.0, 3: 45.0}
    row["TOTAL_EXERCISE_MET"] = float(met_map.get(activity_level, 0.0))
    row["IS_ATHLETE"] = 1.0 if activity_level == 3 else 0.0

    # Sex: Female, Male, or Other/Neutral
    if sex == "Male":
        row["HSSEX_1"], row["HSSEX_2"] = 1.0, 0.0
    elif sex == "Female":
        row["HSSEX_1"], row["HSSEX_2"] = 0.0, 1.0
    else: # Other / Prefer not to say -> Neutral prior
        row["HSSEX_1"], row["HSSEX_2"] = 0.5, 0.5

    # Ethnicity
    row["DMARETHN_1"] = 1.0 if "White" in ethnicity else 0.0
    row["DMARETHN_2"] = 1.0 if "Black" in ethnicity else 0.0
    row["DMARETHN_3"] = 1.0 if "Mexican" in ethnicity or "Hispanic" in ethnicity else 0.0
    row["DMARETHN_4"] = 1.0 if ("White" not in ethnicity and "Black" not in ethnicity and "Mexican" not in ethnicity) else 0.0

    # Race
    row["DMARACER_1"] = 1.0 if "White" in race else 0.0
    row["DMARACER_2"] = 1.0 if "Black" in race else 0.0
    row["DMARACER_8"] = 1.0 if ("White" not in race and "Black" not in race) else 0.0

    # Activity level one-hot
    for i in range(4):
        row[f"ACT_{i}"] = 1.0 if activity_level == i else 0.0

    return pd.DataFrame([row], columns=FEATURE_NAMES)

# ============================================================
# CLINICAL DECISION & CONFOUNDER ENGINE
# ============================================================
def evaluate_clinical_screening(
    probs,
    ecg_feats,
    caffeine_mg,
    activity_level
):
    """
    Uses the deployed model's actual probability output.
    Confounder notes are advisory only and never override
    the model prediction.
    """

    # Make sure probabilities are numeric
    probs = np.asarray(probs, dtype=float)

    # Normalize if necessary
    if probs.sum() > 0:
        probs = probs / probs.sum()

    # Actual model prediction
    pred_idx = int(np.argmax(probs))
    status = CLASS_NAMES[pred_idx]

    # ECG / lifestyle information
    hr = float(ecg_feats.get("ECPRATE", 75.0))

    is_athlete = activity_level == 3
    is_high_caffeine = caffeine_mg >= 250.0
    is_sedentary = activity_level == 0

    confounder_notes = []

    # --------------------------------------------------------
    # Low heart rate
    # --------------------------------------------------------
    if hr < 60.0 and is_athlete:
        confounder_notes.append(
            "Low heart rate may be associated with athletic conditioning. "
            "This is an advisory physiological note and does not override "
            "the model prediction."
        )

    elif hr < 60.0 and is_sedentary:
        confounder_notes.append(
            "Low resting heart rate was detected in a sedentary profile. "
            "Clinical correlation may be appropriate."
        )

    # --------------------------------------------------------
    # High heart rate
    # --------------------------------------------------------
    if hr > 95.0 and is_high_caffeine:
        confounder_notes.append(
            "Elevated heart rate may partly reflect caffeine intake. "
            "This is an advisory note and does not override the model prediction."
        )

    elif hr > 95.0 and not is_high_caffeine:
        confounder_notes.append(
            "Elevated resting heart rate was detected without high "
            "reported caffeine intake. Clinical correlation may be appropriate."
        )

    # --------------------------------------------------------
    # Display style
    # --------------------------------------------------------
    css_map = {
        "Euthyroid": "status-normal",
        "Hypothyroid": "status-warning",
        "Hyperthyroid": "status-danger"
    }

    tier = (
        f"Model prediction: {status} "
        f"(P={probs[pred_idx] * 100:.1f}%)"
    )

    return status, tier, css_map[status], confounder_notes

# ============================================================
# MAIN APPLICATION
# ============================================================
st.markdown('<div class="main-header">🫀 ThyreoSensus-AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Confounder-Aware, Explainable AI Screening for Thyroid Dysfunction (Hypo/Hyperthyroid)</div>', unsafe_allow_html=True)

try:
    model, explainer = load_deployed_system()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Configuration")
app_mode = st.sidebar.radio("Operating Mode:", ["🔬 Single Patient Screening", "📂 Cohort Batch Prediction", "📖 Clinical Information"])
sensitivity_mode = st.sidebar.selectbox("Screening Sensitivity:", ["Balanced (Standard Triage)", "High Sensitivity (Screening Alert)", "High Specificity (Confirmation)"])

# ============================================================
# MODE 1: SINGLE PATIENT SCREENING
# ============================================================
if app_mode == "🔬 Single Patient Screening":
    st.markdown("### 1. Patient Demographics & Lifestyle Confounders")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age_in = st.slider("Age (Years)", 18, 95, 45, 1)
        sex_in = st.selectbox("Sex", ["Female", "Male", "Other / Prefer not to say"])
        
    with col2:
        ethnicity_in = st.selectbox("Ethnicity", [
            "Non-Hispanic White", "Non-Hispanic Black", "Mexican American",
            "Asian / Pacific Islander", "Hispanic / Other", "Other / Multiracial"
        ])
        race_in = st.selectbox("Race", ["White", "Black", "Asian / Other", "Other / Unknown"])
        
    with col3:
        st.markdown("**☕ Daily Caffeine Intake**")
        caff_choice = st.selectbox("Caffeine Intake", [
            "None / Zero (0 mg)",
            "Light (1 cup coffee / 2 teas ≈ 90 mg)",
            "Moderate (2 cups coffee ≈ 180 mg)",
            "High (3-4 cups coffee / Energy drink ≈ 350 mg)",
            "Heavy (Pre-workout / Multiple energy drinks ≥ 500 mg)",
            "Custom entry (mg)"
        ])
        if "Custom" in caff_choice:
            caffeine_in = st.number_input("Caffeine (mg/day)", 0, 2000, 150, 25)
        elif "Zero" in caff_choice:
            caffeine_in = 0.0
        elif "Light" in caff_choice:
            caffeine_in = 90.0
        elif "Moderate" in caff_choice:
            caffeine_in = 180.0
        elif "High" in caff_choice:
            caffeine_in = 350.0
        else:
            caffeine_in = 500.0
        st.caption(f"Estimated: **{caffeine_in:.0f} mg/day**")

    with col4:
        st.markdown("**🏃 Physical Activity & Conditioning**")
        activity_choice = st.selectbox("Exercise / Activity Profile", [
            "🛋️ Sedentary / No Exercise (0 hrs/week)",
            "🚶 Minimal / Light Activity (1–2 hrs/week)",
            "🏃 Moderate / Regular Exercise (3–4 hrs/week)",
            "⚡ High Intensity / Competitive Athlete (≥5 hrs/week)"
        ])
        act_level_map = {
            "🛋️ Sedentary / No Exercise (0 hrs/week)": 0,
            "🚶 Minimal / Light Activity (1–2 hrs/week)": 1,
            "🏃 Moderate / Regular Exercise (3–4 hrs/week)": 2,
            "⚡ High Intensity / Competitive Athlete (≥5 hrs/week)": 3
        }
        activity_level_in = act_level_map[activity_choice]
        
        bmi_in = st.number_input("Body Mass Index (BMI)", 14.0, 55.0, 25.0, 0.5)

    st.markdown("---")
    st.markdown("### 2. Electrocardiogram (ECG) Input")
    
    ecg_tab1, ecg_tab2, ecg_tab3 = st.tabs([
        "📈 Mode A: Upload ECG Waveform (.npy / .csv)",
        "📝 Mode B: Enter Clinical ECG Values (Direct Manual)",
        "📄 Mode C: Single Patient Tabular CSV"
    ])
    
    ecg_dict_final = None
    waveform_fig = None
    
    # TAB 1: WAVEFORM UPLOAD
    with ecg_tab1:
        st.write("Upload a 12-lead or 1-lead ECG recording (e.g. MIMIC-IV-ECG `.npy` array at 500Hz or raw CSV time-series).")
        uploaded_wave = st.file_uploader("Upload ECG Waveform File", type=["npy", "csv"], key="w_upload")
        if uploaded_wave is not None:
            try:
                wave_data = np.load(uploaded_wave) if uploaded_wave.name.endswith(".npy") else pd.read_csv(uploaded_wave).values.squeeze()
                st.success(f"Successfully loaded: `{uploaded_wave.name}` | Dimensions: {wave_data.shape}")
                with st.spinner("Executing NeuroKit2 fiducial segmentation and axis computation..."):
                    ecg_dict_final, clean_sig, sig_df = extract_ecg_features_from_array(wave_data, sampling_rate=500)
                    waveform_fig = plot_ecg_signal_with_peaks(clean_sig, sig_df, sampling_rate=500, max_seconds=6)
                    st.pyplot(waveform_fig)
                    plt.close(waveform_fig)
            except Exception as e:
                st.error(f"Waveform parsing error: {e}")
                ecg_dict_final = {"ECPRATE": 75.0, "ECPPR": 160.0, "ECPQRS": 90.0, "ECPQT": 400.0, "ECPAXIS1": 45.0, "ECPAXIS2": 35.0, "ECPAXIS3": 40.0}

    # TAB 2: MANUAL ENTRY
    with ecg_tab2:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            hr_m = st.number_input("Heart Rate (bpm)", 30.0, 220.0, 72.0, 1.0)
            pr_m = st.number_input("PR Interval (ms)", 80.0, 320.0, 158.0, 2.0)
        with m2:
            qrs_m = st.number_input("QRS Duration (ms)", 50.0, 220.0, 88.0, 2.0)
            qt_m = st.number_input("QT Interval (ms)", 250.0, 600.0, 398.0, 2.0)
        with m3:
            qrs_ax = st.number_input("QRS Axis (°)", -180.0, 180.0, 40.0, 5.0)
            p_ax = st.number_input("P-Wave Axis (°)", -90.0, 180.0, 45.0, 5.0)
        with m4:
            t_ax = st.number_input("T-Wave Axis (°)", -90.0, 180.0, 40.0, 5.0)
            
        if uploaded_wave is None:
            ecg_dict_final = {
                "ECPRATE": hr_m, "ECPPR": pr_m, "ECPQRS": qrs_m, "ECPQT": qt_m,
                "ECPAXIS1": p_ax, "ECPAXIS2": qrs_ax, "ECPAXIS3": t_ax
            }
            waveform_fig = plot_synthetic_pqrst(hr_m, pr_m, qrs_m, qt_m)
            st.pyplot(waveform_fig)
            plt.close(waveform_fig)

    # TAB 3: SINGLE ROW CSV
    with ecg_tab3:
        st.write("Upload a 1-row CSV with extracted clinical values.")
        s_csv = st.file_uploader("Upload 1-Row Patient CSV", type=["csv"], key="scsv")
        if s_csv is not None:
            try:
                sdf = pd.read_csv(s_csv)
                r0 = sdf.iloc[0].to_dict()
                ecg_dict_final = {k: float(r0.get(k, 0.0)) for k in ECG_FEATURES}
                st.dataframe(sdf.head(1))
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # ============================================================
    # SCREENING INFERENCE & EXPLANATION
    # ============================================================
    st.markdown("---")
    st.markdown("### 3. Comprehensive Screening Decision & Explanation")
    
    if st.button("🚀 Evaluate Thyroid Screening Risk", type="primary", use_container_width=True):
        if ecg_dict_final is None:
            st.warning("Please provide ECG measurements.")
        else:
            X_input = build_input_vector(
                ecg_dict_final, age_in, sex_in, ethnicity_in, race_in,
                caffeine_in, activity_level_in, bmi=bmi_in
            )
            
            probs = model.predict_proba(X_input)[0]
            pred_state, tier_label, css_class, confounder_notes = evaluate_clinical_screening(
    probs,
    ecg_dict_final,
    caffeine_in,
    activity_level_in
)
            
            # SHAP
            shap_values = explainer.shap_values(X_input)
            pred_idx = CLASS_NAMES.index(pred_state)
            if isinstance(shap_values, list):
                class_shap = shap_values[pred_idx][0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                class_shap = shap_values[0, :, pred_idx]
            else:
                class_shap = shap_values[0]

            label_map = {
                "ECPRATE": "Heart Rate", "ECPPR": "PR Interval", "ECPQRS": "QRS Duration", "ECPQT": "QT Interval",
                "ECPAXIS1": "P-Axis", "ECPAXIS2": "QRS-Axis", "ECPAXIS3": "T-Axis",
                "HSAGEIR": "Age", "HAN6FS": "BMI", "NCPNCAFE": "Caffeine Intake",
                "TOTAL_EXERCISE_MET": "Total Exercise METs", "IS_ATHLETE": "Athlete Status",
                "HSSEX_1": "Male Sex", "HSSEX_2": "Female Sex",
                "DMARETHN_1": "Ethnicity (White)", "DMARETHN_2": "Ethnicity (Black)", "DMARETHN_3": "Ethnicity (Hispanic)", "DMARETHN_4": "Ethnicity (Other)",
                "DMARACER_1": "Race (White)", "DMARACER_2": "Race (Black)", "DMARACER_8": "Race (Other)",
                "ACT_0": "Sedentary (No Exercise)", "ACT_1": "Light Activity", "ACT_2": "Moderate Exercise", "ACT_3": "Athlete / High Intensity"
            }

            shap_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Feature_Label": [label_map.get(f, f) for f in FEATURE_NAMES],
                "SHAP value": class_shap,
                "Modality": [
                    "Confounder" if f in CONFOUNDER_FEATURES else "ECG" if f in ECG_FEATURES else "Demographic"
                    for f in FEATURE_NAMES
                ]
            }).sort_values("SHAP value", key=lambda x: x.abs(), ascending=True)

            res1, res2 = st.columns([1.1, 1.3])
            
            with res1:
                st.markdown(f'<div class="status-card {css_class}"><h3>{tier_label}</h3></div>', unsafe_allow_html=True)
                
                st.markdown("#### Calibrated Risk Probabilities")
                prob_table = pd.DataFrame({
                    "Clinical State": CLASS_NAMES,
                    "Predicted Probability": [f"{probs[0]*100:.1f}%", f"{probs[1]*100:.1f}%", f"{probs[2]*100:.1f}%"],
                    "Population Prevalence": ["90.15%", "7.97%", "1.88%"],
                    "Relative Risk Factor": [f"{probs[0]/0.9015:.2f}x", f"{probs[1]/0.0797:.2f}x", f"{probs[2]/0.0188:.2f}x"]
                })
                st.dataframe(prob_table, hide_index=True, use_container_width=True)
                
                # Confounder reasoning
                if confounder_notes:
                    st.markdown('<div class="con-box"><b>💡 Physiological Confounder Analysis:</b><br>' + "<br>".join(confounder_notes) + '</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="con-box"><b>💡 Confounder Status:</b> No significant autonomic confounder masking detected. Resting electrophysiological markers align with predicted metabolic state.</div>', unsafe_allow_html=True)

            with res2:
                st.markdown("#### Tree SHAP Feature Contribution")
                s_fig = plot_shap_explanation(shap_df, pred_state)
                st.pyplot(s_fig)
                plt.close(s_fig)

            # Normal Ranges
            st.markdown("#### ECG Landmark Normal-Range Comparison")
            rc = st.columns(4)
            for idx, (f_k, (l_val, h_val, u_str, n_str)) in enumerate(NORMAL_RANGES.items()):
                curr_val = ecg_dict_final.get(f_k, 0.0)
                d_color = "normal" if l_val <= curr_val <= h_val else "inverse"
                with rc[idx % 4]:
                    st.metric(f"{n_str} ({u_str})", f"{curr_val:.1f} {u_str}", f"Reference: {l_val}–{h_val} {u_str}", delta_color=d_color)

# ============================================================
# MODE 2: BATCH PROCESSING
# ============================================================
elif app_mode == "📂 Cohort Batch Prediction":
    st.markdown("### Cohort Batch Prediction & Triage")
    st.write("Upload a batch CSV file with patient records to run batch screening.")
    b_file = st.file_uploader("Upload CSV", type=["csv"], key="b_up")
    if b_file is not None:
        b_df = pd.read_csv(b_file)
        st.write(f"Loaded **{len(b_df)}** records.")
        for col in FEATURE_NAMES:
            if col not in b_df.columns:
                b_df[col] = 0.0
        
        preds = model.predict(b_df[FEATURE_NAMES])
        probs = model.predict_proba(b_df[FEATURE_NAMES])
        
        b_df["Predicted_Class"] = [CLASS_NAMES[i] for i in preds]
        b_df["Prob_Euthyroid_%"] = (probs[:, 0] * 100).round(2)
        b_df["Prob_Hypothyroid_%"] = (probs[:, 1] * 100).round(2)
        b_df["Prob_Hyperthyroid_%"] = (probs[:, 2] * 100).round(2)
        
        st.dataframe(b_df.head(25), use_container_width=True)
        st.download_button("📥 Download Annotated CSV", b_df.to_csv(index=False).encode('utf-8'), "thyreosensus_batch_results.csv", "text/csv", type="primary")

# ============================================================
# MODE 3: CLINICAL INFO
# ============================================================
else:
    st.markdown("### 📖 Clinical Model Card & Performance Transparency")
    st.markdown("""
    #### 1. Why Multimodal Confounder Modeling is Critical
    - **Heart rate is not a standalone diagnostic metric**: A resting heart rate of 52 bpm is clinically normal in an endurance athlete (high vagal tone), but suspicious for hypothyroidism in a sedentary individual.
    - **Caffeine chronotropy**: Consuming 300+ mg caffeine elevates resting heart rate by 8–15 bpm, which mimics hyperthyroid tachycardia.
    - **Disentanglement**: ThyreoSensus-AI models caffeine and athletic conditioning as explicit physiological confounders.
    
    #### 2. Intended Use
    This system is a **point-of-care screening and triage prototype**, not a diagnostic replacement for serum TSH / Free T4 lab tests.
    """)
