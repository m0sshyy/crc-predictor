import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')
 
# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CRC Risk Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
 
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
 
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #050d1a;
    color: #e8edf5;
}
 
.main .block-container {
    padding: 2rem 3rem 4rem;
    max-width: 1200px;
}
 
/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0d2045 40%, #0a2d1f 100%);
    border: 1px solid rgba(0,200,120,0.15);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,200,120,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 400px; height: 200px;
    background: radial-gradient(ellipse, rgba(0,100,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #00c878;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero h1 {
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.3;
    color: #ffffff;
    margin-bottom: 0.8rem;
    max-width: 700px;
}
.hero h1 span { color: #00c878; }
.hero p {
    font-size: 0.92rem;
    color: #8da0b8;
    max-width: 580px;
    line-height: 1.7;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,200,120,0.1);
    border: 1px solid rgba(0,200,120,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #00c878;
    margin-top: 1.2rem;
}
 
/* ── Section Headers ── */
.section-title {
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.18em;
    color: #00c878;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(0,200,120,0.3), transparent);
}
 
/* ── Card ── */
.card {
    background: #0b1629;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
}
.card-accent {
    border-left: 3px solid #00c878;
}
 
/* ── Inputs override ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label,
div[data-testid="stSlider"] label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #a8bcd4 !important;
    letter-spacing: 0.02em;
    margin-bottom: 0.2rem;
}
div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] {
    background: #0f1e35 !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: #e8edf5 !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.88rem !important;
}
div[data-baseweb="select"]:focus-within {
    border-color: rgba(0,200,120,0.4) !important;
    box-shadow: 0 0 0 2px rgba(0,200,120,0.1) !important;
}
 
/* ── Predict Button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00c878, #00a060) !important;
    color: #050d1a !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,200,120,0.3) !important;
}
 
/* ── Risk Result Cards ── */
.result-high {
    background: linear-gradient(135deg, #1a0808, #2d0f0f);
    border: 1px solid rgba(255,80,80,0.3);
    border-radius: 16px;
    padding: 2rem;
}
.result-medium {
    background: linear-gradient(135deg, #1a1008, #2d1f08);
    border: 1px solid rgba(255,170,0,0.3);
    border-radius: 16px;
    padding: 2rem;
}
.result-low {
    background: linear-gradient(135deg, #081a0e, #0a2d18);
    border: 1px solid rgba(0,200,120,0.3);
    border-radius: 16px;
    padding: 2rem;
}
.risk-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.risk-value {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.risk-high { color: #ff5050; }
.risk-medium { color: #ffaa00; }
.risk-low { color: #00c878; }
.risk-confidence {
    font-size: 0.85rem;
    color: #8da0b8;
}
 
/* ── Probability Bar ── */
.prob-bar-wrap {
    margin: 0.5rem 0;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.6rem;
}
.prob-label {
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    width: 60px;
    color: #8da0b8;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}
.prob-pct {
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    width: 40px;
    text-align: right;
    color: #8da0b8;
}
 
/* ── SHAP Section ── */
.shap-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1rem;
}
.shap-badge {
    background: rgba(100,120,255,0.12);
    border: 1px solid rgba(100,120,255,0.25);
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #8090ff;
    letter-spacing: 0.1em;
}
 
/* ── Feature Pills ── */
.feat-pill {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.2rem;
}
.feat-risk  { background: rgba(255,80,80,0.12);  border: 1px solid rgba(255,80,80,0.3);  color: #ff8080; }
.feat-prot  { background: rgba(0,200,120,0.10); border: 1px solid rgba(0,200,120,0.3); color: #00c878; }
 
/* ── Disclaimer ── */
.disclaimer {
    background: rgba(255,200,0,0.05);
    border: 1px solid rgba(255,200,0,0.15);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.78rem;
    color: #8da0b8;
    line-height: 1.6;
    margin-top: 2rem;
}
.disclaimer strong { color: #ffc800; }
 
/* ── Divider ── */
hr { border: none; border-top: 1px solid rgba(255,255,255,0.05); margin: 2rem 0; }
 
/* ── Streamlit Overrides ── */
.stApp { background-color: #050d1a; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #080f1e; }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('crc_xgboost_model.pkl', 'rb') as f:
        return pickle.load(f)
 
try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found. Ensure `crc_xgboost_model.pkl` is in the same directory.")
    st.stop()
 
# Feature column order (must match training)
FEATURE_COLS = [
    'Age', 'Gender', 'Family_History', 'Smoking_History', 'Alcohol_Consumption',
    'Diabetes', 'Inflammatory_Bowel_Disease', 'Genetic_Mutation',
    'Obesity_Risk_Level', 'Diet_Risk_Level', 'Physical_Inactivity_Risk',
    'Screening_History_Irregular', 'Screening_History_Never', 'Screening_History_Regular',
    'Genetic_Age_Interaction', 'Medical_Comorbidity_Score', 'Lifestyle_Index'
]
 
FEATURE_LABELS = {
    'Age': 'Age',
    'Gender': 'Gender',
    'Family_History': 'Family History',
    'Smoking_History': 'Smoking',
    'Alcohol_Consumption': 'Alcohol Use',
    'Diabetes': 'Diabetes',
    'Inflammatory_Bowel_Disease': 'IBD',
    'Genetic_Mutation': 'Genetic Mutation',
    'Obesity_Risk_Level': 'Obesity Level',
    'Diet_Risk_Level': 'Diet Risk',
    'Physical_Inactivity_Risk': 'Physical Inactivity',
    'Screening_History_Irregular': 'Irregular Screening',
    'Screening_History_Never': 'Never Screened',
    'Screening_History_Regular': 'Regular Screening',
    'Genetic_Age_Interaction': 'Genetic × Age',
    'Medical_Comorbidity_Score': 'Comorbidity Score',
    'Lifestyle_Index': 'Lifestyle Index'
}
 
# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-label">🔬 Clinical Decision Support Tool</div>
  <h1>AI-Based Predictive Model for<br><span>Early Detection</span> of Colorectal Cancer Risk Factors</h1>
  <p>Using XGBoost machine learning with SHAP explainability to assess individual CRC risk based on clinical, genetic, and lifestyle parameters.</p>
  <div class="hero-badge">⚡ XGBoost · SHAP Explainability · v1.0</div>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
#  INPUT FORM
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Patient Data Input</div>', unsafe_allow_html=True)
 
with st.container():
    col_a, col_b, col_c = st.columns(3)
 
    with col_a:
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        st.markdown("**🧍 Demographics**")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        screening = st.selectbox("Screening History", ["Regular", "Irregular", "Never"])
        st.markdown('</div>', unsafe_allow_html=True)
 
    with col_b:
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        st.markdown("**🧬 Clinical & Genetic**")
        family_history = st.selectbox("Family History of CRC", ["No", "Yes"])
        genetic_mutation = st.selectbox("Genetic Mutation (BRCA/MLH1/etc.)", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        ibd = st.selectbox("Inflammatory Bowel Disease", ["No", "Yes"])
        st.markdown('</div>', unsafe_allow_html=True)
 
    with col_c:
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        st.markdown("**🍔 Lifestyle Factors**")
        smoking = st.selectbox("Smoking History", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
        obesity = st.selectbox("Obesity Level", ["Normal", "Overweight", "Obese"])
        diet_risk = st.selectbox("Diet Risk Level", ["Low", "Moderate", "High"])
        activity = st.selectbox("Physical Activity Level", ["High", "Moderate", "Low"])
        st.markdown('</div>', unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍  Analyse Risk Profile", use_container_width=True)
 
# ─────────────────────────────────────────────
#  PREDICTION + SHAP
# ─────────────────────────────────────────────
if predict_btn:
    # ── Encode ──
    gen_val   = 1 if gender == "Male" else 0
    fam_val   = 1 if family_history == "Yes" else 0
    smok_val  = 1 if smoking == "Yes" else 0
    alc_val   = 1 if alcohol == "Yes" else 0
    diab_val  = 1 if diabetes == "Yes" else 0
    ibd_val   = 1 if ibd == "Yes" else 0
    gene_val  = 1 if genetic_mutation == "Yes" else 0
 
    obesity_map  = {"Normal": 0, "Overweight": 1, "Obese": 2}
    diet_map     = {"Low": 0, "Moderate": 1, "High": 2}
    activity_map = {"High": 0, "Moderate": 1, "Low": 2}
 
    genetic_age_interaction = gene_val * age
    medical_score  = ibd_val + diab_val
    lifestyle_idx  = smok_val + alc_val + diet_map[diet_risk] + obesity_map[obesity] + activity_map[activity]
 
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gen_val,
        'Family_History': fam_val,
        'Smoking_History': smok_val,
        'Alcohol_Consumption': alc_val,
        'Diabetes': diab_val,
        'Inflammatory_Bowel_Disease': ibd_val,
        'Genetic_Mutation': gene_val,
        'Obesity_Risk_Level': obesity_map[obesity],
        'Diet_Risk_Level': diet_map[diet_risk],
        'Physical_Inactivity_Risk': activity_map[activity],
        'Screening_History_Irregular': 1 if screening == "Irregular" else 0,
        'Screening_History_Never': 1 if screening == "Never" else 0,
        'Screening_History_Regular': 1 if screening == "Regular" else 0,
        'Genetic_Age_Interaction': genetic_age_interaction,
        'Medical_Comorbidity_Score': medical_score,
        'Lifestyle_Index': lifestyle_idx
    }])[FEATURE_COLS]
 
    prediction    = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = {0: "Low", 1: "Moderate", 2: "High"}
    result  = classes[prediction]
 
    # ── SHAP values ──
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        # For multi-class, shap_values is a list; pick the predicted class
        if isinstance(shap_values, list):
            sv = shap_values[int(prediction)][0]
        else:
            sv = shap_values[0]
        shap_ok = True
    except Exception as e:
        shap_ok = False
 
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Risk Assessment Result</div>', unsafe_allow_html=True)
 
    # ── Result + probability bars side by side ──
    r1, r2 = st.columns([1, 1.4])
 
    with r1:
        css_class = f"result-{result.lower()}"
        risk_css  = f"risk-{result.lower()}"
        conf      = np.max(probabilities) * 100
        icon      = "🔴" if result == "High" else ("🟡" if result == "Moderate" else "🟢")
 
        st.markdown(f"""
        <div class="{css_class}">
          <div class="risk-label" style="color:#8da0b8">Predicted Risk Level</div>
          <div class="risk-value {risk_css}">{icon} {result}</div>
          <div class="risk-confidence">Model confidence: <strong style="color:#e8edf5">{conf:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)
 
    with r2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Class Probability Distribution**")
        colors = {"Low": "#00c878", "Moderate": "#ffaa00", "High": "#ff5050"}
        for i, (label, prob) in enumerate(zip(["Low", "Moderate", "High"], probabilities)):
            pct = prob * 100
            color = colors[label]
            st.markdown(f"""
            <div class="prob-row">
              <span class="prob-label">{label}</span>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
              </div>
              <span class="prob-pct">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
 
    # ── SHAP Waterfall Chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="shap-header">
      <div class="section-title" style="margin-bottom:0">XAI Explanation</div>
      <div class="shap-badge">SHAP · SHapley Additive exPlanations</div>
    </div>
    """, unsafe_allow_html=True)
 
    if shap_ok:
        feature_labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
        shap_series = pd.Series(sv, index=feature_labels)
        shap_sorted = shap_series.reindex(shap_series.abs().sort_values(ascending=True).index)
        top_n = 12
        shap_top = shap_sorted.iloc[-top_n:]
 
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor('#0b1629')
        ax.set_facecolor('#0b1629')
 
        bar_colors = ['#ff5050' if v > 0 else '#00c878' for v in shap_top.values]
        bars = ax.barh(shap_top.index, shap_top.values, color=bar_colors,
                       height=0.62, edgecolor='none')
 
        # Value labels
        for bar, val in zip(bars, shap_top.values):
            x_pos = val + (0.003 if val >= 0 else -0.003)
            ha = 'left' if val >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', va='center', ha=ha,
                    fontsize=8, color='#c8d8e8',
                    fontfamily='monospace')
 
        ax.axvline(0, color='rgba(255,255,255,0.15)', linewidth=0.8, linestyle='--')
        ax.set_xlabel('SHAP Value  (impact on predicted risk class)', color='#8da0b8',
                      fontsize=9, labelpad=10)
        ax.tick_params(axis='x', colors='#8da0b8', labelsize=8)
        ax.tick_params(axis='y', colors='#c8d8e8', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('rgba(255,255,255,0.08)')
        ax.spines['left'].set_color('rgba(255,255,255,0.08)')
 
        # Legend
        patch_risk = mpatches.Patch(color='#ff5050', label='↑ Increases risk')
        patch_prot = mpatches.Patch(color='#00c878', label='↓ Decreases risk')
        ax.legend(handles=[patch_risk, patch_prot], loc='lower right',
                  framealpha=0, labelcolor='#8da0b8', fontsize=8)
 
        title_text = f'Feature Contributions → Predicted: {result} Risk'
        ax.set_title(title_text, color='#e8edf5', fontsize=11, fontweight='600',
                     pad=14, loc='left')
 
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
        # ── Top Risk vs Protective Factors (pill summary) ──
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
 
        risk_factors = shap_series[shap_series > 0].sort_values(ascending=False)
        prot_factors = shap_series[shap_series < 0].sort_values(ascending=True)
 
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**🔴 Top Risk-Increasing Factors**")
            if len(risk_factors) > 0:
                pills = ""
                for feat, val in risk_factors.head(6).items():
                    pills += f'<span class="feat-pill feat-risk">{feat}  +{val:.3f}</span>'
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown(f"<br><small style='color:#8da0b8'>These features push the model toward a <strong style='color:#ff5050'>{result}</strong> risk prediction.</small>", unsafe_allow_html=True)
            else:
                st.markdown("<small style='color:#8da0b8'>No risk-increasing features detected.</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
 
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**🟢 Top Risk-Decreasing Factors**")
            if len(prot_factors) > 0:
                pills = ""
                for feat, val in prot_factors.head(6).items():
                    pills += f'<span class="feat-pill feat-prot">{feat}  {val:.3f}</span>'
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown(f"<br><small style='color:#8da0b8'>These features lower the predicted risk level for this patient.</small>", unsafe_allow_html=True)
            else:
                st.markdown("<small style='color:#8da0b8'>No protective features detected.</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
 
    else:
        st.warning("SHAP explanation could not be generated for this model. Ensure the model is a native XGBoost or scikit-learn compatible object.")
 
    # ── Clinical Notes ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical Interpretation</div>', unsafe_allow_html=True)
 
    interpretations = {
        "High": {
            "icon": "🔴",
            "summary": "This patient profile is associated with **high** colorectal cancer risk.",
            "actions": [
                "Immediate referral for colonoscopy is recommended.",
                "Genetic counselling if mutation markers are present.",
                "Lifestyle intervention: smoking cessation, diet modification, weight management.",
                "Increase screening frequency to annual or biennial.",
            ]
        },
        "Moderate": {
            "icon": "🟡",
            "summary": "This patient profile is associated with **moderate** colorectal cancer risk.",
            "actions": [
                "Schedule colonoscopy within 1–3 years.",
                "Encourage lifestyle modifications (diet, exercise, alcohol reduction).",
                "Monitor comorbidities (diabetes, IBD) closely.",
                "Educate patient on early warning symptoms.",
            ]
        },
        "Low": {
            "icon": "🟢",
            "summary": "This patient profile is associated with **low** colorectal cancer risk.",
            "actions": [
                "Continue regular screening as per national guidelines.",
                "Maintain healthy lifestyle: balanced diet, physical activity.",
                "Reassess annually or if new risk factors emerge.",
                "Patient education on CRC prevention.",
            ]
        }
    }
    info = interpretations[result]
    action_items = "".join([f"<li style='margin-bottom:0.4rem;color:#8da0b8'>{a}</li>" for a in info['actions']])
    st.markdown(f"""
    <div class="card">
      <p style="font-size:0.95rem;margin-bottom:1rem">{info['icon']} {info['summary']}</p>
      <p style="font-size:0.78rem;font-family:'JetBrains Mono',monospace;color:#00c878;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem">Recommended Actions</p>
      <ul style="padding-left:1.2rem;font-size:0.88rem">
        {action_items}
      </ul>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer">
      <strong>⚠️ Medical Disclaimer:</strong> This tool is developed for <strong>academic and research purposes only</strong>.
      It is not a substitute for professional medical diagnosis, advice, or treatment.
      All clinical decisions must be made by a qualified healthcare professional.
      SHAP values represent feature contributions to the model's prediction and do not imply direct medical causality.
    </div>
    """, unsafe_allow_html=True)
 
