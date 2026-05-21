import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
#  GLOBAL CSS  (light theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #f4f7fb;
    color: #1a2540;
}

.main .block-container {
    padding: 1.5rem 2.5rem 4rem;
    max-width: 1280px;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Top nav bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border-radius: 14px;
    padding: 0.9rem 1.6rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.topbar-brand {
    display: flex; align-items: center; gap: 0.6rem;
    font-weight: 800; font-size: 1.05rem; color: #1a2540;
}
.topbar-brand span { color: #1a6fd4; }
.topbar-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.12em;
    background: #eef4ff; color: #1a6fd4;
    border: 1px solid #c4d9f8;
    border-radius: 20px; padding: 0.2rem 0.7rem;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1a3a6b 0%, #1a6fd4 55%, #0ea5a0 100%);
    border-radius: 20px;
    padding: 2.8rem 3rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 2rem;
}
.hero-text { flex: 1; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.2em;
    color: rgba(255,255,255,0.7); text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.hero h1 {
    font-size: 1.6rem; font-weight: 800; line-height: 1.3;
    color: #ffffff; margin-bottom: 0.7rem;
}
.hero h1 em { font-style: normal; color: #7ee8df; }
.hero p { font-size: 0.88rem; color: rgba(255,255,255,0.75); line-height: 1.7; max-width: 520px; }
.hero-art { flex-shrink: 0; opacity: 0.95; }
.hero::before {
    content: ''; position: absolute;
    top: -80px; right: 280px; width: 260px; height: 260px;
    background: rgba(255,255,255,0.04); border-radius: 50%;
    pointer-events: none;
}

/* ── Section label ── */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: #1a6fd4; font-weight: 600;
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 1rem;
}
.sec-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(to right, #c4d9f8, transparent);
}

/* ── White cards ── */
.wcard {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}
.wcard-title {
    font-size: 0.78rem; font-weight: 700;
    color: #1a2540; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #eef4ff;
}

/* ── Streamlit input overrides for light theme ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    font-size: 0.8rem !important; font-weight: 600 !important;
    color: #4a5568 !important;
}
div[data-testid="stNumberInput"] input {
    background: #f8faff !important;
    border: 1.5px solid #dce8f8 !important;
    border-radius: 9px !important;
    color: #1a2540 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
div[data-baseweb="select"] > div {
    background: #f8faff !important;
    border: 1.5px solid #dce8f8 !important;
    border-radius: 9px !important;
}
div[data-baseweb="select"] span {
    color: #1a2540 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a6fd4, #0ea5a0) !important;
    color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.8rem 2rem !important; width: 100% !important;
    letter-spacing: 0.03em;
    box-shadow: 0 4px 16px rgba(26,111,212,0.3) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(26,111,212,0.4) !important;
}

/* ── Risk result ── */
.result-box {
    border-radius: 16px; padding: 1.8rem;
    text-align: center;
}
.result-high  { background: linear-gradient(135deg,#fff5f5,#ffe4e4); border: 2px solid #ffb3b3; }
.result-moderate { background: linear-gradient(135deg,#fffbf0,#fff3cc); border: 2px solid #ffd966; }
.result-low   { background: linear-gradient(135deg,#f0fff8,#d4f7ea); border: 2px solid #6eddb8; }
.risk-badge {
    display: inline-block; border-radius: 30px; padding: 0.3rem 1rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.6rem;
}
.badge-high     { background:#ffb3b3; color:#8b0000; }
.badge-moderate { background:#ffd966; color:#7a5000; }
.badge-low      { background:#6eddb8; color:#00593a; }
.risk-num {
    font-size: 3rem; font-weight: 800; line-height: 1; margin-bottom: 0.3rem;
}
.risk-num-high     { color: #c0392b; }
.risk-num-moderate { color: #d4860a; }
.risk-num-low      { color: #0a7a4e; }
.risk-conf { font-size: 0.82rem; color: #6b7280; }

/* ── Prob bars ── */
.prob-row { display:flex; align-items:center; gap:0.7rem; margin-bottom:0.55rem; }
.prob-lbl { font-size:0.75rem; font-weight:600; color:#4a5568; width:68px; }
.prob-track { flex:1; height:10px; background:#f0f4ff; border-radius:5px; overflow:hidden; }
.prob-fill  { height:100%; border-radius:5px; }
.prob-pct   { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b7280; width:38px; text-align:right; }

/* ── Pill tags ── */
.pill { display:inline-block; border-radius:20px; padding:0.22rem 0.65rem;
        font-size:0.72rem; font-family:'JetBrains Mono',monospace; margin:0.18rem; }
.pill-r { background:#fff0f0; border:1px solid #ffb3b3; color:#c0392b; }
.pill-g { background:#f0fff8; border:1px solid #6eddb8; color:#0a7a4e; }

/* ── SHAP badge ── */
.shap-badge {
    display:inline-block; background:#eef4ff; border:1px solid #c4d9f8;
    border-radius:6px; padding:0.15rem 0.5rem;
    font-family:'JetBrains Mono',monospace; font-size:0.62rem;
    color:#1a6fd4; letter-spacing:0.08em; margin-left:0.5rem;
}

/* ── Disclaimer ── */
.disclaimer {
    background:#fffbf0; border:1px solid #ffd966; border-radius:12px;
    padding:1rem 1.2rem; font-size:0.78rem; color:#6b7280; line-height:1.6;
    margin-top:2rem;
}
.disclaimer strong { color:#d4860a; }

/* ── Clinical card ── */
.clin-card {
    background:#f8faff; border-left:4px solid #1a6fd4;
    border-radius:0 12px 12px 0; padding:1.2rem 1.4rem;
}
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

FEATURE_COLS = [
    'Age', 'Gender', 'Family_History', 'Smoking_History', 'Alcohol_Consumption',
    'Diabetes', 'Inflammatory_Bowel_Disease', 'Genetic_Mutation',
    'Obesity_Risk_Level', 'Diet_Risk_Level', 'Physical_Inactivity_Risk',
    'Screening_History_Irregular', 'Screening_History_Never', 'Screening_History_Regular',
    'Genetic_Age_Interaction', 'Medical_Comorbidity_Score', 'Lifestyle_Index'
]
FEATURE_LABELS = {
    'Age': 'Age', 'Gender': 'Gender', 'Family_History': 'Family History',
    'Smoking_History': 'Smoking', 'Alcohol_Consumption': 'Alcohol Use',
    'Diabetes': 'Diabetes', 'Inflammatory_Bowel_Disease': 'IBD',
    'Genetic_Mutation': 'Genetic Mutation', 'Obesity_Risk_Level': 'Obesity Level',
    'Diet_Risk_Level': 'Diet Risk', 'Physical_Inactivity_Risk': 'Physical Inactivity',
    'Screening_History_Irregular': 'Irregular Screening',
    'Screening_History_Never': 'Never Screened',
    'Screening_History_Regular': 'Regular Screening',
    'Genetic_Age_Interaction': 'Genetic × Age',
    'Medical_Comorbidity_Score': 'Comorbidity Score',
    'Lifestyle_Index': 'Lifestyle Index'
}

# ─────────────────────────────────────────────
#  TOP BAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">🔬 <span>CRC</span>&nbsp;Risk Predictor</div>
  <div class="topbar-tag">XGBoost · SHAP · v1.0</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO  (with inline SVG colon illustration)
# ─────────────────────────────────────────────
COLON_SVG = """
<svg width="210" height="190" viewBox="0 0 210 190" xmlns="http://www.w3.org/2000/svg">
  <!-- Glow background circle -->
  <circle cx="105" cy="95" r="85" fill="rgba(255,255,255,0.07)"/>

  <!-- Large intestine / colon shape – simplified cartoon -->
  <!-- Ascending colon (right) -->
  <path d="M130 160 Q145 155 150 130 Q155 105 148 80 Q142 60 128 52"
        stroke="#7ee8df" stroke-width="18" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <!-- Transverse colon (top) -->
  <path d="M128 52 Q105 38 82 52"
        stroke="#7ee8df" stroke-width="18" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <!-- Descending colon (left) -->
  <path d="M82 52 Q68 60 62 80 Q55 105 60 130 Q65 155 80 160"
        stroke="#7ee8df" stroke-width="18" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <!-- Sigmoid / bottom -->
  <path d="M80 160 Q90 172 105 168 Q120 165 130 160"
        stroke="#7ee8df" stroke-width="18" fill="none" stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Haustra folds (decorative bumps) -->
  <path d="M130 160 Q145 155 150 130 Q155 105 148 80 Q142 60 128 52"
        stroke="rgba(255,255,255,0.18)" stroke-width="5" fill="none"
        stroke-dasharray="1 18" stroke-linecap="round"/>
  <path d="M82 52 Q68 60 62 80 Q55 105 60 130 Q65 155 80 160"
        stroke="rgba(255,255,255,0.18)" stroke-width="5" fill="none"
        stroke-dasharray="1 18" stroke-linecap="round"/>

  <!-- Small intestine squiggle (center) -->
  <path d="M100 85 Q112 75 108 90 Q104 105 116 100 Q126 97 118 112 Q110 125 100 118 Q88 112 96 100"
        stroke="rgba(255,255,255,0.35)" stroke-width="6" fill="none" stroke-linecap="round"/>

  <!-- Polyp highlight dot -->
  <circle cx="148" cy="108" r="7" fill="#ff8c8c" opacity="0.9"/>
  <circle cx="148" cy="108" r="4" fill="#ff4f4f"/>

  <!-- Label -->
  <text x="158" y="112" fill="#ffb3b3" font-size="9" font-family="sans-serif" font-weight="600">polyp</text>
  <line x1="155" y1="108" x2="162" y2="108" stroke="#ff8c8c" stroke-width="1"/>

  <!-- Decorative dots -->
  <circle cx="55" cy="62" r="3" fill="rgba(126,232,223,0.4)"/>
  <circle cx="160" cy="150" r="4" fill="rgba(126,232,223,0.3)"/>
  <circle cx="105" cy="35" r="2.5" fill="rgba(255,255,255,0.3)"/>
</svg>
"""

st.markdown(f"""
<div class="hero">
  <div class="hero-text">
    <div class="hero-eyebrow">🔬 Clinical Decision Support · Academic Research Tool</div>
    <h1>AI-Based Predictive Model for <em>Early Detection</em> of Colorectal Cancer Risk Factors</h1>
    <p>Powered by XGBoost machine learning with SHAP explainability — assess individual CRC risk from clinical, genetic, and lifestyle parameters.</p>
  </div>
  <div class="hero-art">{COLON_SVG}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INPUT FORM  — using native Streamlit widgets
#  (no HTML div wrapping widgets — that breaks layout)
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">Patient Data Input</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown('<div class="wcard"><div class="wcard-title">🧍 Demographics</div>', unsafe_allow_html=True)
    age       = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
    gender    = st.selectbox("Gender", ["Male", "Female"])
    screening = st.selectbox("Screening History", ["Regular", "Irregular", "Never"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="wcard"><div class="wcard-title">🧬 Clinical & Genetic</div>', unsafe_allow_html=True)
    family_history   = st.selectbox("Family History of CRC", ["No", "Yes"])
    genetic_mutation = st.selectbox("Genetic Mutation (MLH1/BRCA)", ["No", "Yes"])
    diabetes         = st.selectbox("Diabetes", ["No", "Yes"])
    ibd              = st.selectbox("Inflammatory Bowel Disease (IBD)", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_c:
    st.markdown('<div class="wcard"><div class="wcard-title">🥗 Lifestyle Factors</div>', unsafe_allow_html=True)
    smoking   = st.selectbox("Smoking History", ["No", "Yes"])
    alcohol   = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    obesity   = st.selectbox("Obesity Level", ["Normal", "Overweight", "Obese"])
    diet_risk = st.selectbox("Diet Risk Level", ["Low", "Moderate", "High"])
    activity  = st.selectbox("Physical Activity Level", ["High", "Moderate", "Low"])
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍  Analyse CRC Risk Profile", use_container_width=True)

# ─────────────────────────────────────────────
#  PREDICTION + SHAP
# ─────────────────────────────────────────────
if predict_btn:
    gen_val  = 1 if gender == "Male" else 0
    fam_val  = 1 if family_history == "Yes" else 0
    smok_val = 1 if smoking == "Yes" else 0
    alc_val  = 1 if alcohol == "Yes" else 0
    diab_val = 1 if diabetes == "Yes" else 0
    ibd_val  = 1 if ibd == "Yes" else 0
    gene_val = 1 if genetic_mutation == "Yes" else 0

    obesity_map  = {"Normal": 0, "Overweight": 1, "Obese": 2}
    diet_map     = {"Low": 0, "Moderate": 1, "High": 2}
    activity_map = {"High": 0, "Moderate": 1, "Low": 2}

    genetic_age_interaction = gene_val * age
    medical_score  = ibd_val + diab_val
    lifestyle_idx  = smok_val + alc_val + diet_map[diet_risk] + obesity_map[obesity] + activity_map[activity]

    input_df = pd.DataFrame([{
        'Age': age, 'Gender': gen_val, 'Family_History': fam_val,
        'Smoking_History': smok_val, 'Alcohol_Consumption': alc_val,
        'Diabetes': diab_val, 'Inflammatory_Bowel_Disease': ibd_val,
        'Genetic_Mutation': gene_val, 'Obesity_Risk_Level': obesity_map[obesity],
        'Diet_Risk_Level': diet_map[diet_risk], 'Physical_Inactivity_Risk': activity_map[activity],
        'Screening_History_Irregular': 1 if screening == "Irregular" else 0,
        'Screening_History_Never':     1 if screening == "Never" else 0,
        'Screening_History_Regular':   1 if screening == "Regular" else 0,
        'Genetic_Age_Interaction': genetic_age_interaction,
        'Medical_Comorbidity_Score': medical_score,
        'Lifestyle_Index': lifestyle_idx
    }])[FEATURE_COLS]

    prediction    = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = {0: "Low", 1: "Moderate", 2: "High"}
    result  = classes[int(prediction)]

    # ── SHAP ──
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        pred_class  = int(prediction)
        if isinstance(shap_values, list):
            sv = np.array(shap_values[pred_class]).flatten()
        else:
            sv_arr = np.array(shap_values)
            if sv_arr.ndim == 3:
                sv = sv_arr[0, :, pred_class]
            elif sv_arr.ndim == 2:
                sv = sv_arr[0]
            else:
                sv = sv_arr.flatten()
        sv = np.array(sv).flatten()
        if sv.shape[0] != len(FEATURE_COLS):
            raise ValueError(f"Shape mismatch: {sv.shape[0]} vs {len(FEATURE_COLS)}")
        shap_ok = True
    except Exception as e:
        shap_ok  = False
        shap_err = str(e)

    st.divider()
    st.markdown('<div class="sec-label">Risk Assessment Result</div>', unsafe_allow_html=True)

    # ── Result card + prob bars ──
    r1, r2 = st.columns([1, 1.4])

    with r1:
        css   = result.lower()
        icon  = "🔴" if result == "High" else ("🟡" if result == "Moderate" else "🟢")
        conf  = np.max(probabilities) * 100
        st.markdown(f"""
        <div class="result-box result-{css}">
          <div class="risk-badge badge-{css}">Predicted Risk Level</div><br>
          <div class="risk-num risk-num-{css}">{icon} {result}</div>
          <div class="risk-conf">Model confidence: <strong>{conf:.1f}%</strong></div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="wcard">', unsafe_allow_html=True)
        st.markdown("**Class Probability Distribution**")
        clr = {"Low": "#0a7a4e", "Moderate": "#d4860a", "High": "#c0392b"}
        bg  = {"Low": "#6eddb8", "Moderate": "#ffd966", "High": "#ffb3b3"}
        for label, prob in zip(["Low", "Moderate", "High"], probabilities):
            pct = prob * 100
            st.markdown(f"""
            <div class="prob-row">
              <span class="prob-lbl">{label}</span>
              <div class="prob-track">
                <div class="prob-fill" style="width:{pct:.1f}%;background:{bg[label]};"></div>
              </div>
              <span class="prob-pct" style="color:{clr[label]}">{pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SHAP Chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sec-label">XAI Explanation <span class="shap-badge">SHAP · SHapley Additive exPlanations</span></div>', unsafe_allow_html=True)

    if shap_ok:
        feat_labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
        shap_s = pd.Series(sv, index=feat_labels)
        shap_sorted = shap_s.reindex(shap_s.abs().sort_values(ascending=True).index).iloc[-12:]

        fig, ax = plt.subplots(figsize=(10, 5.2))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8faff')

        bar_clrs = ['#e74c3c' if v > 0 else '#0a7a4e' for v in shap_sorted.values]
        bars = ax.barh(shap_sorted.index, shap_sorted.values,
                       color=bar_clrs, height=0.58, edgecolor='none')

        for bar, val in zip(bars, shap_sorted.values):
            x = val + (0.002 if val >= 0 else -0.002)
            ha = 'left' if val >= 0 else 'right'
            ax.text(x, bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', va='center', ha=ha,
                    fontsize=8, color='#1a2540', fontfamily='monospace')

        ax.axvline(0, color=(0.2, 0.2, 0.2, 0.2), linewidth=0.9, linestyle='--')
        ax.set_xlabel('SHAP Value  (impact on predicted risk class)',
                      color='#6b7280', fontsize=9, labelpad=8)
        ax.tick_params(axis='x', colors='#6b7280', labelsize=8)
        ax.tick_params(axis='y', colors='#1a2540', labelsize=9)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color((0.8, 0.8, 0.9, 1.0))
        ax.spines['left'].set_color((0.8, 0.8, 0.9, 1.0))

        p1 = mpatches.Patch(color='#e74c3c', label='↑ Increases risk')
        p2 = mpatches.Patch(color='#0a7a4e', label='↓ Decreases risk')
        ax.legend(handles=[p1, p2], loc='lower right', framealpha=0.8,
                  labelcolor='#1a2540', fontsize=8)
        ax.set_title(f'Feature Contributions → Predicted: {result} Risk',
                     color='#1a2540', fontsize=11, fontweight='700', pad=12, loc='left')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Pills ──
        st.markdown("<br>", unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        risk_f = shap_s[shap_s > 0].sort_values(ascending=False)
        prot_f = shap_s[shap_s < 0].sort_values(ascending=True)

        with p1:
            st.markdown('<div class="wcard">', unsafe_allow_html=True)
            st.markdown("**🔴 Risk-Increasing Factors**")
            if len(risk_f):
                pills = "".join([f'<span class="pill pill-r">{f}  +{v:.3f}</span>'
                                 for f, v in risk_f.head(6).items()])
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown(f"<br><small style='color:#6b7280'>These push the model toward <strong style='color:#c0392b'>{result}</strong> risk.</small>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<small style='color:#6b7280'>None detected.</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with p2:
            st.markdown('<div class="wcard">', unsafe_allow_html=True)
            st.markdown("**🟢 Risk-Protective Factors**")
            if len(prot_f):
                pills = "".join([f'<span class="pill pill-g">{f}  {v:.3f}</span>'
                                 for f, v in prot_f.head(6).items()])
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown("<br><small style='color:#6b7280'>These lower the predicted risk level.</small>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<small style='color:#6b7280'>None detected.</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning(f"SHAP could not be generated. Error: {shap_err}")

    # ── Clinical Interpretation ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Clinical Interpretation</div>', unsafe_allow_html=True)

    interp = {
        "High": {
            "summary": "This profile is associated with <strong>HIGH</strong> colorectal cancer risk.",
            "actions": [
                "Immediate referral for colonoscopy is recommended.",
                "Genetic counselling if mutation markers are present.",
                "Lifestyle intervention: smoking cessation, diet, weight management.",
                "Increase screening frequency to annual or biennial.",
            ]
        },
        "Moderate": {
            "summary": "This profile is associated with <strong>MODERATE</strong> colorectal cancer risk.",
            "actions": [
                "Schedule colonoscopy within 1–3 years.",
                "Lifestyle modifications: diet, exercise, alcohol reduction.",
                "Monitor comorbidities (diabetes, IBD) closely.",
                "Educate patient on early warning symptoms.",
            ]
        },
        "Low": {
            "summary": "This profile is associated with <strong>LOW</strong> colorectal cancer risk.",
            "actions": [
                "Continue regular screening per national guidelines.",
                "Maintain healthy lifestyle: balanced diet and physical activity.",
                "Reassess annually or if new risk factors emerge.",
                "Patient education on CRC prevention.",
            ]
        }
    }
    info = interp[result]
    items = "".join([f"<li style='margin-bottom:0.4rem;color:#4a5568'>{a}</li>" for a in info['actions']])
    st.markdown(f"""
    <div class="clin-card">
      <p style="font-size:0.92rem;margin-bottom:0.9rem;color:#1a2540">{info['summary']}</p>
      <p style="font-size:0.7rem;font-family:'JetBrains Mono',monospace;color:#1a6fd4;
         letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">Recommended Actions</p>
      <ul style="padding-left:1.2rem;font-size:0.86rem">{items}</ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      <strong>⚠️ Medical Disclaimer:</strong> This tool is developed for
      <strong>academic and research purposes only</strong>. It is not a substitute for
      professional medical diagnosis, advice, or treatment. All clinical decisions must be
      made by a qualified healthcare professional. SHAP values represent feature contributions
      to the model's prediction and do not imply direct medical causality.
    </div>
    """, unsafe_allow_html=True)
