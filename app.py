import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb

# Set Page Config for the Blue Theme
st.set_page_config(page_title="CRC Risk Predictor", layout="centered")

# Custom CSS for the Blue Theme
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #004a99;
        color: white;
        border-radius: 10px;
        width: 100%;
    }
    h1, h2, h3 {
        color: #004a99;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the saved model
@st.cache_resource
def load_model():
    # Make sure this filename matches exactly what you uploaded to GitHub
    with open('crc_xgboost_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'crc_xgboost_model.pkl' is uploaded to GitHub.")
    st.stop()

st.title("AI-Based CRC Risk Factor Detection")
st.write("Enter patient data below to predict Colorectal Cancer risk levels using the XGBoost model.")

# Inputs
st.header("Patient Data Input")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    family_history = st.selectbox("Family History of CRC", options=["No", "Yes"])
    smoking = st.selectbox("Smoking History", options=["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", options=["No", "Yes"])

with col2:
    diabetes = st.selectbox("Diabetes Status", options=["No", "Yes"])
    ibd = st.selectbox("Inflammatory Bowel Disease", options=["No", "Yes"])
    genetic_mutation = st.selectbox("Genetic Mutation", options=["No", "Yes"])
    obesity = st.selectbox("Obesity Level", options=["Normal", "Overweight", "Obese"])
    diet_risk = st.selectbox("Diet Risk Level", options=["Low", "Moderate", "High"])
    activity = st.selectbox("Physical Activity", options=["High", "Moderate", "Low"])

screening = st.selectbox("Screening History", options=["Regular", "Irregular", "Never"])

# Preprocessing logic
if st.button("Predict Risk Level"):
    # Encoding
    gen_val = 1 if gender == "Male" else 0
    fam_val = 1 if family_history == "Yes" else 0
    smok_val = 1 if smoking == "Yes" else 0
    alc_val = 1 if alcohol == "Yes" else 0
    diab_val = 1 if diabetes == "Yes" else 0
    ibd_val = 1 if ibd == "Yes" else 0
    gene_val = 1 if genetic_mutation == "Yes" else 0
    
    obesity_map = {"Normal": 0, "Overweight": 1, "Obese": 2}
    diet_map = {"Low": 0, "Moderate": 1, "High": 2}
    activity_map = {"High": 0, "Moderate": 1, "Low": 2}

    # Feature Engineering
    genetic_age_interaction = gene_val * age
    medical_score = ibd_val + diab_val
    lifestyle_idx = smok_val + alc_val + diet_map[diet_risk] + obesity_map[obesity] + activity_map[activity]

    # Build DataFrame
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
    }])

    # Ensure column order matches training
    # (The order above matches your "X_train.columns" from the notebook)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    classes = {0: "Low", 1: "Medium", 2: "High"}
    result = classes[prediction]

    st.markdown("---")
    st.subheader(f"Results")
    
    if result == "High":
        st.error(f"Predicted Risk: {result}")
    elif result == "Medium":
        st.warning(f"Predicted Risk: {result}")
    else:
        st.success(f"Predicted Risk: {result}")
        
    st.write(f"Confidence: {np.max(probabilities)*100:.2f}%")

st.info("Note: This app is for research purposes and does not replace professional medical advice.")
