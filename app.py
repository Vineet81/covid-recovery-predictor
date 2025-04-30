# covid_app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('covid_recovery_model.pkl')

st.title("🦠 COVID-19 Recovery Risk Predictor")

st.markdown("""
Enter the patient's details below. The model will predict their COVID-19 recovery risk level.
""")

# User Inputs
age = st.slider("Age", min_value=18, max_value=80, value=30)

vaccination = st.radio("Vaccination Status", ["Not Vaccinated (0)", "Vaccinated (1)"])
vaccination = 1 if "Vaccinated" in vaccination else 0

liver = st.radio("Liver Function", ["Abnormal (0)", "Normal (1)"])
liver = 1 if "Normal" in liver else 0

gfr = st.radio("GFR (Kidney)", ["Abnormal (0)", "Normal (1)"])
gfr = 1 if "Normal" in gfr else 0

immuno = st.radio("Immunoglobulin Levels", ["Abnormal (0)", "Normal (1)"])
immuno = 1 if "Normal" in immuno else 0

spirometry = st.radio("Spirometry", ["Abnormal (0)", "Normal (1)"])
spirometry = 1 if "Normal" in spirometry else 0

t_cell = st.radio("T Cell Count", ["Abnormal (0)", "Normal (1)"])
t_cell = 1 if "Normal" in t_cell else 0

# Predict button
if st.button("Predict Recovery Risk"):
    input_df = pd.DataFrame([[age, vaccination, liver, gfr, immuno, spirometry, t_cell]],
                            columns=["AGE", "VACCINATION", "LIVER", "GFR", "IMMUNOGLOBULIN", "SPIROMETRY", "T_CELL_COUNT"])
    prediction = model.predict(input_df)[0]
    st.success(f"🩺 Predicted Recovery Status: **{prediction}**")
