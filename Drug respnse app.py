import pandas as pd
import numpy as np
import joblib
import streamlit as st
#Load the Model

model=joblib.load(open("svm_rbf_model.joblib", 'rb'))

st.title("Drug Response app")
#Input feature
drug_dosage_mg = st.number_input("Drug Dosage (mg)",min_value=0.0)
systolic_blood_pressure_mmHg = st.number_input("Systolic Blood Pressure (mmHg)",min_value=0.0)
heart_rate_BPM = st.number_input("Heart Rate (BPM)",min_value=0.0)
liver_toxicity_index_U_L = st.number_input("Liver Toxicity Index (U/L)",min_value=0.0)
blood_glucose_level_mg_dL = st.number_input("Blood Glucose Level (mg/dL)",min_value=0.0)
#Make Pred
if st.button('Drug Response Prediction'):
	input_data=np.array([[drug_dosage_mg, systolic_blood_pressure_mmHg, heart_rate_BPM, liver_toxicity_index_U_L, blood_glucose_level_mg_dL]])
	prediction=model.predict(input_data)[0]

	st.success(f'Predict Drug Response:{prediction:.2f}')







