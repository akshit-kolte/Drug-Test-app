import pandas as pd
import numpy as np
import joblib
import streamlit as st
#Load the Model

model=joblib.load(open("svm_rbf_model.joblib", 'rb'))

st.title("Drug Response app")
#Input feature
Drug Dosage (mg)=st.number_input("Drug Dosage (mg)",min_value=0.0)
Systolic Blood Pressure (mmHg)=st.number_input("Systolic Blood Pressure (mmHg)",min_value=0.0)
Heart Rate (BPM)=st.number_input("Heart Rate (BPM)",min_value=0.0)
Liver Toxicity Index (U/L)=st.number_input("Liver Toxicity Index (U/L)",min_value=0.0)
Blood Glucose Level (mg/dL)=st.number_input("Blood Glucose Level (mg/dL)",min_value=0.0)
#Make Pred
if st.button('Drug Response Prediction'):
	input_data=np.array([[Drug Dosage (mg), Systolic Blood Pressure (mmHg), Heart Rate (BPM), Liver Toxicity Index (U/L), Blood Glucose Level (mg/dL)]])
	prediction=model.predict(input_data)[0]

	st.success(f'Predict Diabetes:{prediction:.2f}')
