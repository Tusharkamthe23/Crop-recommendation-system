import streamlit as st
import numpy as np
import joblib  
from sklearn.preprocessing import StandardScaler


model = joblib.load("random_forest_model.joblib")

st.title("Crop Recommandation")

with st.form("prediction_form"):
    
    Nitrogen = st.number_input("Enter Nitrogen:", min_value=0, max_value=1000, value=90)
    Phosphorus = st.number_input("Enter  Phosphorus:", min_value=0, max_value=100000, value=42)
    Potassium = st.number_input("Enter Potassium:", min_value=0, max_value=4000, value=43)
    Temperature = st.number_input("Enter Temperature:", min_value=0, max_value=100, value=20)
    Humidity= st.number_input("Enter Humidity:", min_value=0, max_value=40000, value=82)
    Ph = st.number_input("Enter Ph:", min_value=0, max_value=14, value=6)
    Rainfall= st.number_input("Enter Rainfall:", min_value=0, max_value=40000, value=202)

    
    submit = st.form_submit_button("Submit")


if submit:
    
    input_data = np.array([[Nitrogen, Phosphorus,Potassium,Temperature,Humidity,Ph,Rainfall]])

    
    scaler = StandardScaler()  
    input_data_scaled = scaler.fit_transform(input_data)

    prediction = model.predict(input_data_scaled)


    st.success(f"Predicted Output: {prediction[0]}")
