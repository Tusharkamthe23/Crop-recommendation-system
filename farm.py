import streamlit as st
import numpy as np
import joblib  # Use joblib 
from sklearn.base import BaseEstimator

# Load the pre-trained ML model and scaler using joblib
try:
    model = joblib.load("random_forest_model.joblib")  # Load your model
    scaler = joblib.load("scaler.joblib")  # Load the scaler used during training

    # Check if the loaded model is a valid scikit-learn estimator
    if not isinstance(model, BaseEstimator):
        raise ValueError("Loaded object is not a valid scikit-learn model.")
except Exception as e:
    st.error(f"Error loading the model or scaler: {e}")
    st.stop()

# Create a Streamlit app
st.title("ML Model Prediction")

# Define the form to accept input values
with st.form("prediction_form"):
    
    Nitrogen = st.number_input("Enter Nitrogen:", min_value=0, max_value=1000, value=90)
    Phosphorus = st.number_input("Enter Phosphorus:", min_value=0, max_value=100000, value=42)
    Potassium = st.number_input("Enter Potassium:", min_value=0, max_value=4000, value=43)
    Temperature = st.number_input("Enter Temperature:", min_value=0, max_value=100, value=20)
    Humidity= st.number_input("Enter Humidity:", min_value=0, max_value=40000, value=82)
    Ph = st.number_input("Enter Ph:", min_value=0, max_value=14, value=6)
    Rainfall= st.number_input("Enter Rainfall:", min_value=0, max_value=40000, value=202)

    # Submit button
    submit = st.form_submit_button("Predict")

# When the form is submitted, process the input and make a prediction
if submit:
    # Collect the input features into an array
    input_data = np.array([[Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]])
    
    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    try:
        prediction = model.predict(input_data_scaled)
        st.write(f"Predicted Output: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
