import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model, scalers, and polynomial transformer
poly_model = joblib.load('model.pkl')
scaler_features = joblib.load('scaler_features.pkl')
scaler_target = joblib.load('scaler_target.pkl')
poly_reg = joblib.load('poly_features.pkl')

# Define the feature names based on the scaler's feature names
expected_features = scaler_features.feature_names_in_

# Streamlit App Title
st.title("Glucose Level Prediction App")

# Create input fields for each feature
input_values = []
for feature in expected_features:
    value = st.number_input(f"Enter {feature}:", min_value=0.0, format="%.2f")
    input_values.append(value)

# Predict Button
if st.button('Predict Glucose Level'):
    if len(input_values) != len(expected_features):
        st.error("Input size mismatch. Please ensure all fields are filled.")
    else:
        # Prepare the input for prediction
        input_df = pd.DataFrame([input_values], columns=expected_features)
        st.write("Input DataFrame:", input_df)  # Debugging
        
        try:
            # Scale and transform the input
            input_scaled = scaler_features.transform(input_df)
            input_poly = poly_reg.transform(input_scaled)
            st.write("Transformed Input Shape:", input_poly.shape)  # Debugging
            
            # Predict and inverse transform to get the original glucose level
            output_scaled = poly_model.predict(input_poly)
            st.write("Scaled Prediction:", output_scaled)  # Debugging
            
            output = scaler_target.inverse_transform(output_scaled.reshape(-1, 1))
            st.success(f"Predicted Glucose Level: {output[0][0]:.2f} mg/dL")
        
        except ValueError as e:
            st.error(f"Prediction error: {e}")

