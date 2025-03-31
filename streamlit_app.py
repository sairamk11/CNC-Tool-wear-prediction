import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler
import base64

def add_bg_from_local(image_path, blur_strength=5):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            filter: blur({blur_strength}px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("projectcncbackgroundimage.webp", blur_strength=8)


# Load the trained LSTM model
@st.cache_resource
def load_lstm_model():
    return load_model('lstm_model.h5')

# Load the scaler
@st.cache_resource
def load_scaler():
    return joblib.load('scaler (2).pkl')

# Load label encoders
@st.cache_resource
def load_label_encoders():
    label_encoders = {}
    categorical_columns = ['tool_condition', 'machining_finalized', 'passed_visual_inspection']
    for column in categorical_columns:
        with open(f'label_encoder_{column}.pkl', 'rb') as file:
            label_encoders[column] = pickle.load(file)
    return label_encoders

# Load resources
model = load_lstm_model()
scaler = load_scaler()
label_encoders = load_label_encoders()

# Get feature names used during training
expected_features = scaler.feature_names_in_

def preprocess_data(uploaded_file):
    """Preprocess uploaded CSV data."""
    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    
    # Ensure all required features are present
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value
    
    # Reorder columns to match training set
    df = df[expected_features]
    
    # Scale the data
    X_scaled = scaler.transform(df)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    return X_reshaped, df

def make_predictions(X):
    """Make predictions using the trained LSTM model."""
    predictions = model.predict(X)
    decoded_predictions = {}
    
    # Decode predictions
    for i, column in enumerate(label_encoders.keys()):
        predicted_labels = np.argmax(predictions[i], axis=1)
        decoded_predictions[column] = label_encoders[column].inverse_transform(predicted_labels)
    
    return pd.DataFrame(decoded_predictions)

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center; color: black; font-size: 60px;'>🔧 Tool Wear Prediction App</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    """
    <h2 style='font-size: 28px; color: brown'>📌 Overview</h2>
    <p style='font-size: 24px;color: green'>
    This project focuses on analyzing <b>CNC milling machine performance</b> and detecting faults using <b>deep learning techniques</b>. 
    The primary objective is to predict:
    </p>
    <ul style='font-size: 23px;color: brown'>
        <li>🛠 <b>Tool Condition</b> (Unworn/Worn)</li>
        <li>⚙️ <b>Machining Finalization</b> (Yes/No)</li>
        <li>🔍 <b>Passed Visual Inspection</b> (Yes/No)</li>
    </ul>
    <p style='font-size: 20px;color: green'>
    Using <b>LSTM-based deep learning models</b>, the project processes <b>sensor data</b> collected from CNC milling experiments. 
    The web app is built with <b>Streamlit</b> for an interactive experience.
    </p>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    "<p style='font-size: 22px; font-weight: bold;color: brown'>📤 Upload your CSV file to predict tool wear conditions.</p>", 
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    X, original_data = preprocess_data(uploaded_file)
    predictions_df = make_predictions(X)
    
    # Display results
    st.write("### Predictions")
    st.dataframe(predictions_df)
    
    # Combine original data with predictions
    final_results = pd.concat([original_data, predictions_df], axis=1)
    st.write("### Full Data with Predictions")
    st.dataframe(final_results)
    
    # Option to download predictions
    csv = final_results.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
