import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ------------------------------
# Load model + feature config
# ------------------------------
MODEL_PATH = Path("model_output/ctr_model.pkl")
FEATURE_PATH = Path("model_output/feature_info.json")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_features():
    with open(FEATURE_PATH, "r") as f:
        data = json.load(f)
    return data["selected_features"]

pipe = load_model()
selected_features = load_features()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("CTR Probability Calculator")
st.write("Enter feature values to calculate estimated click-through probability.")

# Create input widgets
user_input = {}

st.subheader("Input Features")
for feature in selected_features:
    # Treat everything as numeric input
    user_input[feature] = st.number_input(
        f"{feature}", 
        value=0.0,
        step=0.1,
        format="%.4f"
    )

# Convert to DataFrame for prediction
input_df = pd.DataFrame([user_input])

# ------------------------------
# Prediction (NO predicted class)
# ------------------------------
if st.button("Calculate Click-Through Probability"):
    try:
        prob = pipe.predict_proba(input_df)[0][1]  # probability of click = class "1"
        st.success(f"Estimated Click-Through Probability: **{prob:.4f}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
