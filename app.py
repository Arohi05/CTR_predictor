import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# Paths to the saved model and feature info
MODEL_PATH = Path("model_output/ctr_model.pkl")
FEATURE_PATH = Path("model_output/feature_info.json")

# Load Model + Feature Info with caching
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH) as f:
        feat_info = json.load(f)
    selected_features = feat_info.get("selected_features", [])
    return model, selected_features

st.set_page_config(page_title="CTR Predictor", layout="centered")
st.title("CTR Predictor App")

model, selected_features = load_model()

st.subheader("Enter Feature Values")
st.write("Provide values for each selected important feature used during training:")

with st.form("predict_form"):
    user_input = {}
    for feat in selected_features:
        user_input[feat] = st.number_input(f"{feat}", value=0.0)

    submit = st.form_submit_button("Predict CTR")

if submit:
    input_df = pd.DataFrame([user_input])
    # Reorder columns to match training order
    input_df = input_df.reindex(columns=selected_features, fill_value=0.0)

    try:
        prob = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]

        st.success("✅ Prediction Successful!")
        st.metric("Predicted Click Probability", f"{prob:.4f}")
        st.write("Predicted Class:", int(pred))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
