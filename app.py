import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# Paths
MODEL_PATH = Path("model_output/ctr_model.pkl")
FEATURE_PATH = Path("model_output/feature_info.json")


# Load Model + Feature Info with caching
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    # Load feature info safely
    with open(FEATURE_PATH, "r") as f:
        feat_info = json.load(f)

    selected_features = feat_info.get("selected_features", [])

    # Validate feature list
    if not isinstance(selected_features, list):
        raise ValueError("'selected_features' in feature_info.json must be a list.")

    if len(selected_features) == 0:
        raise ValueError("'selected_features' list is empty. Cannot build input form.")

    return model, selected_features


# Page setup
st.set_page_config(page_title="CTR Predictor", layout="centered")
st.title("CTR Predictor App")

# Load model + features
try:
    model, selected_features = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model or feature info: {e}")
    st.stop()


st.subheader("Enter Feature Values")
st.write("Provide values for each important feature used during training:")

# ---- Input Form ----
with st.form("predict_form"):
    user_input = {}

    for feat in selected_features:
        user_input[feat] = st.number_input(f"{feat}", value=0.0)

    submit = st.form_submit_button("Predict CTR")

# ---- Prediction ----
if submit:
    input_df = pd.DataFrame([user_input])

    # Ensure correct column order
    input_df = input_df.reindex(columns=selected_features, fill_value=0.0)

    try:
        prob = model.predict_proba(input_df)[0, 1]  # Probability of click

        st.success("✅ Prediction Successful!")
        st.metric("Predicted Click Probability", f"{prob:.4f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
