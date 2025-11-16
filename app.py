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

    # Load saved feature names
    with open(FEATURE_PATH, "r") as f:
        feat_info = json.load(f)

    feature_names = feat_info.get("selected_features", [])

    if not isinstance(feature_names, list):
        raise ValueError("'selected_features' must be a list.")

    if len(feature_names) == 0:
        raise ValueError("'selected_features' is empty.")

    return model, feature_names


# Page setup
st.set_page_config(page_title="CTR Predictor", layout="centered")
st.title("CTR Predictor App")

# Load model + features
try:
    model, feature_names = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model or feature info: {e}")
    st.stop()


# Identify numeric vs categorical input groups
numeric_features = [f for f in feature_names if "_" not in f]  # original numeric features
categorical_prefixes = sorted({f.split("_")[0] for f in feature_names if "_" in f})

st.subheader("Enter Input Values")
st.write("Provide input values for each raw feature. The model handles encoding internally.")


with st.form("predict_form"):
    user_input = {}

    # Numeric Inputs
    st.markdown("### 🔢 Numeric Features")
    for feat in numeric_features:
        user_input[feat] = st.number_input(f"{feat}", value=0.0)

    # Categorical Inputs
    st.markdown("### 🔠 Categorical Features")
    for prefix in categorical_prefixes:
        user_input[prefix] = st.text_input(f"{prefix} (category)", value="")

    submit = st.form_submit_button("Predict CTR")


# ---- Prediction ----
if submit:
    input_df = pd.DataFrame([user_input])

    try:
        prob = model.predict_proba(input_df)[0, 1]  # probability of click

        st.success("✅ Prediction Successful!")
        st.metric("Predicted Click Probability", f"{prob:.4f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
