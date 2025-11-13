# Streamlit App
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

MODEL_PATH = Path("model_output//ctr_model.pkl")
FEATURE_PATH = Path("model_output//feature_info.json")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH) as f:
        feats = json.load(f)
    return model, feats

st.set_page_config(page_title="CTR Predictor", layout="centered")
st.title("CTR Predictor App")

model, feats = load_model()

st.write("Enter the input features below:")

with st.form("predict_form"):
    inputs = {}
    for f in feats:
        if f["type"] == "numeric":
            inputs[f["name"]] = st.number_input(f["name"], value=0.0)
        else:
            opts = f.get("sample_values", [])
            if opts:
                inputs[f["name"]] = st.selectbox(f["name"], ["missing"] + opts)
            else:
                inputs[f["name"]] = st.text_input(f["name"])
    submit = st.form_submit_button("Predict CTR")

if submit:
    input_df = pd.DataFrame([inputs])
    try:
        pred_prob = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]
        st.metric("Predicted Click Probability", f"{pred_prob:.4f}")
        st.write("Predicted Class:", int(pred))
    except Exception as e:
        st.error(f"Prediction failed: {e}")




