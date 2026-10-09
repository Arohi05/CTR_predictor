"""Shared configuration and helpers for the CTR predictor.

Everything that has to be identical between training (main.py) and serving
(app.py) lives here, so the two can never drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset.csv"
MODEL_DIR = BASE_DIR / "model_output"
MODEL_PATH = MODEL_DIR / "ctr_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
PROVENANCE_PATH = MODEL_DIR / "provenance.json"

# Only information that is known BEFORE an ad runs may be used as an input.
# `impressions`, `clicks`, `media_cost_usd` and the reach columns are results of
# the campaign, so using them would be data leakage.
CATEGORICAL = [
    "ext_service_name",
    "channel_name",
    "advertiser_name",
    "advertiser_currency",
    "search_tag_cat",
    "creative_size",
    "template_id",
    "timezone",
    "weekday_cat",
]
NUMERIC = ["campaign_day", "campaign_budget_usd"]
FEATURES = CATEGORICAL + NUMERIC

# CTR above 20 % in this data is a reporting glitch (impressions < clicks).
MAX_VALID_CTR = 0.20


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def build_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw campaign-day rows into (features, ctr, campaign id)."""
    df = raw.dropna(subset=["clicks", "impressions"]).copy()
    df = df[df["impressions"] > 0]
    df["ctr"] = df["clicks"] / df["impressions"]
    df = df[df["ctr"] <= MAX_VALID_CTR]

    df["campaign_day"] = df["no_of_days"]  # day number inside the campaign
    w = df["creative_width"].fillna(-1).astype(int).astype(str)
    h = df["creative_height"].fillna(-1).astype(int).astype(str)
    df["creative_size"] = (w + "x" + h).replace({"-1x-1": "unknown", "0x0": "none"})
    df["template_id"] = df["template_id"].fillna(-1).astype(int).astype(str)

    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    return df[FEATURES + ["ctr", "campaign_item_id"]].reset_index(drop=True)


def load_meta() -> dict:
    with open(META_PATH) as f:
        return json.load(f)


def make_input_row(values: dict) -> pd.DataFrame:
    """One-row DataFrame in the exact column order the model expects."""
    row = {c: str(values[c]) for c in CATEGORICAL}
    row.update({c: float(values[c]) for c in NUMERIC})
    return pd.DataFrame([row], columns=FEATURES)


def expected_clicks(ctr: float, impressions: float) -> float:
    return float(np.round(ctr * impressions, 1))
