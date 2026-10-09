"""Train the CTR model.

Run:  python main.py

What changed compared with the first version
- The target is now the real CTR (clicks / impressions), not the raw click count.
- Only inputs known before an ad runs are used (no leakage).
- The model is a fast gradient-boosting regressor (trains in seconds, loads fast).
- The train/test split is by *campaign*, so the test score reflects campaigns
  the model has never seen.
- Dataset, model and metrics are fingerprinted with SHA-256 (provenance.json),
  ready to be registered on the blockchain.
"""
import json
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import ctr_core as core
from ledger import canonical_json, hash_file, sha256_hex

warnings.filterwarnings("ignore")
SEED = 42


def build_pipeline() -> Pipeline:
    encoder = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                core.CATEGORICAL,
            ),
            ("num", "passthrough", core.NUMERIC),
        ]
    )
    n_cat = len(core.CATEGORICAL)
    is_categorical = [True] * n_cat + [False] * len(core.NUMERIC)
    model = HistGradientBoostingRegressor(
        categorical_features=is_categorical,
        max_iter=300,
        learning_rate=0.05,
        random_state=SEED,
    )
    return Pipeline([("encode", encoder), ("model", model)])


def main() -> None:
    t0 = time.time()
    raw = core.load_raw()
    print(f"Loaded {raw.shape[0]:,} rows, {raw.shape[1]} columns")
    df = core.build_training_frame(raw)
    print(f"Usable rows after cleaning: {len(df):,}  (campaigns: {df['campaign_item_id'].nunique()})")

    X, y, groups = df[core.FEATURES], df["ctr"], df["campaign_item_id"]

    # 1) honest evaluation: hold out whole campaigns
    train_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y, groups))
    pipe = build_pipeline().fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = pipe.predict(X.iloc[test_idx])
    y_test = y.iloc[test_idx]
    baseline = np.full(len(test_idx), y.iloc[train_idx].mean())
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, pred)), 6),
        "baseline_mae": round(float(mean_absolute_error(y_test, baseline)), 6),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "test_note": "test set contains only campaigns not seen in training",
    }
    print(f"Held-out MAE {metrics['mae']:.5f} (always-predict-average: {metrics['baseline_mae']:.5f}), R2 {metrics['r2']:.3f}")

    # 2) final model uses all data
    final = build_pipeline().fit(X, y)
    core.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, core.MODEL_PATH, compress=3)

    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sklearn_version": sklearn.__version__,
        "features": core.FEATURES,
        "categorical": core.CATEGORICAL,
        "numeric": core.NUMERIC,
        "categories": {c: sorted(df[c].unique().tolist()) for c in core.CATEGORICAL},
        "defaults": {
            **{c: df[c].mode().iloc[0] for c in core.CATEGORICAL},
            "campaign_day": int(df["campaign_day"].median()),
            "campaign_budget_usd": round(float(df["campaign_budget_usd"].median()), 2),
        },
        "numeric_ranges": {
            "campaign_day": [int(df["campaign_day"].min()), int(df["campaign_day"].max())],
            "campaign_budget_usd": [0.0, round(float(df["campaign_budget_usd"].max()), 2)],
        },
        "average_ctr": round(float(y.mean()), 6),
        "metrics": metrics,
    }
    with open(core.META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    # 3) provenance: fingerprints that get registered on-chain
    provenance = {
        "version": time.strftime("v%Y%m%d-%H%M%S"),
        "dataset_hash": hash_file(core.DATA_PATH),
        "model_hash": hash_file(core.MODEL_PATH),
        "metrics_hash": sha256_hex(canonical_json(metrics)),
    }
    with open(core.PROVENANCE_PATH, "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"Saved {core.MODEL_PATH.name} ({core.MODEL_PATH.stat().st_size / 1e6:.2f} MB)")
    print("Provenance:", json.dumps(provenance, indent=2))
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
