import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
import gc
import warnings
warnings.filterwarnings("ignore")

# 1️⃣ Load Dataset with Downcasting
DATA_PATH = "Dataset.csv"
TARGET_COL = "clicks"
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"✅ Data loaded! Shape: {df.shape}")

# Downcast numeric columns for memory efficiency
for col in df.select_dtypes(include=['int', 'float']).columns:
    df[col] = pd.to_numeric(df[col], downcast='unsigned' if df[col].min() >= 0 else 'integer')

# 2️⃣ Data Analysis (Can comment out on reruns)
print(df.info(memory_usage='deep'))  # See memory usage

# 3️⃣ Prepare Data
df = df.dropna(subset=[TARGET_COL])
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])
del df
gc.collect()

# 4️⃣ Feature Engineering with Efficient Dummy Encoding
num_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print(f"\nNumeric: {len(num_cols)}, Categorical: {len(cat_cols)}")

X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True, sparse=True)
del X
gc.collect()

# 5️⃣ Feature Selection using RandomForest Importance (top 6 features)
rf_sel = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1, max_depth=10, warm_start=True)
rf_sel.fit(X_encoded, y)
feat_importances = pd.Series(rf_sel.feature_importances_, index=X_encoded.columns)
selected_features = feat_importances.nlargest(6).index.tolist()  # Top 6 features
X_selected = X_encoded[selected_features]
del X_encoded
gc.collect()

# 6️⃣ Train/Test Split
X_selected, y = shuffle(X_selected, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
del X_selected, y
gc.collect()

# 7️⃣ Model Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),  # works with sparse data
    ('model', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1))
])
pipe.fit(X_train, y_train)
print("\n✅ Model trained!")

# 8️⃣ Evaluation
y_pred = pipe.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 9️⃣ Save Model + Features
output_dir = Path("model_output")
output_dir.mkdir(parents=True, exist_ok=True)  # Create if not exists

MODEL_PATH = output_dir / "ctr_model.pkl"
FEATURE_PATH = output_dir / "feature_info.json"

joblib.dump(pipe, MODEL_PATH, compress=3)
with open(FEATURE_PATH, "w") as f:
    json.dump({"selected_features": [str(f) for f in selected_features]}, f)

print(f"Model saved to {MODEL_PATH}")
print(f"Feature info saved to {FEATURE_PATH}")