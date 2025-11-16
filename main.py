import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
import gc
import warnings
warnings.filterwarnings("ignore")

# 1️⃣ Load Dataset
DATA_PATH = "Dataset.csv"
TARGET_COL = "clicks"

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"✅ Data loaded! Shape: {df.shape}")

# Downcast numeric columns
for col in df.select_dtypes(include=['int', 'float']).columns:
    df[col] = pd.to_numeric(df[col], downcast='unsigned' if df[col].min() >= 0 else 'integer')

print(df.info(memory_usage='deep'))

# 2️⃣ Prepare Data
df = df.dropna(subset=[TARGET_COL])
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

del df
gc.collect()

# 3️⃣ Define Columns
num_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print(f"Numeric Features: {len(num_cols)}, Categorical Features: {len(cat_cols)}")

# 4️⃣ Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# 5️⃣ Full model pipeline
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=1,
    ))
])

# 6️⃣ Train/Test Split
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ Train model
pipe.fit(X_train, y_train)
print("\n✅ Model trained successfully!")

# 8️⃣ Extract final transformed feature names
ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"]
ohe_features = list(ohe.get_feature_names_out(cat_cols))
final_features = num_cols + ohe_features

# 9️⃣ Evaluate model
y_pred = pipe.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 🔟 Save model + feature names
output_dir = Path("model_output")
output_dir.mkdir(exist_ok=True)

MODEL_PATH = output_dir / "ctr_model.pkl"
FEATURE_PATH = output_dir / "feature_info.json"

joblib.dump(pipe, MODEL_PATH, compress=3)

with open(FEATURE_PATH, "w") as f:
    json.dump({"selected_features": final_features}, f)

print(f"Model saved to: {MODEL_PATH}")
print(f"Feature info saved to: {FEATURE_PATH}")
