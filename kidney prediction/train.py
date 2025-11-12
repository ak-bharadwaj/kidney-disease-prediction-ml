# ==============================
# CKD Prediction Training Script
# ==============================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ---------------------- Load Dataset ----------------------
df = pd.read_csv("ckd.csv")
print("Data shape:", df.shape)
print("Columns:", df.columns)

# ---------------------- Clean Target ----------------------
df["classification"] = df["classification"].str.strip().replace(
    {"ckd": 1, "notckd": 0, "CKD":1, "NOTCKD":0}
)
df = df.dropna(subset=["classification"])
df["classification"] = df["classification"].astype(int)

# ---------------------- Clean Numeric Columns ----------------------
df.replace(['?', '\t?'], np.nan, inplace=True)

numeric_cols = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod",
    "pot", "hemo", "pcv", "wc", "rc"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric values with median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ---------------------- Feature Engineering ----------------------
selected_raw = numeric_cols.copy()

def add_engineered_features(df):
    df["kidney_function_index"] = np.log(df["sc"] * df["bu"] + 1)
    df["hydration_index"] = df["sod"] / (df["pot"] + 1e-6)
    df["anemia_risk"] = (15 - df["hemo"]) / (df["age"] + 1e-6)
    df["urine_protein_index"] = df["al"] * df["sg"]
    df["bp_age_ratio"] = df["bp"] / (df["age"] + 1e-6)
    return df

X = df[selected_raw].copy()
X = add_engineered_features(X)
y = df["classification"]

selected_features = X.columns.tolist()
print("Final selected features:", selected_features)

# ---------------------- Train-Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------- Scaling ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- Ensemble Model ----------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss'
)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    voting="soft"
)

ensemble.fit(X_train_scaled, y_train)

# ---------------------- Evaluate ----------------------
y_pred = ensemble.predict(X_test_scaled)
y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

print("\n--- Ensemble (RF + XGB) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# ---------------------- Save ----------------------
joblib.dump(ensemble, "ensemble_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features, "selected_features.pkl")
print("\n✅ Model, scaler, and selected_features saved!")
