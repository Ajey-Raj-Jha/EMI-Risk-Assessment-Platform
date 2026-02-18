# app/model_training.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.utils import resample

os.makedirs("models", exist_ok=True)
DATA_PATH = "data/cleaned_EMI_dataset.csv"

print("Loading dataset...", DATA_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Dataset shape:", df.shape)

# Ensure the expected one-hot eligibility columns exist
ohe_cols = ["emi_eligibility_High_Risk", "emi_eligibility_Not_Eligible"]
if not all(c in df.columns for c in ohe_cols):
    raise KeyError(f"Missing one-hot target columns. Expected: {ohe_cols}. Found: {df.columns.tolist()}")

# --- TARGET: use emi_eligibility_Not_Eligible as the target
# We'll create a single numerical target column:
df["target_not_eligible"] = df["emi_eligibility_Not_Eligible"].astype(int)  # 1 if Not_Eligible, 0 otherwise

# --- FEATURES: drop the OHE eligibility columns and the regression target (max_monthly_emi)
drop_cols = ohe_cols + ["target_not_eligible", "max_monthly_emi"]
features = [c for c in df.columns if c not in drop_cols]

X = df[features].copy()
y = df["target_not_eligible"].copy()

print("Initial class counts:")
print(y.value_counts())

# --- BALANCE: downsample majority class (simpler, no extra dependency)
df_full = pd.concat([X, y], axis=1)
major_class = df_full[df_full["target_not_eligible"] == df_full["target_not_eligible"].mode()[0]]
minor_class = df_full[df_full["target_not_eligible"] != df_full["target_not_eligible"].mode()[0]]

# Ensure we identify which is majority / minority correctly:
if len(major_class) < len(minor_class):
    majority = minor_class
    minority = major_class
else:
    majority = major_class
    minority = minor_class

major_down = resample(majority,
                      replace=False,
                      n_samples=len(minority),
                      random_state=42)

df_balanced = pd.concat([major_down, minority])
df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle

X_bal = df_balanced.drop(columns=["target_not_eligible"])
y_bal = df_balanced["target_not_eligible"]

print("After balancing class counts:")
print(y_bal.value_counts())

# --- Train / test split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

# --- Train XGBoost Classifier
print("Training XGBoost classifier...")
clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(f"Classifier Accuracy (test): {acc:.4f}")

# --- Save classifier and training columns (important for Streamlit alignment)
joblib.dump(clf, "models/xgb_classifier.pkl")
joblib.dump(list(X.columns), "models/train_columns_class.pkl")
print("Saved: models/xgb_classifier.pkl and models/train_columns_class.pkl")


