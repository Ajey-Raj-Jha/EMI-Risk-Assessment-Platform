# app/regression_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import math


# Load Dataset

data_path = "data/cleaned_EMI_dataset.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path, low_memory=False)
print(" Dataset loaded:", df.shape)


# Target Column: max_monthly_emi
target = "max_monthly_emi"
if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' not found. Columns available: {df.columns.tolist()}"
    )


# Use SAME base features as classifier

 
exclude_cols = ["emi_eligibility", "max_monthly_emi"]
base_features = [col for col in df.columns if col not in exclude_cols]

X = df[base_features].copy()
y = df[target]

# 
# Train/Test Split
# 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 
# Train Linear Regression
# 
model = LinearRegression()
model.fit(X_train, y_train)

# 
# Predictions & Metrics
# 
y_pred = model.predict(X_test)

# NOTE: some sklearn versions don't accept squared=False, so compute RMSE manually
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f" Linear Regression Model Trained | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# 
# Save Model + Training Columns

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/linear_regression.pkl")
joblib.dump(list(X.columns), "models/train_columns_reg.pkl")

print(" Linear Regression model saved successfully.")



