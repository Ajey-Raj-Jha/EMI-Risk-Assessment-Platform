import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="EMI Risk Assessment", layout="centered")

st.title("EMI Risk Assessment Platform")

MODEL_DIR = "models"

def safe_load(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        raise FileNotFoundError(path)
    return joblib.load(path)

# Load models
clf = safe_load(f"{MODEL_DIR}/xgb_classifier.pkl")
reg = safe_load(f"{MODEL_DIR}/linear_regression.pkl")

# Load training columns
train_cols_class = safe_load(f"{MODEL_DIR}/train_columns_class.pkl")
train_cols_reg = safe_load(f"{MODEL_DIR}/train_columns_reg.pkl")


# --------------------------------------------------
# MODEL PERFORMANCE METRICS
# --------------------------------------------------

st.header("Model Performance")

st.subheader("Regression Model (Max EMI Prediction)")

col1, col2, col3, col4 = st.columns(4)

col1.metric("RMSE", "3861.29")
col2.metric("MAE", "2771.22")
col3.metric("R² Score", "0.747")
col4.metric("MAPE", "166.41%")

st.subheader("Classification Model (EMI Eligibility)")

col5, col6, col7, col8, col9 = st.columns(5)

col5.metric("Accuracy", "96.85%")
col6.metric("Precision", "98.21%")
col7.metric("Recall", "95.43%")
col8.metric("F1 Score", "96.80%")
col9.metric("ROC-AUC", "0.996")

st.divider()


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.header("Enter Customer Details")

with st.form("form"):
    age = st.number_input("Age", 18, 70, 30)
    monthly_salary = st.number_input("Monthly Salary", 0, 1000000, 50000)
    years_of_employment = st.number_input("Years of Employment", 0, 50, 3)
    monthly_rent = st.number_input("Monthly Rent", 0, 300000, 8000)
    family_size = st.number_input("Family Size", 1, 20, 4)
    dependents = st.number_input("Dependents", 0, 10, 1)
    school_fees = st.number_input("School Fees", 0, 200000, 0)
    college_fees = st.number_input("College Fees", 0, 200000, 0)
    travel_expenses = st.number_input("Travel Expenses", 0, 200000, 1500)
    groceries_utilities = st.number_input("Groceries & Utilities", 0, 200000, 8000)
    other_monthly_expenses = st.number_input("Other Expenses", 0, 200000, 2000)
    existing_loans = st.number_input("Existing Loans", 0, 10000000, 0)
    current_emi_amount = st.number_input("Current EMI", 0, 200000, 5000)
    credit_score = st.number_input("Credit Score", 300, 900, 700)
    bank_balance = st.number_input("Bank Balance", 0, 10000000, 20000)
    emergency_fund = st.number_input("Emergency Fund", 0, 10000000, 10000)
    requested_amount = st.number_input("Requested Loan Amount", 0, 10000000, 200000)
    requested_tenure = st.number_input("Requested Tenure (months)", 6, 120, 24)

    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["High School", "Post Graduate", "Professional"])
    employment_type = st.selectbox("Employment Type", ["Private", "Self-employed"])
    company_type = st.selectbox("Company Type", ["MNC", "Mid-size", "Small", "Startup"])
    house_type = st.selectbox("House Type", ["Own", "Rented"])
    emi_scenario = st.selectbox("EMI Scenario", [
        "Education EMI",
        "Home Appliances EMI",
        "Personal Loan EMI",
        "Vehicle EMI"
    ])

    submit = st.form_submit_button("Predict")


# --------------------------------------------------
# RAW INPUT DATA
# --------------------------------------------------

raw_input = {
    "age": age,
    "monthly_salary": monthly_salary,
    "years_of_employment": years_of_employment,
    "monthly_rent": monthly_rent,
    "family_size": family_size,
    "dependents": dependents,
    "school_fees": school_fees,
    "college_fees": college_fees,
    "travel_expenses": travel_expenses,
    "groceries_utilities": groceries_utilities,
    "other_monthly_expenses": other_monthly_expenses,
    "existing_loans": existing_loans,
    "current_emi_amount": current_emi_amount,
    "credit_score": credit_score,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,
}

input_df = pd.DataFrame([raw_input])


# --------------------------------------------------
# ONE HOT ENCODING
# --------------------------------------------------

def add_onehot(df, prefix, value, possible_values):
    for val in possible_values:
        df[f"{prefix}_{val}"] = 1 if value == val else 0


add_onehot(input_df, "gender", gender, ["Male", "Female"])
add_onehot(input_df, "marital_status", marital_status, ["Single"])
add_onehot(input_df, "education", education, ["High School", "Post Graduate", "Professional"])
add_onehot(input_df, "employment_type", employment_type, ["Private", "Self-employed"])
add_onehot(input_df, "company_type", company_type, ["MNC", "Mid-size", "Small", "Startup"])
add_onehot(input_df, "house_type", house_type, ["Own", "Rented"])
add_onehot(input_df, "emi_scenario", emi_scenario, [
    "Education EMI",
    "Home Appliances EMI",
    "Personal Loan EMI",
    "Vehicle EMI"
])

input_df["emi_eligibility_High_Risk"] = 0
input_df["emi_eligibility_Not_Eligible"] = 0


# --------------------------------------------------
# COLUMN ALIGNMENT
# --------------------------------------------------

def align(df, cols):
    for c in cols:
        if c not in df:
            df[c] = 0
    return df.reindex(cols, axis=1)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submit:

    aligned_reg = align(input_df.copy(), train_cols_reg)
    pred_reg = reg.predict(aligned_reg)[0]

    st.success(f"Estimated EMI Affordable Amount: ₹{pred_reg:,.2f}")

    aligned_class = align(input_df.copy(), train_cols_class)
    pred_raw = clf.predict(aligned_class)[0]

    human_label = "Eligible" if pred_raw == 0 else "High Risk"

    st.success(f"EMI Eligibility: {human_label}")