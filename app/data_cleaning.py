import pandas as pd

# Load raw dataset
df = pd.read_csv('data/EMI_dataset.csv', low_memory=False)
print("🔹 Initial Shape:", df.shape)
#  CLEAN NUMERIC COLUMNS FIRST
numeric_cols = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
    'other_monthly_expenses', 'existing_loans', 'current_emi_amount',
    'credit_score', 'bank_balance', 'emergency_fund', 'requested_amount',
    'requested_tenure', 'max_monthly_emi'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r'[^0-9.-]', '', regex=True)  # remove junk
            .replace('', '0')
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric values after cleaning
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


# Handle missing categorical values (after numeric fix)
df = df.fillna({
    'loan_type': 'Unknown',
    'gender': 'Unknown',
    'marital_status': 'Unknown',
    'residential_status': 'Unknown'
})

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(" Missing values handled and categorical columns encoded")

# Save cleaned dataset
df.to_csv('data/cleaned_EMI_dataset.csv', index=False)
print(" Cleaned dataset saved to data/cleaned_EMI_dataset.csv")
