import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('data/EMI_dataset.csv', low_memory=False)

print("🔹 Initial Shape:", df.shape)

# Convert numeric columns stored as objects
numeric_cols = ['age', 'monthly_salary', 'monthly_rent', 'credit_score',
                'bank_balance', 'emergency_fund', 'requested_amount',
                'requested_tenure', 'max_monthly_emi']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values
df.fillna({
    'education': df['education'].mode()[0],
    'monthly_rent': df['monthly_rent'].median(),
    'credit_score': df['credit_score'].median(),
    'bank_balance': df['bank_balance'].median(),
    'emergency_fund': df['emergency_fund'].median()
}, inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

print("✅ Missing values handled and categorical columns encoded")

# Split into classification and regression targets
X = df.drop(['emi_eligibility', 'max_monthly_emi'], axis=1)
y_class = df['emi_eligibility']
y_reg = df['max_monthly_emi']

# Train-test split (same X used for both)
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

print(" Data split completed")
print("Training samples:", X_train.shape[0], " | Test samples:", X_test.shape[0])

# Save processed data
df.to_csv('data/cleaned_EMI_dataset.csv', index=False)
print(" Cleaned dataset saved to data/cleaned_EMI_dataset.csv")
