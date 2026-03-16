# 💳 EMIPredict AI
## Intelligent Financial Risk Assessment Platform

EMIPredict AI is a machine learning platform that evaluates a customer's financial profile to determine EMI eligibility and predict the maximum affordable monthly EMI.

The system integrates machine learning models, MLflow experiment tracking, and a Streamlit web application to provide real-time financial risk assessment.

## 📌 Project Overview

Financial institutions often rely on manual underwriting to evaluate loan applications, which can be slow and inconsistent.

EMIPredict AI solves this by providing:

- Automated EMI eligibility prediction
- Prediction of maximum safe EMI amount
- MLflow experiment tracking for model comparison
- Streamlit application for real-time predictions

## 🎯 Project Objectives

- Perform data preprocessing and feature engineering
- Train multiple classification and regression models
- Track experiments using MLflow
- Select the best performing models
- Build an interactive Streamlit application
- Deploy the platform for real-time financial risk prediction

## 🧠 Machine Learning Tasks

### 1️⃣ Classification Problem

Predict EMI eligibility.

Classes:
- Eligible
- High Risk
- Not Eligible

### 2️⃣ Regression Problem

Predict maximum safe monthly EMI.

Output:
- Continuous EMI value representing safe repayment capacity.

## 📊 Dataset Information

- Total Records: 400,000
- Input Features: 22
- Target Variables: 2
- EMI Scenarios: 5

## 🏦 EMI Scenarios

1. E-commerce Shopping EMI  
2. Home Appliances EMI  
3. Vehicle EMI  
4. Personal Loan EMI  
5. Education EMI  

## 🧾 Input Features

### Personal Demographics

- Age
- Gender
- Marital Status
- Education

### Employment Information

- Monthly Salary
- Employment Type
- Years of Employment
- Company Type

### Housing and Family

- House Type
- Monthly Rent
- Family Size
- Dependents

### Monthly Financial Obligations

- School Fees
- College Fees
- Travel Expenses
- Groceries and Utilities
- Other Monthly Expenses

### Financial Status

- Existing Loans
- Current EMI Amount
- Credit Score
- Bank Balance
- Emergency Fund

### Loan Application Details

- EMI Scenario
- Requested Amount
- Requested Tenure

## ⚙️ Project Workflow

Dataset  
↓  
Data Cleaning & Preprocessing  
↓  
Exploratory Data Analysis  
↓  
Feature Engineering  
↓  
Machine Learning Model Training  
↓  
MLflow Experiment Tracking  
↓  
Model Evaluation & Selection  
↓  
Streamlit Web Application  
↓  
Cloud Deployment

## 🧹 Data Preprocessing

Tasks performed:

- Handling missing values
- Removing duplicates
- Data validation
- Feature scaling
- Categorical encoding

Notebook used:

DATA_cleaning.ipynb

## 🤖 Machine Learning Models

### Classification Models

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Notebook used:

model_training.ipynb

### Regression Models

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Notebook used:

regression_training.ipynb

## 🧪 MLflow Experiment Tracking

MLflow is used for:

- Tracking experiments
- Logging hyperparameters
- Storing evaluation metrics
- Comparing models
- Maintaining model registry

## 📊 Evaluation Metrics

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Regression Metrics

- RMSE
- MAE
- R² Score
- MAPE

## 🖥 Streamlit Application

The web application allows users to:

- Enter financial information
- Predict EMI eligibility
- Predict maximum EMI amount
- View model insights

## ☁ Deployment

The application is deployed using Streamlit Cloud.

Features:

- Public access
- Real-time predictions
- Interactive interface

## 📁 Project Structure

EMIPredict-AI  
│  
├── DATA_cleaning.ipynb  
├── model_training.ipynb  
├── regression_training.ipynb  
│  
├── dataset/  
├── models/  
├── streamlit_app/  
└── README.md  

## 🛠 Tech Stack

### Programming

Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

### Tools

- MLflow
- Streamlit
- Streamlit Cloud

## 👨‍💻 Author

Ajey Jha  
Data Science | Machine Learning | Financial Analytics
