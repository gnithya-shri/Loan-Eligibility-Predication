import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------
# Load and preprocess data
# ---------------------------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("loan-train.csv")

    # Clean column names (avoid hidden spaces issues)
    train.columns = train.columns.str.strip()

    # Fix Dependents
    train['Dependents'] = train['Dependents'].astype(str).str.strip()
    train['Dependents'] = train['Dependents'].replace('3+', '3')
    train['Dependents'] = pd.to_numeric(train['Dependents'], errors='coerce')

    # Fill missing values
    train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0])
    train['Married'] = train['Married'].fillna(train['Married'].mode()[0])
    train['Dependents'] = train['Dependents'].fillna(train['Dependents'].mode()[0])
    train['Credit_History'] = train['Credit_History'].fillna(train['Credit_History'].mode()[0])
    train['LoanAmount'] = train['LoanAmount'].fillna(train['LoanAmount'].mean())

    # Clean Loan_Status properly (MAIN FIX)
    train['Loan_Status'] = train['Loan_Status'].astype(str).str.strip()
    train['Loan_Status'] = train['Loan_Status'].map({'Y': 1, 'N': 0})

    # Drop invalid rows
    train = train.dropna(subset=['Loan_Status'])

    # Convert types
    train['Loan_Status'] = train['Loan_Status'].astype(int)
    train['Dependents'] = train['Dependents'].astype(int)

    # Encode categorical variables
    train['Gender'] = train['Gender'].map({'Male': 1, 'Female': 0})
    train['Married'] = train['Married'].map({'Yes': 1, 'No': 0})

    return train


# ---------------------------------------------
# Train model
# ---------------------------------------------
@st.cache_resource
def train_model(data):
    X = data[['Gender', 'Married', 'Dependents', 'Credit_History', 'LoanAmount']]
    y = data['Loan_Status']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


# Load data and train model
data = load_data()
model = train_model(data)

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.title("Loan Eligibility Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
credit_history = st.selectbox("Credit History", [0, 1])
loan_amount = st.number_input("Loan Amount", value=120.0)

# Convert inputs
gender_value = 1 if gender == "Male" else 0
marital_value = 1 if marital_status == "Yes" else 0
dependents_value = 3 if dependents == "3+" else int(dependents)

# Prediction
if st.button("Predict Loan Status"):
    input_data = [[gender_value, marital_value, dependents_value, credit_history, loan_amount]]

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")
