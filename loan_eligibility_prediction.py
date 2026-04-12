import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------
# Load and preprocess data
# ---------------------------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("loan-train.csv")

    # Clean column names
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

    # Clean Loan_Status
    train['Loan_Status'] = train['Loan_Status'].astype(str).str.strip()
    train['Loan_Status'] = train['Loan_Status'].map({'Y': 1, 'N': 0})
    train = train.dropna(subset=['Loan_Status'])

    # Convert types
    train['Loan_Status'] = train['Loan_Status'].astype(int)
    train['Dependents'] = train['Dependents'].astype(int)

    # Encode categorical variables
    train['Gender'] = train['Gender'].map({'Male': 1, 'Female': 0})
    train['Married'] = train['Married'].map({'Yes': 1, 'No': 0})

    return train


# ---------------------------------------------
# Train model with scaling
# ---------------------------------------------
@st.cache_resource
def train_model(data):
    X = data[['Gender', 'Married', 'Dependents', 'Credit_History', 'LoanAmount']]
    y = data['Loan_Status']

    scaler = StandardScaler()
    X[['LoanAmount']] = scaler.fit_transform(X[['LoanAmount']])

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, scaler


# Load data and train model
data = load_data()
model, scaler = train_model(data)

# ---------------------------------------------
# UI with LOGO + TITLE
# ---------------------------------------------
col1, col2 = st.columns([1, 4])

with col1:
    st.image("logo.png", width=80)   # Put logo.png in same folder

with col2:
    st.title("Loan Eligibility Prediction")

st.write("Enter applicant details below:")

# ---------------------------------------------
# User Inputs
# ---------------------------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
credit_history = st.selectbox("Credit History", [0, 1])
loan_amount = st.number_input("Loan Amount (in Rupees)", value=120000.0)

# ---------------------------------------------
# Convert Inputs
# ---------------------------------------------
gender_value = 1 if gender == "Male" else 0
marital_value = 1 if marital_status == "Yes" else 0
dependents_value = 3 if dependents == "3+" else int(dependents)

# Convert rupees → dataset scale (IMPORTANT FIX)
loan_amount_value = loan_amount / 1000

# ---------------------------------------------
# Prediction
# ---------------------------------------------
if st.button("Predict Loan Status"):

    input_df = pd.DataFrame([[gender_value, marital_value, dependents_value, credit_history, loan_amount_value]],
                            columns=['Gender', 'Married', 'Dependents', 'Credit_History', 'LoanAmount'])

    # Apply scaling
    input_df[['LoanAmount']] = scaler.transform(input_df[['LoanAmount']])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Output
    if prediction == 1:
        st.success(f"✅ Loan Approved (Confidence: {round(probability*100, 2)}%)")
    else:
        st.error(f"❌ Loan Not Approved (Confidence: {round((1-probability)*100, 2)}%)")
