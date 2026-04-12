import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------
# Load and preprocess data
# ---------------------------------------------
train = pd.read_csv("loan-train.csv")

# Fix Dependents FIRST
train['Dependents'] = train['Dependents'].replace('3+', 3)

# Fill missing values
train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0])
train['Married'] = train['Married'].fillna(train['Married'].mode()[0])
train['Dependents'] = train['Dependents'].fillna(train['Dependents'].mode()[0])
train['Credit_History'] = train['Credit_History'].fillna(train['Credit_History'].mode()[0])
train['LoanAmount'] = train['LoanAmount'].fillna(train['LoanAmount'].mean())

# Now convert to int
train['Dependents'] = train['Dependents'].astype(int)

# Encode categorical variables
train = train.replace({
    'Loan_Status': {'Y': 1, 'N': 0},
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Self_Employed': {'Yes': 1, 'No': 0}
})

# Features and target
X = train[['Gender', 'Married', 'Dependents', 'Credit_History', 'LoanAmount']]
y = train['Loan_Status']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

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

if dependents == "3+":
    dependents_value = 3
else:
    dependents_value = int(dependents)

# Prediction
if st.button("Predict Loan Status"):
    input_data = [[gender_value, marital_value, dependents_value, credit_history, loan_amount]]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")













