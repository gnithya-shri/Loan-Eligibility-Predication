import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Loan Prediction", layout="centered")

# ---------------------------------------------
# HEADER WITH PROFESSIONAL LOGO
# ---------------------------------------------
st.markdown("""
    <style>
    .title-container {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #1f4e79;
    }
    .subtitle {
        color: gray;
        margin-top: -8px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-container">
    <img src="https://cdn-icons-png.flaticon.com/512/2920/2920323.png" width="65">
    <div>
        <div class="main-title">Loan Eligibility Prediction</div>
        <div class="subtitle">AI-powered loan approval system</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("loan-train.csv")

    train.columns = train.columns.str.strip()

    train['Dependents'] = train['Dependents'].astype(str).str.strip()
    train['Dependents'] = train['Dependents'].replace('3+', '3')
    train['Dependents'] = pd.to_numeric(train['Dependents'], errors='coerce')

    train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0])
    train['Married'] = train['Married'].fillna(train['Married'].mode()[0])
    train['Dependents'] = train['Dependents'].fillna(train['Dependents'].mode()[0])
    train['Credit_History'] = train['Credit_History'].fillna(train['Credit_History'].mode()[0])
    train['LoanAmount'] = train['LoanAmount'].fillna(train['LoanAmount'].mean())

    train['Loan_Status'] = train['Loan_Status'].astype(str).str.strip()
    train['Loan_Status'] = train['Loan_Status'].map({'Y': 1, 'N': 0})
    train = train.dropna(subset=['Loan_Status'])

    train['Loan_Status'] = train['Loan_Status'].astype(int)
    train['Dependents'] = train['Dependents'].astype(int)

    train['Gender'] = train['Gender'].map({'Male': 1, 'Female': 0})
    train['Married'] = train['Married'].map({'Yes': 1, 'No': 0})

    return train

# ---------------------------------------------
# TRAIN MODEL
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

data = load_data()
model, scaler = train_model(data)

# ---------------------------------------------
# INPUT SECTION
# ---------------------------------------------
st.markdown("### Enter Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
credit_history = st.selectbox("Credit History", [0, 1])
loan_amount = st.number_input("Loan Amount (in ₹)", value=120000.0)

# ---------------------------------------------
# INPUT CONVERSION
# ---------------------------------------------
gender_value = 1 if gender == "Male" else 0
marital_value = 1 if marital_status == "Yes" else 0
dependents_value = 3 if dependents == "3+" else int(dependents)

loan_amount_value = loan_amount / 1000

# ---------------------------------------------
# PREDICTION
# ---------------------------------------------
if st.button("Predict Loan Status"):

    input_df = pd.DataFrame(
        [[gender_value, marital_value, dependents_value, credit_history, loan_amount_value]],
        columns=['Gender', 'Married', 'Dependents', 'Credit_History', 'LoanAmount']
    )

    input_df[['LoanAmount']] = scaler.transform(input_df[['LoanAmount']])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.write(f"Approval Confidence: **{round(probability*100, 2)}%**")
    else:
        st.error("❌ Loan Not Approved")
        st.write(f"Rejection Confidence: **{round((1-probability)*100, 2)}%**")
