# 🏦 Loan Eligibility Prediction System

An AI-based web application that predicts whether a loan application will be approved or not based on applicant details. This project demonstrates the use of machine learning for decision-making in financial services.

---

## 📌 Project Overview

This application uses a Logistic Regression model to analyze applicant data and predict loan approval status. It is built with an interactive web interface using Streamlit, allowing users to input details and receive instant predictions.

---

## 🚀 Features

- 📊 Predicts loan approval status (Approved / Not Approved)
- 🤖 Machine Learning model using Logistic Regression
- ⚡ Real-time predictions through Streamlit UI
- 📈 Displays prediction confidence score
- 🧹 Handles missing data and preprocessing
- 💰 Supports loan amount input in INR (₹)

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit

---

## 📂 Dataset

The model is trained on a loan dataset containing:

- Gender  
- Marital Status  
- Number of Dependents  
- Credit History  
- Loan Amount  

> Note: Loan amount is scaled internally for better model performance.

---

## ⚙️ How It Works

1. Data preprocessing (handling missing values, encoding categorical data)
2. Feature selection and scaling
3. Model training using Logistic Regression
4. User inputs are transformed and passed to the model
5. Prediction is displayed along with confidence score

---

## 💡 Key Learning Outcomes

- Applied machine learning to a real-world financial problem
- Understood importance of data preprocessing and feature scaling
- Built and deployed an interactive ML web application
- Learned model behavior based on dataset limitations

---

## ⚠️ Limitations

- Model uses limited features (does not include applicant income or loan term)
- Predictions depend heavily on training dataset patterns
- Not intended for real financial decision-making

---

## 📸 Application Preview

<p align="center">
  <img src="Screenshot 2026-04-14 171734.png" width="700">
</p>

---

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/loan-eligibility-prediction.git
cd loan-eligibility-prediction
pip install -r requirements.txt
streamlit run loan_eligibility_prediction.py
