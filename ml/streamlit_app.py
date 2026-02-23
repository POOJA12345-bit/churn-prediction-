import streamlit as st
import joblib
import pandas as pd

st.title("📊 Customer Churn Prediction")

model = joblib.load("models/churn_model.pkl")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "InternetService": internet_service
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is likely to stay (Probability: {probability:.2f})")