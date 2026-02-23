import joblib
import pandas as pd

model = joblib.load("models/churn_model.pkl")

def predict_churn(tenure, monthly_charges, contract, internet_service):
    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "InternetService": internet_service
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability


if __name__ == "__main__":
    pred, prob = predict_churn(12, 70.5, "Month-to-month", "Fiber optic")
    print("Prediction:", "Churn" if pred == 1 else "No Churn")
    print("Probability:", round(prob, 2))