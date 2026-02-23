 📊 Customer Churn Prediction (Machine Learning)

This project predicts whether a customer will churn (leave the service) using Machine Learning.

 🚀 Features Used
- Tenure
- MonthlyCharges
- Contract
- InternetService

 🧠 Model
- Logistic Regression
- Preprocessing using StandardScaler & OneHotEncoder
- Built using Scikit-learn Pipeline

⚙️ How to Run

Train Model
python src/train.py

 Evaluate Model
python src/evaluate.py

Run Streamlit App
streamlit run streamlit_app.py

Model Performance
- Accuracy: ~80%
- Good performance on non-churn customers
- Moderate recall for churn customers

