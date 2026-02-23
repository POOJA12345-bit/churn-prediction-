import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preprocess import load_and_preprocess, SELECTED_FEATURES

# Load data
df = load_and_preprocess("data/telco_churn.csv")

X = df[SELECTED_FEATURES]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model
model = joblib.load("models/churn_model.pkl")

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))