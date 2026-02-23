import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from preprocess import load_and_preprocess, SELECTED_FEATURES

# Load data
data_path = "data/telco_churn.csv"
df = load_and_preprocess(data_path)

X = df[SELECTED_FEATURES]
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature types
numeric_features = ["tenure", "MonthlyCharges"]
categorical_features = ["Contract", "InternetService"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight='balanced'))
    ]
)

# Train
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")

print("✅ Model trained and saved successfully.")