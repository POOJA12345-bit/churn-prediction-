import pandas as pd

SELECTED_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService"
]

TARGET = "Churn"


def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Keep only required columns
    df = df[SELECTED_FEATURES + [TARGET]]

    # Drop missing values
    df = df.dropna()

    # Convert target to binary
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    return df