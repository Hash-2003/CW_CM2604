import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# Load and clean dataset
def load_and_prepare_data(csv_path: str = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(csv_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # 1. Service Aggregation: How embedded is the customer in the ecosystem?
    service_columns = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    # Count how many total extra services they subscribe to
    df['Total_Extra_Services'] = df[service_columns].apply(lambda x: (x == 'Yes').sum(), axis=1)

    # 2. Financial Ratio: Identifying hidden price hikes
    # Calculated monthly average vs what they are currently paying
    # (Replacing tenure of 0 with 1 to avoid division by zero errors)
    df['Calculated_Monthly_Avg'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    df['Price_Difference'] = df['MonthlyCharges'] - df['Calculated_Monthly_Avg']

    # 3. Contract Logic: Grouping long-term vs short-term
    df['Is_Month_to_Month'] = (df['Contract'] == 'Month-to-month').astype(int)

    # Separate features and target
    X = df.drop(["Churn", "customerID"], axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})

    return X, y


# Train / validation / test split
def split_data(X, y, random_state: int = 2025):

    # 70% train
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y
    )

    # 15% validation / 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# Preprocessing for Decision Tree
def build_tree_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    numeric = X.select_dtypes(include=["int64", "float64"]).columns
    categorical = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    return preprocessor


# Preprocessing for Neural Network
def build_nn_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    numeric = X.select_dtypes(include=["int64", "float64"]).columns
    categorical = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    return preprocessor