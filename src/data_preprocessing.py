import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode target
    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0}).fillna(y)
        if y.dtype == "object":
            y = pd.factorize(y)[0]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    feature_names = X.columns

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = np.array(y, dtype=int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ✅ RETURN SCALER ALSO (IMPORTANT FIX)
    return (X_train, X_test, y_train, y_test), feature_names, scaler