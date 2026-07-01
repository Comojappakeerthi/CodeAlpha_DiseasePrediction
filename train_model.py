
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from encoding import FEATURE_COLUMNS, TARGET_COLUMN, encode_dataframe

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "heart.csv")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Could not find dataset at: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        print("Your dataset is missing these expected columns:", missing)
        print("Available columns:", list(df.columns))
        sys.exit(1)

    df = encode_dataframe(df)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    dropped = before - len(df)
    if dropped:
        print(f"Note: dropped {dropped} row(s) with unrecognized category values or missing data.")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "heart_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    print("Saved models/heart_model.pkl and models/scaler.pkl")


if __name__ == "__main__":
    main()
