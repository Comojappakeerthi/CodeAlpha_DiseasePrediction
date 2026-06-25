import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("\n===== Heart Disease Prediction System =====\n")

features = []

questions = [
    "Age: ",
    "Sex (1 = Male, 0 = Female): ",
    "Chest Pain Type (0-3): ",
    "Resting Blood Pressure: ",
    "Cholesterol: ",
    "Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False): ",
    "Rest ECG (0-2): ",
    "Maximum Heart Rate Achieved: ",
    "Exercise Induced Angina (1 = Yes, 0 = No): ",
    "Oldpeak: ",
    "Slope (0-2): ",
    "Number of Major Vessels (0-3): ",
    "Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect): "
]

for q in questions:
    value = float(input(q))
    features.append(value)

columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]

data = pd.DataFrame([features], columns=columns)

# Scale data
scaled_data = scaler.transform(data)

# Predict
prediction = model.predict(scaled_data)

if prediction[0] == 1:
    print("\n⚠️ Prediction: Heart Disease Detected")
else:
    print("\n✅ Prediction: No Heart Disease Detected")