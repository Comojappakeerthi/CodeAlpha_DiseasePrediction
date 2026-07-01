from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

from encoding import FEATURE_COLUMNS, encode_dataframe

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "heart_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

model = None
scaler = None

# Load trained model and scaler if they exist.
# (Run train_model.py first to create them from your Kaggle dataset.)
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return render_template(
            "index.html",
            prediction="Error",
            message="Model not found. Run train_model.py first to generate models/heart_model.pkl and models/scaler.pkl.",
            icon="❌",
            status="danger",
        )

    try:
        raw = {
            "Age": float(request.form["age"]),
            "Sex": request.form["sex"],                     # M / F
            "ChestPainType": request.form["chestpaintype"],  # TA / ATA / NAP / ASY
            "RestingBP": float(request.form["restingbp"]),
            "Cholesterol": float(request.form["cholesterol"]),
            "FastingBS": float(request.form["fastingbs"]),   # 0 / 1
            "RestingECG": request.form["restingecg"],        # Normal / ST / LVH
            "MaxHR": float(request.form["maxhr"]),
            "ExerciseAngina": request.form["exerciseangina"],  # Y / N
            "Oldpeak": float(request.form["oldpeak"]),
            "ST_Slope": request.form["stslope"],              # Up / Flat / Down
        }

        patient_raw = pd.DataFrame([raw])
        patient_encoded = encode_dataframe(patient_raw)[FEATURE_COLUMNS]

        scaled = scaler.transform(patient_encoded)
        prediction = model.predict(scaled)[0]

        summary = {
            "Age": int(raw["Age"]),
            "Sex": "Male" if raw["Sex"] == "M" else "Female",
            "Chest Pain Type": raw["ChestPainType"],
            "Resting Blood Pressure": int(raw["RestingBP"]),
            "Cholesterol": int(raw["Cholesterol"]),
            "Max Heart Rate": int(raw["MaxHR"]),
        }

        if prediction == 1:
            return render_template(
                "index.html",
                prediction="HIGH RISK OF HEART DISEASE",
                message="Please consult a healthcare professional.",
                icon="⚠️",
                status="danger",
                summary=summary,
            )
        else:
            return render_template(
                "index.html",
                prediction="LOW RISK OF HEART DISEASE",
                message="Maintain a healthy lifestyle and regular checkups.",
                icon="💚",
                status="success",
                summary=summary,
            )

    except Exception as e:
        return render_template(
            "index.html",
            prediction="Error",
            message=str(e),
            icon="❌",
            status="danger",
        )


if __name__ == "__main__":
    app.run(debug=False)