# Heart Disease Prediction — Flask App

## Dataset
This app is built for the **fedesoriano "Heart Failure Prediction Dataset"**
from Kaggle, which uses these columns exactly:

```
Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,
MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease
```

with text categories:
- `Sex`: M / F
- `ChestPainType`: TA / ATA / NAP / ASY
- `RestingECG`: Normal / ST / LVH
- `ExerciseAngina`: Y / N
- `ST_Slope`: Up / Flat / Down
- `HeartDisease`: 0 (no disease) / 1 (disease) — this is the target column

## What was fixed
1. **Page refreshed and cleared your inputs** — the original `index.html` had a broken/duplicated `<form>` tag, and the result section never actually rendered. The form now posts to `/predict`, the result card is fully wired up, and every field re-fills with what you typed after submitting.
2. **Logo not showing/unclear** — replaced with a crisp `static/logo.svg` vector icon. Swap it for your own image any time.
3. **Schema mismatch** — the original code was written for the older UCI numeric-code schema (cp 0-3, thal, ca, etc.), which doesn't match your actual Kaggle dataset. `app.py`, `train_model.py`, and `index.html` have all been rewritten to match your real column names and category values.
4. **Consistent encoding** — `encoding.py` is a shared module both `train_model.py` and `app.py` import from, so the text categories (M/F, ATA/NAP/ASY/TA, etc.) are always converted to numbers the exact same way at training time and prediction time. This avoids a common bug where the two scripts silently disagree on encoding.

## Folder structure
```
heart-disease-app/
├── app.py              # Flask backend
├── train_model.py      # Trains the model from your Kaggle CSV
├── encoding.py         # Shared category encoding (used by both scripts)
├── requirements.txt
├── data/                # put your Kaggle heart.csv here
├── models/              # heart_model.pkl and scaler.pkl get saved here
├── static/
│   ├── style.css
│   └── logo.svg
└── templates/
    └── index.html
```

## Setup

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Add your Kaggle dataset**
   Save your downloaded CSV as:
   ```
   data/heart.csv
   ```

3. **Train the model**
   ```
   python3 train_model.py
   ```
   You'll see a test accuracy printed (with your ~918-row dataset, expect roughly 85-90%, not the 100% you'd see on a tiny sample). This confirms `models/heart_model.pkl` and `models/scaler.pkl` were created.

4. **Run the app**
   ## 🔗 Live Demo
https://codealpha-diseaseprediction-zcrg.onrender.com

> Note: hosted on Render's free tier — the first load after inactivity may take 30-60 seconds to wake up.

## Notes
- On macOS, use `python3` and `pip3` (not `python`/`pip`) unless you've specifically aliased them.
- Some rows in this dataset have `Cholesterol` or `RestingBP` recorded as `0` (a known data quality quirk of this Kaggle dataset, from merged source data). This doesn't break the model — RandomForest handles it fine — but it's why you might occasionally see slightly odd predictions for edge-case inputs.
- If you ever see "Model not found" on the predict page, run `train_model.py` again — that step must complete successfully before `app.py` can make predictions.