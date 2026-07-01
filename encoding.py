
"""
Shared feature definitions and category encodings for the Heart Failure
Prediction dataset (fedesoriano / Kaggle).
"""

FEATURE_COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope"
]

TARGET_COLUMN = "HeartDisease"

SEX_MAP = {"M": 1, "F": 0}
CHEST_PAIN_MAP = {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3}
RESTING_ECG_MAP = {"Normal": 0, "ST": 1, "LVH": 2}
EXERCISE_ANGINA_MAP = {"N": 0, "Y": 1}
ST_SLOPE_MAP = {"Up": 0, "Flat": 1, "Down": 2}


def encode_dataframe(df):
    df = df.copy()
    df["Sex"] = df["Sex"].map(SEX_MAP)
    df["ChestPainType"] = df["ChestPainType"].map(CHEST_PAIN_MAP)
    df["RestingECG"] = df["RestingECG"].map(RESTING_ECG_MAP)
    df["ExerciseAngina"] = df["ExerciseAngina"].map(EXERCISE_ANGINA_MAP)
    df["ST_Slope"] = df["ST_Slope"].map(ST_SLOPE_MAP)
    return df
