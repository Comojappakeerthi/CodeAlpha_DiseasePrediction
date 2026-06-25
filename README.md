# Heart Disease Prediction System

## Project Overview

This project predicts whether a person is likely to have heart disease based on medical attributes using Machine Learning techniques.

## Objective

To predict the possibility of heart disease using patient medical data.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

## Dataset

The project uses the Heart Disease Dataset containing medical attributes such as:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate Achieved
* Exercise Induced Angina
* Oldpeak
* Slope
* Number of Major Vessels
* Thal

## Machine Learning Algorithm

* Random Forest Classifier

## Model Evaluation Metrics

* Accuracy: 98.54%
* Precision: 99%
* Recall: 99%
* F1-Score: 99%

## Project Structure

CodeAlpha_DiseasePrediction/
│
├── dataset/
│   └── heart.csv
├── models/
│   ├── heart_model.pkl
│   └── scaler.pkl
├── screenshots/
├── train_model.py
├── app.py
├── requirements.txt
└── README.md

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

### Train the Model

```bash
python3 train_model.py
```

### Run the Prediction System

```bash
python3 app.py
```

## Output

The system predicts whether a patient is likely to have heart disease or not based on the input medical values.

## Author

COMOJAPPA KEERTHI
