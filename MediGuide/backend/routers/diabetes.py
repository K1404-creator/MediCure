
import os

import joblib
import numpy as np
from fastapi import APIRouter

router = APIRouter()
model_path = os.path.join(os.path.dirname(__file__), '../../models/diabetes_model.pkl')
model = joblib.load(os.path.abspath(model_path))

@router.post("/")
def predict(data: dict):
    features = np.array([[
        data["Pregnancies"], data["Glucose"], data["BloodPressure"],
        data["SkinThickness"], data["Insulin"], data["BMI"],
        data["DiabetesPedigreeFunction"], data["Age"]
    ]])
    prediction = model.predict(features)[0]
    return {"prediction": "Diabetic" if prediction == 1 else "Not Diabetic"}