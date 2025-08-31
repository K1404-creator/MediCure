import os

import joblib
import numpy as np
from fastapi import APIRouter

router = APIRouter()
model_path = os.path.join(os.path.dirname(__file__), '../../models/heart.pkl')
model = joblib.load(os.path.abspath(model_path))

@router.post("/")
def predict(data: dict):
    # Example features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    features = np.array([[
        data["age"], data["sex"], data["cp"], data["trestbps"],
        data["chol"], data["fbs"], data["restecg"], data["thalach"],
        data["exang"], data["oldpeak"], data["slope"], data["ca"], data["thal"]
    ]])
    
    prediction = model.predict(features)[0]
    return {"prediction": "Heart Disease Detected" if prediction == 1 else "No Heart Disease"}