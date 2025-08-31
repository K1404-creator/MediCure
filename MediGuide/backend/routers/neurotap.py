import os

import joblib
import numpy as np
from fastapi import APIRouter

router = APIRouter()
model_path = os.path.join(os.path.dirname(__file__), '../../models/neurotap_model.pkl')
model = joblib.load(os.path.abspath(model_path))

@router.post("/")
def predict(data: dict):
    # Example features: typing_speed, error_rate, reaction_time, mouse_speed, scroll_frequency
    features = np.array([[
        data["typing_speed"], data["error_rate"], data["reaction_time"],
        data["mouse_speed"], data["scroll_frequency"]
    ]])
    
    prediction = model.predict(features)[0]
    return {"prediction": "Fatigued" if prediction == 1 else "Not Fatigued"}