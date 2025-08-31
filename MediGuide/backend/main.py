import os

import joblib
import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ---------------- CORS setup ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load models ----------------
diabetes_model = joblib.load(r"C:\MediGuide\models\diabetes_model.pkl")
heart_model = joblib.load(r"C:\MediGuide\models\heart.pkl")
neurotap_model = joblib.load(r"C:\MediGuide\models\neurotap_model.pkl")

# ---------------- ML PREDICTIONS ----------------
@app.post("/predict/diabetes")
async def predict_diabetes(request: Request):
    data = await request.json()
    input_data = np.array(list(data.values())).reshape(1, -1)
    prediction = diabetes_model.predict(input_data)[0]
    return {"prediction": int(prediction)}

@app.post("/predict/heart")
async def predict_heart(request: Request):
    data = await request.json()
    input_data = np.array(list(data.values())).reshape(1, -1)
    prediction = heart_model.predict(input_data)[0]
    return {"prediction": int(prediction)}

@app.post("/predict/fatigue")
async def predict_fatigue(request: Request):
    data = await request.json()
    input_data = np.array(list(data.values())).reshape(1, -1)
    prediction = neurotap_model.predict(input_data)[0]
    return {"prediction": int(prediction)}

# ---------------- GPT-OSS CHATBOT ----------------
GPTOSS_API = "https://api.gptoss.com/v1/chat/completions"
GPTOSS_KEY = os.getenv("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")  # ✅ set your key in environment variable, not here

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    headers = {
        "Authorization": f"Bearer {GPTOSS_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",   # or whichever model GPT-OSS supports
        "messages": [{"role": "user", "content": user_message}]
    }

    response = requests.post(GPTOSS_API, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        return {"reply": reply}
    else:
        return {"error": response.text}