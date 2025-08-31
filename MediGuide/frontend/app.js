const API_BASE = "http://127.0.0.1:8000";  // Flask backend

// Handle Heart Disease Prediction
document.getElementById("heart-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target).entries());
  const res = await fetch(`${API_BASE}/predict_heart`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  const result = await res.json();
  document.getElementById("heart-result").innerText = `Prediction: ${result.prediction}`;
});

// Handle Diabetes Prediction
document.getElementById("diabetes-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target).entries());
  const res = await fetch(`${API_BASE}/predict_diabetes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  const result = await res.json();
  document.getElementById("diabetes-result").innerText = `Prediction: ${result.prediction}`;
});

// Handle Fatigue Detection
document.getElementById("fatigue-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target).entries());
  const res = await fetch(`${API_BASE}/predict_fatigue`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  const result = await res.json();
  document.getElementById("fatigue-result").innerText = `Prediction: ${result.prediction}`;
});