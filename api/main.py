from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "data" / "models"

# === FastAPI app ===
app = FastAPI(title="HAR API", version="1.0")

# === Load model and encoder ===
def get_latest_file(prefix: str):
    files = sorted(MODEL_DIR.glob(f"{prefix}_*.pkl"), reverse=True)
    return files[0] if files else None

model = joblib.load(get_latest_file("model_har"))
label_encoder = joblib.load(get_latest_file("labels_har"))
print(f"✅ Loaded model: {get_latest_file('model_har').name}")

# Expect exactly 561 features (float values)
class HARInput(BaseModel):
    features: list[float] = Field(..., min_items=561, max_items=561)

@app.post("/har/predict")
def predict_activity(input_data: HARInput):
    try:
        X = np.array(input_data.features).reshape(1, -1)
        y_pred = model.predict(X)[0]
        activity = label_encoder.inverse_transform([y_pred])[0]
        return {"activity": activity}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/har/info")
def model_info():
    return {
        "model_file": get_latest_file("model_har").name,
        "label_encoder": get_latest_file("labels_har").name,
        "input_shape": [561],
        "output_labels": list(label_encoder.classes_)
    }
