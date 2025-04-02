import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "har_merged.parquet"
MODEL_DIR = BASE_DIR / "data" / "models"
PRED_DIR = BASE_DIR / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

def get_latest_file(path: Path, prefix: str):
    files = sorted(path.glob(f"{prefix}_*.pkl"), reverse=True)
    return files[0] if files else None

def load_model_and_encoder():
    model_file = get_latest_file(MODEL_DIR, "model_har")
    encoder_file = get_latest_file(MODEL_DIR, "labels_har")

    if not model_file or not encoder_file:
        raise FileNotFoundError("Model or label encoder not found.")

    model = joblib.load(model_file)
    encoder = joblib.load(encoder_file)
    print(f"✅ Loaded model: {model_file.name}")
    return model, encoder

def run_predictions():
    df = pd.read_parquet(RAW_PATH)
    X = df.drop(columns=["subject", "activity", "activity_name", "set"])

    model, encoder = load_model_and_encoder()
    preds = model.predict(X)
    labels = encoder.inverse_transform(preds)

    df["predicted_activity"] = labels
    return df

def save_predictions(df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PRED_DIR / f"har_predictions_{timestamp}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"📦 Predictions saved to {out_path}")

if __name__ == "__main__":
    print("🚀 Running HAR batch prediction...")
    df_pred = run_predictions()
    save_predictions(df_pred)

