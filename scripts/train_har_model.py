import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "har_merged.parquet"
MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Missing HAR dataset at: {RAW_DATA}")
    df = pd.read_parquet(RAW_DATA)
    return df

def prepare_data(df):
    X = df.drop(columns=["subject", "activity", "activity_name", "set"])
    y = df["activity_name"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return X, y_encoded, encoder

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    return model

def save_model(model, encoder):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"model_har_{timestamp}.pkl"
    encoder_path = MODEL_DIR / f"labels_har_{timestamp}.pkl"

    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Label encoder saved to: {encoder_path}")

if __name__ == "__main__":
    print("🚀 Training HAR classifier...")
    df = load_data()
    X, y, encoder = prepare_data(df)
    model = train_model(X, y)
    save_model(model, encoder)

