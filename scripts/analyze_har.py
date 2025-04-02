import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "har_merged.parquet"
MODEL_DIR = BASE_DIR / "data" / "models"

def get_latest_file(prefix: str):
    files = sorted(MODEL_DIR.glob(f"{prefix}_*.pkl"), reverse=True)
    return files[0] if files else None

def load_data():
    df = pd.read_parquet(RAW_DATA)
    X = df.drop(columns=["subject", "activity", "activity_name", "set"])
    y = df["activity_name"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def analyze_model():
    X_train, X_test, y_train, y_test = load_data()
    model = joblib.load(get_latest_file("model_har"))
    encoder = joblib.load(get_latest_file("labels_har"))

    y_pred = model.predict(X_test)
    y_pred_labels = encoder.inverse_transform(y_pred)

    print("📋 Classification Report:\n")
    print(classification_report(y_test, y_pred_labels))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_labels, labels=encoder.classes_)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Class distribution
    plt.figure(figsize=(8, 4))
    y_test.value_counts().plot(kind="bar", color="skyblue")
    plt.title("True Activity Distribution (Test Set)")
    plt.ylabel("Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🔍 Analyzing HAR model performance...")
    analyze_model()
