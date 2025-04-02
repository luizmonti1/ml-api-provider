import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "har_merged.parquet"
MODELS_DIR = BASE_DIR / "data" / "models"
REPORTS_DIR = BASE_DIR / "data" / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("🔍 Analyzing HAR model performance...")

# Load merged dataset
df = pd.read_parquet(RAW_PATH)

# Load the latest model and label encoder
latest_model = sorted(MODELS_DIR.glob("model_har_*.pkl"))[-1]
latest_encoder = sorted(MODELS_DIR.glob("labels_har_*.pkl"))[-1]

model = joblib.load(latest_model)
le = joblib.load(latest_encoder)

# Match feature set used during training
X = df[model.feature_names_in_]

# Encode labels using the same label encoder
y = df["activity_name"]
y = le.transform(y)

# Train/test split (same as train script)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Predict
y_pred = model.predict(X_test)

# Decode for readable report
y_test_decoded = [le.classes_[i] for i in y_test]
y_pred_decoded = [le.classes_[i] for i in y_pred]

# Generate classification report
report = classification_report(y_test_decoded, y_pred_decoded, digits=4)
print("\n📋 Classification Report:\n")
print(report)

# Save classification report
report_path = REPORTS_DIR / "classification_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(f"✅ Report saved to: {report_path}")

# Create and save confusion matrix
cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=le.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

cm_path = REPORTS_DIR / "confusion_matrix.png"
plt.savefig(cm_path)
print(f"✅ Confusion matrix saved to: {cm_path}")
