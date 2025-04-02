# 🧠 Human Activity Recognition API — Built for Real-World Integration

This project is more than a machine learning script — it's a complete, scalable system for serving predictions from a trained model through a production-ready API.

It uses smartphone sensor data to classify human activities like walking, sitting, standing, or laying down. Built with FastAPI and trained on the [UCI HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), this system mimics the type of architecture you'd expect in enterprise ML infrastructure — from ingestion and feature extraction to training, prediction, and visualization.

---

## 🚀 Key Features

- **Live ML API with FastAPI** — `/har/predict` accepts raw input and returns the predicted activity in real time.
- **Full Pipeline** — Includes data ingestion, preprocessing, training, evaluation, and prediction logic.
- **Batch Processing** — Run predictions on entire datasets, not just individual vectors.
- **Exploratory Analysis** — Confusion matrix, classification reports, and visual validation built in.
- **GitHub-Friendly** — Modular structure, proper `.gitignore`, easy-to-run scripts.
- **Ready to Expand** — Docker-ready architecture, flexible model swapping, easy to monitor and scale.

---

## 🛠️ Quickstart

Clone the repo and install dependencies:

```bash
git clone https://github.com/luizmonti1/ml-api-provider.git
cd ml-api-provider
python -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🧪 Run the Model Locally

Step 1: Import and prepare the dataset

```bash
python scripts/import_har_data.py
```

Step 2: Train the model and save artifacts

```bash
python scripts/train_har_model.py
```

Step 3: Run batch predictions (optional)

```bash
python scripts/predict_har.py
```

Step 4: Visualize model performance

```bash
python scripts/analyze_har.py
```

---

## 🌐 Start the API Server

Run the FastAPI service locally:

```bash
uvicorn api.main:app --reload
```

Interactive API docs available at:

```
http://127.0.0.1:8000/docs
```

### Example Request:

```bash
POST /har/predict
Content-Type: application/json

{
  "features": [0.257178, -0.023285, -0.014654, ..., 0.182527]
}
```

---

## 📂 Project Structure

```
.
├── api/                   # FastAPI server
├── scripts/               # Data pipeline scripts
│   ├── import_har_data.py
│   ├── train_har_model.py
│   ├── predict_har.py
│   └── analyze_har.py
├── data/                  # Ignored in repo — stores model, predictions, raw files
├── external_data/         # Ignored — contains UCI HAR Dataset
├── pipeline.py            # Optional: Orchestrated CLI pipeline
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📈 Model Performance

After training, the classifier achieves ~98% accuracy on the UCI HAR test set. Confusion matrix and classification reports are automatically generated with the `analyze_har.py` script for transparency and insight.

---

## 🤝 Contributing

If you're into ML ops, signal processing, model explainability, or just want to collaborate — fork it, clone it, build on it. Pull requests are welcome.

---

## 📄 License

MIT — use it, learn from it, build on it, fork it into your startup.

---

## 👋 Contact

Built by [Luiz Monti](https://github.com/luizmonti1)  
Feel free to reach out via [GitHub Issues](https://github.com/luizmonti1/ml-api-provider/issues) for questions or suggestions.

# 🧠 Human Activity Recognition API

![Build Time](https://img.shields.io/badge/Built%20in-Under%2024%20Hours-blueviolet?style=for-the-badge)


