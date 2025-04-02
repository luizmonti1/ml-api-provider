# ML API: Human Activity Recognition with FastAPI

🚀 Real-time and batch ML pipeline for classifying smartphone motion sensor data (UCI HAR dataset).

## Features

- 🧠 RandomForestClassifier trained on 561 HAR features
- ⚙️ Full pipeline: import → train → predict → analyze
- 🌐 Live prediction API via FastAPI (`/har/predict`)
- 📊 Confusion matrix and evaluation visualization

## Endpoints

- `POST /har/predict`: Predicts activity from a 561-feature vector
- `GET /har/info`: Model metadata

## Setup

```bash
git clone https://github.com/<your-username>/ml-api-provider.git
cd ml-api-provider
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```
