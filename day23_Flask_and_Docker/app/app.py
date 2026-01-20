from flask import Flask, request, jsonify
import pandas as pd
import pickle
import logging
import os

app = Flask(__name__)

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_PATH = "models/model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    model = None


@app.route("/")
def home():
    return "API is running"


@app.route("/health")
def health():
    return jsonify({"status": "UP"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()

    df = pd.DataFrame([data["features"]])
    prediction = model.predict(df)

    return jsonify({"prediction": float(prediction[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
