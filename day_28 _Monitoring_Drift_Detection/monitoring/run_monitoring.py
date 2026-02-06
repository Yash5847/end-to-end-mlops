import pandas as pd
from datetime import datetime

from data_drift import detect_data_drift
from prediction_drift import detect_prediction_drift

LOG_FILE = "logs/monitoring_log.txt"

REF_DATA_PATH = "data/reference_data.csv"
CURR_DATA_PATH = "data/current_data.csv"
MODEL_PATH = "models/model.pkl"

ref_df = pd.read_csv(REF_DATA_PATH)
curr_df = pd.read_csv(CURR_DATA_PATH)

if "target" in ref_df.columns:
    X_ref = ref_df.drop("target", axis=1)
else:
    X_ref = ref_df.copy()

X_curr = curr_df.copy()

data_drift_result = detect_data_drift(
    REF_DATA_PATH,
    CURR_DATA_PATH
)

prediction_drift_result = detect_prediction_drift(
    MODEL_PATH,
    X_ref,
    X_curr
)

with open(LOG_FILE, "a") as log:
    log.write(f"\n--- {datetime.now()} ---\n")
    log.write(f"Data Drift Detected: {data_drift_result}\n")
    log.write(f"Prediction Drift Detected: {prediction_drift_result}\n")

print("Day 28 Monitoring completed successfully.")
