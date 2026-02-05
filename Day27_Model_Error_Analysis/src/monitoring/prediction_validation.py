import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

model = joblib.load(MODEL_DIR / "model.pkl")

csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV file found in data folder")

df = pd.read_csv(csv_files[0])

TARGET = "Item_Outlet_Sales"

X = df.drop(TARGET, axis=1)
y = df[TARGET]

predictions = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

print("Validation RMSE:", rmse)
print("Validation R2:", r2)

result_df = X.copy()
result_df["Actual"] = y
result_df["Predicted"] = predictions
result_df["Error"] = y - predictions

result_df.to_csv(OUTPUT_DIR / "prediction_validation.csv", index=False)

print("Day 27: Prediction validation completed.")
