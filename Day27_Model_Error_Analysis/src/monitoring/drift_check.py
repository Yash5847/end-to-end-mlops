import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found in data folder")

train_df = pd.read_csv(csv_files[0])

numerical_cols = train_df.select_dtypes(include=["int64", "float64"]).columns

drift_report = []

for col in numerical_cols:
    train_mean = train_df[col].mean()
    new_mean = train_df[col].mean()  

    drift_report.append({
        "feature": col,
        "train_mean": train_mean,
        "new_mean": new_mean,
        "mean_diff": abs(train_mean - new_mean)
    })

drift_df = pd.DataFrame(drift_report)
drift_df.to_csv(OUTPUT_DIR / "data_drift_report.csv", index=False)

print("Day 27: Data drift report generated successfully.")
