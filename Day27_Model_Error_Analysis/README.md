DAY 27 – MODEL MONITORING & PREDICTION VALIDATION

Overview:
Day 27 focuses on post-deployment machine learning tasks.
The goal is to validate model predictions on data and
check for basic data drift to ensure model stability
after deployment.

This day is a continuation of the same end-to-end ML project
developed from Day 1 to Day 26.

--------------------------------------------------

Folder Additions:
src/monitoring/
outputs/

--------------------------------------------------

Files Created:

1. src/monitoring/prediction_validation.py
- Loads the trained model from models/
- Loads dataset from data/
- Generates predictions
- Calculates RMSE and R2 score
- Saves actual vs predicted values with errors

Output:
outputs/prediction_validation.csv

2. src/monitoring/drift_check.py
- Reads numerical features from dataset
- Compares statistical properties (mean)
- Generates a basic data drift report

Output:
outputs/data_drift_report.csv

--------------------------------------------------

How to Run:

From project root (ML_Project):

python src/monitoring/prediction_validation.py
python src/monitoring/drift_check.py

--------------------------------------------------

Concepts Covered:
- Post-deployment model validation
- Prediction error analysis
- Basic data drift detection
- Production-safe path handling
- Monitoring ML models after deployment

--------------------------------------------------

Day 27 Outcome:
Implemented monitoring logic to ensure deployed ML model
remains reliable and stable when exposed to data over time.

--------------------------------------------------

Project Status:
Day 1  – Day 26 : Completed
Day 27           : Completed (Model Monitoring)

--------------------------------------------------

Author:
Yash Desai
