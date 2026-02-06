DAY 28 – PRODUCTION MODEL MONITORING & DRIFT DETECTION

Project Overview
This project implements production-level monitoring for a machine learning regression model.
It focuses on detecting silent model degradation after deployment using drift detection techniques.

This work follows:
Day 27 – Model Error Analysis
Day 28 – Monitoring & Drift Detection

--------------------------------------------------

Objectives
- Monitor incoming production data
- Detect data drift between training and current data
- Detect prediction drift without requiring target values
- Enforce feature schema consistency
- Log monitoring results for audit and retraining decisions

--------------------------------------------------

Project Structure

ML_Project/
│
├── data/
│   ├── reference_data.csv
│   └── current_data.csv
│
├── models/
│   └── model.pkl
│
├── monitoring/
│   ├── data_drift.py
│   ├── prediction_drift.py
│   └── run_monitoring.py
│
├── logs/
│   └── monitoring_log.txt
│
├── requirements.txt
└── README.md

--------------------------------------------------

Key Concepts Implemented

1. Data Drift Detection
- Uses Kolmogorov–Smirnov (KS) test
- Compares numerical feature distributions
- Detects changes between reference and current data

2. Prediction Drift Detection
- Compares model predictions on reference vs current data
- Does not require target values
- Suitable for real production environments

3. Schema Enforcement
- Aligns incoming data with training feature schema
- Adds missing features with default values
- Drops unexpected features
- Prevents model crashes due to feature mismatch

--------------------------------------------------

Why Target Is Not Used
In real production systems, ground truth labels are not immediately available.
Therefore:
- Error analysis is done offline (Day 27)
- Drift monitoring is used online (Day 28)

This approach reflects real-world MLOps practices.

--------------------------------------------------

How to Run

1. Install dependencies
pip install -r requirements.txt

2. Run monitoring pipeline
python monitoring/run_monitoring.py

--------------------------------------------------

Output

- Console message confirming successful execution
- Monitoring results appended to:
  logs/monitoring_log.txt

--------------------------------------------------

Production Value
This monitoring system helps:
- Detect silent model degradation
- Identify schema drift early
- Decide when retraining is required
- Maintain long-term model reliability

--------------------------------------------------
