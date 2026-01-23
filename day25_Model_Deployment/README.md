BigMart Sales Prediction - ML Project

Project Overview
This project demonstrates an end-to-end Machine Learning deployment using Flask.
A trained regression model predicts product sales based on item and outlet details
and exposes the prediction through a REST API.

The project covers model loading, feature handling, API creation, and local testing
using PowerShell curl commands.

--------------------------------------------------

Model Details
Algorithm: Regression (Scikit-learn)
Input Features:
1. Item_Weight
2. Item_Fat_Content
3. Item_Visibility
4. Item_Type
5. Item_MRP
6. Outlet_Identifier
7. Outlet_Size
8. Outlet_Location_Type
9. Outlet_Type
10. Outlet_Years

Output:
Predicted Sales Value

--------------------------------------------------

Project Structure

ML_Project/
|
|-- app/
|   |-- app.py        Flask application
|
|-- models/
|   |-- model.pkl     Trained ML model
|
|-- data.json         Sample input request
|-- test_api.py       Model feature inspection script
|-- README.md         Project documentation

--------------------------------------------------

Setup Instructions

1. (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate

2. Install required libraries
pip install flask numpy pandas scikit-learn

3. Run Flask application
python app/app.py

Server starts at:
http://127.0.0.1:5000

--------------------------------------------------

API Usage

Endpoint:
POST /predict

Headers:
Content-Type: application/json

--------------------------------------------------

Sample Request (data.json)

{
  "Item_Weight": 9.3,
  "Item_Fat_Content": "Low Fat",
  "Item_Visibility": 0.016,
  "Item_Type": "Dairy",
  "Item_MRP": 249.8,
  "Outlet_Identifier": "OUT049",
  "Outlet_Size": "Medium",
  "Outlet_Location_Type": "Tier 1",
  "Outlet_Type": "Supermarket Type1",
  "Outlet_Years": 14
}

--------------------------------------------------

API Call (PowerShell)

curl.exe --% -X POST http://127.0.0.1:5000/predict `
-H "Content-Type: application/json" `
-d @data.json

--------------------------------------------------

Sample Response

{
  "prediction": 4220.998306567135
}

--------------------------------------------------

Technologies Used
Python
Flask
Scikit-learn
Pandas
NumPy
Pickle

--------------------------------------------------

Key Learnings
Handled categorical feature encoding during inference
Resolved feature mismatch issues during deployment
Built and tested REST APIs for ML models
Debugged curl usage in Windows PowerShell

--------------------------------------------------

Future Improvements
Save preprocessing and model as a single pipeline
Add frontend UI for predictions
Deploy application on cloud
Improve validation and logging

--------------------------------------------------