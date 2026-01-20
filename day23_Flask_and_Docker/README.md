ML Model Deployment using Flask & Docker

This project demonstrates deploying a trained Machine Learning model using Flask API and Docker.

Project Structure
ML_Project/
│
├── app/
│   └── app.py
│
├── models/
│   └── model.pkl
│
├── Dockerfile
├── requirements.txt
└── README.md

Description
A trained RandomForest Regression model is loaded from model.pkl
A Flask API exposes a /predict endpoint
Input is accepted as JSON with 10 features
The app is containerized using Docker

API Endpoints
Health Check
GET /health

Response:
{"status":"UP"}

Prediction
POST /predict

Expected JSON format:
{
  "features": {
    "f1": 1,
    "f2": 2,
    "f3": 3,
    "f4": 4,
    "f5": 5,
    "f6": 6,
    "f7": 7,
    "f8": 8,
    "f9": 9,
    "f10": 10
  }
}


Response:
{"prediction": 781.25}



Run Locally (Without Docker)
Install dependencies
pip install -r requirements.txt

Run the app
python app/app.py

API runs at
http://localhost:5000



Run with Docker
Build image
docker build -t mlops-day23 .

Run container
docker run -d -p 5000:5000 mlops-day23

API available at
http://localhost:5000

Technologies Used
Python
Flask
Scikit-learn
Pandas
Docker