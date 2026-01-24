PROJECT NAME: BigMart Sales Prediction

DAY 26 - MODEL DEPLOYMENT (Flask + UI + CSS)

1. PROJECT STRUCTURE:
ML_Project
│
├── app
│   └── app.py
├── models
│   └── model.pkl
├── templates
│   └── index.html
├── static
│   └── style.css
├── requirements.txt

2. OBJECTIVE:
Deploy the trained BigMart Sales prediction model using Flask. Create a UI for user input and show predicted sales.

3. TECHNOLOGIES USED:
Python, Flask, HTML, CSS, Bootstrap, Pickle

4. STEPS:

STEP 1: CREATE MODEL FILE
- Save trained model as model.pkl inside models folder.

STEP 2: CREATE FLASK APP
- app/app.py contains Flask routes and prediction code.

STEP 3: CREATE UI
- templates/index.html contains HTML form for user input.

STEP 4: ADD CSS
- static/style.css contains custom CSS for better UI.

STEP 5: RUN THE APP
- Install requirements:
  pip install -r requirements.txt
- Run Flask app:
  python app/app.py
- Open:
  http://127.0.0.1:5000/

5. PREDICTION:
- Enter 10 features in the form and click Predict.
- Predicted Sales will display on the same page.

6. IMAGE:
![alt text](<Screenshot 2026-01-23 104709.png>)

7. NOTES:
- Model expects 10 input features:
  Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP,
  Outlet_Identifier, Outlet_Size, Outlet_Location_Type, Outlet_Type, Outlet_Years
