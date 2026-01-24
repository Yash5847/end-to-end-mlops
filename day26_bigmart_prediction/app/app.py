from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder="../templates")

model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    features = [
        data['Item_Weight'],
        data['Item_Fat_Content'],
        data['Item_Visibility'],
        data['Item_Type'],
        data['Item_MRP'],
        data['Outlet_Identifier'],
        data['Outlet_Size'],
        data['Outlet_Location_Type'],
        data['Outlet_Type'],
        data['Outlet_Years']
    ]

    input_data = np.array(features, dtype=float).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    output = round(prediction, 2)

    return render_template('index.html', prediction_text=f"Predicted Sales: {output}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
