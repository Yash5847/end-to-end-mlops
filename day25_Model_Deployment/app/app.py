from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

fat_content_map = {
    "Low Fat": 0,
    "Regular": 1
}

item_type_map = {
    "Dairy": 0,
    "Soft Drinks": 1,
    "Meat": 2,
    "Fruits and Vegetables": 3,
    "Household": 4,
    "Baking Goods": 5,
    "Snack Foods": 6,
    "Frozen Foods": 7,
    "Breakfast": 8,
    "Health and Hygiene": 9,
    "Hard Drinks": 10,
    "Canned": 11,
    "Breads": 12,
    "Starchy Foods": 13,
    "Others": 14,
    "Seafood": 15
}

outlet_size_map = {
    "Small": 0,
    "Medium": 1,
    "High": 2
}

outlet_location_map = {
    "Tier 1": 0,
    "Tier 2": 1,
    "Tier 3": 2
}

outlet_type_map = {
    "Grocery Store": 0,
    "Supermarket Type1": 1,
    "Supermarket Type2": 2,
    "Supermarket Type3": 3
}


@app.route("/")
def home():
    return "ML Project API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        input_data = [[
            float(data["Item_Weight"]),
            fat_content_map[data["Item_Fat_Content"]],
            float(data["Item_Visibility"]),
            item_type_map[data["Item_Type"]],
            float(data["Item_MRP"]),
            int(data["Outlet_Identifier"].replace("OUT", "")),
            outlet_size_map[data["Outlet_Size"]],
            outlet_location_map[data["Outlet_Location_Type"]],
            outlet_type_map[data["Outlet_Type"]],
            int(data["Outlet_Years"])
        ]]

        prediction = model.predict(input_data)[0]

        return jsonify({
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
