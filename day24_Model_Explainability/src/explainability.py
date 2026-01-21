import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "BigMartSales.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

TARGET_COLUMN = "Item_Outlet_Sales" 

data = pd.read_csv(DATA_PATH)

X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

model = joblib.load(MODEL_PATH)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:\n")
print(feature_importance)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, show=False)
plt.show()


shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X.iloc[0],
    matplotlib=True
)
