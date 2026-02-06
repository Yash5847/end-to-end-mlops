import joblib
import pandas as pd
from scipy.stats import ks_2samp

def align_features(model, df):
    trained_features = model.feature_names_in_

    for col in trained_features:
        if col not in df.columns:
            df[col] = 0


    df = df[trained_features]

    return df

def detect_prediction_drift(model_path, X_ref, X_curr, threshold=0.05):
    model = joblib.load(model_path)

    X_ref_aligned = align_features(model, X_ref.copy())
    X_curr_aligned = align_features(model, X_curr.copy())

    ref_preds = model.predict(X_ref_aligned)
    curr_preds = model.predict(X_curr_aligned)

    _, p_value = ks_2samp(ref_preds, curr_preds)

    return p_value < threshold
