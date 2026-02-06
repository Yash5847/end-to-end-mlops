import pandas as pd
from scipy.stats import ks_2samp

def detect_data_drift(ref_path, curr_path, threshold=0.05):
    ref = pd.read_csv(ref_path)
    curr = pd.read_csv(curr_path)

    drift = {}

    for col in ref.columns:
        if ref[col].dtype != "object":
            _, p_value = ks_2samp(ref[col], curr[col])
            drift[col] = p_value < threshold

    return drift
