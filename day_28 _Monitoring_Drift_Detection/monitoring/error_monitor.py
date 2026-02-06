import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def monitor_errors(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm.tolist()
    }
