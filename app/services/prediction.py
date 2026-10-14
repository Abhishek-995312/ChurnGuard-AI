from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ml" / "artifacts" / "churn_pipeline_model.joblib"

model = joblib.load(MODEL_PATH)


def predict_churn(customer_data: dict) -> dict:
    customer_df = pd.DataFrame([customer_data])

    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = "Churn" if probability >= 0.5 else "No Churn"

    if probability >= 0.70:
        risk_level = "High"
    elif probability >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
    }