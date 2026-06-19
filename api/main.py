from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
import shap

from api.features import engineer_features

app = FastAPI(title="Behavioral Credit Risk API")

# load model + imputer once, when the server starts — not on every request
with open('models/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

explainer = shap.TreeExplainer(model)

FEATURES = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'emi_to_income', 'credit_to_income',
    'age_years', 'employment_ratio',
    'income_per_person',
]

BEST_THRESHOLD = 0.58  # from your cost curve analysis


class Applicant(BaseModel):
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: float = Field(..., gt=0)
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    CNT_FAM_MEMBERS: float = Field(..., gt=0)
    EXT_SOURCE_1: float | None = None
    EXT_SOURCE_2: float | None = None
    EXT_SOURCE_3: float | None = None


@app.get("/")
def home():
    return {"status": "alive", "message": "Credit risk API is running"}


@app.post("/predict-default")
def predict_default(applicant: Applicant):
    raw = pd.DataFrame([applicant.dict()])
    engineered = engineer_features(raw)

    X = engineered[FEATURES]
    X_imputed = imputer.transform(X)

    probability = model.predict_proba(X_imputed)[0][1]
    decision = "REJECT" if probability >= BEST_THRESHOLD else "APPROVE"

    # SHAP values for this one applicant — how much each feature pushed risk up/down
    shap_values = explainer.shap_values(X_imputed)[0]
    feature_impact = list(zip(FEATURES, shap_values))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    top_factors = [
        {"feature": name, "impact": round(float(val), 4)}
        for name, val in feature_impact[:3]
    ]

    return {
        "default_probability": round(float(probability), 4),
        "decision": decision,
        "threshold_used": BEST_THRESHOLD,
        "top_risk_factors": top_factors
    }
