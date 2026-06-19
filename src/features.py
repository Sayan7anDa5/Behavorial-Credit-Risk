# src/features.py  ← create this file in VSCodium
import numpy as np
import pandas as pd
print("numpy:", np.__version__)
print("pandas:", pd.__version__)

def engineer_features(df):
    """Takes raw Home Credit rows, returns clean behavioral features.
    Used in BOTH training and the API — never change one without the other."""
    d = df.copy()

    # clean monsters
    d['DAYS_EMPLOYED'] = d['DAYS_EMPLOYED'].replace(365243, np.nan)

    # behavioral features
    d['emi_to_income']     = d['AMT_ANNUITY'] / d['AMT_INCOME_TOTAL']
    d['credit_to_income']  = d['AMT_CREDIT'] / d['AMT_INCOME_TOTAL']
    d['age_years']         = -d['DAYS_BIRTH'] / 365
    d['employment_ratio']  = d['DAYS_EMPLOYED'] / d['DAYS_BIRTH']
    d['income_per_person'] = d['AMT_INCOME_TOTAL'] / d['CNT_FAM_MEMBERS']

    return d