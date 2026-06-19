import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Credit Risk Assessor", layout="wide")

st.title("Behavioral Credit Risk & Lifestyle Default Assessor")
st.caption("Real-time loan risk scoring with explainable AI")

API_URL = "http://127.0.0.1:8000"

st.header("Applicant Profile")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income (₹)", min_value=50000, value=200000, step=10000)
    credit = st.number_input("Loan Amount (₹)", min_value=50000, value=500000, step=10000)
    annuity = st.number_input("Annual EMI / Annuity (₹)", min_value=5000, value=25000, step=1000)
    family_members = st.number_input("Family Members", min_value=1, value=2, step=1)

with col2:
    age = st.slider("Age", min_value=18, max_value=70, value=35)
    years_employed = st.slider("Years Employed", min_value=0, max_value=40, value=5)
    ext_score_1 = st.slider("External Credit Score 1", 0.0, 1.0, 0.5)
    ext_score_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.5)
    ext_score_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.5)

if st.button("Assess Risk", type="primary"):
    payload = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "DAYS_BIRTH": -age * 365,
        "DAYS_EMPLOYED": -years_employed * 365,
        "CNT_FAM_MEMBERS": family_members,
        "EXT_SOURCE_1": ext_score_1,
        "EXT_SOURCE_2": ext_score_2,
        "EXT_SOURCE_3": ext_score_3,
    }

    response = requests.post(f"{API_URL}/predict-default", json=payload)
    result = response.json()

    st.divider()

    prob = result["default_probability"]
    decision = result["decision"]

    if decision == "APPROVE":
        st.success(f"### ✅ APPROVE — Default Probability: {prob:.1%}")
    else:
        st.error(f"### ❌ REJECT — Default Probability: {prob:.1%}")

    st.subheader("Top Risk Factors")
    for factor in result["top_risk_factors"]:
        direction = "🔺 increases risk" if factor["impact"] > 0 else "🔻 decreases risk"
        st.write(f"- **{factor['feature']}**: {direction} (impact: {factor['impact']:.3f})")