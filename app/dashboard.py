import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Credit Risk Assessor", layout="wide")

if "prob" not in st.session_state:
    st.session_state.prob = None
    st.session_state.decision = None

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
    st.session_state.prob = prob
    st.session_state.decision = decision

    if decision == "APPROVE":
        st.success(f"### ✅ APPROVE — Default Probability: {prob:.1%}")
    else:
        st.error(f"### ❌ REJECT — Default Probability: {prob:.1%}")

    st.subheader("Top Risk Factors")
    for factor in result["top_risk_factors"]:
        direction = "🔺 increases risk" if factor["impact"] > 0 else "🔻 decreases risk"
        st.write(f"- **{factor['feature']}**: {direction} (impact: {factor['impact']:.3f})")
        st.divider()
st.header("📉 Economic Downturn Simulator")
st.caption("Stress-test this applicant against adverse economic scenarios")

salary_cut = st.slider("Simulate Salary Cut (%)", 0, 70, 0)
inflation_shock = st.slider("Simulate Essential Expense Inflation (%)", 0, 70, 0)

if st.button("Run Stress Test"):
    if st.session_state.prob is None:
        st.warning("Please click 'Assess Risk' first to establish a baseline.")
    else:
        stressed_income = income * (1 - salary_cut / 100)
        stressed_annuity = annuity * (1 + inflation_shock / 100)

        stressed_payload = {
            "AMT_INCOME_TOTAL": stressed_income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": stressed_annuity,
            "DAYS_BIRTH": -age * 365,
            "DAYS_EMPLOYED": -years_employed * 365,
            "CNT_FAM_MEMBERS": family_members,
            "EXT_SOURCE_1": ext_score_1,
            "EXT_SOURCE_2": ext_score_2,
            "EXT_SOURCE_3": ext_score_3,
        }
        
        stressed_response = requests.post(f"{API_URL}/predict-default", json=stressed_payload)
        stressed_result = stressed_response.json()

        stressed_prob = stressed_result["default_probability"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Original Risk", f"{st.session_state.prob:.1%}")
        with col_b:
            st.metric("Stressed Risk", f"{stressed_prob:.1%}", delta=f"{(stressed_prob - st.session_state.prob):.1%}")

        if stressed_result["decision"] != st.session_state.decision:
            st.warning(f"⚠️ Decision FLIPS from {st.session_state.decision} to {stressed_result['decision']} under this scenario.")