from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
DATA_PATH = ROOT / "data" / "customer_churn.csv"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


model = load_model()
df = load_data()

st.title("Customer Churn Prediction Dashboard")
st.caption("Predict churn probability, identify high-risk customers, and support retention decisions.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(df):,}")
m2.metric("Overall churn", f"{df['churn'].eq('Yes').mean():.1%}")
m3.metric("Avg. monthly charge", f"${df['monthly_charges'].mean():.2f}")
m4.metric("Avg. tenure", f"{df['tenure_months'].mean():.1f} months")

single_tab, batch_tab, insights_tab = st.tabs(["Single customer", "Batch scoring", "Business insights"])

with single_tab:
    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure_months = st.slider("Tenure (months)", 0, 72, 12)
        satisfaction_score = st.slider("Satisfaction score", 1, 5, 3)
    with c2:
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
    with c3:
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        contract_type = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    c4, c5, c6, c7 = st.columns(4)
    monthly_charges = c4.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=75.0, step=1.0)
    total_charges = c5.number_input("Total charges ($)", min_value=0.0, max_value=15000.0, value=900.0, step=10.0)
    num_support_tickets = c6.slider("Support tickets", 0, 12, 2)
    avg_monthly_gb = c7.number_input("Avg monthly data (GB)", min_value=0.0, max_value=500.0, value=100.0, step=5.0)

    row = pd.DataFrame(
        [{
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": num_support_tickets,
            "avg_monthly_gb": avg_monthly_gb,
            "satisfaction_score": satisfaction_score,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract_type": contract_type,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
        }]
    )

    if st.button("Predict churn risk", type="primary"):
        probability = float(model.predict_proba(row)[0, 1])
        band = risk_band(probability)
        p1, p2 = st.columns(2)
        p1.metric("Churn probability", f"{probability:.1%}")
        p2.metric("Risk band", band)
        st.progress(min(max(probability, 0.0), 1.0))
        if band == "High":
            st.error("High-risk customer — prioritize proactive retention outreach.")
        elif band == "Medium":
            st.warning("Medium-risk customer — consider targeted retention offers.")
        else:
            st.success("Low-risk customer — standard engagement is likely sufficient.")

with batch_tab:
    st.subheader("Score a customer file")
    st.write("Upload a CSV containing the same input columns used by the model. The app will append churn probability and risk band.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        try:
            probs = model.predict_proba(batch)[:, 1]
            scored = batch.copy()
            scored["churn_probability"] = probs.round(4)
            scored["predicted_churn"] = ["Yes" if p >= 0.5 else "No" for p in probs]
            scored["risk_band"] = [risk_band(float(p)) for p in probs]
            st.dataframe(scored.head(100), use_container_width=True)
            buffer = BytesIO()
            buffer.write(scored.to_csv(index=False).encode("utf-8"))
            st.download_button("Download scored CSV", buffer.getvalue(), "churn_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")

with insights_tab:
    st.subheader("Dataset insights")
    left, right = st.columns(2)
    contract = df.assign(churn_flag=df["churn"].eq("Yes").astype(int)).groupby("contract_type")["churn_flag"].mean().sort_values(ascending=False)
    left.markdown("**Churn rate by contract**")
    left.bar_chart(contract)

    satisfaction = df.assign(churn_flag=df["churn"].eq("Yes").astype(int)).groupby("satisfaction_score")["churn_flag"].mean()
    right.markdown("**Churn rate by satisfaction score**")
    right.line_chart(satisfaction)

    st.markdown("**Top customer segments by churn rate**")
    segment = (
        df.assign(churn_flag=df["churn"].eq("Yes").astype(int))
        .groupby(["contract_type", "internet_service"], observed=False)
        .agg(customers=("customer_id", "count"), churn_rate=("churn_flag", "mean"))
        .query("customers >= 50")
        .sort_values("churn_rate", ascending=False)
        .reset_index()
    )
    segment["churn_rate"] = segment["churn_rate"].map(lambda x: f"{x:.1%}")
    st.dataframe(segment, use_container_width=True)
