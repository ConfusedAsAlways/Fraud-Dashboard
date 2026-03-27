import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# PAGE CONFIG + UI STYLE
# -----------------------------
st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    layout="wide",
    page_icon="🚨"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #ffffff; }
    .metric-container {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 Fraud Intelligence Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\maxki\Downloads\bank_transactions_data_2.csv")

df = load_data()

numeric_cols = [
    'TransactionAmount',
    'AccountBalance',
    'TransactionDuration',
    'LoginAttempts',
    'CustomerAge'
]

# -----------------------------
# MODEL PIPELINE
# -----------------------------
@st.cache_data
def compute_scores(df):
    df = df.copy()

    iso = IsolationForest(contamination=0.02, random_state=42)
    iso.fit(df[numeric_cols])

    df['iso_score'] = -iso.score_samples(df[numeric_cols])
    df['iso_score'] = MinMaxScaler().fit_transform(df[['iso_score']])

    df['iqr_score'] = 0
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df['iqr_score'] += ((df[col] < lower) | (df[col] > upper)).astype(int)

    df['iqr_score'] /= len(numeric_cols)

    df['fraud_confidence'] = (
        0.6 * df['iso_score'] +
        0.4 * df['iqr_score']
    )

    df['fraud_flag'] = df['fraud_confidence'] > df['fraud_confidence'].quantile(0.95)

    df['risk_level'] = pd.cut(
        df['fraud_confidence'],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

    return df

df = compute_scores(df)

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Controls")

threshold = st.sidebar.slider(
    "Fraud Sensitivity Threshold",
    0.0, 1.0, 0.75
)

min_amount = st.sidebar.number_input(
    "Minimum Transaction Amount",
    value=0
)

df = df[df["TransactionAmount"] >= min_amount]
df["dynamic_flag"] = df["fraud_confidence"] > threshold

filtered = df[df["dynamic_flag"]]

# -----------------------------
# KPI CARDS
# -----------------------------
st.markdown("## 📊 System Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", len(df))
col2.metric("Flagged Transactions", int(filtered.shape[0]))
col3.metric("Avg Fraud Score", round(df["fraud_confidence"].mean(), 3))
col4.metric("Max Risk Score", round(df["fraud_confidence"].max(), 3))

st.markdown("---")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Risk Overview",
    "📈 Behavior Patterns",
    "🚨 Fraud Investigation"
])

# -----------------------------
# TAB 1 - RISK OVERVIEW
# -----------------------------
with tab1:

    col1, col2 = st.columns(2)

    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]

    fig1 = px.bar(
        risk_counts,
        x="risk_level",
        y="count",
        color="risk_level",
        title="Risk Level Distribution"
    )

    fig2 = px.histogram(
        df,
        x="fraud_confidence",
        nbins=40,
        title="Fraud Score Distribution",
        color="risk_level"
    )

    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# TAB 2 - PATTERNS
# -----------------------------
with tab2:

    col1, col2 = st.columns(2)

    fig1 = px.scatter(
        df,
        x="TransactionAmount",
        y="TransactionDuration",
        color="fraud_confidence",
        size="AccountBalance",
        title="Amount vs Duration"
    )

    fig2 = px.scatter(
        df,
        x="TransactionAmount",
        y="AccountBalance",
        color="fraud_confidence",
        size="TransactionDuration",
        title="Amount vs Balance"
    )

    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# TAB 3 - FRAUD INVESTIGATION
# -----------------------------
with tab3:

    st.markdown("### 🚨 High-Risk Transactions")

    top_fraud = filtered.sort_values(
        "fraud_confidence",
        ascending=False
    )

    st.dataframe(
        top_fraud.style.background_gradient(
            subset=["fraud_confidence"],
            cmap="Reds"
        ),
        use_container_width=True
    )

    st.markdown("### 🧠 Risk Cluster View")

    fig = px.scatter(
        top_fraud,
        x="TransactionDuration",
        y="CustomerAge",
        size="TransactionAmount",
        color="fraud_confidence",
        title="Fraud Behavior Clusters",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("🚨 Fraud Intelligence System | ML-powered risk scoring dashboard")