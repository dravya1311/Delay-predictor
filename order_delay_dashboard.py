# order_delay_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import re

st.set_page_config(page_title="Order Delay Analysis", layout="wide")

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def norm_col(c):
    c = str(c).strip().lower()
    return re.sub(r'[^a-z0-9]+', '_', c)

def load_csv(path):
    try:
        if path.startswith("http"):
            return pd.read_csv(path)
        if os.path.exists(path):
            return pd.read_csv(path)
        return None
    except Exception:
        return None

def detect_delay_column(df):
    candidates = ["delay", "delay_status", "status", "delivery_status", "order_delay"]
    norm_map = {norm_col(c): c for c in df.columns}
    for c in candidates:
        if norm_col(c) in norm_map:
            return norm_map[norm_col(c)]
    return None

def normalize_delay(series):
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        try:
            iv = int(v)
            if iv in [-1, 0, 1]:
                out.append(iv)
                continue
        except:
            pass
        s = str(v).lower().strip()
        if "early" in s or s == "-1":
            out.append(-1)
        elif "delay" in s or "late" in s or s == "1":
            out.append(1)
        elif "on" in s and "time" in s:
            out.append(0)
        else:
            out.append(np.nan)
    return pd.Series(out)

# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------
st.sidebar.title("Load Data")

path = st.sidebar.text_input("Enter CSV path or GitHub raw link")
uploaded_file = st.sidebar.file_uploader("Or upload CSV", type=["csv"])

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif path:
    df = load_csv(path)

if df is None:
    st.warning("Upload a CSV or enter a valid file path.")
    st.stop()

df.columns = [norm_col(c) for c in df.columns]

# -------------------------------------------------------------------
# Delay column detection
# -------------------------------------------------------------------
delay_col = detect_delay_column(df)
if delay_col is None:
    st.error("No delay column found. Add a column like delay / delay_status / delivery_status.")
    st.stop()

df["delay_flag"] = normalize_delay(df[delay_col])

if df["delay_flag"].isna().all():
    st.error("Delay column found but values cannot be interpreted as -1/0/1.")
    st.stop()

# -------------------------------------------------------------------
# Required column checks (soft)
# -------------------------------------------------------------------
required_cols = [
    "order_region", "order_country", "category_name",
    "product_name", "sales", "profit", "quantity", "shipping_mode"
]

for c in required_cols:
    if c not in df.columns:
        st.warning(f"Column '{c}' missing. KPI depending on it may be skipped.")

# -------------------------------------------------------------------
# KPIs
# -------------------------------------------------------------------
st.header("Order Delay Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Early Deliveries %", round((df["delay_flag"]==-1).mean()*100, 2))
col2.metric("On-Time Deliveries %", round((df["delay_flag"]==0).mean()*100, 2))
col3.metric("Delayed Deliveries %", round((df["delay_flag"]==1).mean()*100, 2))

st.divider()

# -------------------------------------------------------------------
# 1) Average sales per customer based on order_region
# -------------------------------------------------------------------
if "order_region" in df.columns and "sales" in df.columns:
    st.subheader("1) Average Sales per Customer by Region")
    fig = px.bar(df.groupby("order_region")["sales"].mean().reset_index(),
                 x="order_region", y="sales", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# 2) Average Profit per order by order_region
# -------------------------------------------------------------------
if "order_region" in df.columns and "profit" in df.columns:
    st.subheader("2) Average Profit per Order by Region")
    fig = px.bar(df.groupby("order_region")["profit"].mean().reset_index(),
                 x="order_region", y="profit", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# 3) Top 5 order country and order region
# -------------------------------------------------------------------
if "order_country" in df.columns:
    st.subheader("3) Top 5 Countries by Order Count")
    fig = px.bar(df["order_country"].value_counts().head(5).reset_index(),
                 x="index", y="order_country")
    st.plotly_chart(fig, use_container_width=True)

if "order_region" in df.columns:
    st.subheader("Top Regions by Order Count")
    fig = px.bar(df["order_region"].value_counts().reset_index(),
                 x="index", y="order_region")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# 4) Top 8 most profitable category_name
# -------------------------------------------------------------------
if "category_name" in df.columns and "profit" in df.columns:
    st.subheader("4) Most Profitable Categories (Top 8)")
    fig = px.bar(df.groupby("category_name")["profit"].sum().nlargest(8).reset_index(),
                 x="category_name", y="profit")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# 5) Most profitable product for each region
# -------------------------------------------------------------------
if "order_region" in df.columns and "product_name" in df.columns and "profit" in df.columns:
    st.subheader("5) Most Profitable Product per Region")
    top_products = df.groupby(["order_region", "product_name"])["profit"].sum().reset_index()
    top_products = top_products.sort_values(["order_region", "profit"], ascending=[True, False])
    top_products = top_products.groupby("order_region").head(1)
    st.dataframe(top_products)

# -------------------------------------------------------------------
# 6) Top 5 most sold categories based on quantity & revenue
# -------------------------------------------------------------------
if "category_name" in df.columns and "quantity" in df.columns and "sales" in df.columns:
    st.subheader("6) Top 5 Categories by Quantity Sold")
    st.dataframe(df.groupby("category_name")["quantity"].sum().nlargest(5))

    st.subheader("Top 5 Categories by Revenue")
    st.dataframe(df.groupby("category_name")["sales"].sum().nlargest(5))

# -------------------------------------------------------------------
# 7) Preferred shipping mode by region
# -------------------------------------------------------------------
if "order_region" in df.columns and "shipping_mode" in df.columns:
    st.subheader("7) Preferred Shipping Mode by Region")
    mode = df.groupby(["order_region", "shipping_mode"]).size().reset_index(name="count")
    fig = px.bar(mode, x="order_region", y="count", color="shipping_mode", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# 8) Delayed orders by shipping mode
# -------------------------------------------------------------------
if "shipping_mode" in df.columns:
    st.subheader("8) Delayed Orders by Shipping Mode")
    delay_ship = df[df["delay_flag"]==1]["shipping_mode"].value_counts().reset_index()
    fig = px.bar(delay_ship, x="index", y="shipping_mode")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.success("Dashboard loaded successfully.")
