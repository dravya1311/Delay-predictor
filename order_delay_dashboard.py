# order_delay_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Order Delay Dashboard", layout="wide")

# -------------------------------------------------------------
# AUTO LOAD CSV (No upload, no user input)
# -------------------------------------------------------------
CSV_URL = "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/Delay_Model.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_URL)
    except Exception as e:
        st.error(f"Failed to load CSV from GitHub: {e}")
        return None
    return df

df = load_data()
if df is None:
    st.stop()

# -------------------------------------------------------------
# Clean column names
# -------------------------------------------------------------
def norm(c):
    return re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower())

df.columns = [norm(c) for c in df.columns]

# -------------------------------------------------------------
# Delay mapping
# -------------------------------------------------------------
if "delay_status" not in df.columns:
    st.error("delay_status column missing.")
    st.stop()

def map_delay(x):
    try:
        x = int(x)
        if x in [-1, 0, 1]:
            return x
    except:
        pass
    return np.nan

df["delay_flag"] = df["delay_status"].apply(map_delay)

# -------------------------------------------------------------
# KPIs
# -------------------------------------------------------------
st.header("Order Delay Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Early %", round((df["delay_flag"]==-1).mean()*100, 2))
col2.metric("On-Time %", round((df["delay_flag"]==0).mean()*100, 2))
col3.metric("Delayed %", round((df["delay_flag"]==1).mean()*100, 2))

st.divider()

# -------------------------------------------------------------
# 1) Average sales per customer by order_region
# -------------------------------------------------------------
if "order_region" in df.columns and "sales" in df.columns:
    st.subheader("1) Average Sales per Customer by Region")
    fig = px.bar(df.groupby("order_region")["sales"].mean().reset_index(),
                 x="order_region", y="sales", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 2) Average Profit per order by order_region
# -------------------------------------------------------------
if "order_region" in df.columns and "profit" in df.columns:
    st.subheader("2) Average Profit per Order by Region")
    fig = px.bar(df.groupby("order_region")["profit"].mean().reset_index(),
                 x="order_region", y="profit", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 3) Top 5 order countries and regions (marketwise)
# -------------------------------------------------------------
if "order_country" in df.columns:
    st.subheader("3) Top 5 Countries by Order Count")
    fig = px.bar(df["order_country"].value_counts().head(5).reset_index(),
                 x="index", y="order_country", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

if "order_region" in df.columns:
    st.subheader("Top 5 Regions by Order Count")
    fig = px.bar(df["order_region"].value_counts().head(5).reset_index(),
                 x="index", y="order_region", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 4) Top 8 most profitable category_name
# -------------------------------------------------------------
if "category_name" in df.columns and "profit" in df.columns:
    st.subheader("4) Top 8 Most Profitable Categories")
    fig = px.bar(df.groupby("category_name")["profit"].sum().nlargest(8).reset_index(),
                 x="category_name", y="profit", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 5) Most profitable product for each region
# -------------------------------------------------------------
if "order_region" in df.columns and "product_name" in df.columns and "profit" in df.columns:
    st.subheader("5) Most Profitable Product by Region")
    top_prof_prod = df.groupby(["order_region", "product_name"])["profit"].sum()
    top_prof_prod = top_prof_prod.reset_index()
    top_prof_prod = top_prof_prod.loc[top_prof_prod.groupby("order_region")["profit"].idxmax()]

    fig = px.bar(top_prof_prod, x="order_region", y="profit", color="product_name", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 6) Top 5 most sold categories (quantity + revenue)
# -------------------------------------------------------------
if "category_name" in df.columns and "quantity" in df.columns and "sales" in df.columns:
    st.subheader("6) Top 5 Most Sold Categories")
    
    qty_top = df.groupby("category_name")["quantity"].sum().nlargest(5).reset_index()
    sales_top = df.groupby("category_name")["sales"].sum().nlargest(5).reset_index()

    colq, cols = st.columns(2)

    with colq:
        st.write("Top 5 by Quantity")
        fig = px.bar(qty_top, x="category_name", y="quantity", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    with cols:
        st.write("Top 5 by Revenue")
        fig = px.bar(sales_top, x="category_name", y="sales", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 7) Preferred shipping mode by region
# -------------------------------------------------------------
if "shipping_mode" in df.columns and "order_region" in df.columns:
    st.subheader("7) Preferred Shipping Mode by Region")
    fig = px.histogram(df, x="order_region", color="shipping_mode", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 8) Delayed orders by shipping mode
# -------------------------------------------------------------
if "shipping_mode" in df.columns:
    st.subheader("8) Delay % by Shipping Mode")
    delay_rate = df.groupby("shipping_mode")["delay_flag"].apply(lambda x: (x==1).mean()*100).reset_index()
    fig = px.bar(delay_rate, x="shipping_mode", y="delay_flag", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
