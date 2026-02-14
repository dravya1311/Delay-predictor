# -------------------------------------------------------------
# SUPPLY CHAIN & LOGISTICS PERFORMANCE DASHBOARD
# FINAL STABLE VERSION
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain & Logistics Performance Dashboard",
    layout="wide"
)

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.markdown("""
<div style="background-color:#0A1A2F;padding:22px;border-radius:8px">
<h1 style="color:white;text-align:center;">Supply Chain & Logistics Performance Dashboard</h1>
<h4 style="color:#A3C4F3;text-align:center;">Diagnostic Insights for Delivery Performance</h4>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
def load_data():

    local_files = ["Delay_Model.csv", "Delay Model.csv"]

    for f in local_files:
        if os.path.exists(f):
            return pd.read_csv(f)

    github = "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/Delay_Model.csv"
    return pd.read_csv(github)


df = load_data()

# -------------------------------------------------------------
# COLUMN NORMALIZATION (CRITICAL FIX)
# -------------------------------------------------------------
df.columns = df.columns.str.strip().str.lower()

# required columns
required = [
    "label",
    "shipping mode",
    "order region",
    "order country",
    "order city",
    "customer country",
    "customer city",
    "sales per customer",
    "profit per order",
    "category name",
    "product name"
]

missing = [c for c in required if c not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# delay flag
df["is_delayed"] = df["label"] == -1

# -------------------------------------------------------------
# FILTER
# -------------------------------------------------------------
regions = ["All"] + sorted(df["order region"].dropna().unique())
sel_region = st.sidebar.selectbox("Filter by Order Region", regions)

df_view = df if sel_region == "All" else df[df["order region"] == sel_region]

# -------------------------------------------------------------
# KPI CARDS
# -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

total_orders = len(df_view)
delayed_orders = df_view["is_delayed"].sum()
delay_rate = (delayed_orders / total_orders * 100) if total_orders else 0
avg_sales = df_view["sales per customer"].mean()

c1.metric("Total Orders", f"{total_orders:,}")
c2.metric("Delayed Orders", f"{delayed_orders:,}")
c3.metric("Delay %", f"{delay_rate:.1f}%")
c4.metric("Avg Sales / Customer", f"${avg_sales:.2f}")

st.markdown("---")

# -------------------------------------------------------------
# ORDER STATUS PIE
# -------------------------------------------------------------
status_pct = df_view["label"].value_counts(normalize=True) * 100
status_pct = status_pct.reset_index()
status_pct.columns = ["status", "pct"]

fig = px.pie(status_pct, names="status", values="pct", hole=0.5)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# SALES BY REGION
# -------------------------------------------------------------
grp = df_view.groupby("order region")["sales per customer"].mean().reset_index()

fig = px.bar(grp, x="order region", y="sales per customer",
             text=grp["sales per customer"].round(2),
             color="sales per customer")

fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# PROFIT BY REGION
# -------------------------------------------------------------
grp = df_view.groupby("order region")["profit per order"].mean().reset_index()

fig = px.bar(grp, x="order region", y="profit per order",
             text=grp["profit per order"].round(2),
             color="profit per order")

fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# TOP PROFITABLE CATEGORIES
# -------------------------------------------------------------
cat = (
    df_view.groupby("category name")["profit per order"]
    .mean().reset_index()
    .sort_values("profit per order", ascending=False)
    .head(8)
)

fig = px.bar(cat, x="category name", y="profit per order",
             text=cat["profit per order"].round(2),
             color="profit per order")

fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# DELAY % BY SHIPPING MODE
# -------------------------------------------------------------
total_mode = df_view.groupby("shipping mode").size().reset_index(name="total")
delay_mode = df_view[df_view["is_delayed"]].groupby("shipping mode").size().reset_index(name="delay")

mode = total_mode.merge(delay_mode, on="shipping mode", how="left").fillna(0)
mode["delay_pct"] = (mode["delay"] / mode["total"]) * 100

fig = px.bar(mode, x="shipping mode", y="delay_pct",
             text=mode["delay_pct"].round(2),
             color="delay_pct")

fig.update_traces(textposition="outside", texttemplate="%{text:.2f}%")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# TOP 5 MOST DELAYED PRODUCTS
# -------------------------------------------------------------
prod = (
    df_view.groupby("product name")["label"]
    .mean().reset_index()
    .nsmallest(5, "label")
)

fig = px.bar(prod, x="label", y="product name", orientation="h",
             text=prod["label"].round(2), color="label")

fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# TOP 10 MOST DELAYED ROUTES
# -------------------------------------------------------------
df_view["route"] = (
    df_view["order city"].astype(str) + " → " +
    df_view["customer city"].astype(str)
)

routes = (
    df_view.groupby("route")["label"]
    .mean().reset_index()
    .nsmallest(10, "label")
)

fig = px.bar(routes, x="label", y="route", orientation="h",
             text=routes["label"].round(2), color="label")

fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="position:fixed;bottom:10px;right:15px;color:#A3C4F3;">
Developed by <b style="color:#0B6EFD;">Ravindra Yadav</b>
</div>
""", unsafe_allow_html=True)
