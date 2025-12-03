# -------------------------------------------------------------
# order_delay_dashboard.py  (FINAL FULL VERSION)
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, re

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain & Logistics Performance Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.markdown(
    """
    <div style="background-color:#0A1A2F; padding:25px; border-radius:8px;">
        <h1 style="color:white; text-align:center; font-size:42px; margin-bottom:5px;">
            Order Delay Intelligence Dashboard
        </h1>
        <h3 style="color:#A3C4F3; text-align:center; font-weight:300; margin-top:0;">
            Predictive & Diagnostic Insights for E-commerce Delivery Performance
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)
st.write(" ")

ACCENT = "#0B6EFD"
ALERT  = "#E03E3E"
GOOD   = "#2CB67D"
NEUTRAL = "#6C757D"

# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def norm_col(c: str) -> str:
    return re.sub(r'[^0-9a-z]+', '_', str(c).strip().lower())

def try_load():
    local = ["Delay_Model.csv", "Delay Model.csv", "Delay-Model.csv", "DelayModel.csv"]
    for p in local:
        if os.path.exists(p):
            try: return pd.read_csv(p)
            except: pass

    base = "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/"
    gh = ["Delay_Model.csv", "Delay%20Model.csv", "Delay-Model.csv", "DelayModel.csv"]
    for f in gh:
        try: return pd.read_csv(base + f)
        except: continue

    return None

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
df_raw = try_load()
if df_raw is None:
    st.error("Could not load Delay_Model.csv. Ensure file exists at repo root.")
    st.stop()

df = df_raw.copy()
df.columns = [norm_col(c) for c in df.columns]

# Validate required columns
required = [
    "label", "shipping_mode", "order_region", "order_country",
    "sales_per_customer", "profit_per_order",
    "category_name", "product_name", "order_item_quantity"
]

missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Delay flag
# -1 = delayed, 0 = on-time, 1 = early
df["is_delayed"] = df["label"] == -1

# -------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------
st.sidebar.header("Filters")

region_list = ["All"] + sorted(df["order_region"].dropna().unique())
sel_region = st.sidebar.selectbox("Order Region", region_list)

df_view = df if sel_region == "All" else df[df["order_region"] == sel_region]

# -------------------------------------------------------------
# KPI CARDS
# -------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_orders = len(df_view)
delayed_orders = df_view["is_delayed"].sum()
delay_rate = (delayed_orders / total_orders * 100) if total_orders else 0
avg_sales = df_view["sales_per_customer"].mean()

with col1:
    st.metric("Total Orders", f"{total_orders:,}")
with col2:
    st.metric("Delayed Orders", f"{delayed_orders:,}")
with col3:
    st.metric("Delay Rate", f"{delay_rate:.1f}%")
with col4:
    st.metric("Avg Sales per Customer", f"{avg_sales:.2f}")

st.markdown("---")

# -------------------------------------------------------------
# 1. Delayed Orders by Region
# -------------------------------------------------------------
st.subheader("Delayed Orders in Count by Region")

reg_grp = df_view.groupby("order_region")["is_delayed"].sum().reset_index()

if not reg_grp.empty:
    fig = px.bar(
        reg_grp,
        x="order_region",
        y="is_delayed",
        title="Delayed Orders by Region",
        text="is_delayed",
        color="is_delayed",
        color_continuous_scale="Reds"
    )
    fig.update_traces(textposition="outside", texttemplate="%{text:,}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available.")

# -------------------------------------------------------------
# 2. Average Sales per Customer — Region
# -------------------------------------------------------------
st.subheader("Average Sales per Customer by Region")

grp = df_view.groupby("order_region")["sales_per_customer"].mean().reset_index()

fig = px.bar(
    grp, x="order_region", y="sales_per_customer",
    title="Average Sales per Customer by Region",
    text="sales_per_customer", color="sales_per_customer"
)
fig.update_traces(textposition="outside", texttemplate="%{text:.2f}")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 3. Average Profit per Order — Region
# -------------------------------------------------------------
st.subheader("Average Profit per Order by Region")

grp = df_view.groupby("order_region")["profit_per_order"].mean().reset_index()

fig = px.bar(
    grp, x="order_region", y="profit_per_order",
    title="Average Profit per Order by Region",
    text="profit_per_order", color="profit_per_order"
)
fig.update_traces(textposition="outside", texttemplate="%{text:.2f}")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 4. Top 5 Order Country (by count)
# -------------------------------------------------------------
st.subheader("Top 5 Order Countries by No. of orders")

top_country = (
    df_view.groupby("order_country").size()
    .reset_index(name="orders")
    .sort_values("orders", ascending=False)
    .head(5)
)

fig = px.bar(
    top_country,
    x="order_country",
    y="orders",
    title="Top 5 Order Countries",
    text="orders",
    color="orders"
)
fig.update_traces(textposition="outside", texttemplate="%{text:,}")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 5. Top 8 Most Profitable Categories
# -------------------------------------------------------------
st.subheader("Top 8 Most Profitable Categories")

cat_profit = (
    df_view.groupby("category_name")["profit_per_order"]
    .mean().reset_index().sort_values("profit_per_order", ascending=False).head(8)
)

fig = px.bar(
    cat_profit, x="category_name", y="profit_per_order",
    title="Top 8 Most Profitable Categories",
    text="profit_per_order", color="profit_per_order"
)
fig.update_traces(textposition="outside", texttemplate="%{text:.2f}")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 6. Most Profitable Product — Region-wise
# -------------------------------------------------------------
st.subheader("Most Profitable Product by Region")

prod_region = (
    df_view.groupby(["order_region", "product_name"])["profit_per_order"]
    .mean().reset_index()
)

max_prod = prod_region.loc[prod_region.groupby("order_region")["profit_per_order"].idxmax()]

fig = px.bar(
    max_prod, x="order_region", y="profit_per_order", color="product_name",
    text="profit_per_order", title="Most Profitable Product per Region"
)
fig.update_traces(textposition="outside", texttemplate="%{text:.2f}")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 7. Top 5 Most Sold Categories — Quantity + Revenue
# -------------------------------------------------------------
st.subheader("Top 5 Most Sold Categories")

cat_sales = (
    df_view.groupby("category_name")
    .agg({"order_item_quantity": "sum", "sales_per_customer": "sum"})
    .reset_index()
)

top_qty = cat_sales.sort_values("order_item_quantity", ascending=False).head(5)
top_rev = cat_sales.sort_values("sales_per_customer", ascending=False).head(5)

colA, colB = st.columns(2)

with colA:
    fig = px.bar(
        top_qty, x="category_name", y="order_item_quantity",
        title="Top 5 Categories by Quantity Sold", text="order_item_quantity"
    )
    fig.update_traces(textposition="outside", texttemplate="%{text:,}")
    st.plotly_chart(fig, use_container_width=True)

with colB:
    fig = px.bar(
        top_rev, x="category_name", y="sales_per_customer",
        title="Top 5 Categories by Revenue in dollars", text="sales_per_customer"
    )
    fig.update_traces(textposition="outside", texttemplate="%{text:.2f}")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 8. Preferred Shipping Mode — Region (stacked)
# -------------------------------------------------------------
st.subheader("Preferred Shipping Mode by Region")

pref = df_view.groupby(["order_region", "shipping_mode"]).size().reset_index(name="count")

fig = px.bar(
    pref, x="order_region", y="count", color="shipping_mode",
    title="Shipping Mode Preference by Region", text="count"
)
fig.update_traces(textposition="outside", texttemplate="%{text:,}")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 9. Delayed Orders by Shipping Mode
# -------------------------------------------------------------
st.subheader("Delayed Orders by Shipping Mode")

delayed = (
    df_view[df_view["is_delayed"]]
    .groupby("shipping_mode").size().reset_index(name="delayed_count")
    .sort_values("delayed_count", ascending=False)
)

fig = px.bar(
    delayed, x="shipping_mode", y="delayed_count",
    title="Delayed Orders by Shipping Mode",
    text="delayed_count", color="delayed_count", color_continuous_scale="Reds"
)
fig.update_traces(textposition="outside", texttemplate="%{text:,}")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 10. Delay % by Region (Donut)
# -------------------------------------------------------------
st.subheader("Delay Percentage by Region")

reg_delay = (
    df_view.groupby("order_region")["is_delayed"].mean().reset_index()
)
reg_delay["delay_pct"] = reg_delay["is_delayed"] * 100

fig = go.Figure(
    go.Pie(
        labels=reg_delay["order_region"],
        values=reg_delay["delay_pct"],
        hole=0.45,
        textinfo="label+percent",
    )
)
fig.update_layout(title="Delay Percentage by Region")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="
        position: fixed;
        bottom: 10px;
        right: 15px;
        color: #A3C4F3;
        font-size: 16px;
        font-weight: 500;
    ">
        Created by <span style="color:#0B6EFD;">Ravindra Yadav</span>
    </div>
    """,
    unsafe_allow_html=True
)


