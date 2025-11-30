import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="Order Delay Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Load Data
# --------------------------
GITHUB_FILE_URL = (
    "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/Delay_Model.csv"
)

@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None

df = load_data(GITHUB_FILE_URL)

if df is None:
    st.stop()

# --------------------------
# Preprocessing
# --------------------------
required_cols = [
    "Order_ID", "Customer_ID", "Order_Region", "Order_Country", "Shipping_Mode",
    "Category_Name", "Product_Name", "Sales", "Profit", "Quantity", "label"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df["label"] = pd.to_numeric(df["label"], errors="coerce")

# --------------------------
# Header
# --------------------------
st.title("📦 Order Delay Analysis Dashboard")
st.markdown("A comprehensive KPI dashboard analyzing order delays, profitability, and operational performance.")

# --------------------------
# KPI Metrics
# --------------------------
total_orders = len(df)
delayed_orders = len(df[df["label"] == 1])
on_time_orders = len(df[df["label"] == 0])
early_orders = len(df[df["label"] == -1])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", total_orders)
col2.metric("Delayed Orders", delayed_orders)
col3.metric("On-Time Orders", on_time_orders)
col4.metric("Early Deliveries", early_orders)

st.divider()

# --------------------------
# Section 1: Average KPIs
# --------------------------
st.subheader("1) Sales & Profit Metrics by Region")

avg_sales_region = df.groupby("Order_Region")["Sales"].mean().reset_index()
avg_profit_region = df.groupby("Order_Region")["Profit"].mean().reset_index()

col1, col2 = st.columns(2)
col1.dataframe(avg_sales_region, use_container_width=True)
col2.dataframe(avg_profit_region, use_container_width=True)

# --------------------------
# Section 2: Top Markets
# --------------------------
st.subheader("2) Top 5 Countries & Regions by Order Volume")

top_country = df["Order_Country"].value_counts().head(5)
top_region = df["Order_Region"].value_counts().head(5)

col1, col2 = st.columns(2)
col1.write("Top 5 Countries")
col1.bar_chart(top_country)

col2.write("Top 5 Regions")
col2.bar_chart(top_region)

# --------------------------
# Section 3: Most Profitable Categories
# --------------------------
st.subheader("3) Top 8 Most Profitable Categories")

profit_by_cat = df.groupby("Category_Name")["Profit"].sum().nlargest(8).reset_index()
fig_cat = px.bar(profit_by_cat, x="Category_Name", y="Profit", title="Most Profitable Categories")
st.plotly_chart(fig_cat, use_container_width=True)

# --------------------------
# Section 4: Most Profitable Product per Region
# --------------------------
st.subheader("4) Most Profitable Product by Region")

pp_region = (
    df.groupby(["Order_Region", "Product_Name"])["Profit"]
    .sum()
    .reset_index()
    .sort_values(["Order_Region", "Profit"], ascending=[True, False])
)

top_products = pp_region.groupby("Order_Region").head(1)
st.dataframe(top_products, use_container_width=True)

# --------------------------
# Section 5: Top Selling Categories
# --------------------------
st.subheader("5) Top 5 Categories by Quantity and Revenue")

cat_qty = df.groupby("Category_Name")["Quantity"].sum().nlargest(5)
cat_rev = df.groupby("Category_Name")["Sales"].sum().nlargest(5)

col1, col2 = st.columns(2)
col1.write("By Quantity")
col1.bar_chart(cat_qty)

col2.write("By Revenue")
col2.bar_chart(cat_rev)

# --------------------------
# Section 6: Preferred Shipping Mode by Region
# --------------------------
st.subheader("6) Preferred Shipping Mode by Region")

ship_pref = (
    df.groupby(["Order_Region", "Shipping_Mode"])["Order_ID"]
    .count()
    .reset_index(name="Count")
)

fig_ship = px.bar(
    ship_pref,
    x="Order_Region",
    y="Count",
    color="Shipping_Mode",
    barmode="group",
    title="Shipping Mode Preference by Region"
)
st.plotly_chart(fig_ship, use_container_width=True)

# --------------------------
# Section 7: Delay Analysis
# --------------------------
st.subheader("7) Delay Analysis (label: 1 = Delayed)")

df_delayed = df[df["label"] == 1]

# Delay by Shipping Mode
delay_ship = df_delayed["Shipping_Mode"].value_counts().reset_index()
delay_ship.columns = ["Shipping_Mode", "Delay_Count"]

# Delay by Region
delay_region = df_delayed["Order_Region"].value_counts().reset_index()
delay_region.columns = ["Order_Region", "Delay_Count"]

col1, col2 = st.columns(2)
col1.write("Delayed Orders by Shipping Mode")
col1.dataframe(delay_ship, use_container_width=True)

col2.write("Delayed Orders by Region")
col2.dataframe(delay_region, use_container_width=True)

# Most Delayed Mode
if len(delay_ship):
    worst_mode = delay_ship.iloc[0]
    st.warning(f"Most Delayed Shipping Mode: **{worst_mode['Shipping_Mode']}** ({worst_mode['Delay_Count']} delays)")

st.divider()

st.markdown("Dashboard ready for operational review and management reporting.")
