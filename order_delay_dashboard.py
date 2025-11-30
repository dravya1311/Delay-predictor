import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Order Delay Prediction Dashboard", layout="wide")

CSV_URL = "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/Delay_Model.csv"

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()  # Clean column names

        # Ensure Delay_Status is numeric
        df["Delay_Status"] = pd.to_numeric(df["Delay_Status"], errors="coerce").fillna(0)

        # Replace meanings
        df["Delay_Flag"] = df["Delay_Status"].map({
            -1: "Early Delivery",
            0: "On Time",
            1: "Delayed"
        })

        return df

    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()

df = load_data(CSV_URL)

if df.empty:
    st.stop()

# -------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------
st.sidebar.header("Filters")

region_list = ["All"] + sorted(df["Order_Region"].dropna().unique().tolist())
region_choice = st.sidebar.selectbox("Select Region", region_list)

if region_choice != "All":
    df_f = df[df["Order_Region"] == region_choice]
else:
    df_f = df.copy()

# -------------------------------------------------------
# KPI ROW
# -------------------------------------------------------
st.title("📦 Order Delay Prediction Dashboard")

col1, col2, col3, col4 = st.columns(4)

# KPI 1: Avg Sales per Customer
col1.metric(
    "Avg Sales per Customer",
    round(df_f["Sales"].mean(), 2)
)

# KPI 2: Avg Profit per Order
col2.metric(
    "Avg Profit per Order",
    round(df_f["Profit"].mean(), 2)
)

# KPI 3: Delay %
delay_pct = round((df_f["Delay_Status"] == 1).mean() * 100, 2)
col3.metric("Delay Percentage", f"{delay_pct}%")

# KPI 4: Early Delivery %
early_pct = round((df_f["Delay_Status"] == -1).mean() * 100, 2)
col4.metric("Early Delivery Percentage", f"{early_pct}%")

st.markdown("---")

# -------------------------------------------------------
# 1) Average Sales Per Customer Region
# -------------------------------------------------------
st.subheader("1) Average Sales per Customer by Region")
fig1 = px.bar(
    df.groupby("Order_Region")["Sales"].mean().reset_index(),
    x="Order_Region", y="Sales",
    title="Average Sales per Customer by Region"
)
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------------------------------
# 2) Average Profit per Order Region
# -------------------------------------------------------
st.subheader("2) Average Profit per Order by Region")
fig2 = px.bar(
    df.groupby("Order_Region")["Profit"].mean().reset_index(),
    x="Order_Region", y="Profit",
    title="Average Profit per Order by Region"
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------------
# 3) Top 5 Countries & Regions
# -------------------------------------------------------
st.subheader("3) Top 5 Countries & Regions by Order Volume")

top_country = (
    df["Order_Country"].value_counts().head(5).reset_index()
)
top_country.columns = ["Order_Country", "Orders"]

fig3 = px.bar(top_country, x="Order_Country", y="Orders", title="Top 5 Countries")
st.plotly_chart(fig3, use_container_width=True)

top_region = (
    df["Order_Region"].value_counts().head(5).reset_index()
)
top_region.columns = ["Order_Region", "Orders"]

fig4 = px.bar(top_region, x="Order_Region", y="Orders", title="Top 5 Regions")
st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------------------
# 4) Top 8 Most Profitable Categories
# -------------------------------------------------------
st.subheader("4) Top 8 Most Profitable Categories")

top_cat = (
    df.groupby("Category_Name")["Profit"].sum()
    .sort_values(ascending=False)
    .head(8).reset_index()
)

fig5 = px.bar(top_cat, x="Category_Name", y="Profit", title="Top 8 Profitable Categories")
st.plotly_chart(fig5, use_container_width=True)

# -------------------------------------------------------
# 5) Most Profitable Product per Region
# -------------------------------------------------------
st.subheader("5) Most Profitable Product by Region")

profit_prod = (
    df.groupby(["Order_Region", "Product_Name"])["Profit"].sum()
    .reset_index()
)

max_prod = profit_prod.loc[
    profit_prod.groupby("Order_Region")["Profit"].idxmax()
]

fig6 = px.bar(
    max_prod,
    x="Order_Region", y="Profit", color="Product_Name",
    title="Most Profitable Product by Region"
)
st.plotly_chart(fig6, use_container_width=True)

# -------------------------------------------------------
# 6) Top 5 Category by Quantity & Revenue
# -------------------------------------------------------
st.subheader("6) Top 5 Most Sold Categories (Quantity + Revenue)")

cat_agg = (
    df.groupby("Category_Name")[["Quantity", "Sales"]]
    .sum()
    .sort_values(by="Quantity", ascending=False)
    .head(5)
    .reset_index()
)

fig7 = px.bar(cat_agg, x="Category_Name", y="Quantity", title="Top 5 by Quantity")
st.plotly_chart(fig7, use_container_width=True)

fig8 = px.bar(cat_agg, x="Category_Name", y="Sales", title="Top 5 by Revenue")
st.plotly_chart(fig8, use_container_width=True)

# -------------------------------------------------------
# 7) Preferred Shipping Mode by Region
# -------------------------------------------------------
st.subheader("7) Preferred Shipping Mode by Region")

fig9 = px.histogram(
    df, x="Shipping_Mode", color="Order_Region",
    title="Preferred Shipping Mode by Region"
)
st.plotly_chart(fig9, use_container_width=True)

# -------------------------------------------------------
# 8) Delayed Orders by Shipping Mode
# -------------------------------------------------------
st.subheader("8) Delayed Orders by Shipping Mode")

delay_ship = (
    df[df["Delay_Status"] == 1]
    .groupby("Shipping_Mode")["Order_ID"]
    .count()
    .reset_index()
    .sort_values(by="Order_ID", ascending=False)
)

fig10 = px.bar(delay_ship, x="Shipping_Mode", y="Order_ID", title="Delayed Orders by Shipping Mode")
st.plotly_chart(fig10, use_container_width=True)
