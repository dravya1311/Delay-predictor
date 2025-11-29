import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Order Delay Prediction Dashboard", layout="wide")

# ----------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------
@st.cache_data
def load_data(model_file, desc_file):
    try:
        df1 = pd.read_csv(model_file)
        df2 = pd.read_csv(desc_file)
    except Exception as e:
        st.error(f"Failed to load files. Error: {e}")
        return pd.DataFrame()

    # Strip whitespace
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Merge on Order_ID
    if "Order_ID" not in df1.columns or "Order_ID" not in df2.columns:
        st.error("Order_ID column missing in one of the files.")
        return pd.DataFrame()

    df = pd.merge(df1, df2, on="Order_ID", how="inner")

    return df

df = load_data("Delay Model.csv", "Delay description csv.csv")

if df.empty:
    st.stop()

# ----------------------------------------------------
# 2. CLEAN DELAY STATUS
# ----------------------------------------------------
DELAY_MAP = {
    "0": 0, "on-time": 0, "ontime": 0,
    "1": 1, "late": 1, "delayed": 1, "delay": 1,
    "-1": -1, "early": -1, "early-delivery": -1
}

df["Delay_Status"] = (
    df["Delay_Status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace(DELAY_MAP)
)

df["Delay_Status"] = pd.to_numeric(df["Delay_Status"], errors="coerce").fillna(0).astype(int)

# ----------------------------------------------------
# 3. SIDEBAR FILTERS
# ----------------------------------------------------
st.sidebar.header("Filters")

region_list = sorted(df["Order_Region"].dropna().unique())
country_list = sorted(df["Order_Country"].dropna().unique())
category_list = sorted(df["Category_Name"].dropna().unique())

region_f = st.sidebar.multiselect("Select Region", region_list, default=region_list)
country_f = st.sidebar.multiselect("Select Country", country_list, default=country_list)
category_f = st.sidebar.multiselect("Select Category", category_list, default=category_list)

df_f = df[
    df["Order_Region"].isin(region_f)
    & df["Order_Country"].isin(country_f)
    & df["Category_Name"].isin(category_f)
]

# ----------------------------------------------------
# 4. KPIs
# ----------------------------------------------------
st.title("📦 Order Delay Prediction Dashboard")
st.markdown("### Performance Overview")

col1, col2, col3, col4 = st.columns(4)

total_orders = len(df_f)
avg_sales = round(df_f["Sales"].mean(), 2)
avg_profit = round(df_f["Profit"].mean(), 2)
delay_pct = round((df_f["Delay_Status"] == 1).mean() * 100, 2)

col1.metric("Total Orders", total_orders)
col2.metric("Avg Sales", avg_sales)
col3.metric("Avg Profit", avg_profit)
col4.metric("Delay %", f"{delay_pct}%")

# ----------------------------------------------------
# 5. ANALYSIS SECTION
# ----------------------------------------------------

st.markdown("---")
st.header("🔍 Data Insights")

# 1) Average sales per customer by region
st.subheader("1) Average Sales Per Customer by Region")
fig1 = px.bar(
    df_f.groupby("Order_Region")["Sales"].mean().reset_index(),
    x="Order_Region",
    y="Sales",
    text_auto=True,
    title="Average Sales per Customer (Region-wise)"
)
st.plotly_chart(fig1, use_container_width=True)

# 2) Average Profit per order by region
st.subheader("2) Average Profit Per Order by Region")
fig2 = px.bar(
    df_f.groupby("Order_Region")["Profit"].mean().reset_index(),
    x="Order_Region",
    y="Profit",
    text_auto=True,
    title="Average Profit per Order (Region-wise)"
)
st.plotly_chart(fig2, use_container_width=True)

# 3) Top 5 order country and region market-wise
st.subheader("3) Top 5 Order Countries by Region")
top_countries = (
    df_f.groupby(["Order_Region", "Order_Country"])["Order_ID"]
    .count()
    .reset_index()
    .sort_values("Order_ID", ascending=False)
    .head(5)
)
fig3 = px.bar(
    top_countries,
    x="Order_Country",
    y="Order_ID",
    color="Order_Region",
    text_auto=True,
    title="Top 5 Order Countries"
)
st.plotly_chart(fig3, use_container_width=True)

# 4) Top 8 most profitable categories
st.subheader("4) Top 8 Most Profitable Categories")
top_cat = (
    df_f.groupby("Category_Name")["Profit"]
    .sum()
    .reset_index()
    .sort_values("Profit", ascending=False)
    .head(8)
)
fig4 = px.bar(
    top_cat,
    x="Category_Name",
    y="Profit",
    text_auto=True,
    title="Top 8 Most Profitable Categories"
)
st.plotly_chart(fig4, use_container_width=True)

# 5) Most profitable product by region
st.subheader("5) Most Profitable Product by Region")
best_product = (
    df_f.groupby(["Order_Region", "Product_Name"])["Profit"]
    .sum()
    .reset_index()
    .sort_values(["Order_Region", "Profit"], ascending=[True, False])
    .groupby("Order_Region")
    .head(1)
)

fig5 = px.bar(
    best_product,
    x="Order_Region",
    y="Profit",
    color="Product_Name",
    text_auto=True,
    title="Most Profitable Product by Region"
)
st.plotly_chart(fig5, use_container_width=True)

# 6) Top 5 most sold categories by quantity & revenue
st.subheader("6) Top 5 Most Sold Categories (Quantity & Revenue)")
top_sold = (
    df_f.groupby("Category_Name")[["Quantity", "Sales"]]
    .sum()
    .reset_index()
    .sort_values("Quantity", ascending=False)
    .head(5)
)
fig6 = px.bar(
    top_sold,
    x="Category_Name",
    y="Quantity",
    text_auto=True,
    title="Top 5 Categories by Quantity"
)
st.plotly_chart(fig6, use_container_width=True)

# 7) Preferred shipping mode by region
st.subheader("7) Preferred Shipping Mode by Region")
fig7 = px.bar(
    df_f.groupby(["Order_Region", "Ship_Mode"])["Order_ID"].count().reset_index(),
    x="Order_Region",
    y="Order_ID",
    color="Ship_Mode",
    title="Preferred Shipping Mode by Region"
)
st.plotly_chart(fig7, use_container_width=True)

# 8) Delayed orders by shipping mode
st.subheader("8) Delayed Orders by Shipping Mode")
delay_by_ship = (
    df_f[df_f["Delay_Status"] == 1]
    .groupby("Ship_Mode")["Order_ID"]
    .count()
    .reset_index()
)
fig8 = px.bar(
    delay_by_ship,
    x="Ship_Mode",
    y="Order_ID",
    text_auto=True,
    title="Delayed Orders by Shipping Mode"
)
st.plotly_chart(fig8, use_container_width=True)


st.success("Dashboard Loaded Successfully!")
