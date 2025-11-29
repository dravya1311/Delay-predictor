import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Order Delay Predictor Dashboard",
    layout="wide"
)

st.title("Order Delay Prediction & KPI Dashboard")

# -------------------------------------------------------------------------
# LOAD DATA FROM GITHUB (NO LOCAL PATHS)
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    url_model = "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/Delay%20Model.csv"
    url_desc = "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/Delay%20description%20csv.csv"

    df1 = pd.read_csv(url_model)
    df2 = pd.read_csv(url_desc)

    return df1, df2

try:
    df, df_desc = load_data()
except Exception as e:
    st.error(f"Failed to load files: {e}")
    st.stop()

# -------------------------------------------------------------------------
# BASIC CLEANING
# -------------------------------------------------------------------------
df.columns = df.columns.str.strip()
df_desc.columns = df_desc.columns.str.strip()

# Ensure Delay_Status exists
if "order_status" in df.columns:
    df.rename(columns={"order_status": "Delay_Status"}, inplace=True)

# Convert delay flag to readable labels
df["Delay_Type"] = df["Delay_Status"].map({0: "On-Time", 1: "Delayed"})

# -------------------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------------------
st.sidebar.header("Filters")
region_filter = st.sidebar.multiselect(
    "Select Order Region",
    options=sorted(df["order_region"].unique()),
    default=sorted(df["order_region"].unique())
)

df_f = df[df["order_region"].isin(region_filter)]

# -------------------------------------------------------------------------
# KPI SECTION
# -------------------------------------------------------------------------
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", len(df_f))
col2.metric("Delayed Orders", df_f[df_f["Delay_Type"] == "Delayed"].shape[0])
col3.metric("Delay %", round(df_f["Delay_Status"].mean() * 100, 2))
col4.metric("Avg. Profit Per Order", round(df_f["profit"].mean(), 2))

# -------------------------------------------------------------------------
# 1) AVG SALES PER CUSTOMER BY ORDER REGION
# -------------------------------------------------------------------------
st.subheader("1. Average Sales per Customer by Order Region")
avg_sales = df_f.groupby("order_region")["sales"].mean().reset_index()

fig1 = px.bar(avg_sales, x="order_region", y="sales", title="Avg Sales per Customer")
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------------------------------------------------
# 2) AVG PROFIT PER ORDER BY REGION
# -------------------------------------------------------------------------
st.subheader("2. Average Profit per Order by Order Region")
avg_profit = df_f.groupby("order_region")["profit"].mean().reset_index()

fig2 = px.bar(avg_profit, x="order_region", y="profit",
              title="Avg Profit per Order")
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------------------------------
# 3) TOP 5 ORDER COUNTRIES & ORDER REGION
# -------------------------------------------------------------------------
st.subheader("3. Top 5 Countries & Regions by Order Volume")

colA, colB = st.columns(2)

top_country = df_f["order_country"].value_counts().head(5)
fig3 = px.bar(top_country, title="Top 5 Order Countries")
colA.plotly_chart(fig3, use_container_width=True)

top_region = df_f["order_region"].value_counts().head(5)
fig4 = px.bar(top_region, title="Top 5 Order Regions")
colB.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------------------------------------
# 4) TOP 8 MOST PROFITABLE CATEGORY
# -------------------------------------------------------------------------
st.subheader("4. Top 8 Most Profitable Categories")
top_cat = df_f.groupby("category_name")["profit"].sum().nlargest(8).reset_index()

fig5 = px.bar(top_cat, x="category_name", y="profit",
              title="Top 8 Most Profitable Categories")
st.plotly_chart(fig5, use_container_width=True)

# -------------------------------------------------------------------------
# 5) MOST PROFITABLE PRODUCT PER ORDER REGION
# -------------------------------------------------------------------------
st.subheader("5. Most Profitable Product by Order Region")

profit_region = df_f.groupby(["order_region", "product_name"])["profit"].sum().reset_index()
profit_top = profit_region.loc[profit_region.groupby("order_region")["profit"].idxmax()]

fig6 = px.bar(profit_top, x="order_region", y="profit", color="product_name",
              title="Most Profitable Product per Region")
st.plotly_chart(fig6, use_container_width=True)

# -------------------------------------------------------------------------
# 6) TOP 5 MOST SOLD CATEGORY (QUANTITY & REVENUE)
# -------------------------------------------------------------------------
st.subheader("6. Top 5 Most Sold Categories (Quantity & Revenue)")

colC, colD = st.columns(2)

most_qty = df_f.groupby("category_name")["quantity"].sum().nlargest(5).reset_index()
fig7 = px.bar(most_qty, x="category_name", y="quantity",
              title="Top 5 Categories by Quantity Sold")
colC.plotly_chart(fig7, use_container_width=True)

most_rev = df_f.groupby("category_name")["sales"].sum().nlargest(5).reset_index()
fig8 = px.bar(most_rev, x="category_name", y="sales",
              title="Top 5 Categories by Revenue")
colD.plotly_chart(fig8, use_container_width=True)

# -------------------------------------------------------------------------
# 7) PREFERRED SHIPPING MODE BY REGION
# -------------------------------------------------------------------------
st.subheader("7. Preferred Shipping Mode by Region")
ship_pref = df_f.groupby(["order_region", "ship_mode"])["ship_mode"].count().reset_index(name="Count")

fig9 = px.bar(ship_pref, x="order_region", y="Count",
              color="ship_mode", barmode="group",
              title="Preferred Shipping Mode by Region")
st.plotly_chart(fig9, use_container_width=True)

# -------------------------------------------------------------------------
# 8) DELAYED ORDERS BY SHIPPING MODE
# -------------------------------------------------------------------------
st.subheader("8. Delayed Orders by Shipping Mode")

delayed_ship = df_f[df_f["Delay_Type"] == "Delayed"].groupby("ship_mode")["Delay_Type"].count().reset_index()

fig10 = px.bar(delayed_ship, x="ship_mode", y="Delay_Type",
               title="Delayed Orders by Shipping Mode")
st.plotly_chart(fig10, use_container_width=True)

# -------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------
