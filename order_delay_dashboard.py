import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Order Delay Analytics Dashboard", layout="wide")

# -------------------------------------------------------------------
# Utility Function
# -------------------------------------------------------------------
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()  # clean column names
        return df
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.header("Upload Delay Model CSV")
uploaded_file = st.sidebar.file_uploader("Upload Delay_Model.csv", type=["csv"])

if uploaded_file is None:
    st.warning("Upload Delay_Model.csv to view the dashboard.")
    st.stop()

df = load_data(uploaded_file)
if df is None:
    st.stop()


# -------------------------------------------------------------------
# Mandatory Columns
# -------------------------------------------------------------------
required_cols = [
    "Order id", "customer id", "Order region", "Order Country", 
    "Shipping mode", "Category name", "Product name",
    "Order Profit per order", "Sales per customer", "Order item quantity",
    "label"
]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()


# -------------------------------------------------------------------
# Data Preprocessing
# -------------------------------------------------------------------
# Rename for easier access (optional)
df.rename(columns={
    "Order id": "Order_ID",
    "customer id": "Customer_ID",
    "Order region": "Order_Region",
    "Order Country": "Order_Country",
    "Shipping mode": "Shipping_Mode",
    "Category name": "Category_Name",
    "Product name": "Product_Name",
    "Order Profit per order": "Profit",
    "Sales per customer": "Sales_Per_Customer",
    "Order item quantity": "Quantity",
}, inplace=True)

# Label mapping: -1 delayed, 0 on-time, 1 early
df["Delay_Flag"] = df["label"].map({-1: "Delayed", 0: "On-Time", 1: "Early"})


# -------------------------------------------------------------------
# KPI Section
# -------------------------------------------------------------------
st.title("📦 Order Delay Prediction Dashboard")
st.markdown("---")

col1, col2, col3 = st.columns(3)

# Total Orders
col1.metric("Total Orders", len(df))

# Delay %
delay_pct = (df["label"].eq(-1).sum() / len(df)) * 100
col2.metric("Delay %", f"{delay_pct:.2f}%")

# Top Region by Sales
top_region = df.groupby("Order_Region")["Sales_Per_Customer"].sum().idxmax()
col3.metric("Top Revenue Region", top_region)


st.markdown("---")
st.subheader("Market Insights")


# -------------------------------------------------------------------
# 1) Average sales per customer (order_region)
# -------------------------------------------------------------------
st.markdown("### 1) Average Sales per Customer by Order Region")
avg_sales_region = df.groupby("Order_Region")["Sales_Per_Customer"].mean().reset_index()

fig1 = px.bar(avg_sales_region, x="Order_Region", y="Sales_Per_Customer",
              title="Avg Sales per Customer", text_auto=True)
st.plotly_chart(fig1, use_container_width=True)


# -------------------------------------------------------------------
# 2) Average Profit per order (order_region)
# -------------------------------------------------------------------
st.markdown("### 2) Average Profit per Order by Region")
avg_profit_region = df.groupby("Order_Region")["Profit"].mean().reset_index()

fig2 = px.bar(avg_profit_region, x="Order_Region", y="Profit",
              title="Avg Profit per Order", text_auto=True)
st.plotly_chart(fig2, use_container_width=True)


# -------------------------------------------------------------------
# 3) Top 5 countries and regions (marketwise)
# -------------------------------------------------------------------
st.markdown("### 3) Top 5 Order Countries & Regions by Revenue")

top_countries = df.groupby("Order_Country")["Sales_Per_Customer"].sum().nlargest(5).reset_index()
top_regions = df.groupby("Order_Region")["Sales_Per_Customer"].sum().nlargest(5).reset_index()

colA, colB = st.columns(2)

with colA:
    fig3A = px.bar(top_countries, x="Order_Country", y="Sales_Per_Customer",
                   title="Top 5 Countries")
    st.plotly_chart(fig3A, use_container_width=True)

with colB:
    fig3B = px.bar(top_regions, x="Order_Region", y="Sales_Per_Customer",
                   title="Top 5 Regions")
    st.plotly_chart(fig3B, use_container_width=True)


# -------------------------------------------------------------------
# 4) Top 8 most profitable categories
# -------------------------------------------------------------------
st.markdown("### 4) Top 8 Most Profitable Categories")

top_categories = df.groupby("Category_Name")["Profit"].sum().nlargest(8).reset_index()

fig4 = px.bar(top_categories, x="Category_Name", y="Profit",
              title="Most Profitable Categories")
st.plotly_chart(fig4, use_container_width=True)


# -------------------------------------------------------------------
# 5) Most profitable product by region
# -------------------------------------------------------------------
st.markdown("### 5) Most Profitable Product by Region")

prof_prod = df.groupby(["Order_Region", "Product_Name"])["Profit"].sum()
prof_prod = prof_prod.reset_index().sort_values(["Order_Region", "Profit"], ascending=[True, False])

# pick top 1 per region
prof_top = prof_prod.groupby("Order_Region").head(1)

fig5 = px.bar(prof_top, x="Order_Region", y="Profit", color="Product_Name",
              title="Most Profitable Product by Region", text="Product_Name")
st.plotly_chart(fig5, use_container_width=True)


# -------------------------------------------------------------------
# 6) Top 5 most sold categories (quantity + revenue)
# -------------------------------------------------------------------
st.markdown("### 6) Top 5 Most Sold Categories (Quantity & Revenue)")

cat_qty = df.groupby("Category_Name")["Quantity"].sum().nlargest(5).reset_index()
cat_rev = df.groupby("Category_Name")["Sales_Per_Customer"].sum().nlargest(5).reset_index()

colC, colD = st.columns(2)

with colC:
    fig6A = px.bar(cat_qty, x="Category_Name", y="Quantity",
                   title="Top Categories by Quantity Sold")
    st.plotly_chart(fig6A, use_container_width=True)

with colD:
    fig6B = px.bar(cat_rev, x="Category_Name", y="Sales_Per_Customer",
                   title="Top Categories by Revenue")
    st.plotly_chart(fig6B, use_container_width=True)


# -------------------------------------------------------------------
# 7) Preferred shipping mode by region
# -------------------------------------------------------------------
st.markdown("### 7) Preferred Shipping Mode by Region")

ship_pref = df.groupby(["Order_Region", "Shipping_Mode"]).size().reset_index(name="Count")

fig7 = px.bar(ship_pref, x="Order_Region", y="Count", color="Shipping_Mode",
              title="Shipping Mode Preference by Region", barmode="group")
st.plotly_chart(fig7, use_container_width=True)


# -------------------------------------------------------------------
# 8) Delayed orders by shipping mode
# -------------------------------------------------------------------
st.markdown("### 8) Delayed Orders by Shipping Mode")

delay_ship = df[df["label"] == -1].groupby("Shipping_Mode").size().reset_index(name="Delayed_Count")

fig8 = px.bar(delay_ship, x="Shipping_Mode", y="Delayed_Count",
              title="Delayed Orders by Shipping Mode")
st.plotly_chart(fig8, use_container_width=True)

st.success("Dashboard ready.")
