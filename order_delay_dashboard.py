import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Order Delay & Performance Dashboard",
    layout="wide"
)

GITHUB_RAW_URL = "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/Delay_Model.csv"

# ---------------------------------------------------------------
#  LOAD DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()

        # Ensure label column is numeric
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0)

        return df
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()

df = load_data(GITHUB_RAW_URL)

if df.empty:
    st.stop()

# ---------------------------------------------------------------
#  TITLE
# ---------------------------------------------------------------
st.title("📦 Order Record & Commercial Performance Dashboard")
st.markdown("An end-to-end analysis of order performance, profitability and delay patterns.")

# ---------------------------------------------------------------
#  KPI METRICS
# ---------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_orders = df.shape[0]
delayed_orders = df[df["label"] == -1].shape[0]
ontime_orders = df[df["label"] == 0].shape[0]
early_orders = df[df["label"] == 1].shape[0]

delay_percent = round((delayed_orders / total_orders) * 100, 2)

with col1:
    st.metric("Total Orders", total_orders)

with col2:
    st.metric("Delayed Orders", delayed_orders)

with col3:
    st.metric("Delay %", f"{delay_percent}%")

with col4:
    st.metric("Early Deliveries", early_orders)

st.markdown("---")

# ---------------------------------------------------------------
# 1) Average sales per customer based on order_region
# ---------------------------------------------------------------
st.subheader("1) Average Sales Per Customer by Market")
market = df.groupby("market")["sales"].mean().reset_index()

fig1 = px.bar(
    sales_region,
    x="market",
    y="sales",
    title="Average Sales Per Customer by Market",
    text_auto=True
)
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------------
# 2) Average Profit per Order based on order_region
# ---------------------------------------------------------------
st.subheader("2) Average Profit Per Order by Region")
profit_region = df.groupby("order_region")["profit_per_order"].mean().reset_index()

fig2 = px.bar(
    profit_region,
    x="order_region",
    y="profit_per_order",
    title="Average Profit Per Order by Region",
    text_auto=True,
    color="profit_per_order"
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------
# 3) Top 5 Order Countries & Regions (Marketwise)
# ---------------------------------------------------------------
st.subheader("3) Top 5 Markets – By Order Count")

top_markets = df.groupby(["market", "order_country"]).size().reset_index(name="order_count")
top5_markets = top_markets.sort_values("order_count", ascending=False).head(5)

fig3 = px.bar(
    top5_markets,
    x="market",
    y="order_count",
    color="order_country",
    title="Top 5 Order Markets"
)
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------
# 4) Top 8 Most Profitable Categories
# ---------------------------------------------------------------
st.subheader("4) Top 8 Most Profitable Categories")
cat_profit = df.groupby("category_name")["profit_per_order"].sum().reset_index()
top8_cat = cat_profit.sort_values("profit_per_order", ascending=False).head(8)

fig4 = px.bar(
    top8_cat,
    x="category_name",
    y="profit_per_order",
    title="Top 8 Most Profitable Categories",
    color="profit_per_order",
    text_auto=True
)
st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------
# 5) Most Profitable Product for Each Region
# ---------------------------------------------------------------
st.subheader("5) Most Profitable Product by Region")

profit_prod_region = (
    df.groupby(["order_region", "product_name"])["profit_per_order"]
    .sum()
    .reset_index()
)

idx = profit_prod_region.groupby("order_region")["profit_per_order"].idxmax()
best_products = profit_prod_region.loc[idx]

fig5 = px.bar(
    best_products,
    x="order_region",
    y="profit_per_order",
    color="product_name",
    title="Most Profitable Products by Region",
    text_auto=True
)
st.plotly_chart(fig5, use_container_width=True)

# ---------------------------------------------------------------
# 6) Top 5 Most Sold Categories (Quantity & Revenue)
# ---------------------------------------------------------------
st.subheader("6) Top 5 Most Sold Categories")

colA, colB = st.columns(2)

with colA:
    qty = df.groupby("category_name")["order_item_quantity"].sum().reset_index()
    qty_top5 = qty.sort_values("order_item_quantity", ascending=False).head(5)

    fig6A = px.bar(
        qty_top5,
        x="category_name",
        y="order_item_quantity",
        title="Top 5 Categories (By Quantity Sold)",
        text_auto=True
    )
    st.plotly_chart(fig6A, use_container_width=True)

with colB:
    rev = df.groupby("category_name")["sales"].sum().reset_index()
    rev_top5 = rev.sort_values("sales", ascending=False).head(5)

    fig6B = px.bar(
        rev_top5,
        x="category_name",
        y="sales",
        title="Top 5 Categories (By Revenue)",
        text_auto=True,
        color="sales"
    )
    st.plotly_chart(fig6B, use_container_width=True)

# ---------------------------------------------------------------
# 7) Preferred Shipping Mode by Region
# ---------------------------------------------------------------
st.subheader("7) Preferred Shipping Mode by Region")

ship_pref = df.groupby(["order_region", "shipping_mode"]).size().reset_index(name="count")

fig7 = px.bar(
    ship_pref,
    x="order_region",
    y="count",
    color="shipping_mode",
    title="Preferred Shipping Mode by Region"
)
st.plotly_chart(fig7, use_container_width=True)

# ---------------------------------------------------------------
# 8) Delay Rate by Shipping Mode
# ---------------------------------------------------------------
st.subheader("8) Delay % by Shipping Mode")

delay_by_ship = (
    df.groupby("shipping_mode")["label"]
    .apply(lambda x: (x == -1).mean() * 100)   # % delayed
    .reset_index(name="delay_percent")
)

fig8 = px.bar(
    delay_by_ship,
    x="shipping_mode",
    y="delay_percent",
    title="Delay % by Shipping Mode",
    text_auto=".2f"
)
st.plotly_chart(fig8, use_container_width=True)

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("Dashboard completed successfully by RAVINDRA YADAV.")

