# order_delay_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os, re

st.set_page_config(page_title="Order Delay Analysis (Graphs Only)", layout="wide")

# -----------------------
# Helpers
# -----------------------
def norm_col(c: str) -> str:
    return re.sub(r'[^0-9a-z]+', '_', str(c).strip().lower())

def try_load_csv():
    # local candidates then GitHub raw (space-encoded)
    local_candidates = ["Delay_Model.csv", "Delay Model.csv", "Delay-Model.csv", "DelayModel.csv"]
    for p in local_candidates:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    # GitHub raw variants
    gh_base = "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/"
    gh_candidates = ["Delay_Model.csv", "Delay%20Model.csv", "Delay-Model.csv", "DelayModel.csv"]
    for u in gh_candidates:
        url = gh_base + u
        try:
            df = pd.read_csv(url)
            return df
        except Exception:
            continue
    return None

# -----------------------
# Load data
# -----------------------
df_raw = try_load_csv()
if df_raw is None:
    st.error("Could not locate Delay_Model.csv (local or GitHub). Place file in repo root or ensure the GitHub raw URL exists.")
    st.stop()

# -----------------------
# Normalize columns
# -----------------------
orig_columns = list(df_raw.columns)
df = df_raw.copy()
df.columns = [norm_col(c) for c in df.columns]

# map names we need (normalized lookup)
def find(colnames):
    for cand in colnames:
        k = norm_col(cand)
        if k in df.columns:
            return k
    return None

col_label = find(["label"])
col_shipping = find(["shipping mode", "shipping_mode", "shippingmode"])
col_region = find(["order region", "order_region", "orderregion"])
col_country = find(["order country", "order_country", "ordercountry"])
col_sales = find(["sales per customer", "sales_per_customer", "sales"])
col_profit = find(["profit per order", "profit_per_order", "profit"])
col_category = find(["category name", "category_name", "categoryname"])
col_product = find(["product name", "product_name", "productname"])
col_qty = find(["order item quantity", "order_item_quantity", "order_item_quantity", "quantity", "order_item_quantity"])

# critical column check (must have at least: label, shipping, region)
critical_missing = [name for name,v in (("label",col_label),("shipping mode",col_shipping),("order region",col_region)) if v is None]
if critical_missing:
    st.error(f"Critical columns are missing from Delay_Model.csv: {critical_missing}\n\nAvailable columns: {orig_columns}")
    st.stop()

# set derived/clean columns
df["label_num"] = pd.to_numeric(df[col_label], errors="coerce")

# User mapping specified: -1 = delayed, 0 = on-time, 1 = early
df["is_delayed"] = df["label_num"] == -1

df["shipping_mode"] = df[col_shipping].astype(str).fillna("Unknown")
df["order_region"] = df[col_region].astype(str).fillna("Unknown")
df["order_country"] = df[col_country].astype(str).fillna("Unknown")

# numeric conversions for optional fields
if col_sales:
    df["sales_per_customer"] = pd.to_numeric(df[col_sales], errors="coerce").fillna(0.0)
else:
    df["sales_per_customer"] = 0.0

if col_profit:
    df["profit_per_order"] = pd.to_numeric(df[col_profit], errors="coerce").fillna(0.0)
else:
    df["profit_per_order"] = 0.0

if col_qty:
    df["quantity_num"] = pd.to_numeric(df[col_qty], errors="coerce").fillna(0)
else:
    df["quantity_num"] = 0

# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.header("Filters")
regions = ["All"] + sorted(df["order_region"].unique().tolist())
region_sel = st.sidebar.selectbox("Order Region", regions, index=0)

df_view = df if region_sel == "All" else df[df["order_region"] == region_sel]

# -----------------------
# Top KPI metrics (graphical highlight row)
# -----------------------
total_orders = len(df_view)
delayed_count = int(df_view["is_delayed"].sum())
delayed_pct = (delayed_count / total_orders * 100) if total_orders else 0.0

# Most delayed shipping mode info
ship_grp = df_view.groupby("shipping_mode").agg(total_orders=("shipping_mode","count"), delayed_orders=("is_delayed","sum")).reset_index()
ship_grp["delay_pct"] = ship_grp["delayed_orders"] / ship_grp["total_orders"] * 100
ship_grp = ship_grp.sort_values("delay_pct", ascending=False).reset_index(drop=True)

most_delayed_mode = ship_grp.iloc[0]["shipping_mode"] if not ship_grp.empty else "N/A"
most_delayed_mode_pct = float(ship_grp.iloc[0]["delay_pct"]) if not ship_grp.empty else 0.0

# Most delayed region (if needed)
reg_grp_all = df_view.groupby("order_region").agg(total_orders=("order_region","count"), delayed_orders=("is_delayed","sum")).reset_index()
reg_grp_all["delay_pct"] = reg_grp_all["delayed_orders"] / reg_grp_all["total_orders"] * 100
reg_grp_all = reg_grp_all.sort_values("delay_pct", ascending=False).reset_index(drop=True)
most_delayed_region = reg_grp_all.iloc[0]["order_region"] if not reg_grp_all.empty else "N/A"
most_delayed_region_pct = float(reg_grp_all.iloc[0]["delay_pct"]) if not reg_grp_all.empty else 0.0

k1, k2, k3, k4 = st.columns([2,2,2,3])
k1.metric("Total Orders (view)", total_orders)
k2.metric("Delayed (label = -1) count", delayed_count)
k3.metric("Delayed % (view)", f"{delayed_pct:.2f}%")
k4.metric("Most-delayed Shipping Mode", f"{most_delayed_mode} — {most_delayed_mode_pct:.2f}%")

st.markdown("---")

# -----------------------
# 1) Delay % by Shipping Mode  (bar)
# -----------------------
st.subheader("1) Delay % by Shipping Mode")
if ship_grp.empty:
    st.info("No shipping mode data available.")
else:
    fig_ship_delay = px.bar(ship_grp, x="shipping_mode", y="delay_pct",
                            labels={"shipping_mode":"Shipping Mode","delay_pct":"Delay %"},
                            title="Delay % by Shipping Mode",
                            hover_data=["delayed_orders","total_orders"])
    st.plotly_chart(fig_ship_delay, use_container_width=True)

# -----------------------
# 2) Delay % by Order Region (bar)
# -----------------------
st.subheader("2) Delay % by Order Region")
if reg_grp_all.empty:
    st.info("No order region data available.")
else:
    fig_reg_delay = px.bar(reg_grp_all, x="order_region", y="delay_pct",
                           labels={"order_region":"Order Region","delay_pct":"Delay %"},
                           title="Delay % by Order Region",
                           hover_data=["delayed_orders","total_orders"])
    st.plotly_chart(fig_reg_delay, use_container_width=True)

# -----------------------
# 3) Average sales per customer by order_region (bar)
# -----------------------
st.subheader("3) Average Sales per Customer by Order Region")
if col_sales:
    avg_sales_region = df_view.groupby("order_region")["sales_per_customer"].mean().reset_index().sort_values("sales_per_customer", ascending=False)
    fig_avg_sales = px.bar(avg_sales_region, x="order_region", y="sales_per_customer",
                           labels={"order_region":"Order Region","sales_per_customer":"Avg Sales per Customer"},
                           title="Avg Sales per Customer by Region")
    st.plotly_chart(fig_avg_sales, use_container_width=True)
else:
    st.info("Column for Sales per customer not found; skipping this chart.")

# -----------------------
# 4) Average Profit per order based on order_region (bar)
# -----------------------
st.subheader("4) Average Profit per Order by Order Region")
if col_profit:
    avg_profit_region = df_view.groupby("order_region")["profit_per_order"].mean().reset_index().sort_values("profit_per_order", ascending=False)
    fig_avg_profit = px.bar(avg_profit_region, x="order_region", y="profit_per_order",
                            labels={"order_region":"Order Region","profit_per_order":"Avg Profit per Order"},
                            title="Avg Profit per Order by Region")
    st.plotly_chart(fig_avg_profit, use_container_width=True)
else:
    st.info("Column for Profit per order not found; skipping this chart.")

# -----------------------
# 5) Top 5 order country and order region marketwise (bars)
# -----------------------
st.subheader("5) Top 5 Countries & Regions by Order Volume")
top_countries = df_view["order_country"].value_counts().head(5).reset_index()
top_countries.columns = ["order_country","orders"]
if not top_countries.empty:
    fig_top_countries = px.bar(top_countries, x="order_country", y="orders", title="Top 5 Countries by Orders")
    st.plotly_chart(fig_top_countries, use_container_width=True)
top_regions = df_view["order_region"].value_counts().head(5).reset_index()
top_regions.columns = ["order_region","orders"]
if not top_regions.empty:
    fig_top_regions = px.bar(top_regions, x="order_region", y="orders", title="Top 5 Regions by Orders")
    st.plotly_chart(fig_top_regions, use_container_width=True)

# -----------------------
# 6) Top 8 most profitable category_name (bar)
# -----------------------
st.subheader("6) Top 8 Most Profitable Categories")
if col_category and col_profit:
    cat_profit = df_view.groupby(col_category)["profit_per_order"].sum().sort_values(ascending=False).head(8).reset_index()
    cat_profit.columns = ["category","total_profit"]
    fig_cat_profit = px.bar(cat_profit, x="category", y="total_profit", title="Top 8 Profitable Categories")
    st.plotly_chart(fig_cat_profit, use_container_width=True)
else:
    st.info("Category name and/or profit column not found; skipping category profit chart.")

# -----------------------
# 7) Most profitable product for different order_region
# -----------------------
st.subheader("7) Most Profitable Product by Order Region")
if col_product and col_profit:
    prod_region = df_view.groupby(["order_region", col_product])["profit_per_order"].sum().reset_index()
    idx = prod_region.groupby("order_region")["profit_per_order"].idxmax().dropna()
    best_prod = prod_region.loc[idx].reset_index(drop=True)
    if not best_prod.empty:
        fig_best_prod = px.bar(best_prod, x="order_region", y="profit_per_order", color=col_product,
                               title="Most Profitable Product by Region", labels={"profit_per_order":"Total Profit"})
        st.plotly_chart(fig_best_prod, use_container_width=True)
else:
    st.info("Product or profit column missing; skipping product-by-region chart.")

# -----------------------
# 8) Top 5 most sold category name based on quantity and revenue (two charts)
# -----------------------
st.subheader("8) Top 5 Most Sold Categories — Quantity & Revenue")
if col_category:
    if col_qty:
        cat_qty = df_view.groupby(col_category)["quantity_num"].sum().sort_values(ascending=False).head(5).reset_index()
        cat_qty.columns = ["category","total_quantity"]
        fig_cat_qty = px.bar(cat_qty, x="category", y="total_quantity", title="Top 5 Categories by Quantity Sold")
        st.plotly_chart(fig_cat_qty, use_container_width=True)
    if col_sales:
        cat_rev = df_view.groupby(col_category)["sales_per_customer"].sum().sort_values(ascending=False).head(5).reset_index()
        cat_rev.columns = ["category","total_revenue"]
        fig_cat_rev = px.bar(cat_rev, x="category", y="total_revenue", title="Top 5 Categories by Revenue (sales_per_customer)")
        st.plotly_chart(fig_cat_rev, use_container_width=True)
else:
    st.info("Category column not found; skipping sold-category charts.")

# -----------------------
# 9) Preferred shipping mode based on order_region (stacked/grouped bar)
# -----------------------
st.subheader("9) Preferred Shipping Mode by Order Region")
pref = df_view.groupby(["order_region","shipping_mode"]).size().reset_index(name="count")
if not pref.empty:
    fig_pref = px.bar(pref, x="order_region", y="count", color="shipping_mode", barmode="group",
                      title="Preferred Shipping Mode by Region", labels={"count":"Orders","order_region":"Order Region","shipping_mode":"Shipping Mode"})
    st.plotly_chart(fig_pref, use_container_width=True)
else:
    st.info("No shipping-mode-by-region data available.")

# -----------------------
# 10) Delayed orders based on shipping mode (counts)
# -----------------------
st.subheader("10) Delayed Orders by Shipping Mode (counts)")
delayed_mode_counts = df_view[df_view["is_delayed"]].groupby("shipping_mode").size().reset_index(name="delayed_count").sort_values("delayed_count", ascending=False)
if not delayed_mode_counts.empty:
    fig_delayed_mode = px.bar(delayed_mode_counts, x="shipping_mode", y="delayed_count", title="Delayed Orders by Shipping Mode", labels={"delayed_count":"Delayed Orders","shipping_mode":"Shipping Mode"})
    st.plotly_chart(fig_delayed_mode, use_container_width=True)
else:
    st.info("No delayed orders in current filter.")

# -----------------------
# 11) Order_region wise: delay % for each order_region (chart repeated as section 2 but present as card)
# -----------------------
st.subheader("11) Order Region — Delay % (visual summary)")
if not reg_grp_all.empty:
    fig_reg_pct = px.pie(reg_grp_all, names="order_region", values="delayed_orders", title="Delayed Orders Distribution by Region")
    st.plotly_chart(fig_reg_pct, use_container_width=True)

# -----------------------
# 12) (Already covered ordering) - ensure no tables are shown
# -----------------------

# -----------------------
# Footer: small summary text (no tables)
# -----------------------
st.markdown("---")
st.markdown(
    "Notes: Delay mapping used: **label = -1 → delayed**, **0 → on-time**, **1 → early**. "
    "All visualizations are interactive and reflect the selected Order Region filter (sidebar)."
)
