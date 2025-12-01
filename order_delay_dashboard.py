# order_delay_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import os

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Order Delay Analysis", layout="wide")
st.title("Order Delay — Analysis & Insights")

# ---------------------------
# AUTO-LOAD CSV (no upload)
# Tries multiple candidate locations (raw GitHub URLs and repo-root filenames)
# ---------------------------
CANDIDATE_URLS = [
    "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/Delay_Model.csv",
    "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/Delay%20Model.csv",
    "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/Delay-Model.csv",
    "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/DelayModel.csv"
]
CANDIDATE_LOCAL = ["Delay_Model.csv", "Delay Model.csv", "Delay-Model.csv", "DelayModel.csv"]

@st.cache_data
def try_load():
    # try local first
    for p in CANDIDATE_LOCAL:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    # try raw urls
    for u in CANDIDATE_URLS:
        try:
            df = pd.read_csv(u)
            return df
        except Exception:
            continue
    return None

df = try_load()
if df is None:
    st.error("Could not locate Delay_Model.csv. Place the file in the repo root or ensure the GitHub raw URL is correct.")
    st.stop()

# ---------------------------
# NORMALIZE COLUMN NAMES (map to safe keys)
# ---------------------------
def norm(name: str) -> str:
    return re.sub(r'[^0-9a-z]+', '_', str(name).strip().lower())

original_columns = df.columns.tolist()
df.columns = [norm(c) for c in df.columns]

# ---------------------------
# EXPECTED FIELDS (normalized)
# Use exact columns from your sheet mapped to normalized names:
# 'label' -> label
# 'shipping mode' -> shipping_mode
# 'order region' -> order_region
# 'order country' -> order_country
# 'sales per customer' -> sales_per_customer
# 'profit per order' -> profit_per_order
# 'category name' -> category_name
# 'product name' -> product_name
# 'order item quantity' -> order_item_quantity (note original: "Order item quantity")
# ---------------------------
col_map = {
    "label": "label",
    "shipping_mode": ["shipping_mode", "shipping_mode", "shipping_mode"],  # fallback
    "shipping_mode_alt": ["shipping_mode", "shipping mode"],
    "order_region": ["order_region", "order_region", "order region"],
    "order_country": ["order_country", "order_country", "order country"],
    "sales_per_customer": ["sales_per_customer", "sales_per_customer", "sales per customer", "sales"],
    "profit_per_order": ["profit_per_order", "profit_per_order", "profit per order"],
    "category_name": ["category_name", "category_name", "category name"],
    "product_name": ["product_name", "product_name", "product name"],
    "order_item_quantity": ["order_item_quantity", "order_item_quantity", "order item quantity", "order_item_quantity"]
}

# Build a reverse map of normalized existing columns for lookup
available = {c: c for c in df.columns}

def find_column(candidates):
    for cand in candidates:
        key = norm(cand)
        if key in df.columns:
            return key
    # also allow exact normalized candidate
    for cand in candidates:
        cand_norm = norm(cand)
        if cand_norm in df.columns:
            return cand_norm
    return None

# locate columns
cols = {}
cols['label'] = find_column(["label"])
cols['shipping_mode'] = find_column(["shipping mode", "shipping_mode", "shippingmode", "shipping_mode"])
cols['order_region'] = find_column(["order region", "order_region", "orderregion"])
cols['order_country'] = find_column(["order country", "order_country", "ordercountry"])
cols['sales_per_customer'] = find_column(["sales per customer", "sales_per_customer", "sales"])
cols['profit_per_order'] = find_column(["profit per order", "profit_per_order", "profit"])
cols['category_name'] = find_column(["category name", "category_name"])
cols['product_name'] = find_column(["product name", "product_name"])
cols['order_item_quantity'] = find_column(["order item quantity", "order_item_quantity", "quantity"])

# Warn about missing but continue where possible
missing_req = [k for k,v in cols.items() if v is None and k in ['label','shipping_mode','order_region']]
if missing_req:
    st.error(f"Missing critical columns (after normalization): {missing_req}. Available columns: {original_columns}")
    st.stop()

# ---------------------------
# MAP & CLEAN critical columns
# ---------------------------
# label: user defined mapping: -1 = delay, 0 = on-time, 1 = early
df[cols['label']] = pd.to_numeric(df[cols['label']], errors='coerce')

# define boolean delayed mask where label == -1
df['is_delayed'] = (df[cols['label']] == -1)

# shipping mode and order_region safe strings
df['shipping_mode_clean'] = df[cols['shipping_mode']].astype(str).fillna("Unknown")
df['order_region_clean'] = df[cols['order_region']].astype(str).fillna("Unknown")
df['order_country_clean'] = df[cols['order_country']].astype(str).fillna("Unknown")

# numeric columns
if cols['sales_per_customer']:
    df['sales_per_customer_num'] = pd.to_numeric(df[cols['sales_per_customer']], errors='coerce').fillna(0)
else:
    df['sales_per_customer_num'] = 0.0

if cols['profit_per_order']:
    df['profit_per_order_num'] = pd.to_numeric(df[cols['profit_per_order']], errors='coerce').fillna(0)
else:
    df['profit_per_order_num'] = 0.0

if cols['order_item_quantity']:
    df['quantity_num'] = pd.to_numeric(df[cols['order_item_quantity']], errors='coerce').fillna(0)
else:
    df['quantity_num'] = 0

# ---------------------------
# Sidebar filters (order_region default = all)
# ---------------------------
st.sidebar.header("Filters (applies to charts)")

regions = ["All"] + sorted(df['order_region_clean'].unique().tolist())
region_choice = st.sidebar.selectbox("Order Region", regions)

if region_choice == "All":
    df_view = df.copy()
else:
    df_view = df[df['order_region_clean'] == region_choice].copy()

# ---------------------------
# SECTION: Summary KPIs
# ---------------------------
st.markdown("## Summary Metrics")
total = len(df_view)
delayed_count = int(df_view['is_delayed'].sum())
delayed_pct = round((delayed_count / total * 100) if total>0 else 0,2)

col1, col2, col3 = st.columns(3)
col1.metric("Total Orders (view)", total)
col2.metric("Delayed (label=-1) count", delayed_count)
col3.metric("Delayed % (view)", f"{delayed_pct}%")

st.markdown("---")

# ---------------------------
# 1) Shipping mode: delay % for each mode & most delayed mode
# ---------------------------
st.subheader("1) Delay % by Shipping Mode")

ship_group = df_view.groupby('shipping_mode_clean').agg(
    total_orders=('shipping_mode_clean','count'),
    delayed_orders=('is_delayed','sum')
).reset_index()
ship_group['delay_pct'] = ship_group['delayed_orders'] / ship_group['total_orders'] * 100
ship_group = ship_group.sort_values('delay_pct', ascending=False)

st.dataframe(ship_group[['shipping_mode_clean','total_orders','delayed_orders','delay_pct']].rename(
    columns={'shipping_mode_clean':'Shipping Mode','total_orders':'Total Orders','delayed_orders':'Delayed Orders','delay_pct':'Delay %'}
), use_container_width=True)

if not ship_group.empty:
    top_mode = ship_group.iloc[0]
    st.warning(f"Most delayed shipping mode: **{top_mode['shipping_mode_clean']}** — Delay% = {top_mode['delay_pct']:.2f} ( {int(top_mode['delayed_orders'])} delayed / {int(top_mode['total_orders'])} total )")

fig_ship = px.bar(ship_group, x='shipping_mode_clean', y='delay_pct', title="Delay % by Shipping Mode", labels={'shipping_mode_clean':'Shipping Mode','delay_pct':'Delay %'})
st.plotly_chart(fig_ship, use_container_width=True)

st.markdown("---")

# ---------------------------
# 2) Order region: delay % for each region
# ---------------------------
st.subheader("2) Delay % by Order Region")

reg_group = df_view.groupby('order_region_clean').agg(
    total_orders=('order_region_clean','count'),
    delayed_orders=('is_delayed','sum')
).reset_index()
reg_group['delay_pct'] = reg_group['delayed_orders'] / reg_group['total_orders'] * 100
reg_group = reg_group.sort_values('delay_pct', ascending=False)

st.dataframe(reg_group[['order_region_clean','total_orders','delayed_orders','delay_pct']].rename(
    columns={'order_region_clean':'Order Region','total_orders':'Total Orders','delayed_orders':'Delayed Orders','delay_pct':'Delay %'}
), use_container_width=True)

if not reg_group.empty:
    top_reg = reg_group.iloc[0]
    st.info(f"Highest delay region: **{top_reg['order_region_clean']}** — Delay% = {top_reg['delay_pct']:.2f} ( {int(top_reg['delayed_orders'])} delayed / {int(top_reg['total_orders'])} total )")

fig_reg = px.bar(reg_group, x='order_region_clean', y='delay_pct', title="Delay % by Order Region", labels={'order_region_clean':'Order Region','delay_pct':'Delay %'})
st.plotly_chart(fig_reg, use_container_width=True)

st.markdown("---")

# ---------------------------
# 3) Average sales per customer based on order_region
# ---------------------------
st.subheader("3) Average Sales per Customer by Order Region (Sales per customer)")

if cols['sales_per_customer']:
    avg_sales = df_view.groupby('order_region_clean')['sales_per_customer_num'].mean().reset_index().sort_values('sales_per_customer_num', ascending=False)
    st.dataframe(avg_sales.rename(columns={'order_region_clean':'Order Region','sales_per_customer_num':'Avg Sales per Customer'}), use_container_width=True)
    fig = px.bar(avg_sales, x='order_region_clean', y='sales_per_customer_num', title="Avg Sales per Customer by Region", labels={'order_region_clean':'Order Region','sales_per_customer_num':'Avg Sales'})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Sales per customer column not found; skipping this analysis.")

st.markdown("---")

# ---------------------------
# 4) Average Profit per order based on order_region
# ---------------------------
st.subheader("4) Average Profit per Order by Order Region (Profit per order)")

if cols['profit_per_order']:
    avg_profit = df_view.groupby('order_region_clean')['profit_per_order_num'].mean().reset_index().sort_values('profit_per_order_num', ascending=False)
    st.dataframe(avg_profit.rename(columns={'order_region_clean':'Order Region','profit_per_order_num':'Avg Profit per Order'}), use_container_width=True)
    fig = px.bar(avg_profit, x='order_region_clean', y='profit_per_order_num', title="Avg Profit per Order by Region", labels={'order_region_clean':'Order Region','profit_per_order_num':'Avg Profit'})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Profit per order column not found; skipping this analysis.")

st.markdown("---")

# ---------------------------
# 5) Top 5 order country and order region marketwise
# ---------------------------
st.subheader("5) Top 5 Order Countries & Regions by Volume")

country_counts = df_view['order_country_clean'].value_counts().head(5).reset_index()
country_counts.columns = ['Order Country','Orders']
region_counts = df_view['order_region_clean'].value_counts().head(5).reset_index()
region_counts.columns = ['Order Region','Orders']

c1, c2 = st.columns(2)
c1.write("Top 5 Countries (by orders)")
c1.dataframe(country_counts, use_container_width=True)
c2.write("Top 5 Regions (by orders)")
c2.dataframe(region_counts, use_container_width=True)

st.markdown("---")

# ---------------------------
# 6) Top 8 most profitable category_name
# ---------------------------
st.subheader("6) Top 8 Most Profitable Categories")

if cols['category_name'] and cols['profit_per_order']:
    cat_profit = df_view.groupby(norm("category name"))['profit_per_order_num'].sum() if False else None
# Since we normalized columns, use original mapping:
cat_col = cols['category_name']
if cat_col and cols['profit_per_order']:
    top8 = df_view.groupby(cat_col)['profit_per_order_num'].sum().sort_values(ascending=False).head(8).reset_index()
    top8.columns = ['Category','Total Profit']
    st.dataframe(top8, use_container_width=True)
    st.plotly_chart(px.bar(top8, x='Category', y='Total Profit', title="Top 8 Profitable Categories"), use_container_width=True)
else:
    st.warning("Category name and/or profit per order columns not found; skipping category profitability.")

st.markdown("---")

# ---------------------------
# 7) Most profitable product for different order_region
# ---------------------------
st.subheader("7) Most Profitable Product by Order Region")

prod_col = cols['product_name']
if prod_col and cols['profit_per_order']:
    prod_region = df_view.groupby([ 'order_region_clean', prod_col])['profit_per_order_num'].sum().reset_index()
    idx = prod_region.groupby('order_region_clean')['profit_per_order_num'].idxmax()
    best_prod = prod_region.loc[idx].reset_index(drop=True)
    best_prod = best_prod.rename(columns={'order_region_clean':'Order Region', prod_col:'Product Name','profit_per_order_num':'Total Profit'})
    st.dataframe(best_prod, use_container_width=True)
else:
    st.warning("Product name and/or profit per order columns not found; skipping product profitability.")

st.markdown("---")

# ---------------------------
# 8) Top 5 most sold category name based on quantity and revenue
# ---------------------------
st.subheader("8) Top 5 Most Sold Categories (Quantity & Revenue)")

if cols['category_name']:
    cat_qty = df_view.groupby(cols['category_name'])['quantity_num'].sum().sort_values(ascending=False).head(5).reset_index()
    cat_qty.columns = ['Category','Total Quantity']
    st.dataframe(cat_qty, use_container_width=True)
else:
    st.warning("Category column missing for quantity ranking.")

if cols['category_name'] and cols['sales_per_customer']:
    cat_rev = df_view.groupby(cols['category_name'])['sales_per_customer_num'].sum().sort_values(ascending=False).head(5).reset_index()
    cat_rev.columns = ['Category','Total Revenue (sales_per_customer)']
    st.dataframe(cat_rev, use_container_width=True)
else:
    st.warning("Category and/or sales columns missing for revenue ranking.")

st.markdown("---")

# ---------------------------
# 9) Preferred shipping mode based on order_region
# ---------------------------
st.subheader("9) Preferred Shipping Mode by Order Region")

pref = df_view.groupby(['order_region_clean','shipping_mode_clean']).size().reset_index(name='count')
fig_pref = px.bar(pref, x='order_region_clean', y='count', color='shipping_mode_clean', barmode='group', labels={'order_region_clean':'Order Region','shipping_mode_clean':'Shipping Mode','count':'Orders'}, title="Preferred Shipping Mode by Region")
st.plotly_chart(fig_pref, use_container_width=True)

st.markdown("---")

# ---------------------------
# 10) Delayed orders based on shipping mode (counts)
# ---------------------------
st.subheader("10) Delayed Orders by Shipping Mode (counts)")

delayed_by_mode = df_view[df_view['is_delayed']].groupby('shipping_mode_clean').size().reset_index(name='delayed_count').sort_values('delayed_count', ascending=False)
st.dataframe(delayed_by_mode, use_container_width=True)
st.plotly_chart(px.bar(delayed_by_mode, x='shipping_mode_clean', y='delayed_count', title="Delayed Orders by Shipping Mode", labels={'shipping_mode_clean':'Shipping Mode','delayed_count':'Delayed Orders'}), use_container_width=True)

st.markdown("---")

# ---------------------------
# 11) Order_region wise delay % (already shown earlier but show table)
# ---------------------------
st.subheader("11) Order Region — Delay % and counts")
st.dataframe(reg_group.rename(columns={'order_region_clean':'Order Region','total_orders':'Total Orders','delayed_orders':'Delayed Orders','delay_pct':'Delay %'}), use_container_width=True)

# ---------------------------
# Raw data download and preview
# ---------------------------
st.markdown("---")
st.subheader("Dataset preview (first 200 rows)")
st.dataframe(df_view.head(200), use_container_width=True)

csv = df_view.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered dataset (CSV)", data=csv, file_name="delay_analysis_filtered.csv", mime="text/csv")
