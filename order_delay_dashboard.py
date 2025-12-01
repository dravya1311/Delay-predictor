# order_delay_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, re

# -------------------------
# Page config — Modern Minimal
# -------------------------
st.set_page_config(
    page_title="Order Delay — Modern Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# HEADER — PROFESSIONAL DASHBOARD TITLE & BRANDING
# ============================================================

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

st.markdown(" ")

# Minimal styling helper
ACCENT = "#0B6EFD"        # primary accent (blue)
ALERT = "#E03E3E"         # delay red
GOOD = "#2CB67D"          # early/ok green
NEUTRAL = "#6C757D"       # neutral gray

# -------------------------
# Helpers
# -------------------------
def norm_col(c: str) -> str:
    return re.sub(r'[^0-9a-z]+', '_', str(c).strip().lower())

def try_load():
    # try local first
    local_names = ["Delay_Model.csv", "Delay Model.csv", "Delay-Model.csv", "DelayModel.csv"]
    for p in local_names:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    # try GitHub raw variants
    base = "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/"
    gh_names = ["Delay_Model.csv", "Delay%20Model.csv", "Delay-Model.csv", "DelayModel.csv"]
    for n in gh_names:
        try:
            return pd.read_csv(base + n)
        except Exception:
            continue
    return None

# -------------------------
# Load data
# -------------------------
df_raw = try_load()
if df_raw is None:
    st.error("Could not find `Delay_Model.csv` (local repo root or GitHub). Place the file in repo root or ensure GitHub raw URL exists.")
    st.stop()

# Normalize column names for robust mapping
orig_cols = list(df_raw.columns)
df = df_raw.copy()
df.columns = [norm_col(c) for c in df.columns]

# Mapping: expected columns (based on your provided headers)
map_candidates = {
    "label": ["label"],
    "shipping_mode": ["shipping_mode", "shipping_mode", "shipping_mode", "shipping_mode"],  # fallback
    "order_region": ["order_region", "order_region", "order_region"],
    "order_country": ["order_country", "order_country"],
    "sales_per_customer": ["sales_per_customer", "sales_per_customer", "sales"],
    "profit_per_order": ["profit_per_order", "profit_per_order", "profit_per_order", "profit"],
    "category_name": ["category_name", "category_name"],
    "product_name": ["product_name", "product_name"],
    "order_item_quantity": ["order_item_quantity", "order_item_quantity", "order_item_quantity", "order_item_quantity"]
}

def find_first(df_cols, candidates):
    for c in candidates:
        key = norm_col(c)
        if key in df_cols:
            return key
    return None

cols = {}
cols["label"] = find_first(df.columns, map_candidates["label"])
cols["shipping_mode"] = find_first(df.columns, ["shipping_mode", "shipping_mode", "shipping_mode", "shipping_mode", "shipping_mode", "shippingmode", "shipping_mode"])
cols["order_region"] = find_first(df.columns, ["order_region", "order_region", "order_region", "order_region", "orderregion"])
cols["order_country"] = find_first(df.columns, ["order_country","order_country","ordercountry"])
cols["sales_per_customer"] = find_first(df.columns, ["sales_per_customer","sales_per_customer","sales","sales_per_customer"])
cols["profit_per_order"] = find_first(df.columns, ["profit_per_order","profit_per_order","profit_per_order","profit"])
cols["category_name"] = find_first(df.columns, ["category_name","category_name","category_name"])
cols["product_name"] = find_first(df.columns, ["product_name","product_name","product_name"])
cols["order_item_quantity"] = find_first(df.columns, ["order_item_quantity","order_item_quantity","order_item_quantity","order_item_quantity","order_item_quantity","order_item_quantity"])

# Minimal validation (label, shipping_mode, order_region are critical)
critical = [("label", cols["label"]), ("shipping_mode", cols["shipping_mode"]), ("order_region", cols["order_region"])]
missing = [name for name,key in critical if key is None]
if missing:
    st.error(f"Critical columns missing from dataset: {missing}\nAvailable columns: {orig_cols}")
    st.stop()

# -------------------------
# Clean & derived fields
# -------------------------
df["label_num"] = pd.to_numeric(df[cols["label"]], errors="coerce")  # user specified: -1 = delayed, 0 = on-time, 1 = early
# Interpret label mapping: as per your instruction -1 = delayed
df["is_delayed"] = df["label_num"] == -1
df["is_on_time"] = df["label_num"] == 0
df["is_early"] = df["label_num"] == 1

df["shipping_mode"] = df[cols["shipping_mode"]].astype(str).fillna("Unknown")
df["order_region"] = df[cols["order_region"]].astype(str).fillna("Unknown")
df["order_country"] = df[cols["order_country"]].astype(str).fillna("Unknown")

if cols["sales_per_customer"]:
    df["sales_per_customer"] = pd.to_numeric(df[cols["sales_per_customer"]], errors="coerce").fillna(0.0)
else:
    df["sales_per_customer"] = 0.0

if cols["profit_per_order"]:
    df["profit_per_order"] = pd.to_numeric(df[cols["profit_per_order"]], errors="coerce").fillna(0.0)
else:
    df["profit_per_order"] = 0.0

if cols["order_item_quantity"]:
    df["quantity_num"] = pd.to_numeric(df[cols["order_item_quantity"]], errors="coerce").fillna(0)
else:
    df["quantity_num"] = 0

# -------------------------
# Sidebar filter (region)
# -------------------------
st.sidebar.header("Filter")
regions = ["All"] + sorted(df["order_region"].unique().tolist())
selected_region = st.sidebar.selectbox("Order Region", regions, index=0)
df_view = df if selected_region == "All" else df[df["order_region"] == selected_region]

# -------------------------
# KPI cards — top row
# -------------------------
total = len(df_view)
delayed_cnt = int(df_view["is_delayed"].sum())
delay_pct = (delayed_cnt / total * 100) if total else 0.0
on_time_pct = (df_view["is_on_time"].sum() / total * 100) if total else 0.0
early_pct = (df_view["is_early"].sum() / total * 100) if total else 0.0

k1, k2, k3, k4 = st.columns([2,2,2,3])
k1.metric("Total orders (view)", f"{total:,}")
k2.metric("Delayed (%)", f"{delay_pct:.2f}%", delta=f"{delayed_cnt:,} delayed", delta_color="inverse")
k3.metric("On-time (%)", f"{on_time_pct:.2f}%")
k4.metric("Early (%)", f"{early_pct:.2f}%")

st.markdown("---")

# -------------------------
# Chart 1: Delay % by Shipping Mode (sorted, accent + alert)
# -------------------------
st.subheader("Delay % by Shipping Mode")
ship = df_view.groupby("shipping_mode").agg(
    total_orders=("shipping_mode","count"),
    delayed_orders=("is_delayed","sum")
).reset_index()
ship["delay_pct"] = ship["delayed_orders"] / ship["total_orders"] * 100
ship = ship.sort_values("delay_pct", ascending=False)

if ship.empty:
    st.info("No shipping-mode data available for the selected filter.")
else:
    # highlight highest bar
    colors = [ALERT if r==ship.iloc[0]["shipping_mode"] else ACCENT for r in ship["shipping_mode"]]
    fig = go.Figure(go.Bar(
        x=ship["shipping_mode"],
        y=ship["delay_pct"],
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Delay %: %{y:.2f}%<br>Delayed orders: %{customdata[0]}<br>Total: %{customdata[1]}<extra></extra>",
        customdata=np.stack([ship["delayed_orders"], ship["total_orders"]], axis=1)
    ))
    fig.update_layout(template="simple_white", yaxis_title="Delay %", xaxis_title="", bargap=0.25,
                      title_text="Delay % by Shipping Mode — highest highlighted in red")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Chart 2: Delay % by Order Region
# -------------------------
st.subheader("Delay % by Order Region")
reg = df_view.groupby("order_region").agg(total_orders=("order_region","count"), delayed_orders=("is_delayed","sum")).reset_index()
reg["delay_pct"] = reg["delayed_orders"] / reg["total_orders"] * 100
reg = reg.sort_values("delay_pct", ascending=False)

if reg.empty:
    st.info("No order-region data available.")
else:
    fig2 = px.bar(reg, x="order_region", y="delay_pct",
                  color="delay_pct", color_continuous_scale=["#ffd9d9", ALERT],
                  labels={"order_region":"Order Region","delay_pct":"Delay %"},
                  title="Delay % by Order Region")
    fig2.update_traces(hovertemplate="<b>%{x}</b><br>Delay %: %{y:.2f}%<br>Delayed: %{customdata[0]}<extra></extra>",
                       customdata=np.stack([reg["delayed_orders"]], axis=1))
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Chart 3: Avg Sales per Customer by Region
# -------------------------
st.subheader("Average Sales per Customer — Region")
if "sales_per_customer" in df_view.columns:
    avg_sales = df_view.groupby("order_region")["sales_per_customer"].mean().reset_index().sort_values("sales_per_customer", ascending=False)
    fig3 = px.bar(avg_sales, x="order_region", y="sales_per_customer", labels={"sales_per_customer":"Avg Sales"}, title="Avg Sales per Customer by Region", color_discrete_sequence=[ACCENT])
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("`Sales per customer` column not found — skipping this chart.")

# -------------------------
# Chart 4: Avg Profit per Order by Region
# -------------------------
st.subheader("Average Profit per Order — Region")
if "profit_per_order" in df_view.columns:
    avg_profit = df_view.groupby("order_region")["profit_per_order"].mean().reset_index().sort_values("profit_per_order", ascending=False)
    fig4 = px.bar(avg_profit, x="order_region", y="profit_per_order", labels={"profit_per_order":"Avg Profit"}, title="Avg Profit per Order by Region", color_discrete_sequence=[ACCENT])
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("`Profit per order` column not found — skipping this chart.")

# -------------------------
# Chart 5: Top 5 Countries & Regions by Orders (compact side-by-side)
# -------------------------
st.subheader("Top Markets — Countries & Regions (Top 5)")
top_c = df_view["order_country"].value_counts().nlargest(5).reset_index()
top_c.columns = ["order_country","orders"]
top_r = df_view["order_region"].value_counts().nlargest(5).reset_index()
top_r.columns = ["order_region","orders"]

fig5 = make_subplots_rows = None  # placeholder to avoid linters complaining

fig5 = go.Figure()
fig5.add_trace(go.Bar(x=top_c["order_country"], y=top_c["orders"], name="Countries", marker_color=ACCENT, text=top_c["orders"], textposition='auto'))
fig5.add_trace(go.Bar(x=top_r["order_region"], y=top_r["orders"], name="Regions", marker_color=NEUTRAL, text=top_r["orders"], textposition='auto'))
fig5.update_layout(barmode="group", title="Top 5 Countries & Regions by Orders", template="simple_white", xaxis_title="")
st.plotly_chart(fig5, use_container_width=True)

# -------------------------
# Chart 6: Top 8 Most Profitable Categories
# -------------------------
st.subheader("Top 8 Most Profitable Categories")
if cols["category_name"] and "profit_per_order" in df_view.columns:
    cat_profit = df_view.groupby(cols["category_name"])["profit_per_order"].sum().nlargest(8).reset_index()
    fig6 = px.bar(cat_profit, x=cols["category_name"], y="profit_per_order", title="Top 8 Profitable Categories", color_discrete_sequence=[ACCENT])
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("Category or profit column missing — skipping category profitability chart.")

# -------------------------
# Chart 7: Most Profitable Product per Region (small multiples)
# -------------------------
st.subheader("Most Profitable Product by Region")
if cols["product_name"] and "profit_per_order" in df_view.columns:
    prod_region = df_view.groupby(["order_region", cols["product_name"]])["profit_per_order"].sum().reset_index()
    # get top product per region
    idx = prod_region.groupby("order_region")["profit_per_order"].idxmax()
    top_products = prod_region.loc[idx].reset_index(drop=True)
    fig7 = px.bar(top_products, x="order_region", y="profit_per_order", color=cols["product_name"],
                  title="Most Profitable Product per Region", labels={"profit_per_order":"Total Profit"})
    st.plotly_chart(fig7, use_container_width=True)
else:
    st.info("Product or profit column missing — skipping product-by-region chart.")

# -------------------------
# Chart 8: Top 5 Most Sold Categories (Quantity & Revenue) — dual charts
# -------------------------
st.subheader("Top 5 Categories — Quantity & Revenue")
if cols["category_name"]:
    if "quantity_num" in df_view.columns:
        cat_qty = df_view.groupby(cols["category_name"])["quantity_num"].sum().nlargest(5).reset_index()
        fig8a = px.bar(cat_qty, x=cols["category_name"], y="quantity_num", title="Top 5 Categories by Quantity", color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig8a, use_container_width=True)
    if "sales_per_customer" in df_view.columns:
        cat_rev = df_view.groupby(cols["category_name"])["sales_per_customer"].sum().nlargest(5).reset_index()
        fig8b = px.bar(cat_rev, x=cols["category_name"], y="sales_per_customer", title="Top 5 Categories by Revenue", color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig8b, use_container_width=True)
else:
    st.info("Category column missing — skipping sold-category charts.")

# -------------------------
# Chart 9: Preferred shipping mode by region (stacked)
# -------------------------
st.subheader("Preferred Shipping Mode by Region")
pref = df_view.groupby(["order_region","shipping_mode"]).size().reset_index(name="count")
if not pref.empty:
    fig9 = px.bar(pref, x="order_region", y="count", color="shipping_mode", barmode="stack",
                  title="Preferred Shipping Mode by Region", labels={"count":"Orders"})
    st.plotly_chart(fig9, use_container_width=True)
else:
    st.info("Insufficient data for shipping-mode by region.")

# -------------------------
# Chart 10: Delayed orders by shipping mode (counts)
# -------------------------
st.subheader("Delayed Orders by Shipping Mode (counts)")
delayed_mode = df_view[df_view["is_delayed"]].groupby("shipping_mode").size().reset_index(name="delayed_count").sort_values("delayed_count", ascending=False)
if not delayed_mode.empty:
    fig10 = px.bar(delayed_mode, x="shipping_mode", y="delayed_count",
                   title="Delayed Orders by Shipping Mode (counts)",
                   color="shipping_mode")
    st.plotly_chart(fig10, use_container_width=True)
else:
    st.info("No delayed orders present for current filter.")

# -------------------------
# FIX: Compute reg_grp_all BEFORE using it
# -------------------------
reg_grp_all = (
    df_view.groupby("order_region")["is_delayed"]
    .mean()
    .reset_index()
)
reg_grp_all["delay_percent"] = reg_grp_all["is_delayed"] * 100

# -------------------------
# Chart 11: Order-region wise delay % (donut)
# -------------------------
st.subheader("Order-region — Delay % Distribution (donut)")

if not reg_grp_all.empty:
    fig11 = go.Figure(
        go.Pie(
            labels=reg_grp_all["order_region"],
            values=reg_grp_all["delay_percent"],
            hole=0.45
        )
    )
    fig11.update_layout(
        title="Delay % by Region",
        legend_title="Region"
    )
    st.plotly_chart(fig11, use_container_width=True)
else:
    st.info("No region delay data to show.")

# -------------------------
# Footer notes
# -------------------------
st.markdown("---")
st.markdown(
    "Dashboard by RAVINDRA YADAV . "
    
    "Select an Order Region in the sidebar to filter all charts."
)
