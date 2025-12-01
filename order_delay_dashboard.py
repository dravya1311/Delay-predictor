# order_delay_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Order Delay Analysis", layout="wide")

# ---------- CONFIG ----------
CSV_URL = "https://raw.githubusercontent.com/dravya1311/Delay-predictor/main/Delay_Model.csv"

# ---------- HELPERS ----------
def norm(c: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower())

def load_csv(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        df.columns = [col.strip() for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Failed to load CSV from GitHub: {e}")
        return pd.DataFrame()

def map_cols(df: pd.DataFrame):
    """Return mapping of normalized_name -> actual column name present in df."""
    return {norm(c): c for c in df.columns}

def safe_col(mapper, *names):
    """Return actual column name for the first normalized name that exists, else None."""
    for n in names:
        key = norm(n)
        if key in mapper:
            return mapper[key]
    return None

# ---------- LOAD DATA ----------
df = load_csv(CSV_URL)
if df.empty:
    st.stop()

col_map = map_cols(df)

# ---------- REQUIRED COLUMNS (user-provided names) ----------
# We'll search among these normalized variants
COL_LABEL = safe_col(col_map, "label")
COL_SHIP = safe_col(col_map, "Shipping mode", "shipping mode", "shipping_mode", "shippingmode")
COL_REGION = safe_col(col_map, "Order region", "order region", "order_region", "region")
COL_SALES = safe_col(col_map, "Sales per customer", "sales per customer", "sales_per_customer", "sales")
COL_ORDERID = safe_col(col_map, "Order id", "order id", "Order_ID", "order_id")

missing = []
if COL_LABEL is None: missing.append("label")
if COL_SHIP is None: missing.append("Shipping mode")
if COL_REGION is None: missing.append("Order region")
if COL_SALES is None: missing.append("Sales per customer")

if missing:
    st.error(f"Missing required column(s): {missing}. Check CSV header names and try again.")
    st.stop()

# ---------- NORMALIZE LABEL (user rule) ----------
# User: -1 = delayed, 0 = on-time, 1 = early
df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors="coerce")
# force any invalid to NaN, then keep them (we'll treat NaN as not delayed)
df[COL_LABEL] = df[COL_LABEL].fillna(0).astype(int)

# For clarity create helper columns
df["_is_delayed"] = df[COL_LABEL] == -1
df["_is_early"] = df[COL_LABEL] == 1
df["_is_ontime"] = df[COL_LABEL] == 0

# ---------- METRICS ----------
total_orders = len(df)
total_delayed = int(df["_is_delayed"].sum())
delay_pct_overall = round(total_delayed / total_orders * 100, 2) if total_orders else 0.0
most_delayed_mode = None

# ---------- SHIPPING MODE: delay count & percent ----------
ship_grp = df.groupby(COL_SHIP).agg(
    total_orders=("{}".format(COL_SHIP), "count"),
    delayed_count=("_is_delayed", "sum")
).reset_index()
ship_grp["delay_pct"] = (ship_grp["delayed_count"] / ship_grp["total_orders"] * 100).round(2)
ship_grp = ship_grp.sort_values("delay_pct", ascending=False).reset_index(drop=True)
if not ship_grp.empty:
    most_delayed_mode = ship_grp.iloc[0][COL_SHIP]

# ---------- ORDER REGION: delay count & percent ----------
region_grp = df.groupby(COL_REGION).agg(
    total_orders=("{}".format(COL_REGION), "count"),
    delayed_count=("_is_delayed", "sum")
).reset_index()
region_grp["delay_pct"] = (region_grp["delayed_count"] / region_grp["total_orders"] * 100).round(2)
region_grp = region_grp.sort_values("delay_pct", ascending=False).reset_index(drop=True)

# ---------- LAYOUT ----------
st.title("Order Delay Analysis — (label: -1=Delayed, 0=On-time, 1=Early)")
st.markdown("**Key summary** — delay = `label == -1`")

k1, k2, k3, k4 = st.columns([1.2,1.2,1.2,1.4])
k1.metric("Total Orders", f"{total_orders:,}")
k2.metric("Total Delayed (label = -1)", f"{total_delayed:,}")
k3.metric("Overall Delay %", f"{delay_pct_overall}%")
k4.metric("Most Delayed Shipping Mode", most_delayed_mode if most_delayed_mode is not None else "N/A")

st.markdown("---")

# ---------- SHIPPING MODE VISUALS ----------
st.subheader("Delay % by Shipping Mode")
st.write("Delay % = delayed_count / total_orders for that shipping mode")

if ship_grp.empty:
    st.info("No shipping mode data available.")
else:
    st.dataframe(ship_grp[[COL_SHIP, "total_orders", "delayed_count", "delay_pct"]], use_container_width=True)
    fig_ship = px.bar(ship_grp, x=COL_SHIP, y="delay_pct", text="delay_pct",
                      title="Delay % by Shipping Mode (higher = worse)", labels={"delay_pct":"Delay %"})
    fig_ship.update_traces(texttemplate="%{text}%")
    st.plotly_chart(fig_ship, use_container_width=True)

st.markdown("---")

# ---------- ORDER REGION VISUALS ----------
st.subheader("Delay % by Order Region")
st.write("Delay % = delayed_count / total_orders for that region")

if region_grp.empty:
    st.info("No order region data available.")
else:
    st.dataframe(region_grp[[COL_REGION, "total_orders", "delayed_count", "delay_pct"]], use_container_width=True)
    fig_reg = px.bar(region_grp, x=COL_REGION, y="delay_pct", text="delay_pct",
                     title="Delay % by Order Region", labels={"delay_pct":"Delay %"})
    fig_reg.update_traces(texttemplate="%{text}%")
    st.plotly_chart(fig_reg, use_container_width=True)

st.markdown("---")

# ---------- OPTIONAL: export results ----------
with st.expander("Export / download results"):
    ship_csv = ship_grp.to_csv(index=False)
    reg_csv = region_grp.to_csv(index=False)
    st.download_button("Download shipping-mode delay table (CSV)", ship_csv, "delay_by_shipping_mode.csv", "text/csv")
    st.download_button("Download region delay table (CSV)", reg_csv, "delay_by_region.csv", "text/csv")

st.success("Analysis complete.")
