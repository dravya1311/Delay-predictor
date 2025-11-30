# order_delay_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os
import re

st.set_page_config(page_title="Order Delay & Analytics Dashboard", layout="wide")

# -------------------------
# Utility helpers
# -------------------------
def norm_col(c):
    if pd.isna(c):
        return c
    c2 = str(c).strip()
    # normalize: lowercase, replace spaces and punctuation with underscore
    c2 = re.sub(r'[^0-9a-zA-Z]+', '_', c2).lower().strip('_')
    return c2

def load_csv_try(paths):
    """Try list of possible file paths, return first successfully read DataFrame or None."""
    for p in paths:
        try:
            if p.startswith("http://") or p.startswith("https://"):
                df = pd.read_csv(p)
            else:
                # try local file name directly
                if os.path.exists(p):
                    df = pd.read_csv(p)
                else:
                    continue
            return df
        except Exception:
            continue
    return None

def find_common_key(df1, df2):
    """Find a sensible join key between two dataframes using normalized column names."""
    cols1 = {norm_col(c): c for c in df1.columns}
    cols2 = {norm_col(c): c for c in df2.columns}
    common = set(cols1.keys()).intersection(set(cols2.keys()))
    # prefer variants of order_id, order_no, order
    for candidate in ['order_id','order_no','order_number','order','orderid','order_no']:
        if candidate in common:
            return cols1[candidate], cols2[candidate]
    # otherwise return any common
    if common:
        k = next(iter(common))
        return cols1[k], cols2[k]
    return None, None

def map_column(df, candidates):
    """Return first matching column name in df for any candidate names (normalized)."""
    normalized = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = norm_col(cand)
        if key in normalized:
            return normalized[key]
    return None

def safe_rename(df, mapping):
    """Rename columns in df based on mapping of standard_name -> actual column name"""
    rename_map = {v: k for k, v in mapping.items() if v in df.columns}
    # invert mapping: actual->standard
    rename_map_inverted = {}
    for std, actual in mapping.items():
        if actual in df.columns:
            rename_map_inverted[actual] = std
    if rename_map_inverted:
        df = df.rename(columns=rename_map_inverted)
    return df

def map_delay_values(s):
    """Map delay values (strings/numbers) to -1/0/1 if possible."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.integer, np.floating)):
        try:
            v = int(s)
            if v in (-1, 0, 1):
                return v
            # if 0/1 encoded as floats
            if v in (0,1):
                return v
            return np.nan
        except Exception:
            return np.nan
    ss = str(s).strip().lower()
    ss = re.sub(r'[^0-9a-z\-]+', '_', ss)
    mapping = {
        '0': 0, 'on_time': 0, 'on-time': 0, 'ontime': 0, 'on_time_0': 0, '0_ontime': 0,
        '1': 1, 'delayed': 1, 'late': 1, 'delay': 1, '1_delayed': 1,
        '-1': -1, 'early': -1, 'early_delivery': -1, 'early-delivery': -1, 'earlydelivery': -1
    }
    # try direct mapping
    if ss in mapping:
        return mapping[ss]
    # check if numeric string
    if ss.isdigit() or (ss.startswith('-') and ss[1:].isdigit()):
        try:
            v = int(ss)
            if v in (-1,0,1):
                return v
        except:
            pass
    # try contains keywords
    if 'early' in ss:
        return -1
    if 'delayed' in ss or 'late' in ss:
        return 1
    if 'on' in ss and 'time' in ss:
        return 0
    # unknown
    return np.nan

# -------------------------
# File detection & load
# -------------------------
st.sidebar.header("Data files & load options")

# Candidate local filenames (common variants)
candidate_model_files = [
    "Delay Model.csv", "Delay_Model.csv", "delay_model.csv", "delay model.csv",
    "DelayModel.csv", "Delay-Model.csv"
]
candidate_desc_files = [
    "Delay description csv.csv", "Delay_description_csv.csv", "delay_description.csv",
    "Delay description.csv", "Delay_description.csv", "delay description csv.csv"
]

# Allow user to override via sidebar text inputs (useful for custom names)
model_path_input = st.sidebar.text_input("Model CSV filename (leave blank to auto-detect)", value="")
desc_path_input = st.sidebar.text_input("Delay-desc CSV filename (leave blank to auto-detect)", value="")

if model_path_input.strip():
    candidate_model_files.insert(0, model_path_input.strip())
if desc_path_input.strip():
    candidate_desc_files.insert(0, desc_path_input.strip())

st.sidebar.write("Files in repo root (visibility):")
try:
    st.sidebar.write(os.listdir("."))
except Exception:
    st.sidebar.write("Could not list files in current directory.")

# Also accept raw GitHub URLs via inputs if required
model_url_input = st.sidebar.text_input("Model CSV raw GitHub URL (optional)", value="")
desc_url_input = st.sidebar.text_input("Delay-desc CSV raw GitHub URL (optional)", value="")

if model_url_input.strip():
    candidate_model_files.insert(0, model_url_input.strip())
if desc_url_input.strip():
    candidate_desc_files.insert(0, desc_url_input.strip())

# Try load
df_model = load_csv_try(candidate_model_files)
df_desc = load_csv_try(candidate_desc_files)

if df_model is None:
    st.error("Could not load the model CSV. Tried: " + ", ".join(candidate_model_files[:5]))
    st.stop()

if df_desc is None:
    st.error("Could not load the delay description CSV. Tried: " + ", ".join(candidate_desc_files[:5]))
    st.stop()

st.success("Files loaded successfully.")

# -------------------------
# Normalize columns & detect join key
# -------------------------
# Keep original columns backup
orig_cols_model = list(df_model.columns)
orig_cols_desc = list(df_desc.columns)

# Normalize column names (strip) but keep actual names for mapping
df_model.columns = [c.strip() for c in df_model.columns]
df_desc.columns = [c.strip() for c in df_desc.columns]

# Attempt to find a join key
mkey_model, mkey_desc = find_common_key(df_model, df_desc)
if mkey_model is None:
    st.warning("No obvious common key found between the two files. Displaying columns for diagnosis.")
    st.write("Model file columns:", orig_cols_model)
    st.write("Desc file columns:", orig_cols_desc)
    st.error("Please ensure both files share a common order identifier column (e.g. Order_ID).")
    st.stop()

st.info(f"Auto-detected join key: '{mkey_model}' (model)  <->  '{mkey_desc}' (desc)")

# Merge
try:
    data = pd.merge(df_model, df_desc, left_on=mkey_model, right_on=mkey_desc, how="left", suffixes=("_m","_d"))
except Exception as e:
    st.error(f"Merge failed: {e}")
    st.stop()

# -------------------------
# Standardize important columns (map synonyms)
# -------------------------
# Define candidate synonyms for fields we need
candidates = {
    "order_id": ["Order_ID","order_id","order id","orderno","order_no","order number","OrderID","orderid"],
    "order_region": ["Order_Region","order_region","order region","region"],
    "order_country": ["Order_Country","order_country","order country","country"],
    "sales": ["Sales","sales","order_amount","order_value","revenue","total_sales"],
    "profit": ["Profit","profit","profit_margin","order_profit"],
    "category_name": ["Category_Name","category_name","category","product_category"],
    "product_name": ["Product_Name","product_name","product","item_name","item"],
    "quantity": ["Quantity","quantity","qty","order_quantity"],
    "ship_mode": ["Ship_Mode","ship_mode","shipping_mode","shipmode","shipping"],
    "delay_status": ["Delay_Status","delay_status","delay_flag","is_delayed","delayed","delay"]
}

# Build mapping actual_col -> standard_name if present
actual_to_standard = {}
for std, candlist in candidates.items():
    found = map_column(data, candlist)
    if found:
        actual_to_standard[std] = found

# Rename DataFrame to standard names we will use
rename_map = {v: k for k, v in actual_to_standard.items()}  # actual->std
data = data.rename(columns=rename_map)

# Now ensure we have the standard columns present (some may be missing; we handle gracefully)
# For convenience, create missing columns with NaN if not present
for col in ["order_id","order_region","order_country","sales","profit","category_name","product_name","quantity","ship_mode","delay_status"]:
    if col not in data.columns:
        data[col] = np.nan

# -------------------------
# Clean and map delay values to -1/0/1
# -------------------------
data["delay_mapped"] = data["delay_status"].apply(map_delay_values)
# If still NaN, try to find a delay-like column from either file (scan more columns)
if data["delay_mapped"].isna().all():
    # scan all columns for binary-like 0/1 or keywords
    for c in data.columns:
        if c in ["delay_mapped","delay_status"]:
            continue
        vals = data[c].dropna().unique()
        # if val set is subset of {-1,0,1} or {0,1} attempt mapping
        try:
            # try numeric check
            uniq = set([int(x) for x in vals if pd.notna(x) and str(x).lstrip('-').isdigit()])
            if uniq and uniq.issubset({-1,0,1,0,1}):
                data["delay_mapped"] = pd.to_numeric(data[c], errors='coerce').fillna(0).astype(int)
                break
        except Exception:
            pass
# If still NaN, set default 0 (on-time) to avoid crashes, but warn user
if data["delay_mapped"].isna().any():
    st.warning("Some delay values could not be mapped; unmapped rows set to 0 (On-time).")
    data["delay_mapped"] = data["delay_mapped"].fillna(0).astype(int)

# For convenience, create human label
def delay_label(v):
    if v == 1: return "Delayed"
    if v == 0: return "On-time"
    if v == -1: return "Early"
    return "Unknown"
data["delay_label"] = data["delay_mapped"].apply(delay_label)

# -------------------------
# Convert numeric columns
# -------------------------
for ncol in ["sales","profit","quantity"]:
    if ncol in data.columns:
        data[ncol] = pd.to_numeric(data[ncol], errors="coerce").fillna(0)

# Normalize region/country/category/product/ship_mode strings
for tcol in ["order_region","order_country","category_name","product_name","ship_mode"]:
    if tcol in data.columns:
        data[tcol] = data[tcol].astype(str).fillna("Unknown").str.strip()

# -------------------------
# Sidebar filters for user
# -------------------------
st.sidebar.header("View Filters")
regions = sorted(data["order_region"].dropna().unique())
countries = sorted(data["order_country"].dropna().unique())
categories = sorted(data["category_name"].dropna().unique())

sel_regions = st.sidebar.multiselect("Select Region(s)", options=regions, default=regions)
sel_countries = st.sidebar.multiselect("Select Country(s)", options=countries, default=countries)
sel_categories = st.sidebar.multiselect("Select Category(s)", options=categories, default=categories)

filtered = data[
    data["order_region"].isin(sel_regions) &
    data["order_country"].isin(sel_countries) &
    data["category_name"].isin(sel_categories)
].copy()

st.header("Order Delay Analytics")
st.markdown("Filters applied: Regions: " + ", ".join(sel_regions)[:200])

# -------------------------
# Compute and display requested analytics
# -------------------------

# 1) Average sales per customer based on order_region
st.subheader("1) Average Sales per Customer by Order Region")
if "sales" in filtered.columns and not filtered.empty:
    avg_sales = filtered.groupby("order_region")["sales"].mean().reset_index().sort_values("sales", ascending=False)
    st.dataframe(avg_sales, use_container_width=True)
    fig = px.bar(avg_sales, x="order_region", y="sales", labels={"sales":"Avg Sales"}, title="Avg Sales per Customer by Region", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Sales or order_region column missing or no data after filtering.")

# 2) Average Profit per order based on order_region
st.subheader("2) Average Profit per Order by Order Region")
if "profit" in filtered.columns and not filtered.empty:
    avg_profit = filtered.groupby("order_region")["profit"].mean().reset_index().sort_values("profit", ascending=False)
    st.dataframe(avg_profit, use_container_width=True)
    fig = px.bar(avg_profit, x="order_region", y="profit", labels={"profit":"Avg Profit"}, title="Avg Profit per Order by Region", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Profit or order_region column missing or no data after filtering.")

# 3) Top 5 order country and order region marketwise
st.subheader("3) Top 5 Order Countries & Regions (by order count)")
if "order_country" in filtered.columns:
    top_countries = filtered["order_country"].value_counts().head(5).reset_index()
    top_countries.columns = ["order_country","orders"]
    st.table(top_countries)
else:
    st.warning("order_country column missing.")

if "order_region" in filtered.columns:
    top_regions = filtered["order_region"].value_counts().head(5).reset_index()
    top_regions.columns = ["order_region","orders"]
    st.table(top_regions)
else:
    st.warning("order_region column missing.")

# 4) Top 8 most profitable category_name
st.subheader("4) Top 8 Most Profitable Categories")
if "category_name" in filtered.columns and "profit" in filtered.columns:
    top8 = filtered.groupby("category_name")["profit"].sum().sort_values(ascending=False).head(8).reset_index()
    st.dataframe(top8, use_container_width=True)
    fig = px.bar(top8, x="category_name", y="profit", title="Top 8 Profitable Categories", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("category_name and/or profit columns missing.")

# 5) Most profitable product for different order region
st.subheader("5) Most Profitable Product per Order Region")
if all(col in filtered.columns for col in ["order_region","product_name","profit"]):
    rp = filtered.groupby(["order_region","product_name"])["profit"].sum().reset_index()
    idx = rp.groupby("order_region")["profit"].idxmax()
    best = rp.loc[idx].reset_index(drop=True)
    st.dataframe(best, use_container_width=True)
else:
    st.warning("Required columns for #5 missing.")

# 6) Top 5 most sold category name based on quantity and revenue
st.subheader("6) Top 5 Most Sold Categories (by Quantity & Revenue)")
if "category_name" in filtered.columns:
    if "quantity" in filtered.columns:
        top_qty = filtered.groupby("category_name")["quantity"].sum().sort_values(ascending=False).head(5).reset_index()
        st.write("Top 5 by Quantity")
        st.dataframe(top_qty, use_container_width=True)
        st.plotly_chart(px.bar(top_qty, x="category_name", y="quantity", title="Top 5 Categories by Quantity", template="plotly_white"), use_container_width=True)
    else:
        st.warning("quantity column missing for quantity-based ranking.")
    if "sales" in filtered.columns:
        top_rev = filtered.groupby("category_name")["sales"].sum().sort_values(ascending=False).head(5).reset_index()
        st.write("Top 5 by Revenue")
        st.dataframe(top_rev, use_container_width=True)
        st.plotly_chart(px.bar(top_rev, x="category_name", y="sales", title="Top 5 Categories by Revenue", template="plotly_white"), use_container_width=True)
    else:
        st.warning("sales column missing for revenue-based ranking.")
else:
    st.warning("category_name missing.")

# 7) Preferred shipping mode based on order_region
st.subheader("7) Preferred Shipping Mode by Order Region")
if all(col in filtered.columns for col in ["order_region","ship_mode"]):
    pref = filtered.groupby(["order_region","ship_mode"]).size().reset_index(name="count")
    fig = px.bar(pref, x="order_region", y="count", color="ship_mode", barmode="group", title="Preferred Shipping Mode by Region", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("order_region or ship_mode missing.")

# 8) Delayed orders based on shipping mode
st.subheader("8) Delayed Orders by Shipping Mode")
if "ship_mode" in filtered.columns and "delay_mapped" in filtered.columns:
    delayed = filtered[filtered["delay_mapped"] == 1]
    if not delayed.empty:
        delayed_by_mode = delayed.groupby("ship_mode").size().reset_index(name="delayed_count").sort_values("delayed_count", ascending=False)
        st.dataframe(delayed_by_mode, use_container_width=True)
        st.plotly_chart(px.bar(delayed_by_mode, x="ship_mode", y="delayed_count", title="Delayed Orders by Shipping Mode", template="plotly_white"), use_container_width=True)
    else:
        st.info("No delayed orders in the filtered data.")
else:
    st.warning("ship_mode or delay column missing.")

# -------------------------
# Delay prediction model (optional)
# -------------------------
st.markdown("---")
st.header("Delay Prediction Model (optional)")
# Only attempt model if we have a non-trivial target and some predictors
if "delay_mapped" in filtered.columns:
    y = filtered["delay_mapped"]
    # require at least two classes (e.g., delayed and not)
    if len(y.unique()) > 1:
        # build X using numeric columns and a few encoded categoricals
        X = filtered.select_dtypes(include=[np.number]).copy()
        # drop target if present
        if "delay_mapped" in X.columns:
            X = X.drop(columns=["delay_mapped"])
        # If no numeric features, try to encode a few useful categoricals
        if X.shape[1] == 0:
            cat_use = []
            for c in ["order_region","order_country","category_name","product_name","ship_mode"]:
                if c in filtered.columns:
                    cat_use.append(c)
            X = filtered[cat_use].copy()
            for c in X.columns:
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        X = X.fillna(0)
        # if still nothing, skip
        if X.shape[1] == 0:
            st.info("No features available to train model.")
        else:
            test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y if len(y.unique())>1 else None)
            model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.metric("Model accuracy (test)", f"{acc:.3f}")
            st.text("Classification report (test):")
            st.text(classification_report(y_test, preds, zero_division=0))
            # feature importances if numeric features exist
            try:
                importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
                st.subheader("Top feature importances")
                st.dataframe(importances.reset_index().rename(columns={'index':'feature',0:'importance'}), use_container_width=True)
                st.plotly_chart(px.bar(importances.reset_index(), x='index', y=0, title='Feature importances', labels={'index':'feature',0:'importance'}), use_container_width=True)
            except Exception:
                pass
            # download predictions on test set
            out = X_test.copy()
            out['true_delay'] = y_test
            out['predicted_delay'] = preds
            csv = out.to_csv(index=False)
            st.download_button("Download test predictions", data=csv, file_name="delay_model_predictions.csv", mime="text/csv")
    else:
        st.info("Not enough variation in delay target to train a model (need >1 class).")
else:
    st.warning("Delay target 'delay_mapped' not present; cannot train model.")

st.success("Dashboard ready.")
