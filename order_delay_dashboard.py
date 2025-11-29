# order_delay_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import io

st.set_page_config(page_title="Order Delay Prediction & Insights", layout="wide")

st.title("Order Delay Prediction Dashboard")
st.caption("0 = On-time, 1 = Delayed (auto-detected)")

# -------------------------
# Load data (robust)
# -------------------------
@st.cache_data
def load_files():
    # adjust these paths if needed
    path_main = "/mnt/data/Delay Model.csv"
    path_delay = "/mnt/data/Delay description csv.csv"
    df_main = pd.read_csv(path_main)
    df_delay = pd.read_csv(path_delay)
    return df_main, df_delay

try:
    df_main, df_delay = load_files()
except Exception as e:
    st.error(f"Failed to load files. Check paths and filenames. Error: {e}")
    st.stop()

st.success("Files loaded.")

# normalize column names to help match variations
def norm_cols(df):
    df.columns = [c.strip() for c in df.columns]
    return df

df_main = norm_cols(df_main)
df_delay = norm_cols(df_delay)

# -------------------------
# Detect delay target column in delay file
# -------------------------
possible_targets = ['delay_flag','is_delayed','delayed','delay','late']
target_col = None
for col in df_delay.columns:
    if col.lower() in possible_targets:
        target_col = col
        break

if target_col is None:
    # try scanning columns for binary 0/1 values
    for col in df_delay.columns:
        vals = df_delay[col].dropna().unique()
        if set(np.unique(vals)).issubset({0,1}) or set(np.unique(vals)).issubset({'0','1'}):
            target_col = col
            break

if target_col is None:
    st.error("No suitable delay target column found. Rename the 0/1 target to one of: "
             "delay_flag, is_delayed, delayed, delay, late OR ensure a binary 0/1 column exists in the delay file.")
    st.stop()

st.info(f"Detected target column: `{target_col}` (0 = on-time, 1 = delayed)")

# ensure numeric
df_delay[target_col] = pd.to_numeric(df_delay[target_col], errors='coerce')

# -------------------------
# Merge datasets if possible (on order_id or appropriate key)
# -------------------------
merge_key = None
for k in ['order_id','Order ID','order id','OrderID','orderid']:
    if k in df_main.columns and k in df_delay.columns:
        merge_key = k
        break

if merge_key:
    data = df_main.merge(df_delay, on=merge_key, how='left')
else:
    # if no merge key, try using delay file as full dataset (if it contains columns needed)
    data = df_main.copy()
    # merge delay flag if delay file has order-level and identifiable key
    if any(col in df_delay.columns for col in df_main.columns):
        # attempt best-effort merge by intersection of columns (rare)
        common = [c for c in df_main.columns if c in df_delay.columns]
        if common:
            data = df_main.merge(df_delay, on=common, how='left')
        else:
            # append delay file columns if small
            data = pd.concat([df_main, df_delay], axis=1)

# -------------------------
# Basic cleaning & type conversions
# -------------------------
# standardize commonly used column names to lower/no-space keys for analytics
def safe_lower(col):
    return col.strip()

data.columns = [safe_lower(c) for c in data.columns]

# -------------------------
# 1) Average sales per customer based on order_region
# -------------------------
st.header("1) Average Sales per Customer by Order Region")
if 'order_region' in data.columns and 'sales' in data.columns:
    avg_sales = data.groupby('order_region')['sales'].mean().reset_index().sort_values('sales', ascending=False)
    st.dataframe(avg_sales, use_container_width=True)
    st.plotly_chart(px.bar(avg_sales, x='order_region', y='sales', title='Average Sales per Customer by Region'), use_container_width=True)
else:
    st.warning("Columns 'order_region' or 'sales' not found in merged data.")

# -------------------------
# 2) Average Profit per order based on order_region
# -------------------------
st.header("2) Average Profit per Order by Order Region")
if 'order_region' in data.columns and 'profit' in data.columns:
    avg_profit = data.groupby('order_region')['profit'].mean().reset_index().sort_values('profit', ascending=False)
    st.dataframe(avg_profit, use_container_width=True)
    st.plotly_chart(px.bar(avg_profit, x='order_region', y='profit', title='Average Profit per Order by Region'), use_container_width=True)
else:
    st.warning("Columns 'order_region' or 'profit' not found.")

# -------------------------
# 3) Top 5 order country and order region marketwise
# -------------------------
st.header("3) Top 5 Order Countries and Regions (by order count)")
if 'order_country' in data.columns:
    top_countries = data['order_country'].value_counts().head(5).reset_index()
    top_countries.columns = ['order_country','count']
    st.table(top_countries)
else:
    st.warning("'order_country' missing.")

if 'order_region' in data.columns:
    top_regions = data['order_region'].value_counts().head(5).reset_index()
    top_regions.columns = ['order_region','count']
    st.table(top_regions)
else:
    st.warning("'order_region' missing.")

# -------------------------
# 4) Top 8 most profitable category_name
# -------------------------
st.header("4) Top 8 Most Profitable Categories")
if 'category_name' in data.columns and 'profit' in data.columns:
    top8_cat = data.groupby('category_name')['profit'].sum().sort_values(ascending=False).head(8).reset_index()
    st.dataframe(top8_cat, use_container_width=True)
    st.plotly_chart(px.bar(top8_cat, x='category_name', y='profit', title='Top 8 Profitable Categories'), use_container_width=True)
else:
    st.warning("'category_name' or 'profit' missing.")

# -------------------------
# 5) Most profitable product for different order region
# -------------------------
st.header("5) Most Profitable Product per Order Region")
if 'order_region' in data.columns and 'product_name' in data.columns and 'profit' in data.columns:
    reg_prod = data.groupby(['order_region','product_name'])['profit'].sum().reset_index()
    idx = reg_prod.groupby('order_region')['profit'].idxmax()
    best_prod = reg_prod.loc[idx].reset_index(drop=True)
    st.dataframe(best_prod, use_container_width=True)
else:
    st.warning("Columns required: 'order_region', 'product_name', 'profit'.")

# -------------------------
# 6) Top 5 most sold category name based on quantity and revenue
# -------------------------
st.header("6) Top 5 Most Sold Categories (by quantity & by revenue)")
qty_ok = ('quantity' in data.columns)
sales_ok = ('sales' in data.columns)
if 'category_name' in data.columns and (qty_ok or sales_ok):
    if qty_ok:
        top5_qty = data.groupby('category_name')['quantity'].sum().sort_values(ascending=False).head(5).reset_index()
        st.subheader("Top 5 by Quantity")
        st.dataframe(top5_qty, use_container_width=True)
        st.plotly_chart(px.bar(top5_qty, x='category_name', y='quantity', title='Top 5 Categories by Quantity'), use_container_width=True)
    if sales_ok:
        top5_rev = data.groupby('category_name')['sales'].sum().sort_values(ascending=False).head(5).reset_index()
        st.subheader("Top 5 by Revenue")
        st.dataframe(top5_rev, use_container_width=True)
        st.plotly_chart(px.bar(top5_rev, x='category_name', y='sales', title='Top 5 Categories by Revenue'), use_container_width=True)
else:
    st.warning("Required columns for this analysis missing: 'category_name' and ('quantity' or 'sales').")

# -------------------------
# 7) Preferred shipping mode based on order_region
# -------------------------
st.header("7) Preferred Shipping Mode by Order Region")
if 'order_region' in data.columns and 'ship_mode' in data.columns:
    ship_pref = data.groupby(['order_region','ship_mode']).size().reset_index(name='count')
    fig = px.bar(ship_pref, x='order_region', y='count', color='ship_mode', title='Preferred Shipping Mode by Region')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Columns 'order_region' or 'ship_mode' missing.")

# -------------------------
# 8) Delayed orders based on shipping mode
# -------------------------
st.header("8) Delayed Orders by Shipping Mode")
if 'ship_mode' in data.columns and target_col in data.columns:
    delayed = data[data[target_col]==1]
    delayed_by_mode = delayed.groupby('ship_mode').size().reset_index(name='delayed_count').sort_values('delayed_count', ascending=False)
    st.dataframe(delayed_by_mode, use_container_width=True)
    st.plotly_chart(px.bar(delayed_by_mode, x='ship_mode', y='delayed_count', title='Delayed Orders by Shipping Mode'), use_container_width=True)
else:
    st.warning("Columns 'ship_mode' or target delay column missing in merged data.")

# -------------------------
# Delay Prediction Model
# -------------------------
st.header("Delay Prediction Model (Random Forest)")

# ensure target present in data; if not, try add from df_delay
if target_col not in data.columns and target_col in df_delay.columns:
    # attach by merge_key if available earlier
    if merge_key:
        data = data.merge(df_delay[[merge_key, target_col]], on=merge_key, how='left')

if target_col not in data.columns:
    st.error("Delay target column not present in training data. Model cannot be trained.")
else:
    model_data = data.copy()
    # drop rows where target missing
    model_data = model_data.dropna(subset=[target_col])
    # drop high-cardinality or identifier columns
    drop_cols = [c for c in model_data.columns if c.lower() in ('order_id','order_no','id')]
    # choose features: numeric columns + encoded categoricals (small set)
    y = model_data[target_col].astype(int)
    X = model_data.drop(columns=[target_col] + drop_cols, errors='ignore')

    # handle object columns via label encoding (simple)
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    encoders = {}
    for c in cat_cols:
        enc = LabelEncoder()
        X[c] = enc.fit_transform(X[c].astype(str))
        encoders[c] = enc

    # fill NA numeric with 0
    X = X.fillna(0)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    # train
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    st.subheader("Model performance")
    st.write(f"Accuracy: {acc:.3f}")
    st.text(classification_report(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds)
    st.write("Confusion Matrix (rows=true, cols=pred):")
    st.write(cm)

    # Feature importances (top 15)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    st.subheader("Top feature importances")
    st.dataframe(importances.reset_index().rename(columns={'index':'feature',0:'importance'}), use_container_width=True)
    st.plotly_chart(px.bar(importances.reset_index().rename(columns={'index':'feature',0:'importance'}), x='feature', y=0, title='Feature Importances'), use_container_width=True)

    # Predict on full dataset and provide download
    full_X = X.fillna(0)
    full_preds = model.predict(full_X)
    out = model_data.copy()
    out['predicted_delay'] = full_preds
    csv = out.to_csv(index=False)
    st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.success("Done. Review charts and model outputs above.")
