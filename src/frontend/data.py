"""Cached data loading for the dashboard (nothing is computed here beyond joins and look-up tables)."""
import numpy as np
import pandas as pd
import streamlit as st

from . import recommender as rec
from .config import DATE_COL, PROCESSED_FILE, RAW_FILE, STOCK_COLS, TARGET, TEST_FILE, TRAIN_FILE, get_feature_columns


def missing_data_files():
    return [str(p) for p in (PROCESSED_FILE, TRAIN_FILE, TEST_FILE) if not p.exists()]


@st.cache_data(show_spinner="Loading data...")
def load_processed():
    """The full processed dataset (one row per transaction, readable categories)."""
    return pd.read_csv(PROCESSED_FILE, parse_dates=[DATE_COL])


@st.cache_data(show_spinner="Loading raw data...")
def load_raw():
    """The raw dataset with every original column (None if the file is missing)."""
    if not RAW_FILE.exists():
        return None
    df = pd.read_csv(RAW_FILE)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", format="mixed")
    for col in ("Shipping_Time", "Sales_Amount", "Units_Sold", "Returns"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def lead_by_category():
    """Median Shipping_Time (days) of each product category, used for what-if scenarios."""
    raw = load_raw()
    if raw is None or "Shipping_Time" not in raw:
        return {}
    return raw.groupby("Product_Category")["Shipping_Time"].median().to_dict()


@st.cache_data
def abc_xyz_table():
    """ABC (revenue) and XYZ (demand variability) class of every product, from all processed records."""
    return rec.abc_xyz(load_processed())


@st.cache_data(show_spinner="Loading test period...")
def load_test(_mtime):
    """The held-out test rows: readable details (meta), encoded model inputs (X) and the actual Units_Sold (y)."""
    test = pd.read_csv(TEST_FILE, parse_dates=[DATE_COL])
    labels = (load_processed()[["Transaction_ID", "Product_Category", "Product_Subcategory", "Store_Type"]]
              .drop_duplicates("Transaction_ID"))
    meta = test[[DATE_COL] + STOCK_COLS].merge(labels, on="Transaction_ID", how="left")
    raw = load_raw()
    if raw is not None and "Shipping_Time" in raw:                       # lead time of each record
        meta = meta.merge(raw[["Transaction_ID", "Shipping_Time"]].drop_duplicates("Transaction_ID"), on="Transaction_ID", how="left")
    return meta, test[get_feature_columns(test)], test[TARGET]


@st.cache_data
def train_sales_distribution():
    """Units_Sold of the training period, used to decide what counts as 'high demand'."""
    return pd.read_csv(TRAIN_FILE, usecols=[TARGET])[TARGET].to_numpy()


@st.cache_data
def lookups():
    """Options and typical values for the prediction form."""
    df = load_processed()
    mode = lambda s: s.mode().iloc[0]
    return {
        "stores": sorted(df["Store_ID"].unique()),
        "store_type": df.groupby("Store_ID")["Store_Type"].agg(mode).to_dict(),
        "store_types": sorted(df["Store_Type"].unique()),
        "weather": sorted(df["Weather_Condition"].unique()),
        "discounts": sorted(int(x) for x in df["Discount_Percent"].unique()),
        "subcategories": {c: sorted(g["Product_Subcategory"].unique()) for c, g in df.groupby("Product_Category")},
        "typical": df.groupby(["Product_Category", "Product_Subcategory"]).agg(
            price=("Unit_Price", "median"), marketing=("Marketing_Spend", "median"),
            footfall=("Customer_Footfall", "median"), inventory=("Inventory_Level", "median"),
            price_gap=("Price_Gap", "median")),
        "date_min": df[DATE_COL].min().date(), "date_max": df[DATE_COL].max().date(),
    }


def products_for(category, subcategory):
    df = load_processed()
    sel = df[(df["Product_Category"] == category) & (df["Product_Subcategory"] == subcategory)]
    return sorted(sel["Product_ID"].unique())


def latest_inventory(store, product, category, subcategory):
    """Most recent recorded stock for this store / product / category (None if there is no record)."""
    df = load_processed()
    sel = df[(df["Store_ID"] == store) & (df["Product_ID"] == product)
             & (df["Product_Category"] == category) & (df["Product_Subcategory"] == subcategory)]
    if sel.empty:
        return None
    last = sel.sort_values(DATE_COL).iloc[-1]
    return int(last["Inventory_Level"]), last[DATE_COL].date()