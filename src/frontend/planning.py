"""Loading the processed data and the reorder-point / order-up-to policy."""
from statistics import NormalDist

import numpy as np
import pandas as pd
import streamlit as st

from config import MIN_HISTORY_DAYS, PROCESSED_FILE, RAW_FILE


def prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Minimal version of the pipeline, used only if the processed CSV is missing."""
    df = raw.copy()
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_")
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["forecast_error"] = df["units_sold"] - df["demand_forecast"]
    return df


@st.cache_data(show_spinner="Loading inventory data...")
def load_data() -> pd.DataFrame:
    if PROCESSED_FILE.exists():
        df = pd.read_csv(PROCESSED_FILE, parse_dates=["date"])
    elif RAW_FILE.exists():
        df = prepare_raw(pd.read_csv(RAW_FILE))
    else:
        return pd.DataFrame()
    if "forecast_error" not in df.columns and {"units_sold", "demand_forecast"} <= set(df.columns):
        df["forecast_error"] = df["units_sold"] - df["demand_forecast"]
    return df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner="Estimating demand for each store-product line...")
def demand_stats(lookback: int) -> pd.DataFrame:
    """Rolling expected demand (mu) and forecast-error spread (sigma) per line.

    mu uses demand_forecast. To use a trained model instead, replace that
    column's values with the model's predictions and everything below follows.
    sigma uses yesterday's errors and earlier, so a day's plan never sees its own outcome.
    """
    df = load_data()
    g = df.groupby(["store_id", "product_id"], sort=False)
    out = df.copy()
    out["mu"] = g["demand_forecast"].transform(
        lambda s: s.rolling(lookback, min_periods=MIN_HISTORY_DAYS).mean()
    ).clip(lower=1)
    out["sigma"] = g["forecast_error"].transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=MIN_HISTORY_DAYS).std()
    )
    out["avg_price"] = g["price"].transform(lambda s: s.rolling(lookback, min_periods=1).mean())
    return out.dropna(subset=["mu", "sigma"]).reset_index(drop=True)


def apply_policy(d: pd.DataFrame, lead: int, review: int, service: float, over_days: float) -> pd.DataFrame:
    """Reorder point / order-up-to policy applied to every store-product-day."""
    z = NormalDist().inv_cdf(service)
    d = d.copy()
    d["safety_stock"] = z * d["sigma"] * np.sqrt(lead)
    d["reorder_point"] = d["mu"] * lead + d["safety_stock"]
    d["target_level"] = d["mu"] * (lead + review) + z * d["sigma"] * np.sqrt(lead + review)
    d["days_of_cover"] = d["inventory_level"] / d["mu"]

    under = d["inventory_level"] <= d["reorder_point"]
    over = (d["days_of_cover"] > over_days) & (d["inventory_level"] > d["target_level"])
    d["status"] = np.select([under, over], ["Understock risk", "Overstock"], default="Healthy")

    d["order_qty"] = np.where(under, np.ceil(d["target_level"] - d["inventory_level"]).clip(lower=0), 0)
    d["order_value"] = d["order_qty"] * d["avg_price"]
    d["excess_units"] = np.where(over, np.floor(d["inventory_level"] - d["target_level"]).clip(lower=0), 0)
    d["excess_value"] = d["excess_units"] * d["avg_price"]
    return d


