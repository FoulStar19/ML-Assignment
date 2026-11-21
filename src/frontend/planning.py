"""Loading the processed data and the reorder-point / order-up-to policy."""
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd
import streamlit as st

from config import MIN_HISTORY_RECORDS, PROCESSED_FILE, ROOT_DIR

if str(ROOT_DIR / "src" / "backend") not in sys.path:      # the pipeline code lives outside the frontend folder
    sys.path.insert(0, str(ROOT_DIR / "src" / "backend"))

from data_transforming import UNDERSTOCK_COVER_DAYS  # noqa: E402  (the same band the models were trained on)


@st.cache_data(show_spinner="Loading inventory data...")
def load_data() -> pd.DataFrame:
    """The processed records, oldest first. One record = one sale of a product in a store on a date."""
    if not PROCESSED_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PROCESSED_FILE, parse_dates=["date"])
    return df.sort_values(["date", "transaction_id"]).reset_index(drop=True)


@st.cache_data(show_spinner="Estimating demand for each record...")
def demand_stats() -> pd.DataFrame:
    """Expected demand (mu) and its spread (sigma) for every record, used when the XGBoost demand model is off.

    Store and product IDs are only labels in this data (they carry no history), so demand is estimated from
    the records of the same product subcategory that come before the record. A record never sees its own
    outcome. With the demand model on, its forecast replaces mu and sigma (see predictor.with_model_demand).
    """
    df = load_data()
    g = df.groupby("product_subcategory", sort=False)["units_sold"]
    out = df.copy()
    out["mu"] = g.transform(
        lambda s: s.shift(1).expanding(min_periods=MIN_HISTORY_RECORDS).mean()
    ).clip(lower=1)
    out["sigma"] = g.transform(lambda s: s.shift(1).expanding(min_periods=MIN_HISTORY_RECORDS).std())
    return out.dropna(subset=["mu", "sigma"]).reset_index(drop=True)


def apply_policy(d: pd.DataFrame, lead: int, review: int, service: float, over_days: float) -> pd.DataFrame:
    """Reorder point / order-up-to policy applied to every record.

    The reorder point is never below the Understock edge of the cover band the models were trained on
    (UNDERSTOCK_COVER_DAYS of demand), and the target level is one review period above it.
    """
    z = NormalDist().inv_cdf(service)
    d = d.copy()
    d["safety_stock"] = z * d["sigma"] * np.sqrt(lead)
    d["reorder_point"] = np.maximum(d["mu"] * lead + d["safety_stock"], d["mu"] * UNDERSTOCK_COVER_DAYS)
    d["target_level"] = np.maximum(
        d["mu"] * (lead + review) + z * d["sigma"] * np.sqrt(lead + review),
        d["reorder_point"] + d["mu"] * review,
    )
    d["days_of_cover"] = d["inventory_level"] / d["mu"]
    d["forecast_error"] = d["units_sold"] - d["mu"]

    under = d["inventory_level"] <= d["reorder_point"]
    over = (d["days_of_cover"] > over_days) & (d["inventory_level"] > d["target_level"])
    d["status"] = np.select([under, over], ["Understock risk", "Overstock"], default="Healthy")

    d["order_qty"] = np.where(under, np.ceil(d["target_level"] - d["inventory_level"]).clip(lower=0), 0)
    d["order_value"] = d["order_qty"] * d["price"]
    d["excess_units"] = np.where(over, np.floor(d["inventory_level"] - d["target_level"]).clip(lower=0), 0)
    d["excess_value"] = d["excess_units"] * d["price"]
    return d
