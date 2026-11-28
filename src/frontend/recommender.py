"""Statistical inventory planner: predicted demand vs. stock on hand. Pure functions, no Streamlit.

For each record (units are per day):
  protection period P = lead time + review days
  expected demand     = predicted daily units x P
  safety stock        = z x sigma x sqrt(P)          z from the service level, sigma = forecast error of the category
  reorder point       = expected demand + safety stock
  target stock        = max(reorder point, predicted units x target days + safety stock)
  stockout risk       = chance that demand over P is larger than the stock on hand (normal approximation)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import ABC_CUTS, HIGH_DEMAND_PCT, LEAD_DAYS, OVERSTOCK_DAYS, REVIEW_DAYS, STATUS_ICONS, TARGET_DAYS

UNDER, OVER, OK = "Understocking", "Overstocking", "Sufficient stock"


@dataclass(frozen=True)
class Rules:
    service_level: float = 0.95
    review_days: int = REVIEW_DAYS
    target_days: int = TARGET_DAYS
    overstock_days: int = OVERSTOCK_DAYS
    high_demand_units: float = 47.0        # set from the sales history (HIGH_DEMAND_PCT percentile)


@dataclass(frozen=True)
class LeadTime:
    from_data: bool = True                 # True: Shipping_Time of each record; False: one fixed number of days
    fixed_days: int = LEAD_DAYS
    by_category: dict = field(default_factory=dict)

    def for_rows(self, table):
        if self.from_data and "Shipping_Time" in table:
            ship = table["Shipping_Time"]
            return ship.fillna(ship.median() if ship.notna().any() else self.fixed_days).clip(lower=1).to_numpy(float)
        return np.full(len(table), float(self.fixed_days))

    def for_category(self, category):
        return float(self.by_category.get(category, self.fixed_days)) if self.from_data else float(self.fixed_days)


def high_demand_threshold(sales, pct=HIGH_DEMAND_PCT) -> float:
    return float(np.percentile(sales, pct))


def residual_sigma(category, actual, pred):
    """Standard deviation of the daily forecast error for each category, and over all records."""
    err = pd.Series(np.asarray(pred, float) - np.asarray(actual, float))
    by_cat = err.groupby(np.asarray(category)).std().to_dict()
    return by_cat, float(err.std())


def plan(pred, inventory, lead_days, sigma, rules: Rules) -> pd.DataFrame:
    """Reorder point, safety stock, stockout risk, status, advice and priority for each (prediction, stock) pair."""
    pred = np.maximum(np.asarray(pred, dtype=float), 0.0)
    inv = np.asarray(inventory, dtype=float)
    lead, sig = np.broadcast_arrays(np.asarray(lead_days, dtype=float), np.asarray(sigma, dtype=float))
    z = norm.ppf(rules.service_level)
    period = lead + rules.review_days
    expected = pred * period
    sd = np.maximum(sig, 1e-6) * np.sqrt(period)
    safety = z * sd
    reorder = expected + safety
    target = np.maximum(reorder, pred * rules.target_days + safety)
    risk = norm.sf((inv - expected) / sd)
    cover = inv / np.maximum(pred, 0.5)
    status = np.select([inv < reorder, cover > rules.overstock_days], [UNDER, OVER], OK)
    order_qty = np.where(status == UNDER, np.ceil(np.maximum(target - inv, 0)), 0).astype(int)
    excess = np.where(status == OVER, np.floor(np.maximum(inv - target, 0)), 0).astype(int)
    high = pred >= rules.high_demand_units
    under_p = 50 + 50 * np.clip(risk, 0, 1)
    over_p = 50 * np.clip((cover - rules.overstock_days) / rules.overstock_days, 0, 1)
    priority = np.where(status == UNDER, under_p, np.where(status == OVER, over_p, 0)) + 10 * high
    action = []
    for s, q, e, h in zip(status, order_qty, excess, high):
        text = f"Order about {q:,} units" if s == UNDER else (f"Reduce future orders (about {e:,} units extra)" if s == OVER else "Monitor demand")
        action.append(text + (" - high demand, watch closely" if h and s != UNDER else ""))
    return pd.DataFrame({
        "Status": status, "High_Demand": high, "Lead_Days": np.round(lead, 1), "Expected_Demand": np.round(expected, 1),
        "Safety_Stock": np.round(safety, 1), "Reorder_Point": np.round(reorder, 1), "Target_Stock": np.round(target, 1),
        "Stockout_Risk": np.round(risk, 4), "Cover_Days": np.round(cover, 1), "Order_Qty": order_qty, "Excess_Units": excess,
        "Priority": np.round(np.minimum(priority, 100), 1), "Action": action})


# ─────────────────────────────────────────────
# ABC-XYZ segmentation
# ─────────────────────────────────────────────
SEGMENT_POLICY = {
    "AX": "High value, steady demand: tight control, frequent small orders, high service level.",
    "AY": "High value, some swings: review often, keep a solid safety stock.",
    "AZ": "High value, erratic demand: watch closely, order close to need, extra safety stock.",
    "BX": "Medium value, steady demand: automatic reorder at the reorder point.",
    "BY": "Medium value, some swings: standard reorder point with review.",
    "BZ": "Medium value, erratic demand: moderate safety stock, review regularly.",
    "CX": "Low value, steady demand: simple rules, larger and less frequent orders.",
    "CY": "Low value, some swings: simple rules, check now and then.",
    "CZ": "Low value, erratic demand: keep minimal stock, order on demand.",
}


def abc_xyz(df, abc_cuts=ABC_CUTS) -> pd.DataFrame:
    """One row per product. ABC from its share of revenue, XYZ from the variability (CV) of its units sold:
    X = the steadiest third of products, Y = the middle third, Z = the most erratic third."""
    d = df.assign(Revenue=df["Units_Sold"] * df["Unit_Price"] * (1 - df["Discount_Percent"] / 100))
    g = d.groupby("Product_ID").agg(Revenue=("Revenue", "sum"), Mean_Units=("Units_Sold", "mean"),
                                    Std_Units=("Units_Sold", "std"), Records=("Units_Sold", "size"))
    g["CV"] = g["Std_Units"] / g["Mean_Units"]
    g = g.sort_values("Revenue", ascending=False)
    before = g["Revenue"].cumsum() / g["Revenue"].sum() - g["Revenue"] / g["Revenue"].sum()   # share of revenue ahead of this product
    g["ABC"] = np.select([before < abc_cuts[0], before < abc_cuts[1]], ["A", "B"], "C")
    q1, q2 = g["CV"].quantile([1 / 3, 2 / 3])
    g["XYZ"] = np.select([g["CV"] <= q1, g["CV"] <= q2], ["X", "Y"], "Z")
    g["Segment"] = g["ABC"] + g["XYZ"]
    return g


def label(status: str) -> str:
    return f"{STATUS_ICONS[status]} {status}"
