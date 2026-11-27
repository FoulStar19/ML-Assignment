"""Rule-based inventory recommender: predicted demand vs. stock on hand. Pure functions, no Streamlit."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HIGH_DEMAND_PCT, OVERSTOCK_DAYS, STATUS_ICONS, TARGET_DAYS, UNDERSTOCK_DAYS

UNDER, OVER, OK = "Understocking", "Overstocking", "Sufficient stock"


@dataclass(frozen=True)
class Rules:
    understock_days: int = UNDERSTOCK_DAYS
    overstock_days: int = OVERSTOCK_DAYS
    target_days: int = TARGET_DAYS
    high_demand_units: float = 47.0        # set from the sales history (HIGH_DEMAND_PCT percentile)


def high_demand_threshold(sales, pct=HIGH_DEMAND_PCT) -> float:
    return float(np.percentile(sales, pct))


def assess(pred, inventory, rules: Rules) -> pd.DataFrame:
    """Status, advice and priority for each (predicted units, stock) pair.

    Days of cover = stock / predicted daily units. Below `understock_days` -> understocking,
    above `overstock_days` -> overstocking, otherwise sufficient. Replenish / trim to `target_days` of cover.
    Priority (0-100): understocking 50-100 (the lower the cover, the higher), overstocking 0-50
    (the larger the excess, the higher), sufficient 0; high-demand items get +10.
    """
    pred = np.asarray(pred, dtype=float)
    inv = np.asarray(inventory, dtype=float)
    cover = inv / np.maximum(pred, 0.5)
    status = np.select([cover < rules.understock_days, cover > rules.overstock_days], [UNDER, OVER], OK)
    target_units = pred * rules.target_days
    order_qty = np.where(status == UNDER, np.ceil(np.maximum(target_units - inv, 0)), 0).astype(int)
    excess = np.where(status == OVER, np.floor(np.maximum(inv - target_units, 0)), 0).astype(int)
    high = pred >= rules.high_demand_units
    under_p = 50 + 50 * np.clip(1 - cover / rules.understock_days, 0, 1)
    over_p = 50 * np.clip((cover - rules.overstock_days) / rules.overstock_days, 0, 1)
    priority = np.where(status == UNDER, under_p, np.where(status == OVER, over_p, 0)) + 10 * high
    action = np.select(
        [status == UNDER, status == OVER],
        [[f"Replenish about {q:,} units" for q in order_qty], [f"Reduce future orders (about {e:,} units extra)" for e in excess]],
        "Monitor demand")
    action = np.where(high & (status != UNDER), np.char.add(action.astype(str), " - high demand, watch closely"), action)
    return pd.DataFrame({
        "Status": status, "High_Demand": high, "Cover_Days": np.round(cover, 1), "Order_Qty": order_qty,
        "Excess_Units": excess, "Priority": np.round(np.minimum(priority, 100), 1), "Action": action})


def label(status: str) -> str:
    return f"{STATUS_ICONS[status]} {status}"